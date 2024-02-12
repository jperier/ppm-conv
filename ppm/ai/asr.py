import queue
import json
import time
import whisper
import torch
import base64
import os
from datetime import datetime, timedelta
from safetensors.torch import load
from typing import Dict, Any, Tuple, List

from ..worker import WorkerProcess


class ASRContext:
    def __init__(self, **kwargs):
        """
        TODO
         or None (placeholder)
        """
        self.audio, self.time, self.text = None, None, None

    def reset(self):
        ASRContext.__init__(self)

    def is_relevant(self, current_time: datetime):
        """
        Checks if  context is relevant.
        (True if last audio was recorded less than 2sec before current audio)
        """
        return (self.time is not None) and (current_time - self.time).seconds < 2

    def contextualize(self, audio: torch.Tensor, timestamp: datetime | None):
        """
        Use context to augment audio and get a text prefix, if relevant.
        Automatically updates context memory with provided audio and text.
        :param audio: current audio sample to transcribe.
        :param timestamp: timestamp (str) at which the audio sample was taken.
        :return: audio (torch.Tensor), prefix (None or str)
        """
        return audio, None

    def update_text(self, text: str):
        """
        Update context with last transcription. Must be from the same timestamp as
        the last audio processed by the context.
        :param text: last transcription (str)
        :return:
        """
        pass


class PrefixASRContext(ASRContext):
    def __init__(self, prefix_size_ratio: float = 0.1, prefix_text: bool = False):
        super().__init__()
        self.prefix_size_ratio = prefix_size_ratio
        self.prefix_text = prefix_text

    def contextualize(self, audio: torch.Tensor, timestamp: datetime | None):
        res, prefix = None, None

        # Prefix audio if relevant
        if self.is_relevant(timestamp):
            res = torch.cat(
                tensors=(
                    self.audio[-int(audio.shape[0]*self.prefix_size_ratio):],
                    audio),
                dim=0
            )
            prefix = self.text

        # Update memory
        self.time, self.audio = timestamp, audio

        # Return result (or input audio if context wasn't relevant)
        return (res if res is not None else audio), prefix

    def update_text(self, text: str):
        if self.prefix_text:
            self.text = text


class BufferedASRContext(ASRContext):
    def __init__(self, buffer_size: int = 10):
        super().__init__()
        self.buffer_size = buffer_size
        self.current_buff_size = 0

    def contextualize(self, audio: torch.Tensor, timestamp: datetime | None):
        res = audio

        # Contextualize audio if relevant
        if self.is_relevant(timestamp):
            if self.current_buff_size < self.buffer_size:
                if self.audio is not None:
                    res = torch.cat(tensors=(self.audio, audio), dim=0)
                self.current_buff_size += 1
            # Buffer full
            else:
                self.current_buff_size = 0  # Other attributes are taken care of just after this

        # Update memory
        self.time, self.audio = timestamp, res

        return res, None


@WorkerProcess.register("asr")
class ASRWorker(WorkerProcess):
    def __init__(
            self,
            model_size: str = 'large-v2',
            decoding_options: Dict[str, Any] = None,
            no_speech_max_prob: float = 0.7,
            get_speech_segments: bool = False,
            context_type: str | None = None,
            context_options: Dict[str, Any] = None,
            **kwargs
    ):
        super().__init__(name='asr')
        self.model_size = model_size
        self.decoding_options = decoding_options if decoding_options is not None else {}
        self.no_speech_max_prob = no_speech_max_prob
        self.get_speech_segments = get_speech_segments

        # Context
        if context_options is None:
            context_options = {}
        self.context: ASRContext = {
            "prefix": PrefixASRContext,
            "buffer": BufferedASRContext,
            None: ASRContext
        }[context_type](**context_options)

        self.model: whisper.Whisper | None = None
        self.file_start_time: datetime | None = None

    def _transcribe(
            self,
            audio: torch.Tensor,
            timestamp: datetime | None = None
    ) -> Tuple[str, List[dict]]:
        """
        Returns transcription of given audio via given model using context if timestamp is provided.
        """
        # Context
        audio, prefix = self.context.contextualize(audio, timestamp)

        if self.get_speech_segments:
            result = whisper.transcribe(
                self.model,
                audio,
                no_speech_threshold=self.no_speech_max_prob,
                **self.decoding_options
            )
            return result['text'], result['segments']

        else:
            # Preprocessing
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Update max sample len if using a context buffer
            options = self.decoding_options.copy()
            if 'sample_len' in options and type(self.context) is BufferedASRContext:
                options['sample_len'] = options['sample_len'] * (self.context.current_buff_size+1)

            # Model decoding
            options = whisper.DecodingOptions(
                prefix=prefix,
                **options
            )
            result = whisper.decode(self.model, mel, options)

            # Result and context update
            text = result.text if result.no_speech_prob <= self.no_speech_max_prob else '   '
            self.context.update_text(text)

            return text, []

    def setup(self) -> None:
        # Load model
        self.logger.debug(f'Loading model "{self.model_size}"...')
        if self.model_size in whisper._MODELS:
            self.model = whisper.load_model(self.model_size, in_memory=True)
        else:
            path = os.path.join(os.path.expanduser("~"), ".cache", "whisper", f'{self.model_size}.pt')
            self.model = whisper.load_model(path, in_memory=True)

        # Transcribing a dummy audio sample to ensure that all weights are loaded and model is truly ready
        self.logger.info('ASR model loaded, transcribing dummy sample...')
        start_time = time.time()
        _ = self._transcribe(torch.randn((16000,)))
        self.logger.debug(f'Transcription time: {time.time() - start_time}')
        self.logger.info('ASR ready.')

    def routine(self) -> None:
        data = self.get_input()

        if data['command'] in ('transcribe', 'conv'):
            # If we have file_time information, we use this as a timestamp
            if 'file_time' in data:
                # If start of file or first audio to pass VAD, we use now as start timestamp
                if data['file_time'] == 0.0 or self.file_start_time is None:
                    timestamp = datetime.now()
                    self.file_start_time = timestamp
                # We add file_time to our file_start_time otherwise
                else:
                    timestamp = self.file_start_time + timedelta(seconds=data['file_time'])
            # No file_time info, using regular timestamp
            else:
                timestamp = datetime.fromisoformat(data['timestamp'])

            # Transcribe
            start_time = time.time()
            text, segments = self._transcribe(data['audio'], timestamp)
            transcribe_time = round(time.time() - start_time, 3)
            self.logger.debug(f'Transcribed: "{text}" from {timestamp} in {transcribe_time} sec')

            del data['audio']
            data['text'] = text
            if segments:
                data['segments'] = segments

        elif data['command'] in ('conv-reset', 'conv-silence'):
            self.context.reset()
            self.file_start_time = None
            self.logger.debug('Context was reset.')

        self.output(data)
