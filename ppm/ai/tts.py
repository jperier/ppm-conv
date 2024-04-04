import torch
import logging
import numpy as np
from torchaudio import transforms
from typing import List

logging.getLogger().addHandler(logging.NullHandler())   # avoid duplicate log caused by mimic or one of its dependencies
import mimic3_tts.tts as mimic

from ..worker import WorkerProcess


@WorkerProcess.register('tts')
class TTSWorker(WorkerProcess):
    def __init__(
            self,
            language: str = 'fr_FR',
            voice: str = 'fr_FR/siwis_low',
            speaker: str = 'fr_5',
            rate: float = 1.0,
            voices_directories: List[str] = None,
            sample_rate: int = 16000,
            **kwargs
    ) -> None:
        super().__init__(name='tts')
        self.settings = mimic.Mimic3Settings(
            language=language,
            voice=voice,
            voices_directories=voices_directories,
            speaker=speaker,
            rate=rate
        )
        self.sample_rate = sample_rate
        self.model: mimic.Mimic3TextToSpeechSystem | None = None

    def setup(self) -> None:
        self.model = mimic.Mimic3TextToSpeechSystem(settings=self.settings)
        self.model.preload_voice(self.settings.voice)
        self.logger.debug(f'TTS model loaded. {self.settings}')

    def routine(self) -> None:
        data = self.get_input()

        if data['command'] in ('faq', 'llm'):
            if data['command'] == 'faq':
                text = data['answer'] if data['score'] > 0.30 else 'euuuh'     # TODO move to conversation logic later
            else:
                text = data['text']
            self.logger.debug(f'Generating audio for "{text}"')

            # Model audio generation
            self.model.begin_utterance()
            self.model.speak_text(text)
            results = self.model.end_utterance()

            # Gather results
            audio_arrays = []
            for result in results:
                if isinstance(result, mimic.AudioResult):
                    audio_bytes = result.to_wav_bytes()
                    audio_arrays.append(np.frombuffer(audio_bytes, dtype=np.int16))
                else:
                    self.logger.debug(f'Unexpected result "{result}"')

            # Concatenate results, conversion to float domain & resampling
            audio = np.concatenate(audio_arrays, axis=0)
            audio = audio.astype(np.float32, order='C') / 32768.0
            audio = transforms.Resample(22050, self.sample_rate)(torch.tensor(audio)).numpy()

            self.logger.debug(f'Audio generated {audio.shape}')
            self.output({'command': 'tts', 'audio': audio})
