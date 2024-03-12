import queue
import time
import os
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from torchaudio import transforms
from datetime import datetime
from typing import Union

from ..worker import WorkerProcess


@WorkerProcess.register('device_stream')
class DeviceStreamWorker(WorkerProcess):
    def __init__(
            self,
            src_device: str = "hw:0,0",
            audio_format: str = "alsa",
            segment_length: int = 16000,
            sample_rate: int = 16000,
            command_mode: str = 'conv',
            **kwargs
    ):
        super().__init__(name='device_worker')
        self.src_device = src_device
        self.audio_format = audio_format
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.command = command_mode
        self.stream_iterator = None
        self.streamer = None

    def startup(self) -> None:
        # Build streamer
        self.streamer = torchaudio.io.StreamReader(self.src_device, format=self.audio_format)
        self.streamer.add_basic_audio_stream(frames_per_chunk=self.segment_length, sample_rate=self.sample_rate)
        self.stream_iterator = self.streamer.stream(timeout=-1, backoff=1.0)
        # Log stream infos
        self.logger.debug(self.streamer.get_src_stream_info(0))
        self.logger.debug(self.streamer.get_out_stream_info(0))

    def routine(self) -> None:
        # Get next chunk
        (chunk,) = next(self.stream_iterator)
        # Convert to mono
        with torch.no_grad():
            chunk = chunk.T.mean(dim=0)
        # Output
        timestamp = datetime.now().isoformat()
        self.output({'command': self.command, 'timestamp': timestamp, 'audio': chunk})


@WorkerProcess.register('audio_io')
class AudioIOWorker(WorkerProcess):
    def __init__(
            self,
            device: str = "hw:0,0",
            segment_length: int = 16000,
            sample_rate: int = 16000,
            command_mode: str = 'conv',
            **kwargs
    ):
        super().__init__(name='audio_io')
        self.device = device
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.command = command_mode

        self.stream: sd.Stream | None = None

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status) -> None:
        """
        Callback function called by the sounddevice.Stream.
        """
        if status:
            self.logger.warning((str(status)))

        # Device input to worker output
        timestamp = datetime.now().isoformat()
        self.output({
            'command': self.command,
            'timestamp': timestamp,
            'audio': torch.tensor(np.squeeze(indata), requires_grad=False)
        })

        # Worker input to device output
        outdata.fill(0.)
        try:
            data = self.get_input()
            if data.get('command') == 'conv':
                audio = data.get('audio')
                if type(audio) not in (np.ndarray, torch.Tensor):
                    self.logger.warning(f'Bad audio received: {data}')
                else:
                    audio = np.expand_dims(audio, axis=1)
                    if audio.shape[0] < outdata.shape[0]:
                        outdata[:audio.shape[0], :audio.shape[1]] = audio
                    else:
                        outdata[:] = audio
        except queue.Empty:
            pass

    def setup(self) -> None:
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.segment_length,
            device=self.device,
            channels=1,
            callback=self._callback,
            #finished_callback: Any = None
        )

    def routine(self) -> None:
        if not self.stream.active:
            self.stream.start()
            self.logger.info('Device stream started.')
        time.sleep(0.5)

    def cleanup(self) -> None:
        self.stream.abort()


@WorkerProcess.register('file_stream')
class FileStreamWorker(WorkerProcess):
    def __init__(
            self,
            path: str,
            sample_rate: int = 16000,
            segment_length: Union[int, None] = None,
            pause_time: float = 0.5,
            use_file_time: bool = True,
            command_mode: str = 'transcribe',
            **kwargs
    ):
        super().__init__(name='file_stream')
        self.files = []
        self.current_file_index = 0
        self.pause_time = pause_time
        self.path = path
        self.sample_rate = sample_rate
        self.segment_length = sample_rate if segment_length is None else segment_length
        self.use_file_time = use_file_time
        self.command = command_mode

    def setup(self) -> None:
        # Check file(s)
        if os.path.isfile(self.path):
            self.files = [self.path]
        elif os.path.isdir(self.path):
            self.files = [
                os.path.join(self.path, file)
                for file in os.listdir(self.path)
                if os.path.isfile(os.path.join(self.path, file)) and (file.endswith('.wav') or file.endswith('.mp3'))
            ]
        else:
            raise FileNotFoundError(f'Invalid file or directory {self.path}')

        self.logger.info(f'Going through files: {self.files}')

    def routine(self) -> None:
        # Get next file
        if self.current_file_index < len(self.files):
            file = self.files[self.current_file_index]

            self.logger.info(f'Loading audio file {file}')
            try:
                audio, old_sample_rate = torchaudio.load(file)
            except Exception as e:
                self.logger.warning(f'Could not load file {file}, skipping it. ({e})')
                self.current_file_index += 1
                return

            # Resample if necessary
            if old_sample_rate != self.sample_rate:
                audio = transforms.Resample(old_sample_rate, self.sample_rate)(audio)

            # Convert to mono and/or format tensor
            with torch.no_grad():
                if audio.shape[0] == 2:
                    audio = audio.mean(dim=0)
                else:
                    audio = torch.squeeze(audio)

            # Iterate through file
            n_iters = audio.shape[0] // self.segment_length
            if audio.shape[0] % self.segment_length:
                n_iters += 1

            self.logger.info(f'Streaming file {file}...')
            for i in range(n_iters):
                chunk = audio[i*self.segment_length:(i+1)*self.segment_length]

                self.output({
                    'command': self.command,
                    'timestamp': datetime.now().isoformat(),
                    'audio': chunk,
                    'file': file,
                    'file_time': 0.0 if i == 0 else i*self.segment_length/self.sample_rate
                })
                time.sleep(self.pause_time)     # Wait so that we do not overflow the queue too much

            self.current_file_index += 1
            # Send signal to reset conversation context
            self.output({'command': 'conv-reset', 'timestamp': datetime.now().isoformat()})

        else:
            self.done.set()
