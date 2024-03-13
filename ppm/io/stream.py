import queue
import time
import os
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from queue import Queue
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
            device_blocksize: int = 1000,
            workers_audio_chunk_size: int = 16000,      # TODO share between workers
            sample_rate: int = 16000,
            command_mode: str = 'conv',
            **kwargs
    ):
        # Validations
        if workers_audio_chunk_size % device_blocksize != 0:
            raise ValueError(f"Workers' audio chunk size {workers_audio_chunk_size} should be a multiple of "
                             f"Device blocksize {device_blocksize}")
        if sample_rate % device_blocksize != 0:
            raise ValueError(f"Sample rate {sample_rate} should be a multiple of  device blocksize {device_blocksize}")

        super().__init__(name='audio_io')
        self.device = device
        self.device_blocksize = device_blocksize
        self.workers_audio_chunk_size = workers_audio_chunk_size
        self.sample_rate = sample_rate
        self.command = command_mode

        self.device_input_buffer = Queue()      # sound from input device
        self.device_output_buffer = Queue()     # sound to be played on device
        self.worker_output_buffer = []          # sound chunks coming from input device to be sent to next worker(s)

        self.stream: sd.Stream | None = None

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status) -> None:
        """
        Callback function called by the sounddevice.Stream.
        Uses Worker buffers that are updated in self.routine function.
        """
        if status:
            self.logger.warning((str(status)))

        self.device_input_buffer.put(indata[:])

        try:
            outdata[:] = self.device_output_buffer.get_nowait()
        except queue.Empty:
            outdata.fill(0.)

    def setup(self) -> None:
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.device_blocksize,
            device=self.device,
            channels=1,
            callback=self._callback
        )

    def startup(self) -> None:
        self.stream.start()
        self.logger.info(f'Device stream started.')

    def routine(self) -> None:
        # Device input to worker output
        try:
            # Get from buffer
            self.worker_output_buffer.append(
                np.squeeze(self.device_input_buffer.get_nowait()).copy())

            # Send to other workers if buffer has sufficient size
            if len(self.worker_output_buffer)*self.device_blocksize == self.workers_audio_chunk_size:
                data = np.concatenate(self.worker_output_buffer, axis=0)
                self.worker_output_buffer = []

                self.output({
                    'command': self.command,
                    'timestamp': datetime.now().isoformat(),
                    'audio': torch.tensor(data, requires_grad=False)
                })
        except queue.Empty:
            pass

        # Worker input to device output
        try:
            data = self.get_input()

            if data.get('command') == 'conv':
                # Data validation
                audio = data.get('audio')
                if type(audio) is torch.Tensor:
                    audio = audio.numpy()
                elif type(audio) is not np.ndarray:
                    raise ValueError(f'Bad audio received: {data}')

                # Processing & sending audio to device
                audio = np.expand_dims(audio, axis=1)
                out_arrays = []
                # Audio smaller than blocksize
                if audio.shape[0] < self.device_blocksize:
                    outdata = np.zeros((self.device_blocksize, 1))
                    outdata[:audio.shape[0], :audio.shape[1]] = audio   # TODO remove audio shape 1 ?
                    out_arrays.append(outdata)
                # Audio bigger than blocksize
                elif audio.shape[0] > self.device_blocksize:
                    # Go over audio
                    for i in range(0, audio.shape[0] // self.device_blocksize):
                        j = i * self.device_blocksize
                        out_arrays.append(audio[j:j+self.device_blocksize])
                    # Remaining with zero padding TODO is this problematic ?
                    n_remaining_frames = audio.shape[0] % self.device_blocksize
                    if n_remaining_frames > 0:
                        last_chunk = np.zeros((self.device_blocksize, 1))
                        last_chunk[:n_remaining_frames] = audio[-n_remaining_frames:]
                        out_arrays.append(last_chunk)
                # Exact size
                else:
                    out_arrays.append(audio)

                # Push to queue
                for out_array in out_arrays:
                    self.device_output_buffer.put(out_array)

        except queue.Empty:
            pass

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
