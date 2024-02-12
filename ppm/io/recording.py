import os
import torch
import torchaudio

from ..worker import WorkerProcess


@WorkerProcess.register("recording")
class RecordingWorker(WorkerProcess):
    def __init__(
            self,
            sample_rate: int = 16000,
            buffer_size: int = 60,
            **kwargs
    ):
        super().__init__(name='recording')
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.chunks = None
        self.timestamp = None

    def setup(self) -> None:
        self.chunks = []
        self.timestamp = ""
        if not (os.path.exists('recording') and os.path.isdir('recording')):
            os.mkdir('recording')
        self.logger.debug('setup done')

    def _save_chunks(self):
        # Saving to file
        audio = torch.unsqueeze(torch.cat(self.chunks, 0), 0)
        audio_path = f'recording/{self.timestamp}.wav'
        torchaudio.save(audio_path, audio, self.sample_rate, channels_first=True)
        self.logger.debug(f'Saved audio in {audio_path}.')
        # Free memory
        del self.chunks[:]

    def routine(self) -> None:
        # Get chunk and save timestamp if this is the first chunk of the buffer
        data = self.get_input()
        if len(self.chunks) == 0:
            self.timestamp = data['timestamp']
        self.chunks.append(data['audio'])

        # Save in a file when buffer is full
        if len(self.chunks) >= self.buffer_size:
            self._save_chunks()

    def cleanup(self) -> None:
        if len(self.chunks) > 0:
            self._save_chunks()
