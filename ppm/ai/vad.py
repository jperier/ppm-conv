import queue
import json
import torch
import multiprocessing as mp

from ..logs import mp_logger
from ..worker import WorkerProcess


@WorkerProcess.register('vad')
class VADWorker(WorkerProcess):
    def __init__(
            self,
            repo_str: str = 'snakers4/silero-vad',
            model_str: str = 'silero_vad',
            window_size: int = 512,
            threshold: float = 0.3,
            n_silence: int = 3,
            sample_rate: int = 16000,
            **kwargs
    ):
        super().__init__(name='vad')
        self.repo_str = repo_str
        self.model_str = model_str
        self.window_size = window_size
        self.threshold = threshold
        self.n_silence = n_silence
        self.sample_rate = sample_rate

        self.model: torch.nn.Module | None = None
        self.last_audio_vad: bool = False   # Keeps track if last audio had high VAD score
        self.silence_count: int = 0

    def setup(self) -> None:
        # Load model
        self.model, _ = torch.hub.load(repo_or_dir=self.repo_str, model=self.model_str, verbose=False)
        self.logger.info(f'VAD model {self.repo_str}/{self.model_str} loaded.')
        self.last_audio_vad = False
        self.silence_count = 0

    def routine(self) -> None:
        data = self.get_input()

        # Non-audio commands passthrough
        if 'audio' not in data:
            self.output(data)
            return

        # Audio passes directly if not using VAD or if last score was high enough,
        # current score will still be computed to know if next sample can bypass as well
        if self.last_audio_vad:
            self.output(data)
            self.logger.debug(f'Audio bypassed VAD ({data["timestamp"]})')

        timestamp, audio = data['timestamp'], data['audio']

        # Run VAD on chunks of audio chunk
        speech_probs = []
        for i in range(0, len(audio), self.window_size):
            chunk = audio[i: i + self.window_size]
            if len(chunk) < self.window_size:
                break
            speech_prob = self.model(chunk, self.sample_rate).item()
            speech_probs.append(speech_prob)
        self.model.reset_states()  # reset model states after each audio

        # Compute score (average of the speech_probs, we may use another technique in the future)
        if len(speech_probs) > 0:
            vad_score = sum(speech_probs)/len(speech_probs)
        else:
            vad_score = 0

        if vad_score >= self.threshold:
            # Avoids sending same audio twice
            if not self.last_audio_vad:
                self.output(data)
                self.logger.debug(f'Passed with VAD score: {vad_score} ({timestamp})')
            # Saves info that current audio had high VAD score
            self.last_audio_vad = True
            self.silence_count = 0
        else:
            # Next audio will not directly pass
            self.last_audio_vad = False
            self.silence_count += 1

        # Silence detection signal
        if self.silence_count == self.n_silence:
            self.output({'command': 'conv-silence', 'timestamp': timestamp})
