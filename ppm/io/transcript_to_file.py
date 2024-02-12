import os
import json
from io import TextIOWrapper
from datetime import datetime

from ppm.worker import WorkerProcess


@WorkerProcess.register('transcript_to_file')
class TranscriptToFileWorker(WorkerProcess):
    def __init__(
            self,
            save_dir: str = 'transcript/',
            save_details: bool = False,
            **kwargs
    ) -> None:
        super().__init__(name='trancript_to_file')
        self.save_dir = save_dir
        self.save_details = save_details
        self.last_audio_file = None
        self.details = []
        self.save_details_count = 0

        self.file: TextIOWrapper | None = None
        self.start_time: str | None = None

    def setup(self) -> None:
        if not (os.path.exists(self.save_dir) and os.path.isdir(self.save_dir)):
            self.logger.info(f'Creating {self.save_dir}')
            os.makedirs(self.save_dir)

        self.start_time = datetime.now().isoformat()
        self.file = open(os.path.join(self.save_dir, f'{self.start_time}.txt'), 'w')

    def _save_details(self) -> None:
        with open(os.path.join(self.save_dir, f'{self.start_time}-details-{self.save_details_count}.json'), 'w') as f:
            json.dump(self.details, f)
        self.details = []
        self.save_details_count += 1

    def routine(self) -> None:
        data = self.get_input()
        now = datetime.now().isoformat().split('T')[-1]

        if data['command'] == 'transcribe':

            if 'file' in data and data['file'] != self.last_audio_file:
                self.file.write(f'\n - File: {data["file"]}\n')
                self.last_audio_file = data['file']

            self.file.write(
                f"{data['timestamp'].split('T')[-1]} - {now}:  {data['text']}\n")

        else:
            self.file.write(
                f"{data['timestamp'].split('T')[-1]} - {now}: * 'command': {data['command']} *\n")

        if self.save_details:
            self.details.append(data)
            if len(self.details) > 100:
                self._save_details()

    def cleanup(self) -> None:
        self.file.close()
        if self.save_details and len(self.details) > 0:
            self._save_details()
