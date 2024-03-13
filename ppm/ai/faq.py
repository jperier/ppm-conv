import yaml
import logging
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from ..worker import WorkerProcess

# Remove warning from transformers
logging.getLogger('transformers').setLevel(logging.ERROR)
# Warning from PyTorch
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


@WorkerProcess.register('faq')
class FAQWorker(WorkerProcess):
    def __init__(
            self,
            model_str: str = 'sentence-transformers/use-cmlm-multilingual',
            faq_path: str = 'configs/faq/faq.yml',
            **kwargs
    ) -> None:
        super().__init__(name='faq')
        self.faq_path = faq_path
        self.model_str = model_str

        self.text_buffer: List[str] = []
        self.model: SentenceTransformer | None = None
        self.faq: List[dict] = []
        self.questions: List[str] = []
        self.answer_index: List[int] = []
        self.Q: np.ndarray | None = None

    def setup(self) -> None:
        # Load file
        with open(self.faq_path) as f:
            self.faq = yaml.safe_load(f)
        self.logger.info(f'FAQ loaded, {len(self.faq)} QA entries.')
        # Extract all questions and their indexes
        self.questions, self.answer_index = [], []
        for i, qa in enumerate(self.faq):
            self.questions.extend(qa['questions'])
            self.answer_index.extend([i] * len(qa['questions']))

        # Loading model
        self.model = SentenceTransformer(self.model_str)
        self.logger.info(f'Model loaded, {self.model_str}')

        # Pre-computing embedding for all questions in FAQ
        self.Q = self.model.encode(self.questions)
        self.logger.info(f"Questions' embeddings precomputed.")

    def routine(self) -> None:
        data = self.get_input()

        if data['command'] == 'conv':
            self.text_buffer.append(data['text'])

        elif data['command'] == 'conv-silence':
            # Build text & empty buffer
            text = ' '.join(self.text_buffer)
            self.text_buffer = []
            self.logger.debug(f'Text: {text}')
            # Compute embeddings and similarity scores
            x = self.model.encode(text)
            scores = np.squeeze(x @ self.Q.T)
            # Get best candidate
            max_idx = np.argmax(scores)
            ans_idx = self.answer_index[max_idx]
            self.logger.info(f'Best candidate: "{self.questions[max_idx]}", score: {scores[max_idx]}')

            self.output({
                'command': 'faq',
                'answer': self.faq[ans_idx]['answer'],
                'score': scores[max_idx],
                'timestamp': data['timestamp']
            })

        elif data['command'] == 'conv-reset':
            self.text_buffer = []

        else:
            pass
