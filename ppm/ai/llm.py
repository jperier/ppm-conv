import requests
import json
import re
from typing import List, Dict, Any

from ..worker import WorkerProcess

# If running locally you may need to create SSH Tunnel
# !ssh -L 11434:127.0.0.1:11434 ppm 	# Replace "ppm" by the name in your SSH config


# Global variables
URL = 'http://localhost:11434/'
ROUTE = 'api/chat'
MODEL_ID = 'llama2:13b'
SYS_PROMPT_PATH = 'sys_prompts/prompt_context_faq.txt'
DEFAULT_OPTIONS = {
    'temperature': 0.1,
}


def remove_emojis(string):
    emoj = re.compile('['
                      u'\U0001F600-\U0001F64F'  # emoticons
                      u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                      u'\U0001F680-\U0001F6FF'  # transport & map symbols
                      u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                      u'\U00002500-\U00002BEF'  # chinese char
                      u'\U00002702-\U000027B0'
                      u'\U000024C2-\U0001F251'
                      u'\U0001f926-\U0001f937'
                      u'\U00010000-\U0010ffff'
                      u'\u2640-\u2642'
                      u'\u2600-\u2B55'
                      u'\u200d'
                      u'\u23cf'
                      u'\u23e9'
                      u'\u231a'
                      u'\ufe0f'  # dingbats
                      u'\u3030'
                      ']+', re.UNICODE)
    return re.sub(emoj, '', string)


@WorkerProcess.register('llm')
class LlmWorker(WorkerProcess):
    def __init__(
            self,
            ollama_url: str = 'http://localhost:11434/',
            route: str = 'api/chat',
            model_id: str = 'llama2:13b',
            sys_prompt_path: str | None = None,
            intermediate_sys_prompt: str | None = None,
            options: dict = DEFAULT_OPTIONS,
            **kwargs
    ) -> None:
        super().__init__(name='llm')
        self.ollama_url = ollama_url
        self.url = ollama_url + route
        self.model_id = model_id
        self.sys_prompt_path = sys_prompt_path
        self.intermediate_sys_prompt = intermediate_sys_prompt
        self.options = options

        self.sys_prompt: str = ''
        self.req_options: Dict[str, Any] = {}
        self.text_buffer: List[str] = []

    def setup(self) -> None:
        # TODO add initial connection so that model is loaded at the start of the interaction
        # TODO maybe keep a ping to ollama so that the model will stay loaded
        if self.sys_prompt_path is not None:
            with open(self.sys_prompt_path, 'r') as f:
                self.sys_prompt = f.read()

        # Connection check
        res = requests.get(self.ollama_url)
        if not res.ok:
            raise ConnectionError('Could not connect to ollama :', res.text)

        self.req_options = {
            'model': self.model_id,
            'options': self.options,
            'messages': [{'role': 'system', 'content': self.sys_prompt}]
        }

    def routine(self) -> None:
        data = self.get_input()

        if data['command'] == 'conv':
            if data.get('asr_context_type') == 'buffer':
                self.text_buffer = data['text']
            else:
                self.text_buffer.append(data['text'])

        elif data['command'] == 'conv-silence':
            user_message = ' '.join(self.text_buffer) if isinstance(self.text_buffer, list) else self.text_buffer
            if user_message.strip():
                # User input
                self.req_options['messages'].append({'role': 'user', 'content': user_message})

                # Add intermediate system prompt (asks to be concise usually)
                if self.intermediate_sys_prompt is not None:
                    self.req_options['messages'].append({'role': 'system', 'content': self.intermediate_sys_prompt})

                # Bot response
                current_sentence = []
                bot_message = []
                with requests.Session().post(url=URL + ROUTE, json=self.req_options, stream=True) as res:
                    for line in res.iter_lines():
                        if line:
                            line_data = json.loads(line)
                            if line_data['done']:
                                break
                            else:
                                text = line_data['message']['content']
                                current_sentence.append(text)
                                # If a sentence is over, sending it
                                if any(char in text for char in ('.', '!', '?')):
                                    sentence = ''.join(current_sentence)
                                    self.send_chunk(sentence, data['timestamp'])
                                    bot_message.append(sentence)
                                    current_sentence = []
                    if len(current_sentence) > 0:
                        sentence = ''.join(current_sentence)
                        if sentence.strip():
                            self.send_chunk(sentence, data['timestamp'])
                            bot_message.append(sentence)

                bot_message = ''.join(bot_message)

                # update history & log
                self.req_options['messages'].append({'role': 'assistant', 'content': bot_message})
                self.logger.debug(f'User message: "{user_message}"')
                self.logger.debug(f'LLM response: "{bot_message}"')

        elif data['command'] == 'conv-reset':
            self.req_options['messages'] = [{'role': 'system', 'content': self.sys_prompt}]

    def send_chunk(self, text: str, timestamp: str) -> None:
        # Remove special tokens & emojis
        to_remove = ('*smile*', '*giggle*', '*wink*',
                     '[Inst', '[Inst]', '[/Inst]', '\n')
        for token in to_remove:
            text = text.replace(token, '')
        text = remove_emojis(text)

        # Output
        self.output({
            'command': 'llm',
            'text': text,
            'timestamp': timestamp
        })
