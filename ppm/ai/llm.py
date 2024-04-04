import requests
import json
from typing import List, Dict, Any

from ..worker import WorkerProcess

# If running locally you may need to create SSH Tunnel
# !ssh -L 11434:127.0.0.1:11434 ppm 	# Replace "ppm" by the name in your SSH config


# Global variables
URL = 'http://localhost:11434/'
ROUTE = 'api/chat'
MODEL_ID = 'llama2:13b'
SYS_PROMPT_PATH = 'sys_prompts/prompt_context.txt'
DEFAULT_OPTIONS = {
    'temperature': 0.1,
}


@WorkerProcess.register('llm')
class LlmWorker(WorkerProcess):
    def __init__(
            self,
            ollama_url: str = 'http://localhost:11434/',
            route: str = 'api/chat',
            model_id: str = 'llama2:13b',
            sys_prompt_path: str | None = None,
            options: dict = DEFAULT_OPTIONS,
            **kwargs
    ) -> None:
        super().__init__(name='llm')
        self.ollama_url = ollama_url
        self.url = ollama_url + route
        self.model_id = model_id
        self.sys_prompt_path = sys_prompt_path
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
            # User input
            user_message = ' '.join(self.text_buffer) if isinstance(self.text_buffer, list) else self.text_buffer
            self.req_options['messages'].append({'role': 'user', 'content': user_message})

            # Bot response
            text_list = []
            with requests.Session().post(url=URL + ROUTE, json=self.req_options, stream=True) as res:
                for line in res.iter_lines():
                    if line:
                        line_data = json.loads(line)
                        if line_data['done']:
                            break
                        else:
                            text_list.append(line_data['message']['content'])
            bot_message = ''.join(text_list)

            # Output
            self.output({
                'command': 'llm',
                'user_message': user_message,
                'text': bot_message,
                'timestamp': data['timestamp']
            })

            # update history & log
            self.req_options['messages'].append({'role': 'assistant', 'content': bot_message})
            self.logger.debug(f'User message: "{user_message}"')
            self.logger.debug(f'LLM response: "{bot_message}"')

        elif data['command'] == 'conv-reset':
            self.req_options['messages'] = [{'role': 'system', 'content': self.sys_prompt}]
