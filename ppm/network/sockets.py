import queue
import time

import websockets
import websockets.sync.client
import websockets.sync.server
import json
import base64
import threading
from datetime import datetime
from cryptography.fernet import Fernet
from safetensors.torch import save, load

from ..worker import WorkerProcess
from ..logs import mp_logger


@WorkerProcess.register('socket_client')
class NetworkClientWorker(WorkerProcess):
    def __init__(
            self,
            host: str = '127.0.0.1',
            port: int = 8080,
            key: str = None,
            **kwargs
    ):
        super().__init__(name='socket_client')
        self.host = host
        self.port = port
        self.f = Fernet(key.encode("ascii")) if key is not None else None

        self.websocket: websockets.sync.client.ClientConnection | None = None

    def setup(self) -> None:
        self.websocket = websockets.sync.client.connect(f"ws://{self.host}:{self.port}")
        self.logger.info(f'Server connection established, at {self.host}:{self.port}')

    def routine(self) -> None:
        # Message OUT
        try:
            data = self.get_input()
            # Audio
            if 'audio' in data:
                data['audio'] = base64.b64encode(save({'audio': data['audio']})).decode('utf-8')
                message = json.dumps(data).encode('utf-8')
            # Command
            elif 'command' in data:
                message = json.dumps(data).encode('utf-8')
            # Bad type
            else:
                raise Exception(f'Bad data format sent to client_routine, {data}')
            # Encrypt and send
            if self.f is not None:
                message = self.f.encrypt(message)
            self.websocket.send(message)
        # No message in queue
        except queue.Empty:
            pass

        # Message IN
        try:
            data = self.websocket.recv(0)
            if self.f is not None:
                data = self.f.decrypt(data)
            data = json.loads(data.decode('utf-8'))
            self.output(data)
        # No message from server
        except TimeoutError:
            pass

    def cleanup(self) -> None:
        if self.websocket is not None:
            self.websocket.close()


@WorkerProcess.register('socket_server')
class NetworkServerWorker(WorkerProcess):
    def __init__(
            self,
            host: str = '127.0.0.1',
            port: int = 8080,
            key: str = None,
            **kwargs
    ):
        super().__init__(name='socket_server')
        self.host = host
        self.port = port
        self.f = Fernet(key.encode("ascii")) if key is not None else None
        self.handler_running = False

    def run(self) -> None:
        try:
            self.logger = mp_logger(self.log_queue)

            # Launching server
            self.logger.debug('Launching socket server')
            server = websockets.sync.server.serve(self._client_handler, self.host, self.port)
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.start()
            self.ready.set()
            self.logger.info(f'Server ready on {self.host}:{self.port}')

            while not self.exit_event.is_set():
                try:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    server.shutdown()
                    server_thread.join(timeout=1)
        except Exception as e:
            self.logger.exception(e)

    def _client_handler(self, websocket):
        start_time = datetime.now().isoformat()

        if self.handler_running:
            self.logger.warning('Another handler already running !')
            return
        else:
            self.logger.info(f'New client handler running for {websocket.remote_address}')

        self.handler_running = True

        # Looping until KeyboardInterrupt, exit_event or client closed connection
        try:
            while not self.exit_event.is_set():
                try:
                    data, binary = False, False

                    # Message IN
                    try:
                        data = websocket.recv(0)
                        if self.f is not None:
                            data = self.f.decrypt(data)
                        data = json.loads(data.decode('utf-8'))

                        # load tensor
                        if 'audio' in data:
                            data['audio'] = load(base64.b64decode(data['audio'].encode('utf-8')))['audio']

                        if data['command'] in ('transcribe', 'conv', 'conv-reset', 'conv-silence'):
                            self.output(data)
                    except TimeoutError:
                        pass

                    # Message OUT
                    try:
                        data = self.get_input()
                        if data['timestamp'] > start_time:  # Ignoring messages from previous connection
                            binary = json.dumps(data).encode('utf-8')
                            if self.f is not None:
                                binary = self.f.encrypt(binary)
                            websocket.send(binary)
                    except queue.Empty:
                        pass

                    # Sleep if no messages waiting on both sides
                    if not (data or binary):
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    break
        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info('Connection closed OK.')
        except Exception as e:
            self.logger.exception(e)
        finally:
            # Clear queues from current client's messages
            count = 0
            for q in (*self.output_queues, self.input_queue):
                while True:
                    try:
                        q.get_nowait()
                        count += 1
                    except queue.Empty:
                        break
            self.logger.debug(f'Cleaned {count} messages from queues.')
            self.handler_running = False
