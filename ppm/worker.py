import queue
import time
import logging
import datetime
import multiprocessing as mp
from typing import Type, TypeVar, Dict, Any, List, Union
from pprint import pprint

from ppm.logs import mp_logger


T = TypeVar("T", bound='WorkerProcess')  # This enables us to use typing with a class that will be defined later


class WorkerProcess(mp.Process):
    # Base registry, will be completed with custom classes with register()
    registry: Dict[str, Type] = {}

    @classmethod
    def register(cls: Type[T], name: str):
        """
        Register a model under a particular name.
        """
        registry = WorkerProcess.registry

        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an AssertionError if the key has already been used.
            assert name not in registry
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def from_config(
            cls,
            worker_type: str,
            worker_config: Dict[str, Any] | None = None,
            global_config: Dict[str, Any] | None = None
    ) -> T:

        if worker_config is None:
            worker_config = {}
        if global_config is None:
            global_config = {}

        config = global_config.copy()
        config.update(worker_config)    # Worker conf will override global if conflict

        # Ignore 'to' in config
        if 'to' in config:
            config.pop('to')
        try:
            return WorkerProcess.registry[worker_type](**config)
        except KeyError:
            raise KeyError(f'Could not find "{worker_type}" in registry.'
                           'Check spelling and import of corresponding module in main script.')

    def __init__(self, name: str | None = None, **kwargs) -> None:
        super().__init__(name=name)
        self.log_queue: mp.Queue | None = None
        self.logger: logging.Logger | None = None
        self.exit_event: mp.Event | None = None
        self.start_event: mp.Event | None = None
        self.ready: mp.Event | None = None
        self.done: mp.Event | None = None
        self.output_queues: List[mp.Queue] = []
        self.input_queue: mp.Queue | None = None

    def setup(self) -> None:
        pass

    def startup(self) -> None:
        pass

    def routine(self) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass

    def run(self) -> None:
        try:
            self.logger = mp_logger(self.log_queue)
            self.setup()
            self.ready.set()

            while not (self.start_event.is_set() or self.exit_event.is_set()):
                time.sleep(0.1)

            self.startup()

            while not self.exit_event.is_set():
                try:
                    try:
                        if not self.done.is_set():
                            self.routine()
                        else:
                            time.sleep(0.1)
                    except queue.Empty:
                        time.sleep(0.1)

                except KeyboardInterrupt:
                    break

        except Exception as e:
            self.logger.exception(e)

        finally:
            try:
                self.cleanup()
            except Exception as e:
                self.logger.exception(e)

    def output(self, message: Dict[str, Any]) -> None:
        for q in self.output_queues:
            q.put_nowait(message)

    def get_input(self) -> Dict[str, Any]:
        return self.input_queue.get_nowait()


@WorkerProcess.register('print')
class PrintWorker(WorkerProcess):
    def __init__(self, only_field: str = None, **kwargs) -> None:
        super().__init__(name='print')
        self.only_field = only_field

    def routine(self) -> None:
        message = self.get_input()
        self.logger.info(f'Printing {message}')
        timestamp = datetime.datetime.now().isoformat()
        if self.only_field is not None and self.only_field in message:
            print(timestamp, 'print:', message[self.only_field])
        else:
            print(timestamp, 'print:', end=' ')
            pprint(message)

