import sys
import os
import logging
import logging.handlers
import multiprocessing as mp


def mp_logger(log_q: mp.Queue, name: str = None):
    qh = logging.handlers.QueueHandler(log_q)
    logger = logging.getLogger(mp.current_process().name if name is None else name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)
    return logger


class MainAndWarnFilter(logging.Filter):
    def __init__(self, debug: bool = False):
        super().__init__()
        self.loglevel = logging.DEBUG if debug else logging.WARNING

    def filter(self, record):
        return record.processName == 'MainProcess' or record.levelno >= self.loglevel


def logger_process(log_q: mp.Queue, debug: bool = False):
    logger = logging.getLogger('ppm')

    if not (os.path.exists('logs') and os.path.isdir('logs')):
        os.makedirs('logs')

    # Workers
    h = logging.handlers.RotatingFileHandler('logs/log.log', 'a', 30000, 2)
    f = logging.Formatter('%(asctime)s - %(processName)-15s - %(levelname)-6s - %(message)s')
    h.setFormatter(f)
    logger.addHandler(h)

    # MainProcess and warnings
    h = logging.StreamHandler(stream=sys.stdout)  # Add console out
    h.setFormatter(logging.Formatter('%(asctime)s - %(processName)-10s - %(levelname)-5s - %(message)s'))
    h.addFilter(MainAndWarnFilter(debug))
    logger.addHandler(h)

    while True:
        try:
            record = log_q.get()
            if record is None:
                break
            logger.handle(record)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception(e)
