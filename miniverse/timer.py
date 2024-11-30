import logging
import time
from contextlib import ContextDecorator
from dataclasses import dataclass


def timed(logger: logging.Logger | logging.LoggerAdapter):
    def decorator(function):
        def wrapper(*args, timed_log_str: str = None, **kwargs):
            time_start = time.time()
            result = function(*args, **kwargs)
            time_end = time.time()
            if timed_log_str is None:
                timed_log_str = f'Timing for function {function.__name__} and args [{args}, {kwargs}]:'
            logger.info(f'{timed_log_str} {time_end - time_start:2.3f} sec')
            return result
        return wrapper
    return decorator

@dataclass
class Timer(ContextDecorator):
    name: str
    logger: logging.Logger

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        msg_str = f'Time in seconds for {self.name}: {self.interval:.2f}'
        self.logger.info(msg_str)
