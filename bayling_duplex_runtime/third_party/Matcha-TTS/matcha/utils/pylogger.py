import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)

