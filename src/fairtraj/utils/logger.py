"""Logging helpers for FairTraj command-line entry points."""

import logging
from pathlib import Path


def get_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path) for handler in logger.handlers):
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
