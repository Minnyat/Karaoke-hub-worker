"""Structured logging cho worker. (PR-A1)"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Logger chuẩn: level từ Config.LOG_LEVEL, format có timestamp + worker."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
