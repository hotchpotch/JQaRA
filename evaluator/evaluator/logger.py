import logging
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger():
    # Singleton logger
    logger = logging.getLogger("JQaRA")
    logger.setLevel(logging.INFO)
    return logger
