import logging
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger():
    # Singleton logger
    logger = logging.getLogger("JQaRA")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
