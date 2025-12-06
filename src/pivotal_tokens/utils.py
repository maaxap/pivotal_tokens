import logging
from uuid import uuid1


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def generate_unique_id() -> str:
    return str(uuid1())
