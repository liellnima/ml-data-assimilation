import logging
from pathlib import Path


def setup_logging(run_dir: Path, level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
