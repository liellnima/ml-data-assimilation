from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    log_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    log_dir.mkdir(exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(file_level)

    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / "run.log")
    stream_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Silence third parties
    for lib in ["jax", "jaxlib", "dabench", "zarr", "numcodecs"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
