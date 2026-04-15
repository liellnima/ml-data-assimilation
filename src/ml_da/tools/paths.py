from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ml_da import RUN_DIR
from ml_da.tools.config import ExperimentConfig

logger = logging.getLogger(__name__)


def make_run_dir(cfg: ExperimentConfig) -> Path:
    """
    Create a specific run dir for the experiment we want to run.

    Params:
        cfg (ExperimentConfig): the configs used to create the run dir
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.experiment_name}_{timestamp}"
    run_dir = RUN_DIR / run_name

    if run_dir.exists() and not cfg.output.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. " "Set output.overwrite=true or choose a different run_name."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    logger.info("Created run directory at %s", run_dir)

    # (run_dir / "artifacts").mkdir(exist_ok=True)
    # (run_dir / "figures").mkdir(exist_ok=True)
    # (run_dir / "metrics").mkdir(exist_ok=True)
    return run_dir
