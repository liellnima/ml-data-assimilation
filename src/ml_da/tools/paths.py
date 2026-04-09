from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ml_da import OUTPUT_DIR
from ml_da.tools.config import ExperimentConfig


def make_run_dir(cfg: ExperimentConfig) -> Path:
    """
    Create a run dir for the experiment we want to run.

    Params:
        cfg (ExperimentConfig): the configs used to create the run dir
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.experiment_name}_{timestamp}"
    run_dir = OUTPUT_DIR / run_name

    if run_dir.exists() and not cfg.output.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. " "Set output.overwrite=true or choose a different run_name."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)


# TODO make_data_dir
