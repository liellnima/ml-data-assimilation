from __future__ import annotations

import logging
from pathlib import Path

from ml_da.data.generate import generate_datasets
from ml_da.tools.config import ExperimentConfig
from ml_da.tools.io import save_yaml

logger = logging.getLogger(__name__)

# NEXT
# TODO Git commit all changes
# TODO handle git conflicts

# SOON
# TODO Extend to have ensembles of models
# TODO Training Data vs Assimilation Data
# TODO Methods-Data interface (BaseModel)
# TODO DataBundle: make sure we can access covariance
# TODO DataBundle: make sure we can access Jacobian
# TODO DataBundle: make sure we have the syn numerical model (runnable!)
# TODO How to handle the syn numerical models in the whole framework
# TODO: Figure out how to sync numpy vs torch usage of our data bundle

# LATER
# TODO Tests: ... tests ...
# TODO Seeds: think carefully about where to put seeds at some point
# TODO Important: I am  installing all the dependencies of dabench atm and it causes conflicts.
# I have quick-fixed it for now, but we need to fix that issue later on.
# Option A: Install dabench without dependencies, install only the ones I need along the way (might fail though)
# Option B: Make a fork, adapt the code base and install that one
# Option C: Copy the relevant code over from their code base to our benchmark repo and maintain it there
# Along the way: Build a wrapper around the dabench code I am using

# BACK
# TODO Make PR to lab template (move __init__ to right spot for the Paths / DIRs)
# TODO error_params instead of specific error_sd --> needs re-designing though - small steps to improvement make more sense
# TODO figure out how to organize the run_dir best (probably we know once we run experiments)
# TODO overwrite param is not implemented at the moment for data generation


def run_experiment(cfg: ExperimentConfig, stage: list[str], run_dir: Path):
    """
    Run different stages of the experiment.

    Params:
        cfg (ExperimentConfig): Highest-level config file, linking to all relevant configs
        stage (list[str]): What should be run for the experiments. Can be anything of the following:
            "generate", "run", "evaluate", "aggregate", "visualize".
        run_dir (Path): The directory to which important information (logging, configs, etc) is stored.
            This is not the place where data is stored or larger objects! Just run-related information.
    """
    # Generates datasets
    if "generate" in stage:
        logger.info("Generating dataset")
        data_paths = generate_datasets(
            data_generator_cfg=cfg.data.generator,
            data_core_cfg=cfg.data.core,
        )
        # store the paths where the data is stored in the run directory
        data_paths_str = [str(dp) for dp in data_paths]
        save_yaml(data_paths_str, run_dir / "data_paths.yaml")

    # Runs data assimilation and ML models
    if "run" in stage:
        logger.info("Loading data")
        logger.info("Running model")
        raise NotImplementedError()

    # Compares results against ground truth
    if "evaluate" in stage:
        logger.info("Loading data")  # e.g. ground truth
        logger.info("Loading results")  # e.g. DA / ML model outputs
        logger.info("Evaluating outputs")
        raise NotImplementedError()

    if "aggregate" in stage:
        logger.info("Loading xxx")
        logger.info("Aggregating results across all model runs")  # aggregating results from multiple experiments
        raise NotImplementedError()

    if "visualize" in stage:
        logger.info("Loading aggregated results")
        logger.info("Visualizing results")
        raise NotImplementedError()

    logger.info("Finished run: %s", run_dir)
