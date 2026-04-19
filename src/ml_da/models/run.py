from __future__ import annotations

import logging
from pathlib import Path

import ml_da.models.da_methods  # noqa: F401
from ml_da import DATA_DIR, OUTPUT_DIR
from ml_da.tools.config import ModelConfig, load_data_core_config
from ml_da.tools.io import load_data_bundle, save_yaml
from ml_da.tools.registry import (
    DA_METHOD_REGISTRY,
)

logger = logging.getLogger(__name__)

# TODO separate between training and assimilation data / separate that as running


def run_model(data_id: str, model_cfg: ModelConfig) -> Path:
    """Runs a single model on a single datasets."""
    # load data
    # Future: can also be adapted to run several
    data_path_matches = list(DATA_DIR.glob(f"*Dataset-{data_id}*"))

    if len(data_path_matches) != 1:
        raise RuntimeError("IDs of Datasets were not unique. Expecting unique ids though.")

    data_path = data_path_matches[0]

    # TODO Future: rewrite so it automatically loads all ensembles
    data = load_data_bundle(data_path, ensembles=range(40))
    # TODO Future: make sure we can do this via metadata instead
    data_cfg = load_data_core_config(data_path / "configs.yaml")

    # create the model
    # TODO make a registry and handle it via that
    my_model = DA_METHOD_REGISTRY[model_cfg.name](
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        data=data,
    )

    print(my_model)
    print("Created the model!")
    exit(0)
    # LATER: train the model if necessary

    # run the model
    metrics_dict, run_time = my_model.assimilate()

    # save the results
    results_path = OUTPUT_DIR / "results" / data_path.parts[-1] / f"{model_cfg.name}.yaml"
    results_dict = {
        "metrics": metrics_dict,
        "run_time": run_time,
    }
    save_yaml(results_dict, results_path)

    # return the path of the results
    return results_path
