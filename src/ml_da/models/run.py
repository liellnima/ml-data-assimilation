from __future__ import annotations

import logging
from pathlib import Path

from ml_da import DATA_DIR
from ml_da.models.base_model import BaseAssimilationModel
from ml_da.tools.config import ModelConfig, load_data_core_config
from ml_da.tools.io import load_data_bundle

logger = logging.getLogger(__name__)

# TODO separate between training and assimilation data / separate that as running


def run_model(data_id: str, model: ModelConfig) -> list[Path]:
    """Runs a single model on a single datasets."""
    # load data
    # Future: can also be adapted to run several
    data_path_matches = DATA_DIR.glob(f"*Dataset-{data_id}*")

    for data_path in data_path_matches:
        # TODO Future: rewrite so it automatically loads all ensembles
        data = load_data_bundle(data_path, ensembles=range(40))
        # TODO Future: make sure we can do this via metadata instead
        data_cfg = load_data_core_config(data_path / "configs.yaml")

        # create the model
        # TODO make a registry and handle it via that
        my_model = BaseAssimilationModel(data=data, data_cfg=data_cfg, model_cfg=model)
        print(my_model)
        print("Created the model!")
        exit(0)
        # LATER: train the model if necessary

        # run the model
        # my_model.assimilate()

        # save the results

        # return the path of the results
        # result_paths.append(result_path)

    exit(0)
