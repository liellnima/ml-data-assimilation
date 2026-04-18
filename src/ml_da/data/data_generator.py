from __future__ import annotations

import logging

import ml_da.data.dynamical_models  # noqa: F401
import ml_da.data.observers  # noqa: F401
import ml_da.data.systems  # noqa: F401
from ml_da.data.dataclasses import AssimDataBundle
from ml_da.tools.config import (
    DataCoreConfig,
    DynamicalModelConfig,
    ObserverConfig,
    SystemConfig,
)
from ml_da.tools.io import prepare_for_yaml
from ml_da.tools.registry import (
    DYNAMICAL_MODEL_REGISTRY,
    OBSERVER_REGISTRY,
    SYSTEM_REGISTRY,
)
from ml_da.tools.utils import str_join_ls

logger = logging.getLogger(__name__)


class DataGenerator:
    """Class that generates the data for one particular setting."""

    def __init__(
        self,
        data_cfg: DataCoreConfig,
        sys_cfg: SystemConfig,
        dyn_cfg: DynamicalModelConfig,
        obs_cfg: ObserverConfig,
    ) -> None:
        """Init all relevant params."""
        self.seed = data_cfg.seed
        self.timesteps = data_cfg.timesteps

        # initialize all objects with the relevant configs
        self.system = SYSTEM_REGISTRY[sys_cfg.name](sys_cfg)
        self.dynamical_model = DYNAMICAL_MODEL_REGISTRY[dyn_cfg.name](dyn_cfg, sys_cfg)
        self.observer = OBSERVER_REGISTRY[obs_cfg.name](obs_cfg)

    def get_id_name(self) -> str:
        """Returns a reasonable name, for example for a directory."""
        all_relevant_params = [
            str_join_ls(["SEED", self.seed]),
            str_join_ls(["TIME", self.timesteps]),
            self.system.get_id_str(),
            self.dynamical_model.get_id_str(),
            self.observer.get_id_str(),
        ]
        return "_".join(all_relevant_params)

    # TODO separate between Training and Assimilation Data
    # split this up into two functions (need to be run on top of each other)
    # TODO figure out if it is useful to store states of numerical models
    # after training data generation, so assimilation can be picked up from that state
    def generate(self) -> AssimDataBundle:
        """
        Runs the data generator.

        Returns:
            AssimDataBundle: Contains `truth`, `model`, `observations`, and `metadata`.
        """
        if self.system.name is None:
            raise ValueError("Underlying system for the data generation has not been initialized.")

        # Ground Truth: The System's Data
        logger.info("Generating ground truth ...")
        ground_truth = self.system.generate_ground_truth(n_steps=self.timesteps)

        # Synthetic Model: The Numerical Model Approximating the System
        logger.info("Generating numerical model data ...")
        model_data = self.dynamical_model.generate_model_data(n_steps=self.timesteps)

        # Synthetic Observations: What we can measure of the system
        logger.info("Generating observations ...")
        observations = self.observer.generate_observations(ground_truth)

        # return generated data
        return AssimDataBundle(
            truth=ground_truth,
            dynamical_model=model_data,
            observations=observations,
            metadata={
                "generator": prepare_for_yaml(self.__dict__),
                "system": prepare_for_yaml(self.system.__dict__),
                "dynamical_model": prepare_for_yaml(self.dynamical_model.__dict__),
                "observer": prepare_for_yaml(self.observer.__dict__),
            },
        )
