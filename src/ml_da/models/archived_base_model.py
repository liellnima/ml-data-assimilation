from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.tools.config import ModelConfig

# TODO this template needs to be adapted (probably signficantly) to match with our models and data

logger = logging.getLogger(__name__)


# MOVE THIS INTO DA BASE MODEL
# Return the noise covariance Q
# --> different ways to do this, for now:
# return based on the DataBundle model noise
# LATER: estimate this from the model, assume some type of noise,
# that's not necessarily true

# Return initial covariance P0
# --> different ways to do this, for now
# return based on the DataBundle init model noise (perturbations)
# LATER: estimate this from an ensemble or via climatology / tune it

# Return observation covariance R


# Get Jacobian of observations
# --> get it directly from loaded observations (and its metadata)
#  depends on if stationary_observerser=True or false
# same for all obs if stationary, different if non-stationary
class BaseAssimilationModel(ABC):
    """Base Model Class for both traditional and ML-based data assimilation methods."""

    def __init__(self, cfg: ModelConfig, data: AssimDataBundle) -> None:
        self.name = cfg.name
        self._requires_training = cfg.requires_training
        # TODO unpacking params in individual class
        self.params = cfg.params

        # TODO extract necessary data
        print(data.metadata)
        exit(0)
        # TODO covariance R: observation error
        # TODO covariance Q: model error
        # TODO initial covariance of state P0: perturbation error
        # can later be estimated differently

        # TODO Jacobian of observations for whole time series
        # because we assume stationary_observer=True --> raise not implemented error if false

        # TODO load the model, with return_tlm=True

    @property
    def requires_training(self) -> bool:
        return self._requires_training

    @property
    def is_sequential(self) -> bool:
        return True

    def fit(self, train_data: dict[str, Any]) -> None:
        """
        Optional method for ML models that need offline training.

        Traditional methods can skip that one.
        """
        if self.requires_training:
            raise NotImplementedError(f"{self.__class__.__name__} requires fit() to be implemented.")

    # TODO define what it gets and what it returns
    @abstractmethod
    def initialize(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        Initializes internal state before sequential assimilation.

        Returns:
            internal_state (...) TODO
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, state: dict[str, Any], batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Executes one assimilation step.

        Returns:
            updated_state (...) TODO
            step_output (...) TODO diagnostics, etc, and other results
        """
        raise NotImplementedError

    def run(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """
        Default full-trajectory execution.

        Works for sequential methods. Non-sequential models can override it.
        """
        state = self.initialize(dataset)
        outputs = []

        for batch in self.iter_batches(dataset):
            state, step_output = self.step(state, batch)
            outputs.append(step_output)

        return self.collate_outputs(outputs)

    def iter_batches(self, dataset: dict[str, Any]):
        """
        Default iterator over time steps.

        Assumes dataset contains arrays with timesteps.
        """
        observations = dataset["observations"]
        model_states = dataset["model"]
        true_state = dataset.get("true_state")

        for t in range(len(observations)):
            batch = {
                "timestep": t,
                "observation": observations[t],
                "model": model_states[t],
            }
            if true_state is not None:
                batch["truth"] = true_state[t]
            yield batch

    def collate_outputs(self, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Convert a list of step outputs into final arrays / tensors / diagnostics.

        Override if needed.
        """
        # TODO
        return {"steps": outputs}

    def save(self, path: str | Path) -> None:
        """Optional persistence hook."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save().")

    @classmethod
    def load(cls, path: str | Path) -> "BaseAssimilationModel":
        """Optional persistence hook."""
        raise NotImplementedError(f"{cls.__name__} does not implement load().")
