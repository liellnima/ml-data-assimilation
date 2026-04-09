from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# TODO this template needs to be adapted (probably signficantly) to match with our models and data


class BaseAssimilationModel(ABC):
    """Base Model Class for both traditional and ML-based data assimilation methods."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    @property
    def requires_training(self) -> bool:
        return False

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
