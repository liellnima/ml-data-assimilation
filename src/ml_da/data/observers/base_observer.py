from __future__ import annotations

from abc import ABC, abstractmethod

import xarray as xr

from ml_da.tools.config import ObserverConfig


class Observer(ABC):
    """Class that defines behaviour of observer systems."""

    def __init__(
        self,
        obs_cfg=ObserverConfig,
    ) -> None:
        """Init all relevant parameters."""
        self.density = obs_cfg.density
        self.stationary_observers = obs_cfg.stationary_observers
        self.error_type = obs_cfg.observation_error.error_type
        self.error_params = obs_cfg.observation_error.error_params
        self.error_pos_only = obs_cfg.observation_error.only_positive
        self.seed = obs_cfg.observation_error.seed

    @abstractmethod
    def get_id_str(self) -> str:
        """Return string that identifies the observer instance."""

    @abstractmethod
    def generate_observations(self, ground_truth: xr.Dataset) -> xr.Dataset:
        """
        Generate synthetic observations.

        Params:
            ground_truth (xr.Dataset): Ground truth from which we sample observations
        Returns:
            xr.Dataset: The dataset containg the observation values, times, locatoins, and errors
        """
