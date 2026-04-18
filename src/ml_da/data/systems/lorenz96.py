from __future__ import annotations

import logging

import dabench as dab
import xarray as xr

from ml_da.data.systems.base_system import System
from ml_da.tools.config import SystemConfig
from ml_da.tools.registry import system
from ml_da.tools.utils import str_join_ls

for lib in ["jax", "jaxlib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@system
class Lorenz96(System):
    """Implementation of a System for the Lorenz96 system from the DataAssimBench."""

    def __init__(self, sys_cfg: SystemConfig):
        # handle the whole stuff around the forcing factor
        # map the desired non-linearity to the right params that is controlling it
        self._non_linearity_map = sys_cfg.non_linearity
        self._non_linear_level = sys_cfg.chosen_non_linear_level
        if self._non_linear_level not in self._non_linearity_map:
            raise ValueError(f"Got {self._non_linear_level}, but expected one of {list(self._non_linearity_map)}")
        self._forcing_factor = self._non_linearity_map[self._non_linear_level]

        # set all other relevant parameters
        self._params = sys_cfg.params
        # self.delta_t = sys_cfg.params["delta_t"]
        # self.system_dim = sys_cfg.params["system_dim"]

        # ATTENTION: We need to call the super() after we have set the forcing params here
        # because it calls initialize_system and that relies on some of the special defined values here
        super().__init__(sys_cfg)

    def get_id_str(self) -> str:
        """Return string that identifies the observer instance."""
        return str_join_ls(["SYSTEM", self.name, self._forcing_factor])

    def dim(self) -> tuple:
        """Get the dim of the system."""
        return self._system.system_dim

    def _initialize_system(self) -> dab._data.Data:
        """
        Initialize the system, here the Lorenz96 system. When you create your own system class, you can replace this
        part with other systems, e.g. from DataAssimBench.

        We are using the DataAssimBench data generator objects here.

        Returns:
            Object of  Data class (of DataAssimBench)
        """
        # put the right params to create the object
        l96_obj = dab.data.Lorenz96(
            forcing_term=self._forcing_factor,
            x0=self._initial_state,
            **self._params,
        )
        # returns the system object
        # (which will be accessible via self._system)
        return l96_obj

    def generate_ground_truth(self, n_steps: int) -> xr.Dataset | tuple[xr.Dataset]:
        full_trajectory = self._system.generate(
            x0=self.x,
            n_steps=n_steps,
        )

        # retrieve the state of the last timestep
        # and set this as the newest state
        new_state = full_trajectory.isel(time=-1).to_array().data.flatten()
        if new_state.shape != self.x.shape:
            raise RuntimeError("Expected the new state to be of shape {self.x.shape}, but got {new_state.shape}")
        self.x = new_state

        return full_trajectory
