from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from ml_da.tools.config import SystemConfig
from ml_da.tools.io import load_json

# TODO default initial state: went already through spin-up.
# Can be used as the initial state of ground truth.

# TODO add step size of system to configs and this class at some point


class System(ABC):
    """Abstract class that defines behavior of systems."""

    def __init__(self, sys_cfg: SystemConfig):
        """Set all relevant system related params."""
        self.name = sys_cfg.name
        # specific params are all defined in the specific implementation
        # ATTENTION: only call super.__init__() after you have set the params!

        # load initial state from file (see resources/initial_states/ for the options)
        self._initial_state = load_json(Path(sys_cfg.initial_state))
        if not isinstance(self.initial_state, np.ndarray):
            raise ValueError(f"Expected initial state to be a np.ndarray, but got {type(self.initial_state)}")

        # set current state to initial state
        self.x = self.initial_state.copy()

        # initialize the system
        self._system = self._initialize_system()

        # set more specific params in specific implementation

    # we cannot modify the initial state.
    # each instance has a fixed initial state.
    @property
    def initial_state(self) -> np.ndarray:
        """Gets the initial state that was used to set up the system."""
        return self._initial_state

    @property
    def x(self) -> np.ndarray:
        """Gets the current state of the system."""
        return self._x

    @x.setter
    def x(self, x: np.ndarray):
        """Sets a new current state of the system."""
        self._x = x

    @property
    @abstractmethod
    def dim(self) -> tuple:
        """Get the dim of the system."""

    @abstractmethod
    def get_id_str(self) -> str:
        """Return string that identifies the observer instance."""

    @abstractmethod
    def _initialize_system(self) -> Any:
        """
        Initializes the system with the initial state.

        Returns:
            An object, that differs from specific class to specific class.
        """
        # initialize the obj and return it!

    @abstractmethod
    def generate_ground_truth(self, n_steps: int) -> xr.Dataset | tuple[xr.Dataset]:
        """
        Generate an xr.Dataset for a whole sequence of timesteps. Operates on the self._system object for that.

        Params:
            n_steps (int): how many steps the system should be evolved

        Return:
            xr.Dataset | tuple [xr.Dataset]: dataset that contains the state evolution
        """
        # run a full trajectory

        # adapt the self.x value (to update the state!)
