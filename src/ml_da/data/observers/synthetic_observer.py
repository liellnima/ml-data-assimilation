from __future__ import annotations

import logging

import xarray as xr

from ml_da.data.observers.base_observer import Observer
from ml_da.tools.config import ObserverConfig
from ml_da.tools.registry import observer
from ml_da.tools.utils import str_join_ls

# fmt: off
for lib in ["jax", "jaxlib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
# isort: split
import dabench as dab  # noqa: E402, F401

# fmt: on


logger = logging.getLogger(__name__)

# TODO extend for location and time density


@observer
class SyntheticObserver(Observer):
    """Abstract class representing observer behavior."""

    def __init__(self, observer_cfg: ObserverConfig) -> None:
        """Init all relevant observer parameters."""
        super().__init__(observer_cfg)

    def get_id_str(self) -> str:
        """Return string that identifies the observer instance."""
        return str_join_ls(["OBS", self.density, self.error_type, self.error_params["scale"]])

    # TODO ideally I would like to initialize the obs object during __init__
    # but I cannot do that without ground_truth, that's why everything happens here
    def generate_observations(self, ground_truth: xr.Dataset) -> xr.Dataset:
        """
        Generate synthetic observations.

        Params:
            ground_truth (xr.Dataset): Ground truth from which we sample observations
        Returns:
            xr.Dataset: The dataset containg the observation values, times, locatoins, and errors
        """
        # FUTURE: add custom noise to the output of the system
        # this can be added later if we do not want to use Gaussian noise for the observer
        # in that case: make sure to not add additional noise in the observer object

        # FUTURE: Create lower resolution (_downsample_resolution) if desired

        # FUTURE: split up temporal and local sampling + specific location sampling

        # initialize observer
        obs = dab.observer.Observer(
            ground_truth,
            random_time_density=self.density,
            random_location_density=self.density,
            # random_location_count = 10, # alternative for having measurement stations so to say
            error_sd=self.error_params["scale"],
            random_seed=self.seed,
            stationary_observers=self.stationary_observers,
            error_positive_only=self.error_pos_only,
        )
        # run observer
        obs_data = obs.observe()

        return obs_data
