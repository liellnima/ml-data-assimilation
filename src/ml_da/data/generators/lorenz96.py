from typing import Any

import dabench as dab
import numpy as np

from ml_da.data.generators.base_data_generator import DataGenerator
from ml_da.tools.config import DataCoreConfig, SystemConfig

# default values for 36 variables (the default in dabench):
# final state of a 14400 timestep spinup startin with initial state of all 0s excep the first element which is set to 0.01.
# values are for delta_t (the length of one time step = 0.05 (like the original))
# TODO figure out if this belongs here or somewhere else
DEFAULT_INITIAL_STATE_LORENZ96 = np.ndarray(
    [
        0.90061724,
        2.2108543,
        3.3563306,
        7.0470520,
        7.3828993,
        -2.2906365,
        1.6358340,
        4.5246205,
        -0.8536633,
        2.2018400,
        2.5094680,
        5.6148005,
        -1.7163916,
        -3.5827417,
        0.22293478,
        1.8138107,
        3.7354333,
        5.9006715,
        -4.6722836,
        0.4664867,
        0.36800075,
        7.7004447,
        3.0569422,
        -1.7238870,
        -2.1296368,
        1.6388168,
        5.1955190,
        4.7863874,
        0.8382774,
        -4.0938597,
        0.5181451,
        1.2503184,
        6.0076460,
        7.1161866,
        -3.2190716,
        -2.3532054,
    ]
)


class Lorenz96(DataGenerator):
    """Implementation of a DataGenerator for the Lorenz96 system from the DataAssimBench."""

    def __init__(self, data_cfg: DataCoreConfig, sys_cfg: SystemConfig):
        super().__init__(data_cfg, sys_cfg)

    @property
    def default_initial_state(self):
        return DEFAULT_INITIAL_STATE_LORENZ96

    def _initialize_system(self, initial_state: np.ndarray) -> dab._data.Data:
        # map the desired non-linearity to the right params that is controlling it
        forcing_factor = self.system_non_linearity_mapping[self.chosen_non_linearity]

        # put the right params to create the object
        l96_obj = dab.data.Lorenz96(
            forcing_term=forcing_factor,
            x0=initial_state,
            **self.system_params,
        )
        # returns the system object
        return l96_obj

    def _create_initial_state(
        self,
        error_type: str | None = None,
        error_params: dict[str, Any] = {
            "loc": 0.0,
            "scale": 0.01,
            "size": None,
        },
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Creates initial state needed to initialize a system. If error_type is None, it means that there is no noise that
        should be added to the initial state, and the default initial state is returned.

        Otherwise error / noise is added, representating a slightly perturbed system that can be used as 'synthetic
        numerical model' and can even be used to create ensembles of such models.
        """
        # if we have zero params for the noise, it means we want no noise on our initial state
        if error_type is None:
            return self.default_initial_state

        # else we want to add a certain amount of noise to perturb our model
        perturbed_initial_state = self._add_noise(
            data=self.default_initial_state,
            error_type=error_type,
            error_params=error_params,
            only_positive=self.mod_error_pos_only,
            seed=seed,
        )

        return perturbed_initial_state
