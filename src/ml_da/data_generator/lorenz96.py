import dabench as dab
import numpy as np

from ml_da.data_generator.data_base import DataGenerator
from ml_da.tools.config import DataConfig, SystemConfig

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

    def __init__(self, data_cfg: DataConfig, sys_cfg: SystemConfig):
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

    def _create_initial_state(self, error_sd: float = 0) -> np.ndarray:
        # TODO
        # if error_sd is zero: return default values
        # else: add noise according to noise + seed on the default values
        # ideally: use the _add_noise function provided in this class for this
        if error_sd == 0:
            return self.default_initial_state

        # make the ndarray to a xr.DataSet??
        # TODO
        # CONTINUE HERE: who has the responsibility? should we use two different add_noise funcs?

        raise NotImplementedError()
