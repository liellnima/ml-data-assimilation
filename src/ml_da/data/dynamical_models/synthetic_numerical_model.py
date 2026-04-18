import logging

import numpy as np
import xarray as xr

import ml_da.data.systems  # noqa: F401
from ml_da.data.dynamical_models.base_dynamical_model import DynamicalModel
from ml_da.data.transformations import add_noise
from ml_da.tools.config import DynamicalModelConfig, SystemConfig
from ml_da.tools.registry import SYSTEM_REGISTRY, dynamical_model
from ml_da.tools.utils import str_join_ls

logger = logging.getLogger(__name__)


@dynamical_model
class SyntheticNumericalModel(DynamicalModel):
    """
    Synthetic numerical model mimicking a specific system, our ground truth, that we are able to hand to this numerical
    model.

    We perturbate the initial conditions of our true underlying system, and add some model noise on top.
    """

    def __init__(self, dyn_cfg: DynamicalModelConfig, sys_cfg: SystemConfig, return_tlm: bool = False) -> None:
        """Init relevant parameters."""
        # init system depending on name
        self.true_system = SYSTEM_REGISTRY[sys_cfg.name](sys_cfg)
        self.true_initial_state = self.true_system.initial_state
        self.true_system_dim = self.true_system.system_dim

        # only call super afterwards, because it calls create_initial_state
        super().__init__(dyn_cfg, return_tlm)

    def get_id_str(self) -> str:
        """Return string that identifies the dynamical model instance."""
        return str_join_ls(
            [
                "MODEL",
                self.ensemble_size,
                self.model_error.error_type,
                self.model_error.error_params["scale"],
                self.perturbation_error.error_type,
                self.perturbation_error.error_params["scale"],
            ]
        )

    def _create_initial_states(self) -> list[np.ndarray] | np.ndarray:
        """
        Creates initial state needed to initialize a system.

        By adding a small noise term we can perturb the initial ground truth state to "simulate" our numerical model.
        In the ensemble case, we add different noise perturbations, drawn from the same distribution.

        Attention:
        If your error params are set to zero (i.e. no bias, no scale, etc), it means that there is no noise that
        will be added, meaning no perturbations happen, and the ground truth initial state is returned.
        You you should never do this when simulating a model!
        """
        # repeat the ground truth in case of an ensemble
        if self.ensemble_size > 1:
            true_init = [self.true_initial_state] * self.ensemble_size
        else:
            true_init = self.true_initial_state

        # TODO throw error when all params are zero

        # we want to add a certain amount of noise to perturb our model
        perturbed_initial_state = add_noise(
            data=true_init,
            error_type=self.perturbation_error.error_type,
            error_params=self.perturbation_error.error_params,
            only_positive=self.perturbation_error.only_positive,
            seed=self.perturbation_error.seed,
        )

        return perturbed_initial_state

    def _create_initial_linear(self) -> np.ndarray:
        """"""
        # for the moment only supported for single member models. no ensembles
        if self.ensemble_size > 1:
            raise ValueError("TLM / adjoint model not supported for ensembles at the moment.")
        # run np.eye on our system dimension
        return np.eye(self.true_system_dim)

    # generate data for n time steps

    def run_model(self, state: np.ndarray, n_steps: int, return_tlm: bool) -> xr.Dataset | tuple[xr.Dataset]:
        """
        Generate data for n time steps.

        Here our model is our ground truth system, but we run it with states that have been perturbed
        """
        if isinstance(state, list):
            raise ValueError("Expected a single state, but got an ensemble of states.")

        # make sure we are not running the model on our true initial state on accident
        if np.array_equal(state, self.true_initial_state):
            raise RuntimeError(
                "The model runs a realization that has the same state like the ground truth, i.e. something went wrong. Model runs are supposed to never be equal to the ground truth."
            )

        # call the system directly (not via step, to generate the tlm)
        # they start with the initial states, that's why we need to add +1 here
        # otherwise we only get the initial states for step=1
        return self.true_system._system.generate(
            x0=state,
            n_steps=n_steps + 1,
            return_tlm=return_tlm,
        )
