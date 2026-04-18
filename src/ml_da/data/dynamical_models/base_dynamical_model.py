import concurrent.futures
import logging
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from ml_da.data.transformations import add_noise
from ml_da.tools.config import DynamicalModelConfig
from ml_da.tools.utils import get_state

logger = logging.getLogger(__name__)


# TODO seed should be in configs


class DynamicalModel(ABC):
    """Abstract class that defines behavior of a dynamical model."""

    def __init__(self, dyn_cfg: DynamicalModelConfig, return_tlm: bool = False):
        """Set all relevant dynamical model related params."""
        self.name = dyn_cfg.name
        self.ensemble_size = dyn_cfg.ensemble_size
        self.perturbation_error = dyn_cfg.perturbation_error
        self.model_error = dyn_cfg.model_error  # contains error_type, error_params, only_positive

        # for most methods we either need ensembles or tlms
        # this can be adapted in the future, but we need to rework the whole system in that case
        # which is why I want this to throw an error, so we notice
        if (self.ensemble_size > 1) and return_tlm:
            raise ValueError("Dynamical Models currently supports ensembles generation xor TLMs, but not both.")
        self.return_tlm = return_tlm

        # call to create initial conditions right away here
        # states or list of np.ndarrays for ensembles
        self._state_0 = self._create_initial_states()
        self._state = self._state_0.copy()

        if self.return_tlm:
            self._linear_0 = self._create_initial_linear()
            self._linear = self._linear_0.copy()

        # in specific class: e.g. define the system you are modeling

    @property
    def initial_state(self) -> list[np.ndarray] | np.ndarray:
        """Returns initial conditions of the model (ensemble)"""
        return self._state_0

    @property
    def initial_linear(self) -> np.ndarray:
        """Returns initial conditions of the model (ensemble)"""
        return self._linear_0

    @property
    def state(self) -> list[np.ndarray] | np.ndarray:
        """Returns the state of the model (ensemble)"""
        return self._state

    @property
    def linear(self) -> list[np.ndarray] | np.ndarray:
        """
        Returns the TLM, the adjoint model / the linearized map of our model.

        At the moment we calculate this based on analytical Jacobian of our system if provided in DataAssimBench, and it
        is calculated via autodiff (JAX) if there is no quick/analytical Jacobian available.
        """
        if self.return_tlm:
            return self._linear
        else:
            raise ValueError("linear() is not defined for a DynamicalModel with self.return_tlm = False")

    @abstractmethod
    def get_id_str(self) -> str:
        """Return string that identifies the dynamical model instance."""

    @abstractmethod
    def _create_initial_states(self) -> list[np.ndarray] | np.ndarray:
        """
        Abstract method that returns the initial state.

        Make sure to use values for the initial states
        after spin up time, i.e. when the system has already diverged and we can see the non-linear effects.

        Returns:
            np.ndarray or list[np.ndarray]: The initial state of the system, or a list of them for ensembles
        """
        # e.g. perturbe the initial conditions that have been used for the system

    @abstractmethod
    def _create_initial_linear(self) -> np.ndarray:
        """"""
        # for the moment only supported for single member models. no ensembles
        # run np.eye on our system dimension

    # generate data for n time steps
    @abstractmethod
    def run_model(self, state: np.ndarray, n_steps: int, return_tlm: bool) -> xr.Dataset | tuple[xr.Dataset]:
        """Generate data for n time steps."""
        # depending on self.return_tlm, behave differently

    def _get_state_from_trajectory(
        self,
        full_trajectory: xr.Dataset | list[xr.Dataset],
    ) -> np.ndarray:
        """
        Returns the np.ndarray of the last state of a trajectory, which is a xr.Dataset.

        Can do this for ensembles or single models.
        """
        if isinstance(full_trajectory, list):
            return [get_state(ds, time=-1) for ds in full_trajectory]
        else:
            return get_state(full_trajectory, time=-1)

    def run_ensemble(
        self,
        states: list[np.ndarray] | np.ndarray,
        n_steps: int,
        return_tlm: bool,
    ) -> list[xr.Dataset] | xr.Dataset:
        """Run an ensemble (or a single run) of the numerical model."""
        if (self.ensemble_size > 1) and return_tlm:
            raise ValueError("TLM / Adjoint model not supported for ensembles at the moment.")
        # also handles single model run
        if self.ensemble_size == 1:
            return self.run_model(
                state=states,  # this is a single state
                n_steps=n_steps,
                return_tlm=return_tlm,
            )

        # for debugging purposes:
        # return = self.run_model(states[0], n_steps, return_tlm)

        # ensemble case
        if self.ensemble_size != len(states):
            raise ValueError(f"Expected {self.ensemble_size} members but got {len(states)}.")

        # we are using threads here, since we have already started a process per dataset
        # each ensemble generation should now be a thread within a process
        with concurrent.futures.ThreadPoolExecutor() as thread_executor:
            thread_to_id = {
                thread_executor.submit(self.run_model, state=state, n_steps=n_steps, return_tlm=return_tlm): ensemble_id
                for ensemble_id, state in enumerate(states)
            }

            # empty list that is going to be populated with results
            # sorted according to order of ensemble members
            all_trajectories = [None] * self.ensemble_size
            for thread in concurrent.futures.as_completed(thread_to_id):
                curr_ensemble_id = thread_to_id[thread]
                try:
                    result_ds = thread.result()
                    all_trajectories[curr_ensemble_id] = result_ds
                    # logger.info(f"Generated Ensemble with ID {curr_ensemble_id}.")
                except Exception as e:
                    logger.error(f"ERROR: Ensemble no. {curr_ensemble_id} failed with exception: {e}")

        return all_trajectories

    def step(
        self,
        state: list[np.ndarray] | np.ndarray | None = None,
    ) -> list[np.ndarray] | np.ndarray | tuple[np.ndarray]:
        """
        Steps the system n steps forward and returns the new state in the form of a np.ndarray.

        This might be used by traditional data assimilation methods that update the state in between each numerical
        model step.
        """
        # set the state of our model with the given state
        # default is that we use the current internal state (is automaticall updated)
        if state is not None:
            self._state = state

        # step the model one forward, and return_tlm if desired
        state_xr = self.run_ensemble(
            states=self._state,
            n_steps=1,
            return_tlm=self.return_tlm,
        )

        # unpack data correctly in case a tuple was returned
        if self.return_tlm:
            # this code does not work for ensembles, so catch that
            if self.ensemble_size > 1:
                raise ValueError("TLM / adjoint model not supported for ensembles at the moment.")
            state_xr, linear_xr = state_xr
            # set the new linearized matrix (TLM)
            updated_linear = np.asarray(linear_xr.isel(time=-1).data)

            self._linear = updated_linear

        # add noise to new state
        noisy_state_xr = add_noise(
            data=state_xr,
            error_type=self.model_error.error_type,
            error_params={
                **self.model_error.error_params,
                "size": None,
            },
            only_positive=self.model_error.only_positive,
        )

        # set the new noisy state (for ensembles or single realizations)
        self._state = self._get_state_from_trajectory(noisy_state_xr)

        # TODO later: handle this whole self.return_tlm thingy differently...
        if self.return_tlm:
            return noisy_state_xr, linear_xr

        return noisy_state_xr

    def generate_model_data(
        self,
        state: np.ndarray | list[np.ndarray] | None = None,
        n_steps: int = 1000,
    ) -> list[xr.Dataset]:
        """Generate numerical model data for a whole time-sequence."""
        # update state of the system with given state
        # default is using the internal state that has been initialized already
        if state is not None:
            self._state = state

        # call custom generate
        model_data = self.run_ensemble(
            states=self._state,
            n_steps=n_steps,
            return_tlm=False,
        )  # tlm not supported for time sequence by us (we could though!)

        # FUTURE Resample error: create lower resolution
        # model_data = downsample_resolution(model_data)

        # add custom noise (self.model_error) to whole time series  (representation error Q)
        model_data = add_noise(
            data=model_data,
            error_type=self.model_error.error_type,
            error_params={
                **self.model_error.error_params,
                "size": None,
            },
            only_positive=self.model_error.only_positive,
        )

        # get the state out of the whole dataset
        updated_state = self._get_state_from_trajectory(model_data)

        # set new state of the system
        self._state = updated_state
        self._linear = None  # undefined for now in this case

        # return data
        return model_data
