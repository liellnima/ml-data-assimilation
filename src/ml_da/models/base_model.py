from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr

import ml_da.data.dynamical_models  # noqa: F401
from ml_da.data.dataclasses import AssimDataBundle
from ml_da.data.dynamical_models.base_dynamical_model import DynamicalModel
from ml_da.tools.config import (
    DataCoreConfig,
    DynamicalModelConfig,
    ModelConfig,
    ObserverConfig,
    SystemConfig,
)
from ml_da.tools.registry import DYNAMICAL_MODEL_REGISTRY
from ml_da.tools.utils import get_all_states

logger = logging.getLogger(__name__)

# TODO adapt this later on, so we can access data_cfg directly in the AssimDataBundle


class BaseAssimilationModel(ABC):
    """Base Model Class for both traditional and ML-based data assimilation methods."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle) -> None:
        # get configs from data generation
        self.sys_cfg: SystemConfig = data_cfg.system
        self.dyn_cfg: DynamicalModelConfig = data_cfg.model
        self.obs_cfg: ObserverConfig = data_cfg.observer

        # the basics
        self.name: str = model_cfg.name  # model name
        self.timesteps: int = data_cfg.timesteps  # 1000
        self.system_dim: int = self.sys_cfg.params["system_dim"]  # 36

        # some booleans
        self.requires_training: bool = model_cfg.requires_training  # for ml methods
        self.requires_adjoint: bool = model_cfg.requires_adjoint  # for methods that need jacobian of dyn model
        self.requires_ensemble: bool = model_cfg.requires_ensemble  # for ensemble methods
        if self.requires_adjoint and self.requires_ensemble:
            raise NotImplementedError(
                "Models that require ensembles and adjoint models (TLM / Jacobian) are not supported at the moment."
            )

        # get the relevant error sds (needed for covariances)
        self.obs_sd: float = self.obs_cfg.observation_error.error_params[
            "scale"
        ]  # standard deviation of observation noise
        self.dyn_sd: float = self.dyn_cfg.model_error.error_params[
            "scale"
        ]  # standard deviation of numerical model noise
        self.per_sd: float = self.dyn_cfg.perturbation_error.error_params[
            "scale"
        ]  # standard deviation of perturbation noise (true error!)

        # TODO unpacking params in individual class (whatever the specific model needs)
        self.params: dict = model_cfg.params

        # get data, handle observations, ground truth, and model data
        self.data = data
        self.obs_xr: xr.Dataset = data.observations
        self.obs_np = self._build_np_obs()  # 50% np.arrays (1000 bool, max 18,)
        self.obs_idx = self.obs_xr["system_index"].isel(time=0, variable=0).values
        self.ground_truth: np.ndarray = get_all_states(data.truth)  # np.arrays (1000, 36,) first time, than var
        self.dyn_data: list[np.ndarray] = get_all_states(
            data.dynamical_model
        )  # list[np.array(1000, 36)] # first time, than var
        # self.metadata = data.metadata # we should get configs from that one

        # rows: observations, columns: model dim (max:36, 36)
        # covariance R: observation error
        # covariance Q: model error
        # covariance P0: initial state covariance / perturbation error: can later be estimated differently
        self.R, self.Q, self.P0 = self._get_covariances()

        # Jacobian of observations for whole time series
        # because we assume stationary_observer=True --> raise not implemented error if false
        self.H: np.ndarray = self._get_linearized_observation_operator()  # constant matrix #np.ndarray

        # this can be executed with .step and
        self.dyn = self._init_dynamical_model()  # this is an object you can call
        # e.g.: self.dyn.step(state)

        # TODO write documentation for single steps
        # TODO add explanations how to deal with the ensemble

    def observation_operator(self, state: np.ndarray) -> np.ndarray:
        """The observation operator, a function that takes the full state and returns only the observed components."""
        return np.asarray(state)[self.obs_idx]

    def _get_linearized_observation_operator(self) -> np.ndarray:
        """Get Jacobian of observations, i.e. the linearized observation operator."""
        obs_jac = np.zeros((len(self.obs_idx), self.system_dim))
        obs_jac[np.arange(len(self.obs_idx)), self.obs_idx] = 1.0
        return obs_jac

    # TODO this should probably live somewhere else
    # TODO we should just check if the observation exists in the xarray
    # instead of building a large dataset that contains a ton of nans.
    def _build_np_obs(self) -> np.ndarray:
        """Returns np.darray with shape (time, dim) where np.nan means there is no observation, and the values are the
        actual observations."""
        time_ticker = 0 + np.arange(self.timesteps) * float(self.sys_cfg.params["delta_t"])

        # observation times from DataAssimBench
        obs_times = self.obs_xr.time.values

        # number of observed variables / locations
        n_obs = self.obs_xr.sizes["observations"]

        # initialize full array with NaNs
        obs_full = np.full((self.timesteps, n_obs), np.nan)

        # observed values from xr.Dataset
        obs_values = self.obs_xr["x"].values  # shape: (n_obs_times, n_obs)

        # map observation times onto model time indices
        for obs_i, t_obs in enumerate(obs_times):
            model_t_idx = np.where(np.isclose(time_ticker, t_obs))[0]
            if len(model_t_idx) == 0:
                continue
            obs_full[model_t_idx[0], :] = obs_values[obs_i]

        # for debugging
        # obs_mask = ~np.isnan(obs_full)
        return obs_full

    def _init_dynamical_model(self) -> list[DynamicalModel] | DynamicalModel:
        """Initialized the dynamical model that can be used during assimilation."""
        # overwrite ensemble size (as we always have the ensembles on file, even if the model only wants one)
        # we will still get the same output like the 1st ensemble
        # because the numpy random generators are pseudo-random, and deterministic in that sense
        if self.requires_adjoint:
            self.dyn_cfg.ensemble_size = 1
        # re-create the model from configs (works because of seeds)
        # we ask here for returning the adjoint if desired
        dyn_model = DYNAMICAL_MODEL_REGISTRY[self.dyn_cfg.name](self.dyn_cfg, self.sys_cfg, self.requires_adjoint)

        # TODO add this to tests: compare if the generated data HERE matches with the generated data on disk
        # manually tested: it does
        # comp_data = dyn_model.generate_model_data(n_steps=5)

        return dyn_model

    def _get_covariances(self) -> tuple[float, float, float]:
        """Retrieves the covariances."""
        # Observation covariance
        if not self.obs_cfg.stationary_observers:
            # TODO FUTURE we need to iterate over time in non-stationary case and get m at each timestep
            raise NotImplementedError("Covariances need still to be implemented for non-stationary observations!")

        # number of observations per timestep
        # TODO FUTURE the observer currently draws randomly from a distribution, i.e. our density
        # is NOT exactly 0.9 * 36, but can be something more or less. We might want to fix this later on or keep the random factor
        m = self.data.observations.sizes["observations"]
        # resulting observation covariance matrix
        R = (self.obs_sd**2) * np.eye(m)

        # Model covariance
        # number of variables of true system
        n = self.system_dim
        # resulting model error covariance matrix
        # TODO FUTURE if we allow different noise params on different variables,
        # we needs to adapt this part. each entry in the list is the sd for a specific variable
        # sth like that: Q = np.diag([sigma_i**2 for sigma_i in stds])
        Q = (self.dyn_sd**2) * np.eye(n)

        # Perturbations / initial state covariance (true state)
        # TODO FUTURE use different methods that are more realistic later on
        # number of variables of true system
        n = self.system_dim
        # initial state covariance matrix
        P0 = (self.per_sd**2) * np.eye(n)

        return R, Q, P0

    @abstractmethod
    def assimilate(self) -> tuple[dict[str, Any], float]:
        """
        Implement the assimilation.

        Returns a dicionary of metrics and the running time.
        """

    def compute_assimilation_metrics(self):
        """"""
        raise NotImplementedError()

    def collate_results(self):
        """"""
        raise NotImplementedError()
