from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

import ml_da.data.dynamical_models  # noqa: F401
from ml_da.data.dataclasses import AssimDataBundle
from ml_da.data.dynamical_models.base_dynamical_model import DynamicalModel
from ml_da.tools.config import DataCoreConfig, ModelConfig
from ml_da.tools.registry import DYNAMICAL_MODEL_REGISTRY

logger = logging.getLogger(__name__)

# TODO adapt this later on, so we can access data_cfg directly in the AssimDataBundle


class BaseAssimilationModel(ABC):
    """Base Model Class for both traditional and ML-based data assimilation methods."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle) -> None:
        self.name = model_cfg.name
        self.requires_training = model_cfg.requires_training
        self.requires_adjoint = model_cfg.requires_adjoint
        self.requires_ensemble = model_cfg.requires_ensemble
        if self.requires_adjoint and self.requires_ensemble:
            raise NotImplementedError(
                "Models that require ensembles and adjoint models (TLM / Jacobian) are not supported at the moment."
            )
        # TODO unpacking params in individual class
        self.params = model_cfg.params

        # get configs from data generation
        self.sys_cfg = data_cfg.system
        self.dyn_cfg = data_cfg.model
        self.obs_cfg = data_cfg.observer

        # get data
        self.timesteps: int = data_cfg.timesteps  # 1000
        # timestep 0 --> initial states

        # state
        # self.dyn.step()
        # new_state --> compare to ground truth at time 1

        # TODO numpy
        self.ground_truth = data.truth  # np.arrays (1000, 36,) first time, than var
        self.observations = data.observations  # 50% np.arrays (1000 bool, max 18,)
        self.dyn_data = data.dynamical_model  # list[np.array(1000, 36)] # first time, than var
        # self.metadata = data.metadata # we should get configs from that one

        # covariance R: observation error
        # covariance Q: model error
        # initial covariance of state P0: perturbation error
        # can later be estimated differently
        self.R, self.Q, self.P0 = self.get_covariances()

        # TODO Jacobian of observations for whole time series
        # because we assume stationary_observer=True --> raise not implemented error if false
        # self.obs_linear = self.get_obs_linear() # constant matrix #np.ndarray
        # TODO observation operator (somehow)
        # self.obs_operator = self.get_obs_operator() # is okay if it works the same way

        # this can be executed with .step and
        self.dyn = self.init_dynamical_model()
        self.dyn.initial_linear
        self.dyn.initial_state  # list[(36,)]

        # t 1
        self.dyn.state  # can be ensemble
        self.dyn.linear

        self.dyn.step()

        self.dyn.state  # t 501
        self.dyn.linear  # 2

        # TODO write documentation for single steps

        # TODO add explanations how to deal with the ensemble

    def init_dynamical_model(self) -> list[DynamicalModel] | DynamicalModel:
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

    def get_covariances(self) -> tuple[float, float, float]:
        """Retrieves the covariances."""
        # Observation covariance
        if not self.obs_cfg.stationary_observers:
            # TODO FUTURE we need to iterate over time in non-stationary case and get m at each timestep
            raise NotImplementedError("Covariances need still to be implemented for non-stationary observations!")
        # standard deviation of observation noise
        obs_sd = self.obs_cfg.observation_error.error_params["scale"]
        # number of observations per timestep
        # TODO FUTURE the observer currently draws randomly from a distribution, i.e. our density
        # is NOT exactly 0.9 * 36, but can be something more or less. We might want to fix this later on or keep the random factor
        m = self.observations.sizes["observations"]
        # resulting observation covariance matrix
        R = (obs_sd**2) * np.eye(m)

        # Model covariance
        dyn_sd = self.dyn_cfg.model_error.error_params["scale"]
        # number of variables of true system
        n = self.sys_cfg.params["system_dim"]
        # resulting model error covariance matrix
        # TODO FUTURE if we allow different noise params on different variables,
        # we needs to adapt this part. each entry in the list is the sd for a specific variable
        # sth like that: Q = np.diag([sigma_i**2 for sigma_i in stds])
        Q = (dyn_sd**2) * np.eye(n)

        # Perturbations / initial state covariance
        # standard deviation of perturbation noise (true error!)
        # TODO FUTURE use different methods that are more realistic later on
        per_sd = self.dyn_cfg.perturbation_error.error_params["scale"]
        # number of variables of true system
        n = self.sys_cfg.params["system_dim"]
        # initial state covariance matrix
        P0 = (per_sd**2) * np.eye(n)

        return R, Q, P0

    def get_obs_linear(self):
        """Gets Jacobian of observations."""

    @abstractmethod
    def assimilate(self):
        """Implement the assimilation."""

    def compute_assimilation_metrics(self):
        """"""
        raise NotImplementedError()

    def collate_results(self):
        """"""
        raise NotImplementedError()
