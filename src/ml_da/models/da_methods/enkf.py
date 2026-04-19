import logging
import time

import numpy as np
import scipy.linalg as sla

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.experiments.metrics import compute_metrics, init_metrics
from ml_da.models.da_methods.base_model import BaseAssimilationModel
from ml_da.tools.config import DataCoreConfig, ModelConfig
from ml_da.tools.registry import da_method

logger = logging.getLogger(__name__)


@da_method
class EnKF(BaseAssimilationModel):
    """Ensemble Kalman Filter (ETKF formulation)"""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle, dynamical_model=None):
        super().__init__(model_cfg, data_cfg, data)
        self.metrics = init_metrics()
        self.runtime = None
        self.last_trHK = np.nan  # diagnostic storage
        self.trajectory = []  # TODO delete
        self.forecast_trajectory = []
        self.analysis_trajectory = []
        if dynamical_model is None:
            self.dynamical_model = self.dyn
        else:
            self.dynamical_model = dynamical_model  # can be something else as long as it has a "step" func

    # Main step
    def assimilate(
        self,
    ):
        self.trajectory = []  # TODO delete
        self.forecast_trajectory = []
        self.analysis_trajectory = []

        start_time = time.time()

        # Initial ensemble
        Ens = self.dyn.initial_state

        R_inv_sqrt = self.sym_sqrt_inv(self.R)

        trHK = np.nan

        # stores metrics once at the beginning
        self.log_metrics(
            t=0,
            ensemble=Ens,
            truth=self.ground_truth[0] if self.ground_truth is not None else None,
            observation=self.obs_np[0],
            trHK=trHK,
        )

        self.analysis_trajectory.append(np.array(Ens, copy=True))
        self.trajectory.append(np.array(Ens, copy=True))  # TODO delete

        # Time loop
        for t in range(self.timesteps - 1):
            # print("Timestep", t)
            if t % 50 == 0:
                logger.info(f"EKF Timestep {t}")

            # Forecast
            Ens_forecast = self.dynamical_model.step(state=Ens)
            self.forecast_trajectory.append(np.array(Ens_forecast, copy=True))

            # TODO also log after forecast, so we see the difference to the model?

            # Analysis
            if not (np.isnan(self.obs_np[t + 1]).all()):
                #     HEns = np.asarray(Ens_forecast) @ self.H.T
                #     innovation = self.obs_np[t + 1] - np.mean(HEns, axis=0)
                #     self.innovations.append(np.array(innovation, copy=True))

                Ens = self.EnKF_update(
                    Ens_forecast,
                    self.obs_np[t + 1],
                    R_inv_sqrt,
                    self.H,
                )
                trHK = self.last_trHK
            else:
                trHK = np.nan
                Ens = Ens_forecast

            self.analysis_trajectory.append(np.array(Ens, copy=True))
            self.trajectory.append(np.array(Ens, copy=True))  # TODO delete

            self.log_metrics(
                t=t + 1,
                ensemble=Ens,
                truth=self.ground_truth[t + 1] if self.ground_truth is not None else None,
                observation=self.obs_np[t + 1],
                trHK=trHK,
            )

        self.runtime = time.time() - start_time

        return self.metrics, self.runtime

    # Logging (centralized)
    def log_metrics(self, t, ensemble=None, truth=None, observation=None, trHK=np.nan):
        self.metrics["time"].append(t)

        compute_metrics(
            self.metrics,
            ensemble=ensemble,
            truth=truth,
            observation=observation,
        )

        self.metrics["trHK"].append(trHK)

        if ensemble is not None:
            self.trajectory.append(np.array(ensemble, copy=True))

    # Matrix utilities
    def sym_sqrt_inv(self, R):
        w, V = np.linalg.eigh(R)

        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        eps = 1e-8 * np.max(w)
        idx = w > eps
        w_r = w[idx]
        V_r = V[:, idx]

        inv_sqrt_w = 1.0 / np.sqrt(w_r)

        return (V_r * inv_sqrt_w) @ V_r.T

    # ETKF update
    def EnKF_update(self, Ens, current_obs, R_inv_sqrt, observation_operator):

        Ens = np.stack(Ens)
        N, Nx = Ens.shape
        N1 = N - 1

        Ens_mu = np.mean(Ens, axis=0)
        Ano = Ens - Ens_mu

        inflation = 1.2  # eadd inflation because ensemble is under-dispersed (too small error)
        Ano = inflation * Ano
        Ens = Ens_mu + Ano

        HEns = Ens @ observation_operator.T
        HEns_mu = np.mean(HEns, axis=0)
        HAno = HEns - HEns_mu

        dy = current_obs - HEns_mu

        Y_tilde = HAno @ R_inv_sqrt
        dy_tilde = dy @ R_inv_sqrt

        S = Y_tilde / np.sqrt(N1)

        # V, s, _ = sla.svd(S, full_matrices=False)
        U, s, _ = sla.svd(S, full_matrices=False)

        d = 1.0 + s**2
        Id = np.eye(N)

        UU_T = U @ U.T
        # Pw = (V * (1.0 / d)) @ V.T
        Pw = (U * (1.0 / d)) @ U.T + (Id - UU_T)
        # T = (V * (1.0 / np.sqrt(d))) @ V.T
        T = (U * (1.0 / np.sqrt(d))) @ U.T + (Id - UU_T)

        w = (dy_tilde @ Y_tilde.T @ Pw) / N1

        Ens = Ens_mu + w @ Ano + T @ Ano

        # --- Diagnostic: degrees of freedom for signal ---
        self.last_trHK = np.sum((s**2) / (s**2 + 1))

        return list(Ens)
