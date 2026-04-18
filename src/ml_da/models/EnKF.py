import time

import numpy as np
import scipy.linalg as sla

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.experiments.metrics import compute_metrics, init_metrics
from ml_da.models.base_model import BaseAssimilationModel
from ml_da.tools.config import DataCoreConfig, ModelConfig


class EnKF(BaseAssimilationModel):
    """Ensemble Kalman Filter (ETKF formulation)"""

    def __init__(self, N, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle):
        super().__init__(model_cfg, data_cfg, data)
        self.N = N
        self.metrics = init_metrics()
        self.runtime = None
        self.last_trHK = np.nan  # diagnostic storage
        self.trajectory = []

    # Main step
    def assimilate(
        self,
        ground_truth,
        obs,
        CovX0,
        Covy,
        time_sequence,
        dynamic_model,
        observation_operator,
        add_noise=None,
        dt=None,
    ):
        self.trajectory = []
        start_time = time.time()

        # Initial ensemble
        Ens = self.sample(CovX0)  # self.dyn.inital_state

        R = Covy  # self.R
        R_inv_sqrt = self.sym_sqrt_inv(R)

        # Initial logging
        self.log_metrics(
            t=0,
            ensemble=Ens,  # self.dyn.initial_state
            truth=ground_truth[0] if ground_truth is not None else None,
            observation=obs[0] if obs is not None else None,
            trHK=np.nan,
        )

        # Time loop
        for t in range(time_sequence - 1):
            # Forecast
            Ens = dynamic_model(Ens, t - dt, dt)
            #
            # Ens = self.dyn.step(state=list[np.ndarray])

            if add_noise is not None:
                Ens = add_noise(Ens, dt)  # don't need that

            # Analysis
            if obs[t] is not None:  # if self.obs_avail[t]
                Ens = self.EnKF_update(
                    Ens,
                    obs[t],
                    R,
                    R_inv_sqrt,
                    observation_operator,
                )
                trHK = self.last_trHK
            else:
                trHK = np.nan

            # Log everything (aligned)
            self.log_metrics(
                t=t + 1,
                ensemble=Ens,
                truth=ground_truth[t + 1] if ground_truth is not None else None,
                observation=obs[t + 1],
                trHK=trHK,
            )

        self.runtime = time.time() - start_time

        return Ens, self.metrics, self.runtime

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
            self.trajectory.append(np.mean(ensemble, axis=0))

    # Sampling
    def sample(self, CovX0):
        R = np.linalg.cholesky(CovX0)
        return np.random.randn(self.N, R.shape[0]) @ R.T

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
    def EnKF_update(self, Ens, current_obs, R, R_inv_sqrt, observation_operator):
        N, Nx = Ens.shape
        N1 = N - 1

        Ens_mu = np.mean(Ens, axis=0)
        Ano = Ens - Ens_mu

        HEns = observation_operator(Ens)
        HEns_mu = np.mean(HEns, axis=0)
        HAno = HEns - HEns_mu

        dy = current_obs - HEns_mu

        Y_tilde = HAno @ R_inv_sqrt
        dy_tilde = dy @ R_inv_sqrt

        S = Y_tilde / np.sqrt(N1)

        V, s, _ = sla.svd(S, full_matrices=False)

        d = s**2 + 1

        Pw = (V * (1.0 / d)) @ V.T
        T = (V * (1.0 / np.sqrt(d))) @ V.T

        w = dy_tilde @ Y_tilde.T @ Pw

        Ens = Ens_mu + w @ Ano + T @ Ano

        # --- Diagnostic: degrees of freedom for signal ---
        self.last_trHK = np.sum((s**2) / (s**2 + 1))

        return Ens
