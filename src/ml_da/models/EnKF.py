import time

import numpy as np
import scipy.linalg as sla

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.experiments.metrics import compute_metrics, init_metrics
from ml_da.models.base_model import BaseAssimilationModel
from ml_da.tools.config import DataCoreConfig, ModelConfig


class EnKF(BaseAssimilationModel):
    """Ensemble Kalman Filter (ETKF formulation)"""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle, dynamical_model=None):
        super().__init__(model_cfg, data_cfg, data)
        self.metrics = init_metrics()
        self.runtime = None
        self.last_trHK = np.nan  # diagnostic storage
        self.trajectory = []
        if dynamical_model is None:
            self.dynamical_model = self.dyn
        else:
            self.dynamical_model = dynamical_model

    # Main step
    def assimilate(
        self,
    ):
        self.trajectory = []
        start_time = time.time()

        # Initial ensemble
        Ens = self.dyn.inital_state

        R_inv_sqrt = self.sym_sqrt_inv(self.R)

        trHK = np.nan

        # Time loop
        for t in range(self.timesteps - 1):
            self.log_metrics(
                t=t,
                ensemble=Ens,
                truth=self.ground_truth[t] if self.ground_truth is not None else None,
                observation=self.obs_np[t],
                trHK=trHK,
            )

            # Forecast
            Ens = self.dyn.step(state=Ens)

            # Analysis
            if not (np.isnan(self.obs_np[t]).all()):
                Ens = self.EnKF_update(
                    Ens,
                    self.obs_np,
                    R_inv_sqrt,
                    self.H,
                )
                trHK = self.last_trHK
            else:
                trHK = np.nan

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
            self.trajectory.append(np.mean(ensemble, axis=0))

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
        N, Nx = Ens.shape
        N1 = N - 1

        Ens_mu = np.mean(Ens, axis=0)
        Ano = Ens - Ens_mu

        HEns = Ens @ observation_operator.T
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
