import time

import numpy as np

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.data.transformations import add_noise  # noqa: F401
from ml_da.experiments.metrics import compute_metrics, init_metrics
from ml_da.models.base_model import BaseAssimilationModel
from ml_da.tools.config import DataCoreConfig, ModelConfig


class Var4D(BaseAssimilationModel):
    """Incremental 4D-Var with Gauss–Newton Produces time-series metrics like EnKF."""

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataCoreConfig, data: AssimDataBundle, nIters=10, tol=1e-6):
        super().__init__(model_cfg, data_cfg, data)
        self.nIters = nIters
        self.tol = tol
        self.metrics = init_metrics()
        self.runtime = None

    # Main step
    def assimilate(
        self,
    ):
        start_time = time.time()

        x_b = self.dyn.initial_state
        Nx = len(x_b)
        w = np.zeros(Nx)

        B12 = np.linalg.cholesky(self.Q)
        Rm12 = self.sym_sqrt_inv(self.R)

        # Optimization loop
        for _ in range(self.nIters):

            x = x_b + B12 @ w
            X = B12.copy()

            grad = -w
            Y_list = []
            dy_list = []

            for k in range(self.timesteps - 1):
                x, M_k = self.dyn.step(state=x)
                X = M_k @ X

                if not (np.isnan(self.obs_np[k]).all()):
                    y = self.obs_np[k]

                    y_pred = self.H @ x
                    Y = self.H @ x

                    dy = Rm12 @ (y - y_pred)
                    Y = Rm12 @ Y

                    Y_list.append(Y)
                    dy_list.append(dy)

                    grad += Y.T @ dy

            if len(Y_list) == 0:
                break

            Y_all = np.vstack(Y_list)

            U, s, Vt = np.linalg.svd(Y_all, full_matrices=False)

            d = s**2 + 1.0
            dw = Vt.T @ ((Vt @ grad) / d)

            w += dw

            if np.linalg.norm(dw) < self.tol:
                break

        # final trajectory
        x0_opt = x_b + B12 @ w

        traj = self.forward_trajectory(x0_opt)

        # time-series metrics (like EnKF)
        for t, x in enumerate(traj):

            self.metrics["time"].append(t)

            compute_metrics(
                self.metrics,
                estimate=x,
                truth=self.ground_truth[t] if self.ground_truth is not None else None,
                observation=self.obs_np[t] if self.observations is not None else None,
            )

            # No trHK in 4D-Var
            self.metrics["trHK"].append(np.nan)

        self.runtime = time.time() - start_time

        return self.metrics, self.runtime

    # Forward trajectory
    def forward_trajectory(self, x0, time_sequence, model):
        traj = [x0]
        x = x0.copy()

        for _ in time_sequence:
            x = model(x)
            traj.append(x)

        return traj

    # Matrix utility
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
