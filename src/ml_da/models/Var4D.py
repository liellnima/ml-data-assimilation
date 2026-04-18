import time

import numpy as np

from ml_da.experiments.metrics import compute_metrics, init_metrics


class Var4D:
    """Incremental 4D-Var with Gauss–Newton Produces time-series metrics like EnKF."""

    def __init__(self, nIters, tol):
        self.nIters = nIters
        self.tol = tol
        self.metrics = init_metrics()
        self.runtime = None

    # Main step
    def step(
        self,
        ground_truth,
        x_b,
        obs,
        B,
        R,
        time_sequence,
        dynamic_model,
        dynamic_jacobian,
        observation_operator,
        observation_jacobian,
    ):
        start_time = time.time()

        Nx = len(x_b)
        w = np.zeros(Nx)

        B12 = np.linalg.cholesky(B)
        Rm12 = self.sym_sqrt_inv(R)

        # Optimization loop
        for _ in range(self.nIters):

            x = x_b + B12 @ w
            X = B12.copy()

            grad = -w
            Y_list = []
            dy_list = []

            for k in time_sequence:

                M_k = dynamic_jacobian(x)
                X = M_k @ X
                x = dynamic_model(x)

                if obs[k] is not None:
                    y = obs[k]

                    y_pred = observation_operator(x)
                    Y = observation_jacobian(x) @ X

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

        traj = self.forward_trajectory(x0_opt, time_sequence, dynamic_model)

        # time-series metrics (like EnKF)
        for t, x in enumerate(traj):

            self.metrics["time"].append(t)

            compute_metrics(
                self.metrics,
                estimate=x,
                truth=ground_truth[t] if ground_truth is not None else None,
                observation=obs[t] if obs is not None else None,
            )

            # No trHK in 4D-Var
            self.metrics["trHK"].append(np.nan)

        self.runtime = time.time() - start_time

        return x0_opt, traj, self.metrics, self.runtime

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
