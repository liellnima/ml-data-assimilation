import numpy as np


class Var4D:
    """
    4D-Var.

    nIters: Number of iterations
    tol: Convergence threshold/tolerance
    """

    nIters: int
    tol: float

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
        add_noise=None,
        dt=None,
        noise=None,
    ):
        """
        dynamic_model : model function
        dynamic_jacobian : model Jacobian            x -> dM/dx
        observation_operator : observation operator
        observation_jacobian : observation Jacobian      x -> dH/dx
        B : background covariance
        R : observation covariance
        x_b : background state
        """

        # Control variable
        Nx = len(x_b)

        B12 = np.linalg.cholesky(B)  # B^(1/2)
        Rm12 = self.sym_sqrt_inv(R)  # R^(-1/2)

        # initialize background state
        x_state = x_b.copy()

        self.Compute_evaluation_metrics()

        for t in time_sequence:

            # if no obs at this time, we can just propagate state
            if obs[t] is None:
                x_state = dynamic_model(x_state)
                continue

            y = obs[t]
            w = np.zeros(Nx)

            # Inner loop (Gauss-Newton)
            for _ in range(self.nIters):

                # reconstruct initial condition
                x = x_state + B12 @ w

                # tangent linear propagation
                X = B12.copy()

                x_f = x.copy()

                # forward model + TLM
                x_f = dynamic_model(x_f)
                X = dynamic_jacobian(x_f) @ X

                # observation space
                y_pred = observation_operator(x_f)
                Y = observation_jacobian(x_f) @ X

                # whitening
                dy = Rm12 @ (y - y_pred)
                Yw = Rm12 @ Y

                # Gradient
                grad = Yw.T @ dy - w

                # SVD
                U, s, Vt = np.linalg.svd(Yw, full_matrices=False)

                d = s**2 + 1.0
                dw = Vt.T @ ((Vt @ grad) / d)

                w += dw

                if np.linalg.norm(dw) < self.tol:
                    break

            # Analysis update
            x_a = x_state + B12 @ w

            self.compute_evaluation_metrics()

            # Update background-
            x_state = dynamic_model(x_a)

            self.compute_evaluation_metrics()

    def sym_sqrt_inv(self, R):
        w, V = np.linalg.eigh(R)
        idx = w > 1e-12
        w = w[idx]
        V = V[:, idx]
        return (V * (1.0 / np.sqrt(w))) @ V.T

    def compute_evaluation_metrics(self):
        pass
