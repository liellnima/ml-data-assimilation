import numpy as np


class Var4D:
    """
    4D-Var.

    nIters: Number of iterations
    tol: Convergence threshold
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
        w = np.zeros(Nx)
        B12 = np.linalg.cholesky(B)  # B^(1/2)
        Rm12 = self.sym_sqrt_inv(R)  # R^(-1/2)

        self.Compute_evaluation_metrics()

        for iteration in range(self.nIter):

            # Reconstruct initial state
            x = x_b + B12 @ w
            X = self.B12.copy()  # dx/dw

            # Initialize accumulators
            grad = -w
            Y_list = []
            dy_list = []

            for t in time_sequence:

                # Propagate state + Jacobian
                M_k = dynamic_jacobian(x)
                X = M_k @ X
                x = dynamic_model(x)

                self.Compute_evaluation_metrics()

                # If observation exists
                if obs[t] is not None:

                    y = obs[t]

                    # Observation linearization
                    y_pred = observation_operator(x)
                    Y = observation_jacobian(x) @ X

                    # Whitening
                    dy = Rm12 @ (y - y_pred)
                    Y = Rm12 @ Y

                    # Store contributions
                    Y_list.append(Y)
                    dy_list.append(dy)

                    # Gradient contribution
                    grad += Y.T @ dy

            if len(Y_list) == 0:
                break  # no observations → nothing to do

            Y_all = np.vstack(Y_list)
            np.concatenate(dy_list)

            # SVD-based Gauss–Newton step
            U, s, Vt = np.linalg.svd(Y_all, full_matrices=False)

            # Compute (Y^T Y + I)^-1 grad
            # dw = V ( (V^T grad) / (s^2 + 1) )
            d = s**2 + 1.0
            dw = Vt.T @ ((Vt @ grad) / d)

            # Update control
            w += dw

            self.Compute_evaluation_metrics()

            # Convergence check
            if np.linalg.norm(dw) < self.tol:
                break

        # Final state reconstruction
        x_b + self.B12 @ w
        self.Compute_evaluation_metrics()

    def sym_sqrt_inv(self, R):
        w, V = np.linalg.eigh(R)
        idx = w > 1e-10
        w = w[idx]
        V = V[:, idx]
        return (V * (1.0 / np.sqrt(w))) @ V.T

    def Compute_evaluation_metrics(self, *args, **kwargs):
        # Will probably be implemented in superclass?
        return None
