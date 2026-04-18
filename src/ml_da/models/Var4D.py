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
        x_b : background state (estimate)
        """

        # Control variable
        Nx = len(x_b)
        w = np.zeros(Nx)

        B12 = np.linalg.cholesky(B)  # B^(1/2)
        Rm12 = self.sym_sqrt_inv(R)  # R^(-1/2)

        self.Compute_evaluation_metrics()

        for _ in range(self.nIter):

            # Reconstruct initial state
            x = x_b + B12 @ w
            X = B12.copy()  # Jacobian dx/dw

            # Accumulators
            grad = -w

            Y_list = []
            dy_list = []

            # forward pass over time
            for k in time_sequence:

                # Propagate state + Jacobian
                M_k = dynamic_jacobian(x)
                X = M_k @ X
                x = dynamic_model(x)

                # If observation exists
                if obs[k] is not None:

                    y = obs[k]

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
            # Y_all = U S V^T
            U, s, Vt = np.linalg.svd(Y_all, full_matrices=False)

            # Compute (Y^T Y + I)^-1 grad efficiently
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
        x_final = x_b + self.B12 @ w

        self.Compute_evaluation_metrics()
        return x_final
