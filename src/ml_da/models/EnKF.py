# The EnKF method

import numpy as np
import scipy.linalg as sla


class EnKF:
    """
    Ensemble Kalman Filter.

    N: number of samples (desired ensemble size)
    """

    N: int

    def step(
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
        noise=None,
    ):

        _ = ground_truth
        # CovX0 = covariance of initial states.
        Ens = self.sample(CovX0)

        # Define R^(-1/2)
        R = Covy  # Observation covariance noise
        R_inv_sqrt = self.sym_sqrt_inv(R)

        self.Compute_evaluation_metrics(Ens)  # RMSE

        # time_sequence = information related to time sequence like time, time_step, if a obs is available for the current time, etc.
        for t in time_sequence:
            # Apply dynamix model transformation to every sample in Ensemble
            Ens = dynamic_model(Ens, t - dt, dt)  # Like t-dt and dt
            if add_noise is not None:
                # If the dynamic_model is designed deterinistacilly. If it is defined stochastically and take into account noise, remove line.
                Ens = add_noise(Ens, dt)

            if obs[t] is not None:
                current_obs = obs[t]

                # Propagate obs infos in Ensemble
                Ens = self.EnKF_update(
                    Ens,
                    current_obs,
                    R,
                    R_inv_sqrt,
                    observation_operator,
                )

                self.Compute_evaluation_metrics(Ens)  # RMSE

    def sample(self, CovX0):
        R = np.linalg.cholesky(CovX0)
        D = np.random.randn(self.N, R.shape[0]) @ R.T
        return D

    def sym_sqrt_inv(self, R):
        # Eigendecomposition
        w, V = np.linalg.eigh(R)  # eigenvalues, eigenvectors
        idx = np.argsort(w)[::-1]  # Sort descending
        w = w[idx]
        V = V[:, idx]

        # Truncate small eigenvalues
        eps = 1e-8 * np.max(w)
        idx = w > eps
        w_r = w[idx]
        V_r = V[:, idx]

        # Inverse sqrt of eigenvalues
        inv_sqrt_w = 1.0 / np.sqrt(w_r)

        # Construct matrix
        return (V_r * inv_sqrt_w) @ V_r.T

    def EnKF_update(self, Ens, current_obs, R, R_inv_sqrt, observation_operator):
        """
        Perform Ensemble Kalman Filter update if observation are given via Ensemble Transform Kalman Filter (ETKF).

        x_t^(i)​=x_{t∣t−1}^(i)​+K(y_t^(i)​−H(x_{t∣t−1}^(i))​)
        where K_t=P_tH^⊤(HP_tH^⊤+R)^{−1}; P_t = Model covariance at time t

        In practice, we use the anomalies A where P = A^TA/(N-1).
        Then, instead of updating each state variable directly, we update the ensemble in “ensemble space” (size N) using a linear transform.
        """
        # Seting up variables
        N, Nx = Ens.shape  # Dimensionality
        N1 = N - 1

        Ens_mu = np.mean(Ens, 0)  # Ensemble mean
        Ano = Ens - Ens_mu  # Ensemble anomalies

        HEns = observation_operator(Ens)
        HEns_mu = np.mean(HEns, 0)  # Observation ensemble mean
        HAno = HEns - HEns_mu  # Observation ensemble anomalies
        dy = current_obs - HEns_mu  # Mean "innovation"

        # Compute Pw (Cov(w|y)) and T (sqrt of Pw)
        Y_tilde = HAno @ R_inv_sqrt  # Whitening
        dy_tilde = dy @ R_inv_sqrt

        S = Y_tilde / np.sqrt(N1)  # Scale anomalies so sample covariance is correct (1/(N-1))

        V, s, _ = sla.svd(S, full_matrices=False)  # SVD

        d = s**2 + 1  # Eigenvalues

        # Ensemble-space covariance and transform matrix (ETKF)
        Pw = (V * (1.0 / d)) @ V.T
        T = (V * (1.0 / np.sqrt(d))) @ V.T

        w = dy_tilde @ Y_tilde.T @ Pw  # Transform mean update coefficients in ensemble space
        Ens = Ens_mu + w @ Ano + T @ Ano  # Update ensemble

        trHK = np.sum((s**2 + 1) ** (-1.0) * s**2)
        self.Compute_evaluation_metrics(trHK)  # Relative influence of observations

        return Ens

    def Compute_evaluation_metrics(self, Ens, *args, **kwargs):
        # Will probably be implemented in superclass?
        return None
