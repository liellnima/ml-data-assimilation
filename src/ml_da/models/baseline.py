import numpy as np


def compute_climatology(dynamic_model, x0, steps, burn_in=1000):
    """
    Estimate climatological mean from a long free model run.

    dynamic_model : Model propagator
    x0 : Initial state
    steps : Number of samples to collect AFTER burn-in
    burn_in : Number of initial steps to discard

    Returns
    mean : np.ndarray
    cov  : np.ndarray
    """

    x = x0.copy()

    # Burn_in
    for _ in range(burn_in):
        x = dynamic_model(x)

    # Collect trajectory
    traj = []
    for _ in range(steps):
        x = dynamic_model(x)
        traj.append(x.copy())

    traj = np.array(traj)

    mean = np.mean(traj, axis=0)
    cov = np.cov(traj.T)

    return mean, cov


class Climatology:
    """
    Climatology baseline using model-implied statistics.

    Forecast = constant climatological mean
    """

    def __init__(
        self,
        dynamic_model,
        x0,
        steps=10000,
        burn_in=1000,
    ):
        """
        Compute climatology once at initialization (like DAPPER).

        dynamic_model : function
        x0 : initial state
        steps : number of samples
        burn_in : Number of initial steps to discard
        """

        self.mean, self.cov = compute_climatology(
            dynamic_model,
            x0,
            steps,
            burn_in,
        )

    def step(
        self,
        ground_truth,
        obs,
        time_sequence,
    ):
        x = self.mean.copy()

        self.Compute_evaluation_metrics(x, ground_truth, obs)

        for t in time_sequence:
            # Always revert to climatology
            x = self.mean.copy()

            self.Compute_evaluation_metrics(x, ground_truth, obs)

        return x

    def Compute_evaluation_metrics(self, x, *args, **kwargs):
        return None


class ClimatologyEnsemble:
    """
    Climatology baseline as an ensemble.

    Samples from climatological distribution.
    """

    def __init__(
        self,
        dynamic_model,
        x0,
        N,
        steps=10000,
        burn_in=1000,
    ):
        self.N = N

        self.mean, self.cov = compute_climatology(
            dynamic_model,
            x0,
            steps,
            burn_in,
        )

        self.chol = np.linalg.cholesky(self.cov + 1e-8 * np.eye(len(self.mean)))  # Cholesky factor of covariance matrix

    def sample(self):
        return self.mean + np.random.randn(self.N, len(self.mean)) @ self.chol.T

    def step(
        self,
        time_sequence,
    ):
        Ens = self.sample()

        self.Compute_evaluation_metrics(Ens)

        for t in time_sequence:
            # No dynamics, no update
            Ens = self.sample()

            self.Compute_evaluation_metrics(Ens)

        return Ens

    def Compute_evaluation_metrics(self, Ens, *args, **kwargs):
        return None


class Persistence:
    """
    Persistence baseline.

    Uses the model to propagate Does not use observations
    """

    def step(
        self,
        ground_truth,
        x_b,
        obs,
        time_sequence,
        dynamic_model,
        add_noise=None,
        dt=None,
    ):
        x = x_b.copy()

        self.Compute_evaluation_metrics(x, ground_truth, obs)

        for t in time_sequence:

            # Propagate with model
            x = dynamic_model(x)

            if add_noise is not None:
                x = add_noise(x, dt)

            self.Compute_evaluation_metrics(x, ground_truth, obs)

        return x

    def Compute_evaluation_metrics(self, x, *args, **kwargs):
        return None


class PersistenceEnsemble:
    """
    Persistence ensemble baseline.

    Equivalent to EnKF without the update step.
    """

    def __init__(self, N):
        self.N = N

    def step(
        self,
        ground_truth,
        obs,
        CovX0,
        time_sequence,
        dynamic_model,
        add_noise=None,
        dt=None,
    ):
        # Initial ensemble
        Ens = self.sample(CovX0)

        self.Compute_evaluation_metrics(Ens, ground_truth, obs)

        for t in time_sequence:

            # Forecast step
            Ens = dynamic_model(Ens, t - dt, dt)

            if add_noise is not None:
                Ens = add_noise(Ens, dt)

            self.Compute_evaluation_metrics(Ens, ground_truth, obs)

        return Ens

    def sample(self, CovX0):
        R = np.linalg.cholesky(CovX0)
        return np.random.randn(self.N, R.shape[0]) @ R.T

    def Compute_evaluation_metrics(self, Ens, *args, **kwargs):
        return None
