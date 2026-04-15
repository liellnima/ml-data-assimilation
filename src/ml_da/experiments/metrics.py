# Compute the metrics
import numpy as np


def init_metrics():
    return {
        "rmse": [],
        "obs_error": [],
        "mae": [],
        "bias": [],
        "spread": [],
        "crps": [],
    }


def compute_metrics(metrics, estimate=None, truth=None, observation=None, ensemble=None):
    """
    metrics: dict returned from init_metrics()
    """
    state = None
    if ensemble is not None:
        state = np.mean(ensemble, axis=0)
    elif estimate is not None:
        state = estimate

    # Core metrics (shared across methods)
    if state is not None and truth is not None:
        rmse = np.sqrt(np.mean((state - truth) ** 2))
        mae = np.mean(np.abs(state - truth))
        bias = np.mean(state - truth)

        metrics["rmse"].append(rmse)
        metrics["mae"].append(mae)
        metrics["bias"].append(bias)

    # Observation error
    if state is not None and observation is not None:
        obs_err = np.sqrt(np.mean((observation - state) ** 2))
        metrics["obs_error"].append(obs_err)

    # Ensemble-specific metrics
    if ensemble is not None:
        spread = np.sqrt(np.mean(np.var(ensemble, axis=0)))
        metrics["spread"].append(spread)

        if truth is not None:
            metrics["crps"].append(compute_crps(ensemble, truth))


def compute_crps(ensemble, truth):
    ensemble.shape[0]

    # term 1: distance to truth
    term1 = np.mean(np.abs(ensemble - truth), axis=0)

    # term 2: pairwise ensemble distances
    diffs = np.abs(ensemble[:, None, :] - ensemble[None, :, :])
    term2 = np.mean(diffs, axis=(0, 1)) / 2

    return np.mean(term1 - term2)
