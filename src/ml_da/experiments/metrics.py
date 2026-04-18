import numpy as np


def init_metrics():
    return {
        "time": [],
        "rmse": [],
        "obs_error": [],
        "mae": [],
        "bias": [],
        "spread": [],
        "crps": [],
        "trHK": [],
    }


def compute_metrics(metrics, estimate=None, truth=None, observation=None, ensemble=None):
    """
    Compute metrics for ONE time step and append to lists.

    Missing values are filled with np.nan to keep alignment.
    """

    state = None
    if ensemble is not None:
        state = np.mean(ensemble, axis=0)
    elif estimate is not None:
        state = estimate

    # Core metrics
    if state is not None and truth is not None:
        rmse = np.sqrt(np.mean((state - truth) ** 2))
        mae = np.mean(np.abs(state - truth))
        bias = np.mean(state - truth)
    else:
        rmse = mae = bias = np.nan

    metrics["rmse"].append(rmse)
    metrics["mae"].append(mae)
    metrics["bias"].append(bias)

    # Observation error
    if state is not None and observation is not None:
        obs_err = np.sqrt(np.mean((observation - state) ** 2))
    else:
        obs_err = np.nan

    metrics["obs_error"].append(obs_err)

    # Ensemble metrics
    if ensemble is not None:
        spread = np.sqrt(np.mean(np.var(ensemble, axis=0)))
    else:
        spread = np.nan

    metrics["spread"].append(spread)

    # CRPS
    if ensemble is not None and truth is not None:
        crps = compute_crps(ensemble, truth)
    else:
        crps = np.nan

    metrics["crps"].append(crps)


def compute_crps(ensemble, truth):
    """CRPS for ensemble forecasts."""

    term1 = np.mean(np.abs(ensemble - truth), axis=0)

    diffs = np.abs(ensemble[:, None, :] - ensemble[None, :, :])
    term2 = np.mean(diffs, axis=(0, 1)) / 2

    return np.mean(term1 - term2)
