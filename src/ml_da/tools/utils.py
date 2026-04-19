from typing import Any

import numpy as np
import xarray as xr


def str_join_ls(name_and_params: list[Any]) -> str:
    """Returns a joined string of the items in the list."""
    return "-".join(map(str, name_and_params))


def get_state(data: xr.Dataset, time: int = -1) -> np.ndarray:
    """
    Get state of dataset at time t.

    -1 returns the last state. 0 the first.
    """
    return data.isel(time=time).to_array().data.flatten()


def get_all_states(ds: xr.Dataset | list[xr.Dataset]) -> np.ndarray | list[np.ndarray]:
    """
    Returns all states of a trajectory as a numpy array.

    In case it's an ensemble (a list of xr.Datasets), it returns a list of np.ndarray  states.
    """
    all_states = None
    if isinstance(ds, list):
        all_states = [d.to_array().data[0] for d in ds]
    elif isinstance(ds, xr.Dataset):
        all_states = ds.to_array().data[0]
    else:
        raise ValueError("Expected xr.Dataset or list of those, but got {type(data)}.")
    return all_states
