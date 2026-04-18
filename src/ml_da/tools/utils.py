from typing import Any

import xarray as xr


def str_join_ls(name_and_params: list[Any]) -> str:
    """Returns a joined string of the items in the list."""
    return "-".join(map(str, name_and_params))


def get_state(data: xr.Dataset, time: int = -1):
    """
    Get state of dataset at time t.

    -1 returns the last state. 0 the first.
    """
    return data.isel(time=time).to_array().data.flatten()
