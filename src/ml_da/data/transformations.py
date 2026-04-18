from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def generate_noise(
    error_shape: tuple,
    error_type: str = "normal",
    error_params: dict[str, Any] = {"loc": 0, "scale": 0.01},
    only_positive: bool = False,
    seed: int = 123456,
) -> np.ndarray:
    """
    Generates noise by drawing from a distribution.

    error_shape (Tuple): size (the number of samples we draw). Make sure it aligns with the size of our orginal data.
        We want to add the noise generated here onto a different np.ndarray usually. So we  want one sample per entry.
    error_type (str): The type of distribution used for the error.
        Please check that you only use distributions from this list:
        https://numpy.org/doc/2.1/reference/random/generator.html#distributions
    error_params (dict): the params, such as error_sd and error_bias needed to create the noise.
        Please look up the link above to understand which params need to be handed over.
        The param "size" does not matter - it will always be overwritten here.
    only_positive (bool): if the noise should be clipped because negative values would make no sense

    seed (int or None): The seed used for the random generator. should vary when creating model ensembles.
        Default is None: In that case only one model is realized and we take the default seed.

    Returns:
        np.ndarray: the desired noise, i.e. samples drawn from a distribution
    """
    rng = np.random.default_rng(seed)

    # check if numpy can generate the requested distribution
    all_dists = [dist for dist in dir(rng) if callable(getattr(rng, dist))]
    if error_type not in all_dists:
        raise ValueError(f"Distribution '{error_type}' is not supported by numpy.")

    # get the right distribution func
    func = getattr(rng, error_type)

    # generate the noise data (np.array)
    error_params["size"] = error_shape
    noise = func(**error_params)

    # make sure errors are only positive if needed
    if only_positive:
        noise[noise < 0.0] = 0.0

    return noise


def convert_to_np(
    data: xr.Dataset | np.ndarray | list[xr.Dataset] | list[np.ndarray],
) -> np.ndarray:
    """Converts np.ndarrays, xr.Datasets, and lists of those two types into np.ndarrays."""
    np_data = None
    # single model case: we either get a numpy array or an xr.Dataset
    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, xr.Dataset):
        np_data = data.to_array().values
    # ensemble case: we either get a list of np array or a list of xr.Datasets
    elif isinstance(data, list):
        if isinstance(data[0], np.ndarray):
            np_data = np.stack(data)
        elif isinstance(data[0], xr.Dataset):
            np_data = np.stack([ds.to_array().values for ds in data])
        else:
            raise ValueError(f"Can only work with lists of np.arrays or xr.Datasets, but got list of {type(data[0])}.")
    else:
        raise ValueError(f"Expected xr.Dataset or np.narray, or a list of those, but got {type(data)}.")

    return np_data


def convert_np_to_original_format(
    original_data: xr.Dataset | np.ndarray | list[xr.Dataset] | list[np.ndarray], new_np_data: np.ndarray
) -> xr.Dataset | np.ndarray | list[xr.Dataset] | list[np.ndarray]:
    """Returns the np.ndarray in the original data format."""
    reformatted = None
    # single model case: we either get a numpy array or an xr.Dataset
    if isinstance(original_data, np.ndarray):
        reformatted = new_np_data
    elif isinstance(original_data, xr.Dataset):
        for i, var in enumerate(original_data.data_vars):
            original_data[var].values = new_np_data[i]
        reformatted = original_data
    # ensemble case: we either get a list of np array or a list of xr.Datasets
    elif isinstance(original_data, list):
        if isinstance(original_data[0], np.ndarray):
            reformatted = list(np.unstack(new_np_data, axis=0))
        elif isinstance(original_data[0], xr.Dataset):
            reformatted = []
            for i, ds in enumerate(original_data):
                # no deep copy needed, done beforehand already in add_noise
                for j, var in enumerate(ds.data_vars):
                    ds[var].values = new_np_data[i, j]  # this will be the new data
                reformatted.append(ds)
        else:
            raise ValueError(
                f"Can only work with lists of np.arrays or xr.Datasets, but got list of {type(original_data[0])}."
            )
    else:
        raise ValueError(f"Expected xr.Dataset or np.narray, or a list of those, but got {type(original_data)}.")

    return reformatted


def add_noise(
    data: xr.Dataset | np.ndarray | list[xr.Dataset] | list[np.ndarray],
    error_type: str = "normal",
    error_params: dict[str, Any] = {"loc": 0, "scale": 0.01, "size": None},
    only_positive: bool = False,
    seed: int = 123456,
) -> xr.Dataset | np.ndarray | list[xr.Dataset] | list[np.ndarray]:
    """
    Adds noise to a given xr.Dataset or an np.ndarray.

    Params:
        data (xr.Dataset or np.ndarray or list of those): The data on which the noise / error should be added.
        error_type (str): The type of distribution used for the error.
            Please check that you only use distributions from this list:
            https://numpy.org/doc/2.1/reference/random/generator.html#distributions
        error_params (dict): the params, such as error_sd and error_bias needed to create the noise.
            Please look up the link above to understand which params need to be handed over.
            The param "size" does not matter - it will always be overwritten here.
        only_positive (bool): if the noise should be clipped because negative values would make no sense
        seed (int): The seed that should be used when adding noise
    Returns:
        xr.Dataset or np.ndarray: the same data but with added noise. Returns the same
            type like the provided 'data' type.
    """
    # make sure we are not changing the original data
    data = copy.deepcopy(data)

    # get numpy format of the data we want to manipulate (add noise) if needed
    np_data = convert_to_np(data)

    # generate the noise
    noise = generate_noise(
        error_type=error_type,
        error_params=error_params,
        error_shape=np_data.shape,
        only_positive=only_positive,
        seed=seed,
    )

    # add noise to the np data
    if noise.shape != np_data.shape:
        raise ValueError(f"Expected the same shape, but got noise-shape:{noise.shape} != data-shape:{np_data.shape}.")
    noisy_np_data = np_data + noise

    reformatted_noisy_data = convert_np_to_original_format(
        original_data=data,
        new_np_data=noisy_np_data,
    )

    return reformatted_noisy_data


def downsample_resolution(data: xr.Dataset, resolution: float = 1) -> xr.Dataset:
    """
    Function to downsample the resolution of an xarray dataset.

    Params:
        data (xr.Dataset): The data that should be downsampled
        resolution (float): Default 1 means nothing is changed. Higher values [TODO mean something]
    Returns:
        xr.Dataset: the downsampled data
    """
    # TODO
    # figure out resolution parameter
    # figure out the best way to accumulate / aggregate xr.Datasets without geospatial people getting a heart attack

    if resolution == 1:
        return data

    raise NotImplementedError()
