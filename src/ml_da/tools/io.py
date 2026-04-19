from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
import xarray as xr
import yaml
from pydantic import BaseModel

from ml_da.data.dataclasses import AssimDataBundle

logger = logging.getLogger(__name__)


def save_pickle(data: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in: {path}")
    return data


def save_yaml(data: dict[str, Any] | List, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data_list = json.load(f)
    return np.array(data_list)


def save_json(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(arr.tolist(), f)


def save_xr_dataset_zarr(ds: xr.Dataset | list[xr.Dataset], path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # in case of ensemble data
    if isinstance(ds, list):
        for i, single_ds in enumerate(ds):
            mode = "w" if i == 1 else "a"
            # TODO make sure this is not saved with '{}' around it (happening right now for some reason??)
            single_ds.to_zarr(path, group={f"ensemble_{i}"}, mode=mode)
    # in most cases we just have a single dataset
    elif isinstance(ds, xr.Dataset):
        ds.to_zarr(path, mode=mode)
    else:
        raise ValueError("Can only save xr.Dataset or a list of those.")


def load_xr_dataset_zarr(path: Path, group=None, chunks: dict[str, int] | None = None) -> xr.Dataset:
    return xr.open_zarr(path, group=group, chunks=chunks)


# in case we want to use netcdf instead
def save_xr_dataset_netcdf(ds: xr.Dataset, path: Path) -> None:
    if isinstance(ds, list):
        raise NotImplementedError
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def load_xr_dataset_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def save_data_bundle(bundle: AssimDataBundle, root: Path) -> None:
    """Store data bundle at given path."""
    root.mkdir(parents=True, exist_ok=True)

    save_xr_dataset_zarr(bundle.truth, root / "truth.zarr")
    save_xr_dataset_zarr(bundle.dynamical_model, root / "dynamical_model.zarr")
    save_xr_dataset_zarr(bundle.observations, root / "observations.zarr")
    save_yaml(bundle.metadata, root / "metadata.yaml")


def load_data_bundle(
    root: Path, ensembles: list | None = None, chunks: dict[str, int] | None = None
) -> AssimDataBundle:
    # differentiate between ensemble case and loading a single model
    if ensembles is not None:
        dyn_model_data = [
            load_xr_dataset_zarr(root / "dynamical_model.zarr", group=f"{{'ensemble_{i}'}}", chunks=chunks)
            for i in ensembles
        ]
    else:
        dyn_model_data = load_xr_dataset_zarr(root / "dynamical_model.zarr", chunks=chunks)

    bundle = AssimDataBundle(
        truth=load_xr_dataset_zarr(root / "truth.zarr", chunks=chunks),
        dynamical_model=dyn_model_data,
        observations=load_xr_dataset_zarr(root / "observations.zarr", chunks=chunks),
        metadata=load_yaml(root / "metadata.yaml"),
    )
    return bundle


def is_serializable_type(obj):
    """Helper to check if a type is 'safe' or can be processed further."""
    allowed_basics = (str, int, float, bool, list, tuple, dict, type(None))
    return isinstance(obj, allowed_basics) or isinstance(obj, (BaseModel))


def prepare_for_yaml(data):
    # handle pydantic objects
    if isinstance(data, BaseModel):
        return prepare_for_yaml(data.model_dump())

    # step recursively through list and tuples
    if isinstance(data, (list, tuple)):
        return [prepare_for_yaml(item) for item in data if is_serializable_type(item)]

    # return basic stuff
    if isinstance(data, (str, int, float, bool, type(None))):
        return data

    # fallback, convert to string
    return str(data)
