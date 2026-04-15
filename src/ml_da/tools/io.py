from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import xarray as xr
import yaml

from ml_da.data.dataclasses import AssimDataBundle

logger = logging.getLogger(__name__)


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


def save_xr_dataset_zarr(ds: xr.Dataset, path: Path, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode=mode)


def load_xr_dataset_zarr(path: Path, chunks: dict[str, int] | None = None) -> xr.Dataset:
    return xr.open_zarr(path, chunks=chunks)


# in case we want to use netcdf instead
def save_xr_dataset_netcdf(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def load_xr_dataset_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def save_data_bundle(bundle: AssimDataBundle, root: Path) -> None:
    """Store data bundle at given path."""
    root.mkdir(parents=True, exist_ok=True)

    save_xr_dataset_zarr(bundle.truth, root / "truth.zarr")
    save_xr_dataset_zarr(bundle.numerical_model, root / "numerical_model.zarr")
    save_xr_dataset_zarr(bundle.observations, root / "observations.zarr")
    save_yaml(bundle.metadata, root / "metadata.yaml")


def load_data_bundle(root: Path, chunks: dict[str, int] | None = None) -> AssimDataBundle:
    bundle = AssimDataBundle(
        truth=load_xr_dataset_zarr(root / "truth.zarr", chunks=chunks),
        numerical_model=load_xr_dataset_zarr(root / "numerical_model.zarr", chunks=chunks),
        observations=load_xr_dataset_zarr(root / "observations.zarr", chunks=chunks),
        metadata=load_yaml(root / "metadata.yaml"),
    )
    return bundle
