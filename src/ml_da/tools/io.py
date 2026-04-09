from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in: {path}")
    return data


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_xr_dataset_zarr(
    ds: xr.Dataset,
    path: Path,
    mode: str = "w",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(path, mode=mode)


def load_xr_dataset_zarr(
    path: Path,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    return xr.open_zarr(path, chunks=chunks)


# TODO in case we want to use netcdf instead


def save_xr_dataset_netcdf(
    ds: xr.Dataset,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def load_xr_dataset_netcdf(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


# TODO continue here: add save and load data bundle and figure out the path management
