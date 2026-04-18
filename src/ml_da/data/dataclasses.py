from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xarray as xr

# we create two different data classes here to logically separate the two use cases


@dataclass
class AssimDataBundle:
    """Data used during assimilation."""

    truth: xr.Dataset
    dynamical_model: xr.Dataset
    observations: xr.Dataset
    metadata: dict[str, Any]  # TODO reference the config file in the metadata
    # TODO add sth to get "synthetic numerical model"? (not needed for ML methods)


@dataclass
class TrainDataBundle:
    """Data used during training of the ML models to emulate the 'synthetic numerical models'."""

    truth: xr.Dataset
    dynamical_model: xr.Dataset
    observations: xr.Dataset
    metadata: dict[str, Any]
