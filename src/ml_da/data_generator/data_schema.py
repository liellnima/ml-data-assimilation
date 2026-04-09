from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xarray as xr


@dataclass
class DataBundle:
    truth: xr.Dataset
    model: xr.Dataset
    observations: xr.Dataset
    metadata: dict[str, Any]
