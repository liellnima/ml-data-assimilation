from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml_da.tools.io import load_yaml


class DataConfig(BaseModel):
    seed: int = 42
    timesteps: int = 1000
    # system: Literal["lorenz96"] = "lorenz96"
    non_linearity: Literal["low", "med", "hig"] = "med"
    # synthetic model params
    mod_error_type: Literal["gaussian"] = "gaussian"
    mod_error_sd: float = 1.5
    mod_init_conds_noise: float = 0.01
    mod_error_pos_only: bool = False
    # observation params
    obs_density: float = 0.5
    obs_error_type: str = "gaussian"
    obs_error_sd: float = 1.2
    # should stay fixed
    obs_error_pos_only: bool = True


# this one is to iterate through multiple datasets
# TODO this is not solved elegantly atm
# TODO all of this is probably breaking, to be fixed
# but we accept it as tech debt for now


class DataListConfig(BaseModel):
    # seed: int = 42
    # timesteps: int = 1000
    system: list[Literal["lorenz96"]] = ["lorenz96"]  # keys of dict
    non_linearity: list[Literal["low", "med", "hig"]] = ["med"]
    mod_noise_type: list[Literal["gaussian"]] = ["gaussian"]
    mod_noise: list[float] = [1.5]
    density: list[float] = [0.5]
    obs_noise_type: list[str] = ["gaussian"]
    obs_noise: list[float] = [1.2]


class SystemConfig(BaseModel):
    name: str = "lorenz96"
    non_linearity: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str = "persistence"
    params: dict[str, Any] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["rmse", "mae"])
    # TODO add running time, inference time, and stability measures


class AggrConfig(BaseModel):
    pass
    # TODO add stuff
    #


class VizConfig(BaseModel):
    pass
    # TODO add stuff


class OutputConfig(BaseModel):
    run_name: str | None = None
    overwrite: bool = False


class ConfigReference(BaseModel):
    """
    Represents one entry in the defaults list, e.g.
    - data: data/default_data_generator.yaml
    """

    model_config = ConfigDict(extra="allow")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_name: str = "default-run"
    stage: list[Literal["generate", "run", "evaluate", "aggregate", "visualize"]] = Field(
        default_factory=lambda: ["generate", "run", "evaluate", "aggregate", "visualize"]
    )
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    aggregation: AggrConfig = Field(default_factory=AggrConfig)
    visualization: VizConfig = Field(default_factory=VizConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values in override take precedence over base.
    """
    result = deepcopy(base)

    for key, override_value in override.items():
        base_value = result.get(key)

        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = deepcopy(override_value)

    return result


def _resolve_configs(
    raw_cfg: dict[str, Any],
    config_dir: Path,
    visited: set[Path],
) -> dict[str, Any]:
    """
    Resolve `configs` entries recursively.

    Example:
        defaults:
          - data: data/lorenz.yaml
          - model: models/enkf.yaml
    """
    defaults = raw_cfg.get("configs", [])
    if not defaults:
        return raw_cfg

    merged: dict[str, Any] = {}

    for item in defaults:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each configs entry must look like `- section: path/to/file.yaml`")

        _, rel_path = next(iter(item.items()))
        include_path = (config_dir / rel_path).resolve()

        if include_path in visited:
            chain = " -> ".join(str(p) for p in visited)
            raise ValueError(f"Cyclic config include detected: {chain} -> {include_path}")

        if not include_path.exists():
            raise FileNotFoundError(f"Included config not found: {include_path}")

        visited.add(include_path)
        included_raw = load_yaml(include_path)
        included_resolved = _resolve_configs(
            included_raw,
            include_path.parent,
            visited,
        )
        visited.remove(include_path)

        merged = _deep_merge(merged, included_resolved)

        # main config overrides included configs
        raw_without_defaults = {k: v for k, v in raw_cfg.items() if k != "defaults"}
        merged = _deep_merge(merged, raw_without_defaults)
        return merged


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = load_yaml(path)
    resolved = _resolve_configs(raw, path.parent, visited={path})
    return ExperimentConfig.model_validate(resolved)
