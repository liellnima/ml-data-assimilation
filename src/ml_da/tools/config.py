from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, List, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ml_da import CONFIG_DIR
from ml_da.tools.io import load_yaml


class DataCoreConfig(BaseModel):
    seed: int
    timesteps: int
    system: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    observer: dict[str, Any] = Field(default_factory=dict)

    # TODO delete this if not needed anymore
    # sys_name: str
    # sys_non_linearity: Literal["low", "med", "hig"]
    # # synthetic model params
    # mod_error_type: Literal["gaussian"]
    # mod_error_sd: float = 1.5
    # mod_init_conds_noise: float = 0.01
    # mod_error_pos_only: bool = False
    # # observation params
    # obs_density: float = 0.5
    # obs_error_type: str = "gaussian"
    # obs_error_sd: float = 1.2
    # # should stay fixed
    # obs_error_pos_only: bool = True


class GeneratorConfig(BaseModel):
    # seed: int
    # timesteps: int
    systems: dict[str, List] = Field(default_factory=dict)
    models: dict[str, List] = Field(default_factory=dict)
    observers: dict[str, List] = Field(default_factory=dict)
    # TODO delete this if not needed anymore
    # sys_name: list[Literal["Lorenz96"]] = ["Lorenz96"]  # keys of dict
    # sys_non_linearity: list[Literal["low", "med", "hig"]] = ["med"]
    # mod_error_type: list[Literal["gaussian"]] = ["gaussian"]
    # mod_error_sd: list[float] = [1.5]
    # obs_density: list[float] = [0.5]
    # obs_error_type: list[str] = ["gaussian"]
    # obs_error_sd: list[float] = [1.2]


class SystemConfig(BaseModel):
    name: str = "Lorenz96"
    non_linearity: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    # system: SystemConfig = Field(default_factory=SystemConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    core: DataCoreConfig = Field(default_factory=DataCoreConfig)


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


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    raw = load_yaml(path)
    resolved = _resolve_configs(raw, path.parent, visited={path})
    return ExperimentConfig.model_validate(resolved)


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


def _config_combination_iterator(generator_config: GeneratorConfig) -> Iterator[dict]:
    """Iterator that yields a dict with the relevant params for all possible combinations."""
    data_dict = generator_config.model_dump()
    keys = data_dict.keys()
    values = data_dict.values()

    # itertools.product creates the Cartesian product of all lists
    for combination in itertools.product(*values):
        # Map the keys back to the specific values in this combination
        yield dict(zip(keys, combination))


def get_data_and_system_cfgs(
    data_generator_cfg: GeneratorConfig, data_core_cfg: DataCoreConfig
) -> list[Tuple[DataCoreConfig, SystemConfig]]:
    """
    Given a list of changing dataset params and the core dataset params, this function return a list providing the
    DataCoreConfig and SystemConfig obj for all possible param combinations in the Generator.

    Params:
        data_generator_cfg (GeneratorConfig): The params that are changing for different datasets.
            The function returns all possible datasets that can be created from the params.
        data_core_cfg (DataCoreConfig): The default dataset. Any param that does not show up in the
            generator config is going to stay the same as indicated in this core config.

    Returns:
        List(Tuple(<DataCoreConfig>, <SystemConfig>)): The length of the list is the number
            of possible combinations (i.e. each entry corresponds to one possible dataset).
            It contains the core data configs and the system configs needed to create this dataset.
    """
    # create all possible combinations of datasets given the generator configs
    cfg_iter = _config_combination_iterator(data_generator_cfg)
    all_cfgs = []  # first entry in each tuple is data core cfg, second entry is system cfg

    # iterate over all possible config combinations and overwrite the default values of the DataCoreConfig that way
    for params in cfg_iter:
        # get the default config values of our data object (only some of them are overwritting by the generator config)
        default_core_cfg = data_core_cfg.model_copy()
        # TODO check if unpacking here does actually overwrite the right values as expected
        # unpack the current values into the default dict
        curr_dataset_cfg = default_core_cfg(**params)

        # TODO the whole system thing is not a pretty solution imo,
        # we accept it as tech debt for now
        system_cfg_path = CONFIG_DIR / "data" / "systems" / f"{curr_dataset_cfg.system}.yaml"
        system_cfg = SystemConfig(**load_yaml(system_cfg_path))

        all_cfgs.append((curr_dataset_cfg, system_cfg))
