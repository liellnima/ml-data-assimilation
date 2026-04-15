from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, List, Tuple

from pydantic import BaseModel, Field

from ml_da import CONFIGS_DIR
from ml_da.tools.io import load_yaml


class DataCoreConfig(BaseModel):
    seed: int = 1234
    timesteps: int = 1000
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
    system: dict[str, List] = Field(default_factory=dict)
    model: dict[str, List] = Field(default_factory=dict)
    observer: dict[str, List] = Field(default_factory=dict)
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
    metrics: list[str] = Field(default_factory=lambda: ["rmse", "mae", "mse"])
    # TODO add running time, inference time, and stability measures


class AggrConfig(BaseModel):
    pass
    # TODO add stuff
    #


class VizConfig(BaseModel):
    pass
    # TODO add stuff


class ExperimentConfig(BaseModel):
    experiment_name: str = "default-run"
    overwrite: bool = True
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    aggregation: AggrConfig = Field(default_factory=AggrConfig)
    visualization: VizConfig = Field(default_factory=VizConfig)


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
    nested_configs = raw_cfg.get("configs", [])
    if not nested_configs:
        return raw_cfg

    merged: dict[str, Any] = {}

    for item in nested_configs:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each configs entry must look like `- section: path/to/file.yaml`")

        section, rel_path = next(iter(item.items()))
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

        # add the dictionary key that describes the included (nested) config file
        # if it is not already contained in the newly included file
        if isinstance(included_resolved, dict) and section in included_resolved:
            sectioned_include = included_resolved
        else:
            sectioned_include = {section: included_resolved}

        merged = _deep_merge(merged, sectioned_include)

    # main config overrides included configs
    raw_without_nested_configs = {k: v for k, v in raw_cfg.items() if k != "configs"}
    merged = _deep_merge(merged, raw_without_nested_configs)

    return merged


def _collect_list_leaves(
    obj: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], list[Any]]]:
    """
    Recursively collect all nested leaf paths whose values are lists.

    Example return value:
        [
            (("systems", "name"), ["Lorenz96"]),
            (("systems", "non_linearity"), ["low", "med", "hig"]),
            ...
        ]
    """
    leaves: list[tuple[tuple[str, ...], list[Any]]] = []

    for key, value in obj.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            leaves.extend(_collect_list_leaves(value, path))
        elif isinstance(value, list):
            leaves.append((path, value))
        else:
            raise TypeError(
                f"Expected nested dicts or lists at path {'.'.join(path)}, " f"but got {type(value).__name__}."
            )

    return leaves


def _set_nested_value(obj: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested value in a dict given a path tuple."""
    curr = obj
    for key in path[:-1]:
        curr = curr[key]
    curr[path[-1]] = value


def _config_combination_iterator(generator_config: GeneratorConfig) -> Iterator[dict[str, Any]]:
    """
    Yield all possible concrete configurations from a nested GeneratorConfig.

    Each yielded dict has the same nested structure as generator_config.model_dump(),
    but every list leaf is replaced by a single selected value.

    Example:
        {
            "systems": {"name": "Lorenz96", "non_linearity": "low"},
            "models": {"error_type": "normal", "error_sd": 0.1},
            "observers": {"density": 0.75, "error_type": "gaussian", "error_sd": 0.1},
        }
    """
    data_dict = generator_config.model_dump()
    list_leaves = _collect_list_leaves(data_dict)

    if not list_leaves:
        yield deepcopy(data_dict)
        return

    paths = [path for path, _ in list_leaves]
    value_lists = [values for _, values in list_leaves]

    for combination in itertools.product(*value_lists):
        config_instance = deepcopy(data_dict)
        for path, value in zip(paths, combination):
            _set_nested_value(config_instance, path, value)
        yield config_instance


def _update_data_core_cfg(
    data_core_cfg: DataCoreConfig,
    params: dict[str, Any],
) -> DataCoreConfig:
    """
    Return a new DataCoreConfig where nested fields are deep-updated from the provided params dict.

    Params:
        data_core_cfg (DataCoreConfig): The basic data core config that we want to adapt.
            We want to keep the default values of this obj for all the args that do not show up in params.
        params (dict[str, Any]): The params that should be used to update the data core config.

    Returns:
        DataCoreConfig: A new and updated data core config
    """
    base_dict = data_core_cfg.model_dump()
    merged_dict = _deep_merge(base_dict, params)
    return DataCoreConfig.model_validate(merged_dict)


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
        # and unpack the current values into the default dict
        curr_dataset_cfg = _update_data_core_cfg(data_core_cfg, params)

        # TODO the whole system thing is not a pretty solution imo,
        # we accept it as tech debt for now
        system_id_name = curr_dataset_cfg.system["name"]
        system_cfg_path = CONFIGS_DIR / "data" / "systems" / f"{system_id_name}.yaml"
        system_cfg = SystemConfig(**load_yaml(system_cfg_path))

        all_cfgs.append((curr_dataset_cfg, system_cfg))

    return all_cfgs
