from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing as mp
from pathlib import Path

from ml_da import DATA_DIR
from ml_da.data.data_generator import DataGenerator
from ml_da.tools.config import (
    DataCoreConfig,
    GeneratorConfig,
    build_cfg_combos,
)
from ml_da.tools.io import save_data_bundle, save_yaml

logger = logging.getLogger(__name__)

# Note: If you want to print something in here, you need to set up a new logger within each task
# AND make sure that you have a separate logging file for each task


def generate_single_dataset(
    data_cfg: DataCoreConfig,
    id: int = 0,
) -> Path:
    """
    Generates a single dataset from a data core config and a system config.

    Params:
        data_cfg (DataCoreConfig): The config containing all relevant params for the dataset
        system_cfg (SystemConfig): The config containing all relevant params for the underlying system
        id (int): Used to index the generation of multiple datasets. Must be unique.
    Returns:
        Path: Where the dataset has been stored. You can use load_data_bundle to load it.
    """
    # TODO unclear to me how to setup individual worker logging here

    # create the generator and generate the data
    data_generator = DataGenerator(
        data_cfg,
        data_cfg.system,
        data_cfg.model,
        data_cfg.observer,
    )
    data_bundle = data_generator.generate()

    # create path for this specific dataset
    path = DATA_DIR / f"Dataset-{id:03}_{data_generator.get_id_name()}"

    # store the data and the configs
    save_data_bundle(data_bundle, path)
    # TODO make sure that has the resolved configs
    save_yaml(data_cfg.model_dump(), path / "configs.yaml")

    return path


def generate_datasets(data_generator_cfg: GeneratorConfig, data_core_cfg: DataCoreConfig) -> list[Path]:
    """
    IMPORTANT: This function can only be called from within a if __name__ == "__main__" block!
    Generates multiple datasets across lists of configs. This is trying to
    execute things parallelized.

    Params:
        data_generator_cfg (GeneratorConfig): The params over which should be looped to create all possible datasets.
        data_core_cfg (DataCoreConfig): Default configs - anything not listed in the generator config, will use the default values of the data core configs.

    Returns:
        list(Path): List of Paths where all the data has been stored.
    """
    # get specific data core and system configs for all possible combinations of the dataset params in generator
    all_cfg_combos = build_cfg_combos(data_generator_cfg, data_core_cfg)

    # test this on a single case
    # generate_single_dataset(all_cfg_combos[0], id=999)
    # print("Got one dataset, go check it out :D")
    # exit(0)

    # I get a forking warning with potential deadlock because of jax (already doing some threads)
    # -> ChatGPT suggested to use "spawn" or "forkserver" to make sure stuff is single-threaded
    # TODO Tech Debt warning: I do not fully understand what is going on here
    ctx = mp.get_context("forkserver")

    # parellelize execution of dataset generation
    with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as executor:
        # schedule all tasks and make sure we can track the different processes
        task_id_dict = {
            executor.submit(generate_single_dataset, data_cfg, task_id): task_id
            for task_id, (data_cfg) in enumerate(all_cfg_combos)
        }

        # check out completed tasks
        all_paths = []
        for task in concurrent.futures.as_completed(task_id_dict):
            task_id = task_id_dict[task]
            try:
                data_path = task.result()
                all_paths.append(data_path)
                logger.info(f"Generated Dataset with Task ID {task_id}.")
            except Exception as e:
                logger.error(f"ERROR: Dataset Generation with Task ID {task_id} failed: {e}")

    return all_paths  # reminder for myself: 'with blocks' do not have their own variable scope
