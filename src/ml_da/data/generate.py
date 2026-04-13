import concurrent.futures
import logging
from pathlib import Path

from ml_da import DATA_DIR, LOG_DIR
from ml_da.data.generators.base_data_generator import DataGenerator
from ml_da.tools.config import (
    DataCoreConfig,
    GeneratorConfig,
    SystemConfig,
    get_data_and_system_cfgs,
)
from ml_da.tools.io import save_data_bundle
from ml_da.tools.logger import setup_logging

# NEXT
# TODO Implement the orchestrator (configs handling + running everything)
# TODO Get dataset creation running as it is right now

# SOON
# TODO Extend to have ensembles of models
# TODO Training Data vs Assimilation Data
# TODO Methods-Data interface (BaseModel)
# TODO DataBundle: make sure we can access covariance
# TODO DataBundle: make sure we can access Jacobian
# TODO DataBundle: make sure we have the syn numerical model (runnable!)
# TODO How to handle the syn numerical models in the whole framework
# TODO: Figure out how to sync numpy vs torch usage of our data bundle

# LATER
# TODO consider system being more strongly separated out from data (because we need them for "models" as well)
# TODO should we separate the training vs assimilation data generation
# to create better modularity?
# TODO Tests: ... tests ...
# TODO Seeds: think carefully about where to put seeds at some point

# BACK
# TODO error_params instead of specific error_sd --> needs re-designing though - small steps to improvement make more sense

GENERATOR_REGISTRY = {c.__name__: c for c in DataGenerator.__subclasses__()}

# Note: If you want to print something in here, you need to set up a new logger within each task
# AND make sure that you have a separate logging file for each task


def generate_single_dataset(data_cfg: DataCoreConfig, system_cfg: SystemConfig) -> Path:
    """
    Generates a single dataset from a data core config and a system config.

    Params:
        data_cfg (DataCoreConfig): The config containing all relevant params for the dataset
        system_cfg (SystemConfig): The config containing all relevant params for the underlying system
    Returns:
        Path: Where the dataset has been stored. You can use load_data_bundle to load it.
    """
    # TODO unclear to me how to setup individual worker logging here

    # create the generator and generate the data
    data_generator = GENERATOR_REGISTRY["Lorenz96"](data_cfg, system_cfg)
    data_bundle = data_generator.generate()
    # create path for this specific dataset
    path = DATA_DIR / "Set{idx:03d}_" + data_generator.get_id_name()
    # store the data
    save_data_bundle(data_bundle, path)

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
    # setup logging
    setup_logging(LOG_DIR)
    data_logger = logging.getLogger("DataGeneration")

    # get specific data core and system configs for all possible combinations of the dataset params in generator
    all_data_and_system_cfgs = get_data_and_system_cfgs(data_generator_cfg, data_core_cfg)

    # parellelize execution of dataset generation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # schedule all tasks and make sure we can track the different processes
        task_id_dict = {
            executor.submit(generate_single_dataset, data_cfg, system_cfg): task_id
            for task_id, (data_cfg, system_cfg) in all_data_and_system_cfgs
        }

        # check out completed tasks
        all_paths = []
        for task in concurrent.futures.as_completed(task_id_dict):
            task_id = task_id_dict[task]
            try:
                data_path = task.result()
                all_paths.append(data_path)
                data_logger.info(f"Generated Dataset with Task ID {task_id}.")
            except Exception as e:
                data_logger.error(f"ERROR: Dataset Generation with Task ID {task_id} failed: {e}")

    return all_paths  # reminder for myself: 'with blocks' do not have their own variable scope
