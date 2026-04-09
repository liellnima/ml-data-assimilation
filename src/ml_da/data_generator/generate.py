from typing import Any

from ml_da.tools.config import DataListConfig

# TODO ADD NOISE function (crucial!)

# TODO Loader / Creator / Storage: create something for handling the data (generate vs store vs load)
# TODO Implement the orchestrator which passes configs to this function
# and runs everything
# TODO Prallelized: figure out how to generate a whole Dataset in a parallelized way
# --> should be its own class I assume? do i need a mapping to figure out which DataGenerator needs to be created?

# TODO Interface to models: find out how to make this a nice torch dataset or Hidden Markov Model or sth like that

# TODO Tests: ... tests ...
# TODO Seeds: think carefully about where to put seeds at some point


def generate_dataset(data_list_cfg: DataListConfig) -> dict[str, Any]:
    """
    Generates multiple datasets across lists of configs.

    Creates a mapping for all setups and parallelizes the creation of the datasets.
    """
    # set seed for data generation

    # iterate over all configs and create a mapping for all the datasets that need to be generated

    # create objects for all (parallelize?)

    # run data generation for all (parellize?)
    # and save the output (should be in tools / configs)

    # TODO somewhere else: create a Pytorch DataSet that loads the data from files
