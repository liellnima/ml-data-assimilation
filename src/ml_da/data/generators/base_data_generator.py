from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import dabench as dab
import numpy as np
import xarray as xr

from ml_da.data.dataclasses import AssimDataBundle
from ml_da.tools.config import DataCoreConfig, SystemConfig


class DataGenerator(ABC):
    """Class that generates the data for one particular setting."""

    def __init__(self, data_cfg: DataCoreConfig, sys_cfg: SystemConfig) -> None:
        """Init all relevant params."""
        # TODO handle non-linearity mapping differently, this is gonna be confusing in the future
        self.seed = data_cfg.seed
        self.timesteps = data_cfg.timesteps
        # system parameters
        self.system_name = sys_cfg.name
        self.system_params = sys_cfg.params
        self.system_non_linearity_mapping = sys_cfg.non_linearity
        self.chosen_non_linearity = data_cfg.model["non_linearity"]
        # synthetic model params
        self.mod_error_type = data_cfg.model["error_type"]
        self.mod_error_sd = data_cfg.model["error_sd"]
        self.mod_init_conds_noise = data_cfg.model["noise_on_initial_conditions"]
        self.mod_error_pos_only = data_cfg.model["error_pos_only"]
        # observation params
        self.obs_density = data_cfg.observer["density"]
        self.obs_error_type = data_cfg.observer["error_type"]
        self.obs_error_sd = data_cfg.observer["error_sd"]
        self.obs_error_pos_only = data_cfg.observer["error_pos_only"]

        # at the moment we only support normal dist for noise generation
        if self.mod_error_type != "normal":
            raise NotImplementedError("Code needs to modified to support non-Gaussian noise.")

    def get_id_name(self) -> str:
        """Returns a reasonable name, for example for a directory."""
        relevant_params = [
            self.timesteps,
            self.system_name,
            self.chosen_non_linearity,
            self.mod_error_type,
            self.mod_error_sd,
            self.mod_init_conds_noise,
            self.obs_density,
            self.obs_error_type,
            self.obs_error_sd,
            self.seed,
        ]
        return "_".join(map(str, relevant_params))

    @property
    @abstractmethod
    def default_initial_state(self):
        """The default values of the initial state."""

    @abstractmethod
    def _initialize_system(self, initial_state: np.ndarray) -> dab._data.Data:
        """
        Abstract method to initialize the system, e.g. the Lorenz96 system.

        We are using the DataAssimBench data generator objects here.
        Params:
            initial_state (np.ndarray): The initial state with which we want to initialize the system
        Returns:
            Object of  Data class (of DataAssimBench)
        """

    @abstractmethod
    def _create_initial_state(self, error_sd: float = 0) -> np.ndarray:
        """
        Abstract method that returns the initial state.

        With the default value of zero error_sd,
        we get the ground truth of the initial state. By adding a small noise term we can
        perturb the initial state to "simulate" or model. Make sure to use values for the initial states
        after spin up time, i.e. when the system has already diverged and we can see the non-linear effects.
        Params:
            error_sd (float): Noise [TODO describe what noise/error param this is]. If zero (default): perfect values, no noise added.
            TODO seed? maybe?
        Returns:
            np.ndarray: The initial state of the system
        """

    def generate(self) -> AssimDataBundle:
        """
        Runs the data generator.

        Returns:
            AssimDataBundle: Contains `truth`, `model`, `observations`, and `metadata`.
        """
        if self.system is None:
            raise ValueError("Underlying system for the data generation has not been initialized.")

        # Ground Truth: The System's Data
        ground_truth = self._generate_ground_truth()

        # Synthetic Model: The Numerical Model Approximating the System
        model_data = self._generate_model_data()

        # Synthetic Observations: What we can measure of the system
        observations = self._generate_observations(ground_truth)

        # return generated data
        return AssimDataBundle(
            truth=ground_truth,
            model=model_data,
            observations=observations,
            metadata=self.__dict__,
        )

    def _generate_ground_truth(self) -> xr.Dataset:
        """Run the system and return the generated data."""
        # initialize the system
        initial_state = self._create_initial_state()  # noise is kept as default value: 0 for ground truth
        system_obj = self._initialize_system(initial_state)
        # run the system to create ground truth data
        ground_truth_data = system_obj.generate(n_steps=self.timesteps)
        # return the generated data
        return ground_truth_data

    def _generate_model_data(self) -> xr.Dataset:
        """Generate synthetic model data."""
        # Parameterization/Forcing error: add small noise to initial state
        noisy_initial_state = self._create_initial_state(**self.mod_init_conds_noise)
        noisy_system_obj = self._initialize_system(noisy_initial_state)

        # run the system again to get a different output
        model_data = noisy_system_obj.generate(n_steps=self.timesteps)

        # FUTURE Resample error: create lower resolution
        # model_data = self._downsample_resolution(model_data)

        # optional: add noise again (representation error)
        model_data = self._add_noise(
            data=model_data,
            error_type=self.mod_error_type,
            error_params={
                "loc": 0,
                "scale": self.mod_error_sd,
                "size": None,
            },
            only_positive=self.mod_error_pos_only,
        )

        return model_data

    def _generate_noise(
        self,
        error_shape: Tuple,
        error_type: str = "normal",
        error_params: dict[str, Any] = {"loc": 0, "scale": 0.01},
        only_positive: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generates noise by drawing from a distribution.

        error_shape (Tuple): size (the number of samples we draw). Make sure it aligns with the size of our orginal data.
            We want to add the noise generated here onto a different np.ndarray usually. So we  want one sample per entry.
        error_type (str): The type of distribution used for the error.
            Please check that you only use distributions from this list:
            https://numpy.org/doc/2.1/reference/random/generator.html#distributions
        error_params (dict): the params, such as error_sd and error_bias needed to create the noise.
            Please look up the link above to understand which params need to be handed over.
            The param "size" does not matter - it will always be overwritten here.
        only_positive (bool): if the noise should be clipped because negative values would make no sense

        seed (int or None): The seed used for the random generator. should vary when creating model ensembles.
            Default is None: In that case only one model is realized and we take the default seed.

        Returns:
            np.ndarray: the desired noise, i.e. samples drawn from a distribution
        """
        # seeds are only used when generating ensembles, so if it is none we use the default generator obj seed
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed)

        # check if numpy can generate the requested distribution
        all_dists = [dist for dist in dir(rng) if callable(getattr(rng, dist))]
        if error_type not in all_dists:
            raise ValueError(f"Distribution '{error_type}' is not supported by numpy.")

        # get the right distribution func
        func = getattr(rng, error_type)

        # generate the noise data (np.array)
        noise = func(**error_params, size=error_shape)

        # make sure errors are only positive if needed
        if only_positive:
            noise[noise < 0.0] = 0.0

        return noise

    # TODO collaps all the features that are just handed over from one func to another
    def _add_noise(
        self,
        data: xr.Dataset | np.ndarray,
        error_type: str = "normal",
        error_params: dict[str, Any] = {"loc": 0, "scale": 0.01, "size": None},
        only_positive: bool = False,
        seed: int | None = None,
    ) -> xr.Dataset | np.ndarray:
        """
        Adds noise to a given xr.Dataset or an np.ndarray.

        Params:
            data (xr.Dataset or np.ndarray): The data on which the noise / error should be added.
            error_type (str): The type of distribution used for the error.
                Please check that you only use distributions from this list:
                https://numpy.org/doc/2.1/reference/random/generator.html#distributions
            error_params (dict): the params, such as error_sd and error_bias needed to create the noise.
                Please look up the link above to understand which params need to be handed over.
                The param "size" does not matter - it will always be overwritten here.
            only_positive (bool): if the noise should be clipped because negative values would make no sense
            seed (int): The seed that should be used when adding noise
        Returns:
            xr.Dataset or np.ndarray: the same data but with added noise. Returns the same
                type like the provided 'data' type.
        """
        data = data.copy(deep=True)

        # get numpy format of the data we want to manipulate (add noise) if needed
        np_data = None
        if isinstance(data, np.ndarray):
            np_data = data
        elif isinstance(data, xr.Dataset):
            np_data = data.to_array().values
        else:
            raise ValueError(f"Expected xr.Dataset or np.narray but got {type(data)}.")

        # generate the noise
        noise = self._generate_noise(
            error_type=error_type,
            error_params=error_params,
            error_shape=np_data.shape,
            only_positive=only_positive,
            seed=seed,
        )

        # add noise to the np data
        if noise.shape != np_data.shape:
            raise ValueError(
                f"Expected the same shape, but got noise-shape:{noise.shape} != data-shape:{np_data.shape}."
            )
        noisy_np_data = np_data + noise

        # return noisy xr.Dataset if needed
        if isinstance(data, xr.Dataset):
            for i, var in enumerate(data.data_vars):
                data[var].values = noisy_np_data[i]
            return data  # this is the noisy data we are returning

        return noisy_np_data

    def _downsample_resolution(self, data: xr.Dataset, resolution: float = 1) -> xr.Dataset:
        """
        Function to downsample the resolution of an xarray dataset.

        Params:
            data (xr.Dataset): The data that should be downsampled
            resolution (float): Default 1 means nothing is changed. Higher values [TODO mean something]
        Returns:
            xr.Dataset: the downsampled data
        """
        # TODO
        # figure out resolution parameter
        # figure out the best way to accumulate / aggregate xr.Datasets without geospatial people getting a heart attack

        if resolution == 1:
            return data

        raise NotImplementedError()

    def _generate_observations(self, ground_truth: xr.Dataset) -> xr.Dataset:
        """
        Generate synthetic observations.

        Params:
            ground_truth (xr.Dataset): Ground truth from which we sample observations
        Returns:
            xr.Dataset: The dataset containg the observation values, times, locatoins, and errors
        """
        # FUTURE: add custom noise to the output of the system
        # this can be added later if we do not want to use Gaussian noise for the observer
        # in that case: make sure to not add additional noise in the observer object

        # FUTURE: Create lower resolution (_downsample_resolution) if desired

        # FUTURE: split up temporal and local sampling + specific location sampling

        # initialize observer
        obs = dab.observer.Observer(
            ground_truth,
            random_time_density=self.obs_density,
            random_location_density=self.obs_density,
            # random_location_count = 10, # alternative for having measurement stations so to say
            error_sd=self.obs_error_sd,
            random_seed=self.seed,
            error_positive_only=self.obs_error_pos_only,
        )
        # run observer
        obs_data = obs.observe()

        return obs_data
