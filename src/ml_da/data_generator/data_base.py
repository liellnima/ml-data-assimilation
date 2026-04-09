from __future__ import annotations

from abc import ABC, abstractmethod

import dabench as dab
import numpy as np
import xarray as xr

from ml_da.tools.config import DataConfig, SystemConfig


class DataGenerator(ABC):
    """Class that generates the data for one particular setting."""

    def __init__(self, data_cfg: DataConfig, sys_cfg: SystemConfig) -> None:
        """Init all relevant params."""
        # TODO handle non-linearity mapping differently, this is gonna be confusing in the future
        self.seed = data_cfg.seed
        # system parameters
        self.timesteps = data_cfg.timesteps
        self.system_name = sys_cfg.name
        self.system_params = sys_cfg.params
        self.system_non_linearity_mapping = sys_cfg.params
        self.chosen_non_linearity = data_cfg.non_linearity
        # synthetic model params
        self.mod_error_type = data_cfg.mod_error_type
        self.mod_error_sd = data_cfg.mod_error_sd
        self.mod_init_conds_noise = data_cfg.mod_init_conds_noise
        self.mod_error_pos_only = data_cfg.mod_error_pos_only
        # observation params
        self.obs_density = data_cfg.obs_density
        self.obs_error_type = data_cfg.obs_error_type
        self.obs_error_sd = data_cfg.obs_error_sd
        self.obs_error_pos_only = data_cfg.obs_error_pos_only

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

    def generate(self) -> dict[str, xr.Dataset]:
        """
        Runs the data generator.

        Returns:
            dict(str, xr.Dataset): Contains `ground_truth`, `model_data`, and `observations`.
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
        return {"ground_truth": ground_truth, "model_data": model_data, "observations": observations}

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
        noisy_initial_state = self._create_initial_state(noise=self.mod_init_conds_noise)
        noisy_system_obj = self._initialize_system(noisy_initial_state)

        # run the system again to get a different output
        model_data = noisy_system_obj.generate(n_steps=self.timesteps)

        # Resample error: create lower resolution
        model_data = self._downsample_resolution(model_data)

        # optional: add noise again (representation error)
        model_data = self._add_noise(
            model_data,
            self.mod_error_type,
            self.mod_error_sd,
            self.mod_error_pos_only,
        )

        return model_data

    def _add_noise(
        self, data: xr.Dataset, error_type: str = "gaussian", error_sd: float = 0.01, only_positive: bool = False
    ) -> xr.Dataset:
        """
        Adds noise to a given xr.Dataset.

        Params:
            error_type (str)
        Returns:
            xr.Dataset: the same data but with added noise
        """
        # TODO
        # figure out how to handle data as xr.Dataset vs np.ndarray (should work for both)
        # figure out how to handle seeds
        # create the right random generator according to noise type (create mapping somewhere in a separate file)
        # create the right level of noise (figure out what the best way is for that)
        # consider:
        # errors_vector = rng.normal(loc=error_bias,
        #                     scale=error_sd,
        #                     size=errors_vec_size)
        # errors_vector[errors_vector < 0.] = 0.
        # and apply somehow to dataset --> check out observer.py in DataAssimBench
        if error_sd == 0:
            return data

        raise NotImplementedError()

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
