# Current List of Todos

We need to implement:

- [] Data Generator
- [] Task Setup
- [] Experiments
- [] Metrics
- [] Models

Other:

- [x] Environment Setup
- [] Orchestrator
- [] Config Handler
- [] Plotting
- [] WandB
- [] CLI

## Data Generator

- [] 3 levels scarcity in obs (space/time)
- [] 3 levels of model noise
- [] 3 levels of observation noise
- [] 3 levels of nonlinearity

--> 81 datasets

- [] Separate between training and assimilation data
- [] Provide Covariance + Jacobian
- [] Provide model object
- [] Provide Model Ensembles

## Task Setup

- [] Interface between Data and Models
- [] Adapted to reflect filtering task

## Experiments

- [] How to compare different models (ML vs non-ML methods) code-wise?
- [] WandB setup
- [] overall code setup
- [] Cluster setup
- [] Run each model 3 times

--> I would propose to not do any parameter tuning. I don't think we have time for this.

## Metrics

- [] RMSE / MAE
- [] STD of RMSE across 3 runs
- [] Inference time
- [] Training time
- [] Robustness: how strongly changes the output for small perturbations of the input?

--> Do robustness during inference time: check how strongly the outputs change after applying a noise term?

## Models

- []
- []
- []
- [] ML
- [] ML
- [] ML

## Tests

Client and other stuff around:

- [] Test if you can execute load_config(config) in client.py and assert all entries of the resolved experiment_cfg (for the default.yamls). Consider getting specific testing configs ready for this. --> test the recursive solution, as well as overwriting default values, and filling in missing aspects with default values, and getting a valid ExperimentConfig.
- [] Test make_run_dir
- [] Test if the cli.py setup is working (can we call an empty function)?
- [] Test if the orchestrator is entering the right stage
- [] Test registry by comparing it with manual string list

Configs:

- [] Test config_combination_iterator
- [] Test if the get_data_and_system_cfgs is working as expected. Check for length of output (e.g. expecting 81 for 3x3x3x3 options). Take specific example and test if we get the exact output as expected. Check that two consec items are not the same (should be the same in all but one factor).
- [] Test the \_update_data_core_cfg which deep copies a Default DataCoreConfig object and updates it params with user-defined one (from the combos). Does the output have the right type? Does it have the default values? Does it have the new user-defined values?
- [] Test if the ExperimentConfig is valid
- [] Test if the SystemConfig is valid
- [] Test get_data_and_system cfgs: are there enough items? are the items different from each other? is the first data core cfg and system cfg as expected? Is the FIRST item a data core, and the SECOND item a system cfg?

Dataset Creation:

- [] Test single dataset creation
- [] Test initialization of all data_generators
- [] test ground truth generation: expected dims / shape? expected range of numbers?
- [] test model data in comparison to ground truth
- [] test observation generation
- [] test noise function
- [] do the ground-truth and model data have the same shape together?

Generate.py

- [] check path generation (direct comparison with expected string)

IO:

- [] Test all io functions
