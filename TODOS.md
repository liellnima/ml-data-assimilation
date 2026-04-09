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

- [] RMSE
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
