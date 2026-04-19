#!/bin/bash
#SBATCH --job-name=neuralEnkf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/persistence.yaml"

DATASET_ID="$1"

echo "Starting one neuralEnkf run"
echo "Repo: $(pwd)"
echo "Model: $MODEL"
echo "Dataset: $DATASET_ID"

uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"

echo "One neuralEnKF finished"