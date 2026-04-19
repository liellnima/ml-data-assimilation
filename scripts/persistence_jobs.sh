#!/bin/bash
#SBATCH --job-name=persistence_seq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/<u>/<username>/ml-data-assimilation/slurm_logs/%x_%j.out
#SBATCH --error=/network/scratch/<u>/<username>/ml-data-assimilation/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/persistence.yaml"

echo "Starting sequential persistence runs"
echo "Repo: $(pwd)"
echo "Model: $MODEL"

for i in $(seq 0 80); do
    DATASET_ID=$(printf "%03d" "$i")
    echo "=================================================="
    echo "Running dataset $DATASET_ID"
    echo "=================================================="

    uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"
done

echo "All persistence runs finished"