#!/bin/bash
#SBATCH --job-name=enkf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/enkf.yaml"

echo "Starting missing enkf runs"
echo "Repo: $(pwd)"
echo "Model: $MODEL"


# Explicit list of dataset IDs
DATASETS=(21 22 23 26 41 42 45 47 61)

for i in "${DATASETS[@]}"; do
    DATASET_ID=$(printf "%03d" "$i")
    echo "=================================================="
    echo "Running dataset $DATASET_ID"
    echo "=================================================="

    uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"
done

echo "Missing enkf runs finished"