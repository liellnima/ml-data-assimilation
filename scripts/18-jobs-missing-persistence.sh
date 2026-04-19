#!/bin/bash
#SBATCH --job-name=persistence_seq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/persistence.yaml"

echo "Starting missing sequential persistence runs"
echo "Repo: $(pwd)"
echo "Model: $MODEL"


# Explicit list of dataset IDs
DATASETS=(31 32)

for i in "${DATASETS[@]}"; do
    DATASET_ID=$(printf "%03d" "$i")
    echo "=================================================="
    echo "Running dataset $DATASET_ID"
    echo "=================================================="

    uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"
done

echo "Missing persistence runs finished"