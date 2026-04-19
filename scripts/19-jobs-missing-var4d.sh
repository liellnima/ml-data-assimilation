#!/bin/bash
#SBATCH --job-name=var4d_seq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/var4d.yaml"

echo "Starting missing var4d runs"
echo "Repo: $(pwd)"
echo "Model: $MODEL"


# Explicit list of dataset IDs
DATASETS=(0 1 11 12 13 15 31 52 51 61 62 63 71)

for i in "${DATASETS[@]}"; do
    DATASET_ID=$(printf "%03d" "$i")
    echo "=================================================="
    echo "Running dataset $DATASET_ID"
    echo "=================================================="

    uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"
done

echo "Missing var4d runs finished"