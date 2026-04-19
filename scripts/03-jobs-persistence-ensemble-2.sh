#!/bin/bash
#SBATCH --job-name=persist_ens_02
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd "$SCRATCH/ml-data-assimilation"
mkdir -p slurm_logs

MODEL="configs/models/da/persistence_ensemble.yaml"

echo "Starting ensemble persistence runs"
echo "Repo: $(pwd)"
echo "Model: $MODEL"

for i in $(seq 21 40); do
    DATASET_ID=$(printf "%03d" "$i")
    echo "=================================================="
    echo "Running dataset $DATASET_ID"
    echo "=================================================="

    uv run python src/ml_da/cli.py model --model "$MODEL" --data "$DATASET_ID"
done

echo "Persistence Ensemble 02 finished."