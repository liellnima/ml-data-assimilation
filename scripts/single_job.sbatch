#!/bin/bash
#SBATCH --job-name=ml_da
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/network/scratch/<u>/<username>/ml-data-assimilation/slurm_logs/%x_%j.out
#SBATCH --error=/network/scratch/<u>/<username>/ml-data-assimilation/slurm_logs/%x_%j.err

set -euo pipefail

cd "$SCRATCH/ml-data-assimilation"

echo "PWD=$(pwd)"
echo "Model=$1"
echo "Dataset=$2"

uv run python src/ml-data-assimilation/cli.py model -m "$1" -d "$2"