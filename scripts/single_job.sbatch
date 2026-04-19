	#!/bin/bash
#SBATCH --job-name=ml_data_assim
#SBATCH --ntasks=1
#SBATCH --partition=long                          # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:50:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/<u>/<username>/slurm-%j.out  # Write the log on scratch

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Working dir: $(pwd)"

# -------------------------
# 1. Load Mila environment
# -------------------------
module load python/3.12
module load cuda  # if you use GPU

# -------------------------
# 2. Move to SCRATCH repo
# -------------------------
PROJECT_DIR="$SCRATCH/ml-data-assimilation"   # <-- CHANGE THIS
cd "$PROJECT_DIR"

echo "Running in: $(pwd)"

# -------------------------
# 4. Read manifest
# -------------------------
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

MODEL_FILE=$(echo "$LINE" | cut -f1)
DATASET_ID=$(echo "$LINE" | cut -f2)

echo "MODEL_FILE=$MODEL_FILE"
echo "DATASET_ID=$DATASET_ID"

# -------------------------
# 5. Run job
# -------------------------

# Optional: move to your project root if sbatch might be launched elsewhere
# cd /path/to/your/project

# If you use uv, this matches Mila's documented pattern of running commands under Slurm.
uv run python cli.py model \
  --model "$MODEL_FILE" \
  --data "$DATASET_ID" \
  --log-level INFO