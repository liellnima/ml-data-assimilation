#!/bin/bash
set -euo pipefail

# Usage:
#   bash submit_grid.sh /path/to/main.yaml
#
# Example:
#   bash submit_grid.sh configs/main.yaml

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 MAIN_CONFIG"
  exit 1
fi

MAIN_CONFIG="$1"

if [ ! -f "$MAIN_CONFIG" ]; then
  echo "Main config not found: $MAIN_CONFIG"
  exit 1
fi

mkdir -p slurm_logs manifests

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="manifests/grid_${TIMESTAMP}.tsv"

MODELS=(
  "configs/models/da/persistence.yaml"
  "configs/models/da/persistence_ensemble.yaml"
  "configs/models/da/var4d.yaml"
  "configs/models/da/enkf.yaml"
  "configs/models/ml/neural_enkf.yaml"
)

# Optional safety check: verify model files exist
for model_file in "${MODELS[@]}"; do
  if [ ! -f "$model_file" ]; then
    echo "Model config not found: $model_file"
    exit 1
  fi
done

: > "$MANIFEST"

for model_file in "${MODELS[@]}"; do
  for i in $(seq 0 80); do
    dataset_id=$(printf "%03d" "$i")
    printf "%s\t%s\n" "$model_file" "$dataset_id" >> "$MANIFEST"
  done
done

NUM_JOBS=$(wc -l < "$MANIFEST" | tr -d ' ')

if [ "$NUM_JOBS" -eq 0 ]; then
  echo "No jobs to submit."
  exit 1
fi

echo "Manifest written to $MANIFEST"
echo "Submitting $NUM_JOBS jobs"

# Adjust %4 to control how many run simultaneously
sbatch \
  --array=0-$((NUM_JOBS - 1))%4 \
  --export=ALL,MANIFEST="$MANIFEST" \
  single_job.sbatch