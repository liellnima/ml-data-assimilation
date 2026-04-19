#!/bin/bash

cd "$SCRATCH/ml-data-assimilation"

for i in $(seq 0 80); do
    DATASET_ID=$(printf "%03d" "$i")
    echo "Submitting dataset $DATASET_ID"
    sbatch 00-single-neuralenkf.sh "$DATASET_ID"
done