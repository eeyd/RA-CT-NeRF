#!/bin/bash

BASE_DIR="kits19/data"
BICUBIC_SCRIPT="src/bicubic.py"
N_EVAL=0
SCALE=4

START_INDEX=0
counter=0

for CASE_DIR in "$BASE_DIR"/*; do
    if [ -d "$CASE_DIR" ]; then
        # Increment the counter
        counter=$((counter + 1))

        # Only start processing from the 96th iteration onwards
        if [ $counter -ge $START_INDEX ]; then
            CASE_NAME=$(basename "$CASE_DIR")
            
            echo "Running bicubic.py for case: $CASE_NAME (iteration $counter)"
            python "$BICUBIC_SCRIPT" --case "$CASE_NAME" --n_eval $N_EVAL --scale $SCALE
        fi
    fi
done
