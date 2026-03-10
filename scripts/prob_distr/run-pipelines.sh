#!/bin/bash

# Assign arguments, falling back to defaults if not provided by the user
DIST_TYPE="${1:-baseline}"
DATA_SPLIT="${2:-}" # Defaults to empty string
GPUS="${3:-4}"
MODEL_PATH="${4:-/mnt/data-hps/models/Meta-Llama-3.1-8B}"
GPU_MEM="${5:-0.35}"

# Counter to keep track of any failures
FAILED_JOBS=0

# Helper function to run commands and handle failures gracefully
run_pipeline() {
    local job_name="$1"
    shift # Shift the arguments so $@ only contains the command itself
    local cmd=("$@")

    echo "================================================================="
    echo "▶ Starting: $job_name"
    echo "▶ Command: ${cmd[*]}"
    echo "================================================================="

    # Execute the command
    "${cmd[@]}"
    local status=$?

    # Check the exit status
    if [ $status -eq 0 ]; then
        echo -e "\n[SUCCESS] $job_name completed successfully.\n"
    else
        echo -e "\n[ERROR] $job_name failed with exit code $status.\n"
        FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
}

echo "Starting sequential pipeline runs..."
echo "Distribution Type: $DIST_TYPE | Dataset Split: '$DATA_SPLIT' | GPUs: $GPUS (MFRC: $MFRC_GPUS) | Model: $MODEL_PATH | GPU Mem: $GPU_MEM"
echo ""


# SemEval
run_pipeline "SemEval" bash scripts/prob_distr/pipeline-semeval.sh "$DIST_TYPE" "$DATA_SPLIT" "$GPUS" "$MODEL_PATH" vllm "$GPU_MEM"

# MFRC 
run_pipeline "MFRC" bash scripts/prob_distr/pipeline-MFRC.sh "$DIST_TYPE" "$DATA_SPLIT" "$GPUS" "$MODEL_PATH" vllm "$GPU_MEM"

# GoEmotions
run_pipeline "GoEmotions" bash scripts/prob_distr/pipeline-goemotions.sh "$DIST_TYPE" "$DATA_SPLIT" "$GPUS" "$MODEL_PATH" vllm "$GPU_MEM"

# Final Summary
echo "================================================================="
if [ $FAILED_JOBS -eq 0 ]; then
    echo "✅ All pipelines completed successfully!"
    exit 0
else
    echo "⚠️  Run finished, but $FAILED_JOBS pipeline(s) failed. Check logs above."
    exit 1
fi
