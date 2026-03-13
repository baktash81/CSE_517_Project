#!/bin/bash
# Run from project root so paths like scripts/prob_distr/llm_prob_distr.py resolve
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Usage: run-pipelines.sh [dist_type] [id_list] [train_split] [test_split] [gpus] [model] [gpu_mem]

# Assign arguments, falling back to defaults if not provided by the user
dist_type="${1:-baseline}"
id_list="${2:-}"
train_split="${3:-}"
test_split="${4:-}"
gpus="${5:-4}"
model="${6:-/mnt/data-hps/models/Meta-Llama-3.1-8B}"
gpu_mem="${7:-0.35}"

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
    if [ "$status" -eq 0 ]; then
        echo -e "\n[SUCCESS] $job_name completed successfully.\n"
    else
        echo -e "\n[ERROR] $job_name failed with exit code $status.\n"
        FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
}

echo "Starting sequential pipeline runs..."
echo "Distribution Type: $dist_type | Dataset Split: '$id_list' | GPUs: $gpus | Model: $model | GPU Mem: $gpu_mem | Train: ${train_split:-(default)} | Test: ${test_split:-(default)}"
echo ""


# SemEval
run_pipeline "SemEval" bash scripts/prob_distr/pipeline-semeval.sh "$dist_type" "$id_list" "$train_split" "$test_split" "$gpus" "$model" vllm "$gpu_mem"

# MFRC
run_pipeline "MFRC" bash scripts/prob_distr/pipeline-MFRC.sh "$dist_type" "$id_list" "$train_split" "$test_split" "$gpus" "$model" vllm "$gpu_mem"

# GoEmotions
run_pipeline "GoEmotions" bash scripts/prob_distr/pipeline-goemotions.sh "$dist_type" "$id_list" "$train_split" "$test_split" "$gpus" "$model" vllm "$gpu_mem"

# HateExplain
# run_pipeline "HateExplain" bash scripts/prob_distr/pipeline-hate.sh "$dist_type" "$id_list" "$train_split" "$test_split" "$gpus" "$model" vllm "$gpu_mem"

# Final Summary
echo "================================================================="
if [ $FAILED_JOBS -eq 0 ]; then
    echo "✅ All pipelines completed successfully!"
    exit 0
else
    echo "⚠️  Run finished, but $FAILED_JOBS pipeline(s) failed. Check logs above."
    exit 1
fi
