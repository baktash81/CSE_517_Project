#!/bin/bash
# Pipeline for CoT few-shot MFRC experiments.
#
# Usage:
#   ./pipeline-MFRC-cot-fewshot.sh <distribution> <id_list> <gpu> \
#       [inference_model] [backend] [reasoning_model] [num_shots] [seed] [gpu_mem]
#
# Example:
#   ./pipeline-MFRC-cot-fewshot.sh baseline big_multilabel 0 \
#       meta-llama/Llama-3.1-8B vllm meta-llama/Llama-3.1-8B-Instruct 10 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

model=meta-llama/Llama-3.1-8B
model=${4:-$model}

backend=${5:-"vllm"}
reasoning_model=${6:-"meta-llama/Llama-3.1-8B-Instruct"}
num_shots=${7:-10}
seed=${8:-0}
gpu_mem=${9:-0.95}

export CUDA_VISIBLE_DEVICES="$3"

id_list=$2
id_file=prob_distr_ids/MFRC/$id_list.txt

model_slug=$(echo "$model" | sed 's/\//--/g')
reasoning_model_slug=$(echo "$reasoning_model" | sed 's/\//--/g')
alternative_path="$id_list/cot-fewshot-${num_shots}shot/$model_slug"
reasoning_file="cot_fewshot/reasoning/MFRC_${reasoning_model_slug}_shots${num_shots}_seed${seed}.csv"
ids_file="cot_fewshot/reasoning/MFRC_${reasoning_model_slug}_shots${num_shots}_seed${seed}_ids.txt"

echo "=== CoT Few-Shot MFRC ==="
echo "Inference model: $model"
echo "Reasoning model: $reasoning_model"
echo "Backend: $backend"
echo "Distribution: $1"
echo "Shots: $num_shots"
echo "Seed: $seed"
echo "Testing on IDs in: $id_file"
echo "Running on GPU(s): $3"

# Step 1: Generate reasoning if not cached
if [ ! -f "$reasoning_file" ]; then
    echo ""
    echo "--- Step 1: Generating reasoning ---"
    python cot_fewshot/generate_reasoning.py \
        MFRC \
        --root-dir datasets/mfrc \
        --split train \
        --num-shots "$num_shots" \
        --model "$reasoning_model" \
        --output "$reasoning_file" \
        --seed "$seed" \
        --gpu-memory-utilization "$gpu_mem"

    if [ $? -ne 0 ]; then
        echo "ERROR: Reasoning generation failed."
        exit 1
    fi
else
    echo "Using cached reasoning: $reasoning_file"
fi

echo ""
echo "--- Step 2: Running experiment ---"

if [ "$backend" == "vllm" ]; then
    python scripts/prob_distr/vllm_prob_distr.py \
        MFRC \
        --distribution "$1" \
        --root-dir datasets/mfrc \
        --train-split train \
        --train-ids-filename "$ids_file" \
        --test-split dev test \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}. Let\'s think step by step about what moral foundations are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:\nOutput: {"label": ["foundation1", "foundation2"]}\n' \
        --incontext $'Input: {text}\nReasoning: {cot}\nOutput: {label}\n' \
        --model-name-or-path "$model" \
        --label-format json \
        --max-new-tokens 500 \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --trust-remote-code \
        --seed "$seed" \
        --shot "$num_shots" \
        --cot-csv "$reasoning_file" \
        --alternative "$alternative_path" \
        --test-ids-filename "$id_file" \
        --gpu-memory-utilization "$gpu_mem"
else
    python scripts/prob_distr/llm_prob_distr.py \
        MFRC \
        --distribution "$1" \
        --root-dir datasets/mfrc \
        --train-split train \
        --train-ids-filename "$ids_file" \
        --test-split dev test \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}. Let\'s think step by step about what moral foundations are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:\nOutput: {"label": ["foundation1", "foundation2"]}\n' \
        --incontext $'Input: {text}\nReasoning: {cot}\nOutput: {label}\n' \
        --model-name-or-path "$model" \
        --label-format json \
        --max-new-tokens 500 \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --load-in-4bit \
        --trust-remote-code \
        --seed "$seed" \
        --shot "$num_shots" \
        --cot-csv "$reasoning_file" \
        --alternative "$alternative_path" \
        --test-ids-filename "$id_file" \
        --gpu-memory-utilization "$gpu_mem"
fi
