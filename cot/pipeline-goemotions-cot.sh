while getopts c: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
    esac
done

# Run from project root so paths like scripts/prob_distr/llm_prob_distr.py resolve
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to Llama 3.1 8B base; override from command line if provided
model=meta-llama/Llama-3.1-8B
model=${4:-$model}

# Backend: "hf" (default) or "vllm"
backend=${5:-"hf"}

# Seed for example sampling etc. (optional, default 0)
gpu_mem=${6:-0.95}
seed=${7:-0}

export CUDA_VISIBLE_DEVICES="$3"

id_list=$2
id_file=prob_distr_ids/GoEmotions/$id_list.txt

# Normalize model name for log path (e.g. meta-llama/Llama-3.1-8B -> meta-llama--Llama-3.1-8B)
model_slug=$(echo "$model" | sed 's/\//--/g')
alternative_path="$id_list/baseline-cot/$model_slug"

echo "Using model: $model"
echo "Backend: $backend"
echo "Evaluating distribution type: $1"
echo "Testing on IDs in: $id_file"
echo "Running on GPU(s): $3"
echo "Seed: $seed"

if [ "$backend" == "vllm" ]; then
    echo "Using vLLM backend for CoT GoEmotions."

    python scripts/prob_distr/vllm_prob_distr.py \
        GoEmotions \
        --distribution "$1" \
        --root-dir datasets/goemotions \
        --emotion-clustering-json datasets/goemotions/emotion_clustering.json \
        --train-split train \
        --test-split dev \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Let\'s think step by step about what emotions are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:\nOutput: {"label": ["emotion1", "emotion2"]}\n' \
        --incontext $'Input: {text}\nReasoning: {cot}\nOutput: {label}\n' \
        --model-name-or-path "$model" \
        --label-format json \
        --max-new-tokens 500 \
        --device cpu \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed "$seed" \
        --shot 0 \
        --alternative "$alternative_path" \
        --test-ids-filename "$id_file"
else
    echo "Using HuggingFace backend for CoT GoEmotions."

    python scripts/prob_distr/llm_prob_distr.py \
        GoEmotions \
        --distribution "$1" \
        --root-dir datasets/goemotions \
        --emotion-clustering-json datasets/goemotions/emotion_clustering.json \
        --train-split train \
        --test-split dev \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Let\'s think step by step about what emotions are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:\nOutput: {"label": ["emotion1", "emotion2"]}\n' \
        --incontext $'Input: {text}\nReasoning: {cot}\nOutput: {label}\n' \
        --model-name-or-path "$model" \
        --label-format json \
        --max-new-tokens 500 \
        --device cpu \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --load-in-4bit \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed "$seed" \
        --shot 0 \
        --alternative "$alternative_path" \
        --gpu-memory-utilization $gpu_mem \
        --test-ids-filename "$id_file"
fi

