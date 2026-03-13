# Run from project root so paths like scripts/prob_distr/llm_prob_distr.py resolve
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"


# override from command line, if provided ($1=dist_type, $2=id_list, $3=train_split, $4=test_split, $5=gpus, $6=model, $7=backend, $8=gpu_mem, $9=seed, $10=debug_samples)
id_list=$2
train_split=${3:-dev}
test_split=${4:-train}
export CUDA_VISIBLE_DEVICES="$5"

# model=meta-llama/Llama-3.2-1B
# model=meta-llama/Llama-3.1-8B
model=meta-llama/Llama-3.1-8B-Instruct
model=${6:-$model}

gpu_mem=${8:-0.95}
seed=${9:-0}
debug_samples=${10:-5}

id_file_args=""
if [ -n "$id_list" ]; then
    id_file=prob_distr_ids/GoEmotions/$id_list.txt
    id_file_args="--test-ids-filename $id_file"
    alt_name="$id_list/{distribution}/{model_name_or_path}"
else
    alt_name="full_dataset/{distribution}/{model_name_or_path}"
fi

echo Using model $model
echo Evaluating distribution type $1
echo Testing on IDs: ${id_file:-"(full dataset)"}
echo Running on GPU $5

if [ "$7" == "vllm" ]; then
    echo Using VLLM

    python scripts/prob_distr/vllm_prob_distr.py \
        GoEmotions \
        --distribution $1 \
        --root-dir datasets/goemotions \
        --emotion-clustering-json datasets/goemotions/emotion_clustering.json \
        --train-split $train_split \
        --test-split $test_split \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' \
        --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model \
        --label-format json \
        --max-new-tokens 18 \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $alt_name \
        --gpu-memory-utilization $gpu_mem \
        $([ "$debug_samples" -gt 0 ] && echo "--debug-samples $debug_samples") \
        $id_file_args

else
    echo Using HuggingFace

    python scripts/prob_distr/llm_prob_distr.py \
        GoEmotions \
        --distribution $1 \
        --root-dir datasets/goemotions \
        --emotion-clustering-json datasets/goemotions/emotion_clustering.json \
        --train-split $train_split \
        --test-split $test_split \
        --system ' ' \
        --instruction $'Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Output exactly these emotions and no others.\n' \
        --incontext $'Input: {text}\n{label}\n' \
        --model-name-or-path $model \
        --label-format json \
        --max-new-tokens 18 \
        --device auto \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $alt_name \
        $([ "$debug_samples" -gt 0 ] && echo "--debug-samples $debug_samples") \
        $id_file_args

fi
