while getopts c: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
    esac
done

# model=meta-llama/Llama-2-13b-chat-hf
model=meta-llama/Llama-2-7b-chat-hf
# model=meta-llama/Llama-3.2-1B
# model=meta-llama/Llama-3.1-8B
# model=meta-llama/Llama-3.3-70B-Instruct

# override from command line, if provided
model=${4:-$model}
gpu_mem=${6:-0.95}
seed=${7:-0}

export CUDA_VISIBLE_DEVICES="$3"

id_list=$2

id_file_args=""
if [ -n "$id_list" ]; then
    id_file=prob_distr_ids/Hatexplain/$id_list.txt
    id_file_args="--test-ids-filename $id_file"
    alt_name="$id_list/{distribution}/{model_name_or_path}"
else
    alt_name="full_dataset/{distribution}/{model_name_or_path}"
fi

echo Using model $model
echo Evaluating distribution type $1
echo Testing on IDs: ${id_file:-"(full dataset)"}
echo Running on GPU $3

if [ "$5" == "vllm" ]; then
    echo Using VLLM

    python scripts/prob_distr/vllm_prob_distr.py \
        Hatexplain \
        --distribution $1 \
        --train-split test \
        --test-split train \
        --system ' ' \
        --instruction $'Classify the following inputs into one of the following options per input: {labels}. Output exactly one label and no others.\n\n' \
        --incontext $'Question: {text}\nAnswer: {label}\n\n' \
        --model-name-or-path $model \
        --max-new-tokens 18 \
        --device cpu \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --trust-remote-code \
        --sampling-strategy uniform \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $alt_name \
        --gpu-memory-utilization $gpu_mem \
        $id_file_args

else
    echo Using HuggingFace

    python scripts/prob_distr/llm_prob_distr.py \
        Hatexplain \
        --distribution $1 \
        --train-split test \
        --test-split train \
        --system ' ' \
        --instruction $'Classify the following inputs into one of the following options per input: {labels}. Output exactly one label and no others.\n\n' \
        --incontext $'Question: {text}\nAnswer: {label}\n\n' \
        --model-name-or-path $model \
        --max-new-tokens 18 \
        --device auto \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --trust-remote-code \
        --sampling-strategy uniform \
        --sentence-model all-mpnet-base-v2 \
        --seed $seed \
        --shot 10 \
        --alternative $alt_name \
        $id_file_args

fi