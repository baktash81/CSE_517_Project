while getopts c: flag
do
    case "${flag}" in
        c) cuda=${OPTARG};;
    esac
done

model=meta-llama/Llama-3.1-8B-Instruct

# override from command line, if provided
model=${4:-$model}
gpu_mem=${6:-0.95}
seed=${7:-0}

export CUDA_VISIBLE_DEVICES="$3"

id_list=$2

id_file_args=""
if [ -n "$id_list" ]; then
    id_file=prob_distr_ids/Boxes/$id_list.txt
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
        Boxes \
        --distribution $1 \
        --root-dir datasets/boxes \
        --label-format polysyndeton \
        --train-split dev \
        --test-split train \
        --system ' ' \
        --instruction ./configs/Boxes/instruction.txt \
        --incontext ./configs/Boxes/incontext.txt \
        --model-name-or-path $model \
        --max-new-tokens 30 \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --trust-remote-code \
        --alternative $alt_name \
        --shot 5 \
        --seed $seed \
        --gpu-memory-utilization $gpu_mem \
        $id_file_args

else
    echo Using HuggingFace

    python scripts/prob_distr/llm_prob_distr.py \
        Boxes \
        --distribution $1 \
        --root-dir datasets/boxes \
        --label-format polysyndeton \
        --train-split dev \
        --test-split train \
        --system ' ' \
        --instruction ./configs/Boxes/instruction.txt \
        --incontext ./configs/Boxes/incontext.txt \
        --model-name-or-path $model \
        --max-new-tokens 30 \
        --device auto \
        --accelerate \
        --logging-level debug \
        --annotation-mode aggregate \
        --text-preprocessor false \
        --sampling-strategy multilabel \
        --trust-remote-code \
        --alternative $alt_name \
        --shot 5 \
        --seed $seed \
        $id_file_args

fi
