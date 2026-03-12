#!/usr/bin/env bash

# CoT few-shot versions of the distribution plots.
# Assumes you have already run the CoT few-shot pipelines, e.g.:
#   bash cot_fewshot/pipeline-MFRC-cot-fewshot.sh baseline big_multilabel 1 meta-llama/Llama-3.1-8B-Instruct vllm
#   bash cot_fewshot/pipeline-semeval-cot-fewshot.sh baseline big_multilabel 2 meta-llama/Llama-3.1-8B-Instruct vllm
#   bash cot_fewshot/pipeline-goemotions-cot-fewshot.sh baseline big_multilabel 0 meta-llama/Llama-3.1-8B-Instruct vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Spikiness plot
python cot_fewshot/plot_spikiness_cot_fewshot.py

# Contrast distribution plots
python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/MFRC \
    --name "Llama3 8B Instruct CoT Few-Shot on MFRC" --union --max-rank 1

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/SemEval \
    --name "Llama3 8B Instruct CoT Few-Shot on SemEval 2018 Task 1" --union --max-rank 1

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/GoEmotions \
    --name "Llama3 8B Instruct CoT Few-Shot on GoEmotions" --union --max-rank 1
