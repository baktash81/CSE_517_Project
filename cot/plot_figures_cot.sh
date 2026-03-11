#!/usr/bin/env bash

# CoT versions of the main distribution plots.
# Assumes you have already run the CoT pipelines, e.g.:
#   bash cot/pipeline-MFRC-cot.sh baseline big_multilabel 0 meta-llama/Llama-3.1-8B-Instruct vllm
#   bash cot/pipeline-semeval-cot.sh baseline big_multilabel 0 meta-llama/Llama-3.1-8B-Instruct vllm
#   bash cot/pipeline-goemotions-cot.sh baseline big_multilabel 0 meta-llama/Llama-3.1-8B-Instruct vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/big_multilabel/baseline-cot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/MFRC \
    --name "Llama3 8B Instruct CoT on MFRC" --union --max-rank 1

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/big_multilabel/baseline-cot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/SemEval \
    --name "Llama3 8B Instruct CoT on SemEval 2018 Task 1" --union --max-rank 1

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/big_multilabel/baseline-cot/meta-llama--Llama-3.1-8B-Instruct_0/ \
    --out logs/analysis/ml-distr/GoEmotions \
    --name "Llama3 8B Instruct CoT on GoEmotions" --union --max-rank 1

