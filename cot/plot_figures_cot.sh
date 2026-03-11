#!/usr/bin/env bash

# CoT versions of the main distribution plots.
# Assumes you have already run the CoT pipelines, e.g.:
#   bash cot/pipeline-MFRC-cot.sh baseline main_test_set 0 meta-llama/Llama-3.1-8B
#   bash cot/pipeline-semeval-cot.sh baseline main_test_set 0 meta-llama/Llama-3.1-8B
#   bash cot/pipeline-goemotions-cot.sh baseline main_test_set 0 meta-llama/Llama-3.1-8B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/MFRC/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/ \
    --out logs/analysis/ml-distr/MFRC \
    --name "Llama3 8B Base CoT on MFRC" --union --max-rank 2

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/SemEval/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/ \
    --out logs/analysis/ml-distr/SemEval \
    --name "Llama3 8B Base CoT on SemEval 2018 Task 1" --max-rank 3

python scripts/ml-distribution/contrast_distributions_plot.py \
    --exp ./logs/GoEmotions/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/ \
    --out logs/analysis/ml-distr/GoEmotions \
    --name "Llama3 8B Base CoT on GoEmotions" --max-rank 2

