# Chain-of-Thought (CoT) Experiments

## Overview

**Chain-of-Thought (CoT)** is a prompting technique where the model is asked to produce step-by-step reasoning *before* giving its final answer. Instead of jumping directly to a label prediction, the model first explains its thinking, then outputs the classification result.

The motivation in this project comes from the observation that LLMs produce **spiky distributions** and tend to classify labels one-at-a-time in multi-label emotion/moral-foundation tasks. The hypothesis is that CoT reasoning—by requiring the model to consider multiple labels jointly before committing—may shift the distributions and affect calibration.

---

## Method

CoT is implemented entirely via **prompt modification**—no fine-tuning and no human annotation. The approach uses **0-shot prompting**: the model receives only the instruction and the query, with no in-context demonstration examples.

### Instruction

The instruction asks the model to reason step by step and then emit the label in a strict single-line JSON format:

**GoEmotions / SemEval:**
```
Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Let's think step by step about what emotions are present before classifying. After your reasoning, output exactly one line starting with "Output:" in this format:
Output: {"label": ["emotion1", "emotion2"]}
```

**MFRC:**
```
Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}. Let's think step by step about what moral foundations are present before classifying. After your reasoning, output exactly one line in this format: Moral foundation(s): foundation1, foundation2.
```

### Query Template

```
Input: {text}
Reasoning:
```

At test time the model continues from `Reasoning:`, generating its reasoning, then the `Output:` line with the label. For example:

```
Reasoning: The text expresses pain and shock. The speaker seems hurt and surprised by what they saw.
Output: {"label": ["sadness", "surprise"]}
```

---

## Score Extraction

Label probability scores are extracted from the label line only, not the reasoning. The implementation works as follows:

1. The incontext template `Input: {text}\nReasoning: {cot}\nOutput: {label}\n` defines `prefix_cutoff_str = "\nOutput: "` (the separator between `{cot}` and `{label}`). This specific prefix avoids false splits on newlines within the reasoning chain.
2. The model's full output is split at the first `\nOutput: `: everything before goes into `test_prefix_outs` (the reasoning), everything after into `test_outs` (the label line).
3. `string_overlap_idx_in_token_space` locates the starting token index of the label line within the full generated token sequence.
4. The score tensor is sliced from that index for the length of the label tokens, ensuring `test_scores` reflect the model's probability distribution over labels at the point where it writes the JSON—not during reasoning.
5. `label_first_token_ids` then extracts the logprob of the first token of each label name from the label line scores and applies softmax to produce the final label distribution.

This means `test_scores` in CoT runs have the same semantics as in baseline runs and are directly comparable.

---

## Pipeline Scripts

| Script | Dataset |
|--------|---------|
| `cot/pipeline-goemotions-cot.sh` | GoEmotions |
| `cot/pipeline-MFRC-cot.sh` | MFRC |
| `cot/pipeline-semeval-cot.sh` | SemEval 2018 Task 1 |

Key parameters compared to the baseline pipelines:

| Parameter | Baseline | CoT |
|-----------|----------|-----|
| `--shot` | 10 | 0 |
| `--max-new-tokens` | 18–25 | 500 |
| instruction | direct classification | step-by-step + format constraint |
| incontext | `Input: {text}\n{label}\n` | `Input: {text}\nReasoning: {cot}\nOutput: {label}\n` (GoEmotions/SemEval) or `Input: {text}\nReasoning: {cot}\nMoral foundation(s): {label}\n` (MFRC) |
| model | `Llama-3.1-8B-Instruct` | `Llama-3.1-8B` (base) |
| log path suffix | `baseline/` | `baseline-cot/` |

`max_new_tokens=500` is used to provide enough room for the reasoning chain plus the label line, with headroom for more verbose outputs.

---

## How to Run

Run all commands from the **project root** (`CSE_517_Project/`).

### Pipeline Arguments

| Position | Argument | Default |
|----------|----------|---------|
| 1 | Distribution (e.g. `baseline`) | — |
| 2 | ID list (e.g. `main_test_set`) | — |
| 3 | GPU index (e.g. `0`) | — |
| 4 | Model name or path | `meta-llama/Llama-3.1-8B` |
| 5 | Backend (`hf` or `vllm`) | `hf` |
| 6 | Seed | `0` |

### GoEmotions
```bash
bash cot/pipeline-goemotions-cot.sh baseline main_test_set 0
```
Logs: `logs/GoEmotions/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/`

### MFRC
```bash
bash cot/pipeline-MFRC-cot.sh baseline main_test_set 0
```
Logs: `logs/MFRC/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/`

### SemEval 2018 Task 1
```bash
bash cot/pipeline-semeval-cot.sh baseline main_test_set 0
```
Logs: `logs/SemEval/main_test_set/baseline-cot/meta-llama--Llama-3.1-8B_0/`

---

## Figures

```bash
# Multi-label distribution plots
bash cot/plot_figures_cot.sh

# Spikiness plot (sorted label probability distributions)
python cot/plot_spikiness_cot.py
```

Outputs go to `logs/analysis/ml-distr/{Dataset}/` and `scripts/prob_distr/figures/spikiness_cot.png`, directly comparable to the baseline figures.

---

## Comparing with Baseline

Both baseline and CoT runs produce the same `indexed_metrics.yml` structure (`test_scores`, `test_preds`, `test_all_scores`, `test_outs`, `test_prefix_outs`, etc.), so all downstream metrics and plots apply to both.

- **`test_prefix_outs`**: `null` in baseline, contains the model's reasoning chain in CoT.
- **`test_outs`**: the label line in both cases (e.g. `{"label": ["sadness"]}`).
- **`test_scores`**: label probability distribution, extracted from the label-line tokens in both cases.

Because only the prompts and `max_new_tokens` differ, CoT and baseline runs are directly comparable on all metrics.
