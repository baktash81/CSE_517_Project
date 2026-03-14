# Chain-of-Thought Few-Shot (CoT Few-Shot) Experiments

## Overview

**CoT Few-Shot** extends the [Chain-of-Thought (CoT)](../cot/CoT_Experiments.md) approach by combining step-by-step reasoning with **few-shot in-context demonstrations**. Rather than 0-shot prompting (as in CoT), the model receives 10 labeled examples—each annotated with a short reasoning chain—before classifying the test input.

The reasoning chains for the demonstration examples are **pre-generated** by a separate, stronger model (the *reasoning model*, e.g. `gemma-3-27b-it`) and cached to disk. The smaller *inference model* (e.g. `Llama-3.1-8B`) then conditions on these annotated demonstrations at test time.

---

## Method

CoT Few-Shot runs in two stages:

### Stage 1: Reasoning Generation (`generate_reasoning.py`)

A reasoning model produces 2–3 sentence explanations for a set of training examples selected with **uniform label coverage**.

1. **Shot selection** (`select_shots`): iterates over each label in the label set and picks one training example per label, then fills the remaining slots randomly until `num_shots` is reached (default 10).
2. **Prompt construction** (`build_reasoning_prompt`): each selected example is sent to the reasoning model as a system+user chat turn:
   - *System*: "You are an expert annotator for a {task} task on {domain}. The possible {label_type} are: {labels}."
   - *User*: 'Text: "{text}"\nAssigned {label_type}: {assigned_labels}\n\nIn 2-3 concise sentences, explain why ...'
3. **Generation**: runs the reasoning model via vLLM (local GPU) or the Google Gemini API (`--backend google` for `gemma-*` models). Outputs are truncated to at most 3 sentences.
4. **Output**: a CSV (`id`, `text`, `cot`) and a companion `_ids.txt` file listing the selected training IDs.

### Stage 2: Inference

The pipeline scripts call the same `vllm_prob_distr.py` / `llm_prob_distr.py` as baseline and CoT experiments, but with:

- `--shot 10` (or the configured number of shots)
- `--cot-csv <reasoning_file>` to inject pre-generated reasoning into the in-context examples
- `--train-ids-filename <ids_file>` to restrict train examples to those with generated reasoning

### Instruction

The instruction mirrors the 0-shot CoT prompt but adds a brevity constraint:

**GoEmotions / SemEval:**
```
Classify the following inputs into none, one, or multiple the following emotions per input: {labels}. Let's think step by step about what emotions are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:
Output: {"label": ["emotion1", "emotion2"]}
```

**MFRC:**
```
Classify the following inputs into none, one, or multiple the following moral foundations per input: {labels}. Let's think step by step about what moral foundations are present before classifying. Keep your reasoning brief (2-3 sentences). After your reasoning, output exactly one line starting with "Output:" in this format:
Output: {"label": ["foundation1", "foundation2"]}
```

### Query Template

```
Input: {text}
Reasoning: {cot}
Output: {label}
```

Each of the 10 in-context examples follows this template with its pre-generated reasoning filled in. The test query ends at `Reasoning:` and the model continues from there.

---

## Score Extraction

Score extraction works identically to the 0-shot CoT setup. The incontext template defines `prefix_cutoff_str = "\nOutput: "`, and only the tokens after this separator (the label line) contribute to `test_scores`. See the [CoT Experiments](../cot/CoT_Experiments.md#score-extraction) documentation for the full procedure.

---

## Pipeline Scripts

| Script | Dataset |
|--------|---------|
| `cot_fewshot/pipeline-goemotions-cot-fewshot.sh` | GoEmotions |
| `cot_fewshot/pipeline-MFRC-cot-fewshot.sh` | MFRC |
| `cot_fewshot/pipeline-semeval-cot-fewshot.sh` | SemEval 2018 Task 1 |

Key parameters compared to baseline and 0-shot CoT pipelines:

| Parameter | Baseline | CoT (0-shot) | CoT Few-Shot |
|-----------|----------|--------------|--------------|
| `--shot` | 10 | 0 | 10 |
| `--max-new-tokens` | 18–25 | 500 | 500 |
| instruction | direct classification | step-by-step + format | step-by-step + brevity + format |
| incontext | `Input: {text}\n{label}\n` | `Input: {text}\nReasoning: {cot}\nOutput: {label}\n` | `Input: {text}\nReasoning: {cot}\nOutput: {label}\n` |
| `--cot-csv` | — | — | pre-generated reasoning CSV |
| reasoning model | — | — | `gemma-3-27b-it` (default) |
| inference model | `Llama-3.1-8B-Instruct` | `Llama-3.1-8B` (base) | `Llama-3.1-8B` (base, default) |
| backend | `hf` | `hf` | `vllm` |
| label format (MFRC) | `Moral foundation(s):` | `Moral foundation(s):` | JSON (`Output: {"label": [...]}`) |
| log path suffix | `baseline/` | `baseline-cot/` | `cot-fewshot-{N}shot/` |

---

## Pre-Generated Reasoning Files

Reasoning CSVs and ID files are stored under `cot_fewshot/reasoning/`:

```
cot_fewshot/reasoning/
├── GoEmotions_gemma-3-27b-it_shots10_seed0.csv
├── GoEmotions_gemma-3-27b-it_shots10_seed0_ids.txt
├── MFRC_gemma-3-27b-it_shots10_seed0.csv
├── MFRC_gemma-3-27b-it_shots10_seed0_ids.txt
├── SemEval_gemma-3-27b-it_shots10_seed0.csv
├── SemEval_gemma-3-27b-it_shots10_seed0_ids.txt
└── ... (also Llama-3.1-8B-Instruct variants)
```

The pipeline scripts check whether the reasoning file exists before generating; cached files are reused automatically.

---

## How to Run

Run all commands from the **project root** (`CSE_517_Project/`).

### Pipeline Arguments

| Position | Argument | Default |
|----------|----------|---------|
| 1 | Distribution (e.g. `baseline`) | — |
| 2 | ID list (e.g. `big_multilabel`) | — |
| 3 | GPU index (e.g. `0`) | — |
| 4 | Inference model name or path | `meta-llama/Llama-3.1-8B` |
| 5 | Backend (`hf` or `vllm`) | `vllm` |
| 6 | Reasoning model | `gemma-3-27b-it` |
| 7 | Number of shots | `10` |
| 8 | Seed | `0` |
| 9 | GPU memory utilization | `0.95` |

### GoEmotions
```bash
bash cot_fewshot/pipeline-goemotions-cot-fewshot.sh baseline big_multilabel 0
```
Logs: `logs/GoEmotions/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B_0/`

### MFRC
```bash
bash cot_fewshot/pipeline-MFRC-cot-fewshot.sh baseline big_multilabel 0
```
Logs: `logs/MFRC/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B_0/`

### SemEval 2018 Task 1
```bash
bash cot_fewshot/pipeline-semeval-cot-fewshot.sh baseline big_multilabel 0
```
Logs: `logs/SemEval/big_multilabel/cot-fewshot-10shot/meta-llama--Llama-3.1-8B_0/`

### Standalone Reasoning Generation

To generate reasoning without running the full pipeline:
```bash
python cot_fewshot/generate_reasoning.py GoEmotions \
    --root-dir datasets/goemotions \
    --emotion-clustering-json datasets/goemotions/emotion_clustering.json \
    --split train \
    --num-shots 10 \
    --model gemma-3-27b-it \
    --seed 0
```

Use `--force` to regenerate even if the output file already exists. Use `--backend google` to call the Gemini API instead of running locally via vLLM (requires `GEMINI_API_KEY`).

---

## Figures

```bash
# Multi-label distribution plots
bash cot_fewshot/plot_figures_cot_fewshot.sh

# Spikiness plot (sorted label probability distributions)
python cot_fewshot/plot_spikiness_cot_fewshot.py
```

Outputs go to `logs/analysis/ml-distr/{Dataset}/` and `scripts/prob_distr/figures/spikiness_cot_fewshot.png`, directly comparable to baseline and 0-shot CoT figures.

---

## Comparing with Baseline and CoT

All three setups (baseline, CoT 0-shot, CoT few-shot) produce the same `indexed_metrics.yml` structure, so downstream metrics and plots are directly comparable.

- **`test_prefix_outs`**: `null` in baseline; contains the model's reasoning chain in both CoT variants.
- **`test_outs`**: the label line in all cases.
- **`test_scores`**: label probability distribution, extracted from label-line tokens only.

Key differences from 0-shot CoT:
- The model sees 10 annotated demonstrations (with reasoning) before the test input, rather than reasoning from scratch.
- Reasoning for demonstrations comes from a separate, potentially stronger model.
- All three datasets use a unified JSON label format (`Output: {"label": [...]}`), whereas 0-shot CoT uses a different format for MFRC (`Moral foundation(s): ...`).
- The instruction includes an explicit brevity constraint ("Keep your reasoning brief (2-3 sentences)").
