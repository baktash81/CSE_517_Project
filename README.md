# [EMNLP Main 2025] LLMs do Multi-Label Classification Differently Reproduction

This repo is a reproduction of the official implementation of [Large Language Models do Multi-Label Classification Differently](https://arxiv.org/abs/2505.17510), which appeared in the Main Conference Proceedings of EMNLP 2025. 

## Paper Abstract

> Multi-label classification is prevalent in realworld settings, but the behavior of Large Language Models (LLMs) in this setting is understudied. We investigate how autoregressive
LLMs perform multi-label classification, focusing on subjective tasks, by analyzing the
output distributions of the models at each label
generation step. We find that the initial probability distribution for the first label often does
not reflect the eventual final output, even in
terms of relative order and find LLMs tend to
suppress all but one label at each generation
step. We further observe that as model scale increases, their token distributions exhibit lower
entropy and higher single-label confidence, but
the internal relative ranking of the labels improves. Finetuning methods such as supervised
finetuning and reinforcement learning amplify
this phenomenon. We introduce the task of
distribution alignment for multi-label settings:
aligning LLM-derived label distributions with
empirical distributions estimated from annotator responses in subjective tasks. We propose
both zero-shot and supervised methods which
improve both alignment and predictive performance over existing approaches. We find one
method – taking the max probability over all
label generation distributions instead of just
using the initial probability distribution – improves both distribution alignment and overall
F1 classification without adding any additional
computation.

## Installation
*The instructions in this sections are taken directly from the original repository:*

This repo uses `Python 3.10` (type hints, for example, won't work with some previous versions). After you create and activate your virtual environment (with conda, venv, etc), install local dependencies with:

```bash
pip install -e .[dev]
```

## Data preparation

The datasets required to run these experiments are already included in this repository under the `/datasets` directory. **No additional download steps are necessary**

However, if you wish to pull the raw data from the original sources to recreate the pipeline, they can be found at:
- GoEmotions: https://huggingface.co/datasets/google-research-datasets/go_emotions
- MFRC: https://huggingface.co/datasets/USC-MOLA-Lab/MFRC 
- SemEval 2018 Task 1 E-c: https://competitions.codalab.org/competitions/17751#learn_the_details-datasets 
- HateXplain: https://github.com/hate-alert/HateXplain 

In addition, for evaluation, you will need to generate the human annotation label distributions. These can be generated using the `generate_human_distributions.py` script in each data directory. The splits for the train and test sets can be found under the `prob_distr_ids` directory.

## Preprocessing Code
No additional preprocessing steps are needed.

## Pretrained Models
This experiment uses Llama 3 model family. To use the model, you will have to log in to Hugging Face and request access to the models, then set up an access token. 

### Setup Instructions
1. **Request Access:** Visit the model page for one of the models below (e.g., [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)) on Hugging Face and accept Meta's terms of use. Access is usually granted instantly.
2. **Generate a Token:** Go to your Hugging Face Account Settings > Access Tokens and create a new token with "Read" permissions.
3. **Configure Your Environment:** Create a `.env` file in the root directory of this project and add your token like this:
   `ACCESS_TOKEN=your_huggingface_token_here`

Specifically, experiments are run on:
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)

## Training Code
This experiment uses pretrained models with no fine-tuning, so training is not needed.

## Run Experiments
*The instructions in this sections are taken directly from the original repository:*

Experiments are logged with [legm](https://github.com/gchochla/legm), so refer to the documentation there for an interpretation of the resulting `logs` folder, but navigating should be intuitive enough with some trial and error. Note that some bash scripts have arguments, which are self-explanatory. Make sure to run scripts from the root directory of this repo.

Also, you should create a `.env` file with your OpenAI key if you want to perform experiments with the GPTs.

```bash
OPENAI_API_KEY=<your-openai-key>
```

### Main Experiments

The majority of the scripts are located in `scripts/prob_distr`. The main python file is `llm_prob_distr.py`, but the entrypoint for calling this python function is in all of the `pipeline-*.sh` bash scripts. All the `pipeline-*.sh` scripts take the following ordered arguments:
- Position 1: distribution type to evaluate (`baseline` for most experiments; `unary`/`binary` for results on distribution alignment; `multilabel_icl` for a sweep of multilabel prompts for Figure 6)
- Position 2: IDs to use for testing; most experiments will use `main_test_set`. See the folder `prob_distr_ids` for valid lists of testing IDs
- Position 3: Which GPUs to use (int based, i.e. to fit into `cuda:x`)
- Position 4: Model to use, should be exact Huggingface name
- Position 5: Whether to use `vllm` for efficiency (UNTESTED, might not work: leave blank to run with normal `transformers` library)

For example, to run the main experiments for MFRC, you could run:

```
bash pipeline-MFRC.sh baseline main_test_set 0 meta-llama/Llama-3.1-8B
```

After successfully running these scripts, a folder should appear under `logs/{dataset}/{test_id_set}/{distribution_type}/{model_name}_x`, where `x` is an integer that is usually 0 but sometimes 1 or higher. For example the above script would create the folder `logs/MFRC/main_test_set/baseline/meta-llama--Llama-3.1-8B_0`. There is a file in that folder, `indexed_metrics.yml`, which is the file that contains all of the relevant information for that experiment. It lists each individual test stimulus, along with its logits and probabilities for every generated label.

All of the `plot*.py` files are the files for plotting Figures 4, 5, and 6 and they use various `indexed_metrics.yml` files to process and analyze the data. If you want to run them, some of the initial settings might need to change according to which log files they point to.

## Evaluation Code
Specifically, our reproduction of the original work used the following combinations of datasets, evaluation sets, and Llama 3.1 models. 

To run an evaluation, construct your command using the options in the table below. 

**Base Command Structure:**
```
bash <script_name.sh> <distribution_type> <test_set> <gpu_id> <model_name> <use_vllm>
```

**Evaluation Parameters**

| Parameter | Options | Description |
| :--- | :--- | :--- |
| **Script Name** | `pipeline-goemotions.sh`<br>`pipeline-MFRC.sh`<br>`pipeline-semeval.sh`<br>`pipeline-hate.sh` | Determines the dataset being evaluated. |
| **Position 1** | `baseline` | The distribution type to evaluate. |
| **Position 2** | `main_test_set`<br>`big_multilabel`<br>`''` *(Empty string for train set)* | The evaluation split. See the note below regarding train set evaluations. |
| **Position 3** | `0`, `1`, etc. | The target GPU ID. |
| **Position 4** | `meta-llama/Llama-3.1-8B`<br>`meta-llama/Llama-3.1-8B-Instruct`<br>`meta-llama/Llama-3.1-70B`<br>`meta-llama/Llama-3.1-70B-Instruct` | The exact Hugging Face model name. |
| **Position 5** | `vllm` *(Optional)* | Includes the vLLM backend for efficiency. This depends on your hardware configuration, but it is functional in our environment. |

**Important Execution Notes**

- Evaluating the Train Set: When running an evaluation on the full train set (by passing an empty string `''` to position 2), you must manually open the bash script and flip the train and test parameters before executing:
```
--train-split dev \
--test-split train \
```
- All other parameters inside the shell scripts remain exactly as specified by the original authors

### Chain-of-Thought (CoT) Experiments
The scripts for the CoT experiments utilize an identical setup and parameter structure as explained above. These can be found in the `/cot` directory.

### Plotting Figures
To generate the metrics and visuals for the reproduction, use the python scripts located in `scripts/prob_distr/`. Ensure the log file paths are accurate in your scripts before running.

- Spikiness Graph: `python scripts/prob_distr/plot_spikiness.py`

- Table Metrics (NLL, L1, F1): `python scripts/prob_distr/evaluate_baselines.py`

