#!/usr/bin/env python
"""Generate chain-of-thought reasoning for few-shot examples.

Preselects examples from the training set and uses an LLM (via vLLM) to
generate 2-3 sentence reasoning explaining why the assigned labels are
correct. Saves results as a CSV loadable via --cot-csv, plus an IDs file
that can be passed as --train-ids-filename to restrict the train set to
exactly the examples with generated reasoning.
"""

import argparse
import os
import random

import pandas as pd


DATASET_CONFIGS = {
    "GoEmotions": {
        "task_desc": "multilabel emotion classification",
        "label_type": "emotions",
        "domain": "Reddit comments",
    },
    "SemEval": {
        "task_desc": "multilabel emotion classification",
        "label_type": "emotions",
        "domain": "tweets",
    },
    "MFRC": {
        "task_desc": "multilabel moral foundation classification",
        "label_type": "moral foundations",
        "domain": "Reddit posts",
    },
}


def load_dataset(dataset_name, root_dir, split, args):
    from llm_ml.benchmarks import GoEmotions, SemEval2018Task1Ec, MFRC

    cls_map = {
        "GoEmotions": GoEmotions,
        "SemEval": SemEval2018Task1Ec,
        "MFRC": MFRC,
    }

    extra = {}
    if dataset_name == "GoEmotions":
        extra["emotion_clustering_json"] = args.emotion_clustering_json
    elif dataset_name == "SemEval":
        extra["language"] = args.language

    splits = split if isinstance(split, list) else [split]
    return cls_map[dataset_name](
        root_dir=root_dir,
        splits=splits,
        annotation_mode="aggregate",
        **extra,
    )


def select_shots(dataset, num_shots, seed=0):
    """Select examples with uniform label coverage."""
    random.seed(seed)

    label_set = dataset.label_set
    annotator = "aggregate"
    label_inds = dataset.annotator2label_inds.get(annotator, {})
    all_inds = dataset.annotator2inds.get(annotator, list(range(len(dataset))))

    selected = set()

    for label in label_set:
        if len(selected) >= num_shots:
            break
        candidates = [c for c in label_inds.get(label, []) if c not in selected]
        if candidates:
            selected.add(random.choice(candidates))

    remaining = [i for i in all_inds if i not in selected]
    random.shuffle(remaining)
    while len(selected) < num_shots and remaining:
        selected.add(remaining.pop())

    return [dataset[i] for i in sorted(selected)]


def build_reasoning_prompt(text, labels, label_set, config):
    label_str = ", ".join(labels) if labels else "none"
    all_labels = ", ".join(label_set)

    n_labels = len(labels)
    if n_labels == 0:
        why_clause = f'explain why no {config["label_type"]} apply'
    elif n_labels == 1:
        why_clause = "explain why this label is appropriate"
    else:
        why_clause = f'explain why these {config["label_type"]} are appropriate'

    return [
        {
            "role": "system",
            "content": (
                f'You are an expert annotator for a {config["task_desc"]} task '
                f'on {config["domain"]}. '
                f'The possible {config["label_type"]} are: {all_labels}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f'Text: "{text}"\n'
                f'Assigned {config["label_type"]}: {label_str}\n\n'
                f"In 2-3 concise sentences, {why_clause} for this text."
            ),
        },
    ]


def _generate_reasoning_vllm(
    examples,
    dataset,
    model_name,
    dataset_name,
    gpu_memory_utilization=0.90,
    max_tokens=200,
):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    config = DATASET_CONFIGS[dataset_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    for ex in examples:
        labels = dataset.index_label_set(ex["label"])
        if isinstance(labels, str):
            labels = [labels]
        messages = build_reasoning_prompt(
            ex["text"], labels, dataset.label_set, config
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    print(f"Loading model {model_name} onto {num_gpus} GPU(s)...")
    os.environ["VLLM_USE_V1"] = "0"
    llm = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=True,
        distributed_executor_backend="mp",
    )
    print("Model loaded.")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    print(f"Generating reasoning for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    print("Generation complete.")

    results = []
    for ex, output in zip(examples, outputs):
        reasoning = output.outputs[0].text.strip()
        sentences = reasoning.split(".")
        if len(sentences) > 4:
            reasoning = ".".join(sentences[:3]) + "."
        results.append(
            {"id": ex["id"], "text": ex["text"], "cot": reasoning}
        )

    return results


def _generate_reasoning_google(
    examples,
    dataset,
    model_name,
    dataset_name,
    max_tokens=200,
):
    # Uses the Google Gemini / Gemma API via google-genai:
    #   pip install google-genai
    #
    # Expects the API key in the GEMINI_API_KEY environment variable.
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value
            api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it to your Google Generative AI API key to use Gemma."
        )

    client = genai.Client(api_key=api_key)
    config = DATASET_CONFIGS[dataset_name]

    results = []
    print(f"Generating reasoning with remote model {model_name} via Google API...")

    for ex in examples:
        labels = dataset.index_label_set(ex["label"])
        if isinstance(labels, str):
            labels = [labels]

        messages = build_reasoning_prompt(
            ex["text"], labels, dataset.label_set, config
        )

        system_parts = []
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                user_parts.append(msg["content"])

        system_text = "\n\n".join(system_parts).strip()
        user_text = "\n\n".join(user_parts).strip()
        full_prompt = system_text + "\n\n" + user_text if system_text else user_text

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )

        reasoning = (response.text or "").strip()
        sentences = reasoning.split(".")
        if len(sentences) > 4:
            reasoning = ".".join(sentences[:3]) + "."

        results.append(
            {"id": ex["id"], "text": ex["text"], "cot": reasoning}
        )

    return results


def generate_reasoning(
    examples,
    dataset,
    model_name,
    dataset_name,
    backend="vllm",
    gpu_memory_utilization=0.90,
    max_tokens=200,
):
    if backend == "google":
        return _generate_reasoning_google(
            examples,
            dataset,
            model_name,
            dataset_name,
            max_tokens=max_tokens,
        )

    return _generate_reasoning_vllm(
        examples,
        dataset,
        model_name,
        dataset_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_tokens=max_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning for few-shot examples"
    )
    parser.add_argument("dataset", choices=["GoEmotions", "SemEval", "MFRC"])
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--split", nargs="+", default=["train"])
    parser.add_argument("--num-shots", type=int, default=20)
    parser.add_argument(
        "--model",
        default="gemma-3-27b-it",
        help="Model to use for reasoning generation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: cot_fewshot/reasoning/<dataset>_<model>_shots<N>_seed<S>.csv)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration even if output exists"
    )

    parser.add_argument("--emotion-clustering-json", default=None)
    parser.add_argument("--language", default="english")
    parser.add_argument(
        "--backend",
        choices=["vllm", "google"],
        default=None,
        help=(
            "Backend to use for reasoning generation. "
            "If omitted, 'google' is used for Gemma models and 'vllm' otherwise."
        ),
    )

    args = parser.parse_args()

    if args.output is None:
        model_slug = args.model.replace("/", "--")
        args.output = (
            f"cot_fewshot/reasoning/{args.dataset}_{model_slug}"
            f"_shots{args.num_shots}_seed{args.seed}.csv"
        )

    ids_file = args.output.replace(".csv", "_ids.txt")

    if os.path.exists(args.output) and not args.force:
        print(f"Reasoning file already exists: {args.output}")
        print("Use --force to regenerate.")
        df = pd.read_csv(args.output, index_col=0)
        print(f"Loaded {len(df)} existing reasoning entries.")
        return

    print(f"Loading {args.dataset} dataset from {args.root_dir}...")
    dataset = load_dataset(args.dataset, args.root_dir, args.split, args)

    print(f"Dataset has {len(dataset)} examples, {len(dataset.label_set)} labels")
    print(f"Labels: {', '.join(dataset.label_set)}")

    print(f"Selecting {args.num_shots} shots with seed={args.seed}...")
    examples = select_shots(dataset, args.num_shots, seed=args.seed)

    print(f"Selected {len(examples)} examples:")
    for ex in examples[:5]:
        labels = dataset.index_label_set(ex["label"])
        if isinstance(labels, str):
            labels = [labels]
        print(f"  [{ex['id']}] {ex['text'][:60]}... -> {', '.join(labels)}")
    if len(examples) > 5:
        print(f"  ... and {len(examples) - 5} more")

    if args.backend is None:
        if args.model.startswith("gemma"):
            backend = "google"
        else:
            backend = "vllm"
    else:
        backend = args.backend

    print(f"\nGenerating reasoning with {args.model} (backend={backend})...")
    results = generate_reasoning(
        examples,
        dataset,
        args.model,
        args.dataset,
        backend=backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = pd.DataFrame(results).set_index("id")
    df.to_csv(args.output)
    print(f"Saved {len(df)} reasoning entries to {args.output}")

    raw_ids = [ex_id.split("__")[0] for ex_id in df.index]
    with open(ids_file, "w") as fp:
        fp.write("\n".join(raw_ids) + "\n")
    print(f"Saved {len(raw_ids)} IDs to {ids_file}")

    print("\nSample reasoning:")
    for idx, row in list(df.iterrows())[:3]:
        print(f"  ID: {idx}")
        print(f"  Text: {row['text'][:80]}...")
        print(f"  Reasoning: {row['cot'][:150]}...")
        print()


if __name__ == "__main__":
    main()
