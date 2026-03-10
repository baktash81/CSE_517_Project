"""
CoT variant of scripts/prob_distr/plot_spikiness.py.
Loads CoT runs (baseline‑cot) for each dataset and generates a single
spikiness figure, comparable to the baseline one.
"""
import os
import sys
import yaml
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROB_DISTR_SCRIPTS = os.path.join(PROJECT_ROOT, "scripts", "prob_distr")
sys.path.insert(0, PROB_DISTR_SCRIPTS)

import matplotlib.pyplot as plt  # noqa: E402

from distribution_estimation import load_data_from_yaml  # noqa: E402


def get_graph_probs(data):
    probs = None
    for example_id, example in data.items():
        if len(example["test_scores"]) == 0:
            continue

        distr = example["test_scores"]
        if "none" in distr:
            del distr["none"]

        if probs is None:
            probs = []
            for _ in range(len(distr)):
                probs.append([])
        sorted_probs = sorted(list(distr.values()), reverse=True)
        # ignore 'none' examples and noisy examples (less than 0.3 for top label)
        if sorted_probs[0] < 0.3 or len(example["test_preds"]) < 1:
            continue
        for i, prob in enumerate(sorted_probs):
            probs[i].append(prob)

    return [x for x in probs if len(x) > 0]


def plot_multilabel_icl(yaml_files, save_path):
    colors = ["blue", "red", "green"]
    labels = ["GoEmotions", "MFRC", "SemEval"]

    plt.figure(figsize=(12, 7))

    # Offset each dataset on the x-axis for clearer separation
    x_offset = 0.25

    for i, (yaml_file, color, label) in enumerate(
        zip(yaml_files, colors, labels)
    ):
        data = load_data_from_yaml(yaml_file)
        points = get_graph_probs(data)
        # Legend entry (one per dataset; outside loop so we get it even if points is empty)
        plt.scatter([], [], color=color, alpha=1, label=label, s=80)
        for x, probs in enumerate(points):
            jitter = x_offset * (i - 1)  # -0.25, 0.0, 0.25 for three datasets
            jittered_x = (
                x + jitter + 0.12 * (np.random.random(len(probs)) - 0.5)
            )
            plt.scatter(jittered_x, probs, color=color, alpha=0.08)

    plt.xlabel("Sorted Label Index", fontsize=22)
    plt.ylabel("Label Probability", fontsize=22)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-0.5, 10.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    datasets = [
        "MFRC",
        "SemEval",
        "GoEmotions",
    ]

    models = [
        "meta-llama--Llama-3.1-8B_0",
    ]

    yaml_files = []
    for dataset in datasets:
        for model in models:
            yaml_file = os.path.join(
                PROJECT_ROOT,
                "logs",
                dataset,
                "main_test_set",
                "baseline-cot",
                model,
                "indexed_metrics.yml",
            )
            if os.path.exists(yaml_file):
                yaml_files.append(yaml_file)

    figures_dir = os.path.join(PROB_DISTR_SCRIPTS, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, "spikiness_cot.png")
    plot_multilabel_icl(yaml_files, save_path=save_path)

