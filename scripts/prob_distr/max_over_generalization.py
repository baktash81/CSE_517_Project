import argparse
import os

import yaml
import math
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from distribution_estimation import load_data_from_yaml


def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def calculate_max_over_generations(sample_data):
    """
    Extracts the highest probability for each label across all generation steps.
    """
    
    # Iterate through examples (skipping metadata keys like 'description')
    all_scores = sample_data['test_all_scores']
    # if 'test_all_scores' not in sample_data, grab test_scores instead
    if not all_scores:
        return sample_data['test_scores']
    
    max_scores = {}

    for step_dist in all_scores:
        for label, score in step_dist.items():
            # Save the maximum probability seen so far for each label
            if label not in max_scores or score > max_scores[label]:
                max_scores[label] = score
                
    return max_scores

def calculate_paper_metrics(max_distributions, experiment_data):
    """
    Calculates the exact metrics from the EMNLP 2025 paper:
    Negative Log-Likelihood (NLL), L1 Distance, and F1 Score.
    """
    l1_distances = []
    nlls = []
    
    all_preds_binary = []
    all_gts_binary = []

    first_example = next(iter(max_distributions.values()))
    label_set = sorted(list(first_example['max_over_generations_scores'].keys()))
    
    for example_id, max_scores in max_distributions.items():
        datum = experiment_data[example_id]
        max_scores = max_scores['max_over_generations_scores']

        # Ground truth labels from the YAML
        gt_labels = datum.get('test_gt', [])

        # is empty list, skip
        if not gt_labels:
            continue
        
        # 1. Negative Log-Likelihood (NLL)
        # NLL = -sum(log(P(g))) for g in ground_truth_labels
        nll = 0
        for g in gt_labels:
            prob = max_scores.get(g, 0.0)
            nll -= math.log(prob + 1e-8)    # Add epsilon inside log for stability
        nlls.append(nll)
        
        # 2. L1 Distance
        # L1 = sum |P_model(label) - P_human(label)| over all labels
        l1 = 0
        for label in label_set:
            # Using 1.0 or 0.0 here. If soft human distributions are available, plug them in here!
            human_prob = 1.0 if label in gt_labels else 0.0
            if label not in max_scores:
                max_scores[label] = 0.0
            l1 += abs(max_scores[label] - human_prob)
        l1_distances.append(l1)
        
        # 3. F1 Score
        pred_binary = [1 if label in datum.get('test_preds', []) else 0 for label in label_set]
        gt_binary = [1 if label in gt_labels else 0 for label in label_set]
        
        all_preds_binary.append(pred_binary)
        all_gts_binary.append(gt_binary)
        
    # Convert to numpy array
    all_gts_array = np.array(all_gts_binary)
    all_preds_array = np.array(all_preds_binary)

    # Calculate aggregate F1 (macro/micro)
    micro_f1 = f1_score(all_gts_array, all_preds_array, average='micro', zero_division=0)             # micro: all instances together
    macro_f1 = f1_score(all_gts_array, all_preds_array, average='macro', zero_division=0)             # macro: average of F1 for each class
    
    return {
        "Mean_NLL": np.mean(nlls),
        "Mean_L1_Distance": np.mean(l1_distances),
        "Micro_F1": micro_f1,
        "Macro_F1": macro_f1
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate max-over-generations distributions and metrics.")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the folder containing the input YAML file.")
    parser.add_argument("--input_yaml", type=str, default="indexed_metrics.yml", help="Path to the input YAML file with distributions.")
    parser.add_argument("--output_yaml", type=str, default="max_over_generations_output.yml", help="Path to save the output YAML file with max-over-generations distributions and metrics.")
    args = parser.parse_args()

    # Load the existing indexed_metrics.yml
    yaml_file_path = os.path.join(args.folder_path, args.input_yaml)
    data = load_data_from_yaml(yaml_file_path)

    output_data = {}
    # Calculate max-over-generations distributions for each sample
    print("--- Calculating max-over-generations distributions ---")
    for sample_key, sample_data in data.items():
        max_distributions = calculate_max_over_generations(sample_data)
        
        output_data[sample_key] = {
            "max_over_generations_scores": max_distributions
        }
    # Calculate alignment metrics from the paper
    metrics = calculate_paper_metrics(output_data, data)
    print("--- Distribution Alignment Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    
    output_file_path = os.path.join(args.folder_path, args.output_yaml)
    save_yaml(output_data, output_file_path)
    print(f"\nSaved new distributions and metrics to {output_file_path}")

if __name__ == "__main__":
    main()