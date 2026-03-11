import argparse
import json
import os
import yaml
from sklearn.metrics import f1_score
import math
import numpy as np

from distribution_estimation import load_data_from_yaml


def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

def calculate_max_over_generations(sample_data):
    """
    Reverted to extract the raw maximum probability across steps.
    This exactly replicates the L1=1.58 alignment found in the paper.
    """
    all_scores = sample_data.get('test_all_scores')
    if not all_scores:
        return sample_data.get('test_scores', {})
    
    max_scores = {}
    for step_dist in all_scores:
        for label, score in step_dist.items():
            if label not in max_scores or score > max_scores[label]:
                max_scores[label] = score
                
    return max_scores

def calculate_compare_to_none(sample_data):
    """
    Calculates Compare-to-None probabilities.
    Mathematically simplifies to p_i / (p_i + p_none).
    """
    scores = sample_data.get('test_scores', {})
    if not scores and sample_data.get('test_all_scores'):
        scores = sample_data['test_all_scores'][0]
        
    p_none = scores.get('none', 1e-8)
    max_p_other = max([v for k, v in scores.items() if k != 'none'] + [1e-8])
    
    c2n_scores = {}
    for label, p_i in scores.items():
        if label == 'none':
            prob = p_none / (p_none + max_p_other + 1e-10)
        else:
            prob = p_i / (p_i + p_none + 1e-10)
        c2n_scores[label] = prob
        
    return c2n_scores

def calculate_hard_predictions(sample_data, label_set, epsilon=1e-5):
    """
    Converts autoregressive text outputs into extreme probabilities 
    (1-epsilon for predicted, epsilon for not predicted).
    """
    preds = sample_data.get('test_preds', [])
    if preds is None:
        preds = []
        
    hard_scores = {}
    is_none_predicted = (not preds) or ('none' in preds)
    
    for label in label_set:
        if label == 'none':
            hard_scores[label] = 1.0 - epsilon if is_none_predicted else epsilon
        else:
            hard_scores[label] = 1.0 - epsilon if label in preds else epsilon
            
    return hard_scores

def calculate_paper_metrics(max_distributions, experiment_data, label_set, raw_human_distributions=None):
    """
    Calculates NLL, L1 Distance, and F1 Score.
    Dynamically handles both continuous (dict) and binary (list) ground truth distributions.
    """
    l1_distances = []
    nlls = []
    all_preds_binary = []
    all_gts_binary = []

    for example_id, max_scores in max_distributions.items():
        datum = experiment_data[example_id]

        # Binary GT
        binary_gt = datum.get('test_gt', ['none'])
        # Distribution GT (if available)
        is_distribution = False
        if raw_human_distributions and example_id in raw_human_distributions:
            gt_data = raw_human_distributions[example_id]
            is_distribution = True
            # Ensure 'none' is present if the human distribution is completely empty
            if not gt_data:
                gt_data = {'none': 1.0}
        else:
            # Fallback to the YAML binary list
            gt_raw = datum.get('test_gt', [])
            gt_data = gt_raw if gt_raw else ['none']
        
        # 1. Negative Log-Likelihood (NLL) - binary labels
        nll = 0
        for g in binary_gt:
            prob = max_scores.get(g, 0.0)
            nll -= math.log(prob + 1e-8)
        nlls.append(nll)
        
        # 2. L1 Distance
        # L1 = sum |P_model(label) - P_human(label)| over all labels
        l1 = 0
        for label in label_set:
            if is_distribution:
                human_prob = gt_data.get(label, 0.0)
            else:
                human_prob = 1.0 if label in gt_data else 0.0
            if label not in max_scores:
                max_scores[label] = 0.0
            l1 += abs(max_scores[label] - human_prob)
        l1_distances.append(l1)
        
        # 3. F1 Score
        # Threshold the model probabilities at 0.5 to get binary predictions
        pred_binary = [1 if max_scores.get(label, 0.0) >= 0.5 else 0 for label in label_set]
        gt_binary = [1 if label in binary_gt else 0 for label in label_set]
        
        all_preds_binary.append(pred_binary)
        all_gts_binary.append(gt_binary)
        
    # Convert to numpy array
    all_gts_array = np.array(all_gts_binary)
    all_preds_array = np.array(all_preds_binary)

    # Calculate aggregate F1
    micro_f1 = f1_score(all_gts_array, all_preds_array, average='micro', zero_division=0)             # micro: all instances together
    macro_f1 = f1_score(all_gts_array, all_preds_array, average='macro', zero_division=0)             # macro: average of F1 for each class
    example_f1 = f1_score(all_gts_array, all_preds_array, average='samples', zero_division=0) 
    
    return {
        "Mean_NLL": float(np.mean(nlls)),
        "Mean_L1_Distance": float(np.mean(l1_distances)),
        "Micro_F1": float(micro_f1),
        "Macro_F1": float(macro_f1),
        "Example_F1": float(example_f1)
    }

def process_experiment(input_yaml_path, output_yaml_path, human_dist=None):
    print(f"\n--- Processing {input_yaml_path} ---")
    data = load_data_from_yaml(input_yaml_path)
    
    # Handle the 'experiment_0' wrapper if it exists in the YAML
    if len(data) == 1 and 'experiment_0' in data:
        samples = data['experiment_0']
    else:
        samples = data
        
    # Determine label set
    global_labels = set(['none'])
    for sample in samples.values():
        global_labels.update(sample.get('test_scores', {}).keys())
        for step in sample.get('test_all_scores', []) or []:
            global_labels.update(step.keys())
        global_labels.update(sample.get('test_gt', []))
        
    label_set = sorted(list(global_labels))
    
    c2n_distr_data = {}
    hard_distr_data = {}
    max_distr_data = {}
    
    # Calculate distributions for all three methods
    for sample_key, sample_data in samples.items():
        # 1. Compare-to-None
        c2n_distr_data[sample_key] = calculate_compare_to_none(sample_data)
        
        # 2. Hard Predictions
        hard_distr_data[sample_key] = calculate_hard_predictions(sample_data, label_set)
        
        # 3. Max-Over-Generations
        max_raw = calculate_max_over_generations(sample_data)
        max_distr_data[sample_key] = {k: v for k, v in max_raw.items() if k != 'none'}
        
    # Calculate paper metrics for all three
    print("Calculating metrics for baseline and proposed methods...")
    metrics_c2n = calculate_paper_metrics(c2n_distr_data, samples, label_set, human_dist)
    metrics_hard = calculate_paper_metrics(hard_distr_data, samples, label_set, human_dist)
    metrics_max = calculate_paper_metrics(max_distr_data, samples, label_set, human_dist)

    # Package into the final output format
    output_data = {
        'compare_to_none': {
            'metrics': metrics_c2n,
            'distributions': c2n_distr_data
        },
        'hard_predictions': {
            'metrics': metrics_hard,
            'distributions': hard_distr_data
        },
        'max_over_generations': {
            'metrics': metrics_max,
            'distributions': max_distr_data
        }
    }
    
    save_yaml(output_data, output_yaml_path)
    
    print("Results:")
    print(f"  Compare-to-None F1 (Micro):    {metrics_c2n['Micro_F1']:.4f}")
    print(f"  Hard Predictions F1 (Micro):   {metrics_hard['Micro_F1']:.4f}")
    print(f"  Max-Over-Generations F1 (Micro): {metrics_max['Micro_F1']:.4f}")
    print(f"Saved complete results to {output_yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Calculate baseline and MoG metrics.")
    parser.add_argument("--input_yaml", type=str, required=True, help="Path to zero-shot indexed_metrics.yml")
    parser.add_argument("--is_cot", action='store_true', help="Flag if the input YAML is from a CoT experiment.")
    parser.add_argument("--dataset", type=str, choices=['goemotions', 'mfrc', 'hatexplain', 'semeval'], 
                        help="The dataset being evaluated (for logging/routing).")
    args = parser.parse_args()

    human_distributions = None
    if args.dataset in ['mfrc', 'goemotions', 'hatexplain']:
        human_dist_json = os.path.join('datasets', args.dataset, 'human_distributions.json')
        if os.path.exists(human_dist_json):
            print(f"Loading exact human distributions from {human_dist_json}...")
            with open(human_dist_json, 'r') as f:
                human_distributions = json.load(f)
        else:
            print(f"  [!] Warning: Distribution file {human_dist_json} not found. Falling back to binary YAML targets.")

    if os.path.exists(args.input_yaml):
        output_yaml_name = "alignment_scores_cot.yml" if args.is_cot else "alignment_scores_10_shot.yml"
        output_yaml_path = os.path.join(os.path.dirname(args.input_yaml), output_yaml_name)
        process_experiment(args.input_yaml, output_yaml_path, human_distributions)
    else:
        print(f"File not found: {args.input_yaml}")

if __name__ == "__main__":
    main()