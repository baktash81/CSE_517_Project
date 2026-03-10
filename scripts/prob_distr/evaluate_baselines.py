import argparse
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
    Extracts the highest probability for each label across all generation steps.
    """
    
    # Iterate through examples (skipping metadata keys like 'description')
    all_scores = sample_data.get('test_all_scores')
    # if 'test_all_scores' not in sample_data (e.g. non-CoT baseline), fall back to test_scores
    if not all_scores:
        return sample_data['test_scores']
    
    max_scores = {}

    for step_dist in all_scores:
        for label, score in step_dist.items():
            # Save the maximum probability seen so far for each label
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
    
    c2n_scores = {}
    for label, p_i in scores.items():
        if label == 'none':
            continue  # Exclude 'none' from the final evaluation labels
        
        # P(li=1|di) = p_i / (p_i + p_none)
        prob = p_i / (p_i + p_none + 1e-10) # 1e-10 prevents division by zero
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
    for label in label_set:
        if label == 'none':
            continue
        hard_scores[label] = 1.0 - epsilon if label in preds else epsilon
        
    return hard_scores

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
    label_set = sorted([k for k in first_example.keys() if k != 'none'])
    
    for example_id, max_scores in max_distributions.items():
        datum = experiment_data[example_id]
        gt_labels = datum.get('test_gt', [])
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
            human_prob = 1.0 if label in gt_labels else 0.0
            if label not in max_scores:
                max_scores[label] = 0.0
            l1 += abs(max_scores[label] - human_prob)
        l1_distances.append(l1)
        
        # 3. F1 Score
        # Threshold the model probabilities at 0.5 to get binary predictions
        pred_binary = [1 if max_scores.get(label, 0.0) >= 0.5 else 0 for label in label_set]
        gt_binary = [1 if label in gt_labels else 0 for label in label_set]
        
        all_preds_binary.append(pred_binary)
        all_gts_binary.append(gt_binary)
        
    # Convert to numpy array
    all_gts_array = np.array(all_gts_binary)
    all_preds_array = np.array(all_preds_binary)

    # Calculate aggregate F1 (macro/micro)
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

def process_experiment(input_yaml_path, output_yaml_path):
    print(f"\n--- Processing {input_yaml_path} ---")
    data = load_data_from_yaml(input_yaml_path)
    
    # Handle the 'experiment_0' wrapper if it exists in the YAML
    if len(data) == 1 and 'experiment_0' in data:
        samples = data['experiment_0']
    else:
        samples = data
        
    # Determine label set from the first sample (ignoring 'none')
    first_sample = next(iter(samples.values()))
    label_set = [k for k in first_sample.get('test_scores', {}).keys() if k != 'none']
    
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
    metrics_c2n = calculate_paper_metrics(c2n_distr_data, samples)
    metrics_hard = calculate_paper_metrics(hard_distr_data, samples)
    metrics_max = calculate_paper_metrics(max_distr_data, samples)
    
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
    args = parser.parse_args()

    if os.path.exists(args.input_yaml):
        output_yaml_name = "alignment_scores_cot.yml" if args.is_cot else "alignment_scores_zero_shot.yml"
        output_yaml_path = os.path.join(os.path.dirname(args.input_yaml), output_yaml_name)
        process_experiment(args.input_yaml, output_yaml_path)
    else:
        print(f"File not found: {args.input_yaml}")

if __name__ == "__main__":
    main()