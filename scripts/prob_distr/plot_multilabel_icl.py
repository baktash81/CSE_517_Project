import yaml
import os
import numpy as np

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib.pyplot as plt

from distribution_estimation import load_data_from_yaml

def get_graph_probs(data):
    """Fraction of examples with >1 predicted label. Supports ratio-experiment IDs (ratio_0.0, ...) or baseline IDs (no ratio)."""
    ratios = ['ratio_0.0', 'ratio_0.2', 'ratio_0.4', 'ratio_0.6', 'ratio_0.8', 'ratio_1.0']

    is_multilabel = [0] * len(ratios)
    ratio_counts = [0] * len(ratios)
    baseline_multilabel = 0
    baseline_total = 0

    for example_id, example in data.items():
        n_labels = max(len(example.get('test_all_scores') or []), len(example.get('test_preds') or []))
        is_multi = 1 if n_labels > 1 else 0

        matched_ratio = False
        for i, ratio in enumerate(ratios):
            if ratio in example_id:
                ratio_counts[i] += 1
                is_multilabel[i] += is_multi
                matched_ratio = True
                break
        if not matched_ratio:
            baseline_total += 1
            baseline_multilabel += is_multi

    # If we have ratio-experiment data, return per-ratio fractions
    if sum(ratio_counts) > 0:
        return [
            (is_multilabel[i] / ratio_counts[i]) if ratio_counts[i] > 0 else 0.0
            for i in range(len(ratios))
        ]
    # Baseline data (no ratio in any example_id): one fraction for all x positions
    frac = (baseline_multilabel / baseline_total) if baseline_total > 0 else 0.0
    return [frac] * len(ratios)                
                
def plot_multilabel_icl(yaml_files, save_path, color='blue'):
    
    potential_datasets = ['GoEmotions', 'MFRC', 'SemEval']
    potential_models = ['Llama-3.2-1B', 'Llama-3.1-8B', 'Llama-3.3-70B-Instruct']
    
    # results[dataset][model] = order_accs
    results = {d: {} for d in potential_datasets}
    
    for yml_file in yaml_files:
        upper_yml_file = yml_file
        # yml_file = yml_file.lower()
        if 'semeval' in yml_file.lower():
            dataset = 'SemEval'
        elif 'mfrc' in yml_file.lower():
            dataset = 'MFRC'
        elif 'goemotions' in yml_file.lower():
            dataset = 'GoEmotions'
        else:
            raise ValueError(f'Unknown dataset: {yml_file}')

        if '3.1' in yml_file:
            model = '8B Instruct'
        elif '3.2' in yml_file:
            model = '1B Instruct'
        elif '3.3' in yml_file:
            model = '70B Instruct'
        elif 'qwen' in yml_file.lower():
            model = 'Qwen 7B Instruct'
        else:
            raise ValueError(f'Unknown model: {yml_file}')
        
        results[dataset][model] = []
        yaml_data = yaml.safe_load(open(upper_yml_file, 'r'))
        # for seed in [1, 2, 3, 4]:
        for seed in [0]:
            seed_data = load_data_from_yaml(upper_yml_file, specific_experiment_num=seed-1, existing_data=yaml_data)
            if seed_data is not None:
                results[dataset][model].append(get_graph_probs(seed_data))
        print(dataset, model, results[dataset][model])
        
    plt.figure(figsize=(12, 7))
    
    # Set up colors and line styles
    colors = {'GoEmotions': 'blue', 'MFRC': 'red', 'SemEval': 'green'}
    line_styles = {'1B Instruct': ':',
                   '8B Instruct': '--',
                   '70B Instruct': '-.',
                   'Qwen 7B Instruct': '-'}
    
    # Plot each dataset and model combination
    for dataset in results:
        for model in results[dataset]:
            accuracies = results[dataset][model]
            # Pad arrays with nan to make them same length
            max_len = max(len(acc) for acc in accuracies)
            padded_accuracies = np.array([np.pad(acc, (0, max_len - len(acc)), constant_values=np.nan) for acc in accuracies])
            x = [0, 0.2, 0.4, 0.6, 0.8, 1]
            
            # Calculate mean and std across seeds, ignoring nan values
            mean_accuracies = np.nanmean(padded_accuracies, axis=0)
            std_accuracies = np.nanstd(padded_accuracies, axis=0)
            
            plt.plot(x, mean_accuracies, 
                    color=colors[dataset], 
                    linestyle=line_styles[model],
                    marker='o',
                    alpha=0.6)
            
            # Add shaded standard deviation
            plt.fill_between(x, 
                           mean_accuracies - std_accuracies / 2,
                           mean_accuracies + std_accuracies / 2,
                           color=colors[dataset],
                           alpha=0.1)
    
    plt.ylim(0, 1)
    plt.xlim(-0.05, 1.05)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
    
    plt.ylabel('% of Multi-label Outputs', fontsize=22)
    plt.xlabel('% of Multi-label In-Context Examples', fontsize=22)
    
    # Create separate legends for datasets and models
    dataset_handles = [plt.Line2D([0], [0], color=color, label=f'{dataset}', linestyle='-') 
                      for dataset, color in colors.items()]
    model_handles = [plt.Line2D([0], [0], color='black', label=f'{model}', linestyle=style) 
                    for model, style in line_styles.items()]
    
    plt.legend(handles=dataset_handles + model_handles, 
            #   bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              fontsize=20,
              )
    
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
        
if __name__ == '__main__':
    
    datasets = [
        'MFRC',
        'SemEval',
        'GoEmotions',
    ]
    
    models = [
        # 'meta-llama--Llama-3.2-1B-Instruct_0',
        'meta-llama--Llama-3.1-8B_0',
        # 'meta-llama--Llama-3.3-70B-Instruct_0',
        # 'Qwen--Qwen2.5-7B-Instruct_0',
    ]
    
    yaml_files = []
    for dataset in datasets:
        for model in models:
            yaml_file = os.path.join(PROJECT_ROOT, 'logs', dataset, 'main_test_set', 'baseline', model, 'indexed_metrics.yml')
            if os.path.exists(yaml_file):
                yaml_files.append(yaml_file)

    save_path = os.path.join(PROJECT_ROOT, 'scripts', 'prob_distr', 'figures', 'multilabel_ICL.pdf')
    plot_multilabel_icl(yaml_files, save_path=save_path)