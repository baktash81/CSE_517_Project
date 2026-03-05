"""
Written by Kylie with help of Google Gemini. Reverse-engineered from the plotting scripts.
"""
import yaml

def load_data_from_yaml(yaml_file, seed=None, specific_experiment_num=None, existing_data=None):
    """
    Reverse-engineered function to load indexed_metrics.yml and return a dictionary 
    of {example_id: metrics_dict} formatted for the plotting scripts.
    """
    # 1. Load data from cache or file
    if existing_data is not None:
        data = existing_data
    else:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

    # 2. Handle nested dictionary structures (if grouped by seed/experiment)
    if isinstance(data, dict):
        first_val = next(iter(data.values()), {})
        # If the first item doesn't have metric keys, it's a nested seed dict
        if isinstance(first_val, dict) and 'test_preds' not in first_val and 'test_scores' not in first_val:
            key_to_find = seed if seed is not None else specific_experiment_num
            
            if f"experiment_{key_to_find}" in data:
                data = data[f"experiment_{key_to_find}"]
            elif key_to_find in data:
                data = data[key_to_find]
            elif str(key_to_find) in data:
                data = data[str(key_to_find)]
            else:
                data = first_val # Fallback to the first available split

    # 3. Clean the data: filter out metadata (like 'description') and standardize keys
    cleaned_data = {}
    for example_id, metrics in data.items():
        # Only process actual examples (which are dictionaries)
        if isinstance(metrics, dict) and ('test_scores' in metrics or 'test_preds' in metrics):
            
            if 'test_label' in metrics and 'test_gt' not in metrics:
                metrics['test_gt'] = metrics['test_label']
            if 'test_text' in metrics and 'test_outs' not in metrics:
                metrics['test_outs'] = metrics['test_text']
                
            cleaned_data[example_id] = metrics

    return cleaned_data