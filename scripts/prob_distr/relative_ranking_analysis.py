import argparse
import yaml
import numpy as np

def load_data_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def analyze_relative_ranking(data):
    # Counters for the metrics
    total_intermediate_steps = 0
    second_highest_not_next = 0
    second_highest_never_predicted = 0
    cases_where_second_not_next = 0
    
    # Lists to calculate mean/median probabilities
    intermediate_at_r_plus_1 = [] 
    r_plus_1_pred_at_r = []       

    # Unwrap experiment_0 if it exists
    if len(data) == 1 and 'experiment_0' in data:
        samples = data['experiment_0']
    elif len(data) == 2 and 'experiment_1' in data:
        samples = data['experiment_1']
    else:
        samples = data

    for sample_id, sample in samples.items():
        # Skip metadata keys like 'description'
        if not isinstance(sample, dict):
            continue
            
        test_preds = sample.get('test_preds', [])
        test_all_scores = sample.get('test_all_scores', [])
        
        if not test_preds or not test_all_scores:
            continue
            
        # We only look at "intermediate" steps, meaning the model continues 
        # to predict at least one more label after the current one.
        num_steps = min(len(test_preds), len(test_all_scores))
        
        for r in range(num_steps - 1):
            current_scores = test_all_scores[r]
            next_scores = test_all_scores[r+1]
            next_pred = test_preds[r+1]
            
            # Sort the labels at step r by probability (descending)
            sorted_labels = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_labels) < 2:
                continue
                
            # Extract the label with the second-highest probability
            top_label = sorted_labels[0][0]
            second_label = sorted_labels[1][0]
            
            total_intermediate_steps += 1
            
            # --- Metric 1: Is the second highest label predicted next? ---
            if second_label != next_pred:
                second_highest_not_next += 1
                cases_where_second_not_next += 1
                
                # --- Metric 4: Is it predicted at all in the future? (Section D.3) ---
                if second_label not in test_preds[r+1:]:
                    second_highest_never_predicted += 1
                    
            # --- Metric 2: "intermediate @ r+1" ---
            # Probability of step r's second_label inside step r+1's distribution
            prob_second_at_next = next_scores.get(second_label, 0.0)
            intermediate_at_r_plus_1.append(prob_second_at_next)
            
            # --- Metric 3: "r+1 pred" ---
            # Probability of step r+1's actual prediction inside step r's distribution
            prob_next_at_current = current_scores.get(next_pred, 0.0)
            r_plus_1_pred_at_r.append(prob_next_at_current)

    # --- Final Calculations ---
    not_next_pct = (second_highest_not_next / total_intermediate_steps) * 100 if total_intermediate_steps > 0 else 0
    never_pred_pct = (second_highest_never_predicted / cases_where_second_not_next) * 100 if cases_where_second_not_next > 0 else 0
    
    avg_intermediate_at_r1 = np.mean(intermediate_at_r_plus_1) if intermediate_at_r_plus_1 else 0
    avg_r1_pred_at_r = np.mean(r_plus_1_pred_at_r) if r_plus_1_pred_at_r else 0
    
    median_intermediate_at_r1 = np.median(intermediate_at_r_plus_1) if intermediate_at_r_plus_1 else 0
    median_r1_pred_at_r = np.median(r_plus_1_pred_at_r) if r_plus_1_pred_at_r else 0

    # --- Output ---
    print(f"\n" + "="*50)
    print(f" RELATIVE RANKING ANALYSIS")
    print(f"="*50)
    print(f"Total intermediate steps analyzed: {total_intermediate_steps}")
    print(f"\n[1] Ranking Failures:")
    print(f"    Second-highest label NOT predicted next: {not_next_pct:.1f}%")
    print(f"\n[2] Section D.3 'Never Predicted' Rate:")
    print(f"    If not predicted next, rate it is NEVER predicted: {never_pred_pct:.1f}%")
    print(f"\n[3] 'intermediate @ r+1' (Prob of 2nd-highest evaluated at next step):")
    print(f"    Mean:   {avg_intermediate_at_r1:.4f}")
    print(f"    Median: {median_intermediate_at_r1:.4f}  <-- (Paper notes this clusters near 0)")
    print(f"\n[4] 'r+1 pred' (Prob of actual next prediction evaluated at current step):")
    print(f"    Mean:   {avg_r1_pred_at_r:.4f}")
    print(f"    Median: {median_r1_pred_at_r:.4f}  <-- (Paper notes this clusters near 0)")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze relative ranking across generation steps.")
    parser.add_argument("--input_yaml", type=str, required=True, help="Path to indexed_metrics.yml")
    args = parser.parse_args()

    try:
        data = load_data_from_yaml(args.input_yaml)
        analyze_relative_ranking(data)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.input_yaml}")