import argparse
import json
from collections import Counter

def create_hatexplain_distributions(raw_dataset_path, output_path):
    """
    Reads the raw Hatexplain dataset and calculates the empirical 
    probability distribution of labels based on annotator agreement.
    """
    print(f"Loading raw Hatexplain data from {raw_dataset_path}...")
    with open(raw_dataset_path, 'r') as f:
        raw_data = json.load(f)
        
    human_distributions = {}
    
    # Iterate over every post in the Hatexplain dataset
    for post_id, post_data in raw_data.items():
        annotators = post_data.get('annotators', [])
        
        if not annotators:
            continue
            
        # Count how many times each label was chosen
        label_counts = Counter()
        for annotator in annotators:
            label = annotator.get('label')
            if label:
                label_counts[label] += 1
                
        # Calculate the probability distribution (count / total annotators)
        num_annotators = len(annotators) # typically 3 for Hatexplain
        distribution = {
            label: count / num_annotators 
            for label, count in label_counts.items()
        }
        
        human_distributions[post_id] = distribution
        
    # Save the output to the format expected by evaluate_baselines.py
    with open(output_path, 'w') as f:
        json.dump(human_distributions, f, indent=4)
        
    print(f"Successfully processed {len(human_distributions)} examples.")
    print(f"Saved empirical human distributions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate exact human distributions from GoEmotions CSVs.")
    parser.add_argument("--input", type=str, default="datasets/hatexplain/dataset.json", help="Path to the raw Hatexplain dataset JSON")
    parser.add_argument("--output", type=str, default="datasets/hatexplain/human_distributions.json", help="Output JSON path for human distributions")
    args = parser.parse_args()

    create_hatexplain_distributions(args.input, args.output)