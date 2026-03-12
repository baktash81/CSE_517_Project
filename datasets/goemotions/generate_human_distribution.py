import pandas as pd
import json
import os
import argparse

def load_id_list(filepath):
    """Safely loads a text file of IDs into a set."""
    if not os.path.exists(filepath):
        print(f"  [!] Warning: '{filepath}' not found. Skipping this ID check.")
        return set()
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    parser = argparse.ArgumentParser(description="Calculate exact human distributions from GoEmotions CSVs.")
    parser.add_argument("--csv_dir", type=str, default="datasets/goemotions/", help="Directory containing goemotions_1.csv, etc.")
    parser.add_argument("--clustering_json", type=str, default="datasets/goemotions/emotion_clustering.json", help="Path to your emotion clustering file.")
    parser.add_argument("--main_test", type=str, default="prob_distr_ids/GoEmotions/main_test_set.txt", help="Path to main test IDs")
    parser.add_argument("--multi_test", type=str, default="prob_distr_ids/GoEmotions/big_multilabel.txt", help="Path to multilabel test IDs")
    parser.add_argument("--output", type=str, default="datasets/goemotions/human_distributions.json", help="Output JSON path")
    args = parser.parse_args()

    # 1. Load the raw GoEmotions CSVs
    dfs = []
    print("Loading CSV files...")
    for i in [1, 2, 3]:
        csv_path = os.path.join(args.csv_dir, f"goemotions_{i}.csv")
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"  [!] Warning: {csv_path} not found.")
    
    if not dfs:
        raise FileNotFoundError("No GoEmotions CSV files found! Please check the --csv_dir argument.")
        
    df = pd.concat(dfs, ignore_index=True)
    initial_len = len(df)
    
    # 2. Drop unclear examples (standard GoEmotions preprocessing)
    df = df[df['example_very_unclear'] == False]
    print(f"Filtered out {initial_len - len(df)} unclear annotations.")

    # 3. Load the emotion clustering map (e.g. {'joy': ['joy', 'amusement', 'excitement']})
    clustering = None
    if os.path.exists(args.clustering_json):
        with open(args.clustering_json, 'r') as f:
            clustering = json.load(f)
        print(f"Loaded emotion clusters from {args.clustering_json}")
    else:
        print(f"  [!] Warning: Clustering JSON not found. Will output original 28 raw labels.")

    raw_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

    # 4. Calculate Distributions
    print("Calculating annotator fractions per ID...")
    distributions = {}
    grouped = df.groupby('id')
    
    for name, group in grouped:
        num_raters = len(group)
        
        if clustering:
            # A rater endorses a cluster if they selected ANY sub-emotion within it
            rater_cluster_counts = {cluster: 0 for cluster in clustering.keys()}
            # explicitly ensure 'none' is tracked if neutral maps to it
            if 'none' not in rater_cluster_counts: 
                rater_cluster_counts['none'] = 0

            for _, row in group.iterrows():
                for cluster, sub_emotions in clustering.items():
                    if any(row.get(emo, 0) == 1 for emo in sub_emotions):
                        rater_cluster_counts[cluster] += 1
                
                # Check for the neutral/none condition if it isn't specifically in the clustering map
                if row.get('neutral', 0) == 1:
                    rater_cluster_counts['none'] += 1
            
            dist = {cluster: count / num_raters for cluster, count in rater_cluster_counts.items()}
            
        else:
            # If no clustering JSON, just sum and divide the raw 28 columns
            dist = {emo: group[emo].sum() / num_raters for emo in raw_emotions}
            if 'neutral' in dist:
                dist['none'] = dist.pop('neutral') # Rename neutral to none for pipeline compatibility
            
        distributions[name] = dist

    # 5. Strict Error Checking Against Required IDs
    print("Validating against required test sets...")
    main_ids = load_id_list(args.main_test)
    multi_ids = load_id_list(args.multi_test)
    
    all_required = main_ids.union(multi_ids)
    missing_ids = all_required - set(distributions.keys())
    
    if missing_ids:
        print(f"\n❌ ERROR: {len(missing_ids)} required IDs are completely missing from the CSVs!")
        print(f"   Sample missing IDs: {list(missing_ids)[:10]}\n")
        print("   (These were either excluded for being 'very_unclear' or are absent from the dataset).")
    else:
        if all_required:
            print("\n✅ SUCCESS: All required IDs from both test sets are present in the distributions.")
        else:
            print("\n⚠️ Note: No test ID files were found, so validation was skipped.")
    
    # 6. Save out the final payload
    with open(args.output, 'w') as f:
        json.dump(distributions, f, indent=4)
        
    print(f"Saved {len(distributions)} final human distributions to '{args.output}'")

if __name__ == '__main__':
    main()