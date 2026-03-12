import argparse
import json
import os
import hashlib
import yaml
import pandas as pd
from datasets import load_dataset

def get_text_hash(text):
    """Fallback ID generator if splits.yaml is missing."""
    return hashlib.md5(str(text).encode('utf-8')).hexdigest()

def load_id_list(filepath):
    """Safely loads a text file of IDs into a set."""
    if not os.path.exists(filepath):
        print(f"  [!] Warning: '{filepath}' not found. Skipping this ID check.")
        return set()
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def main():
    parser = argparse.ArgumentParser(description="Calculate exact human distributions matching the MFRC Dataset Class.")
    parser.add_argument("--main_test", type=str, default="prob_distr_ids/MFRC/main_test_set.txt", help="Path to main test IDs")
    parser.add_argument("--multi_test", type=str, default="prob_distr_ids/MFRC/big_multilabel.txt", help="Path to multilabel test IDs")
    parser.add_argument("--splits_yaml", type=str, default="datasets/mfrc/splits.yaml", help="Path to splits.yaml to map text to IDs")
    parser.add_argument("--output", type=str, default="datasets/mfrc/human_distributions.json", help="Output JSON path")
    args = parser.parse_args()

    # 1. Load the dataset directly from HuggingFace
    print("Downloading/Loading MFRC dataset from HuggingFace...")
    ds = load_dataset("USC-MOLA-Lab/MFRC", split="train")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} total annotation rows.")

    # 2. Map text to your exact dataset IDs using splits.yaml
    if os.path.exists(args.splits_yaml):
        print(f"Loading text-to-ID mappings from {args.splits_yaml}...")
        with open(args.splits_yaml, "r") as fp:
            splits_data = yaml.safe_load(fp)
            text2id = {}
            # Combine all splits so we capture every possible ID
            for split_name, mapping in splits_data.items():
                text2id.update(mapping)
                
        df['item_id'] = df['text'].map(text2id)
        df = df.dropna(subset=['item_id']) # Drop text that isn't in your splits
    else:
        print(f"  [!] Warning: {args.splits_yaml} not found. Falling back to MD5 text hashes.")
        df['item_id'] = df['text'].apply(get_text_hash)

    # 3. Calculate Distributions exactly matching the MFRC Dataset Class
    print("Calculating annotator fractions per unique example...")
    distributions = {}
    global_label_set = set(['none'])
    
    grouped = df.groupby('item_id')
    for item_id, group in grouped:
        
        annotator_groups = group.groupby('annotator')
        valid_raters = 0
        rater_class_counts = {}
        
        for annotator, ann_group in annotator_groups:
            # Replicate the exact string splitting logic from your dataset loader
            raw_annotation = ann_group['annotation'].dropna().iloc[0] if len(ann_group['annotation'].dropna()) > 0 else ""
            if not raw_annotation:
                continue
                
            labels = [l.strip() for l in raw_annotation.split(",")]
            
            # THE EXACT FILTERING LOGIC FROM YOUR MFRC CLASS:
            if len(labels) > 1 and ("Non-Moral" in labels or "Thin Morality" in labels):
                continue  # Skip this rater entirely
            elif len(labels) > 0 and (labels[0] == "Non-Moral" or labels[0] == "Thin Morality"):
                labels = []
                
            valid_raters += 1
            
            if not labels:
                rater_class_counts['none'] = rater_class_counts.get('none', 0) + 1
            else:
                for l in labels:
                    rater_class_counts[l] = rater_class_counts.get(l, 0) + 1
                    global_label_set.add(l)
                    
        # Calculate the final distribution for this example
        if valid_raters > 0:
            dist = {cls: count / valid_raters for cls, count in rater_class_counts.items()}
            distributions[item_id] = dist

    # 4. Strict Error Checking Against Required IDs
    print("Validating against required test sets...")
    main_ids = load_id_list(args.main_test)
    multi_ids = load_id_list(args.multi_test)
    
    all_required = main_ids.union(multi_ids)
    missing_ids = all_required - set(distributions.keys())
    
    if missing_ids:
        print(f"\n❌ ERROR: {len(missing_ids)} required IDs are missing from the parsed dataset!")
        print(f"   Sample missing IDs: {list(missing_ids)[:5]}")
    else:
        if all_required:
            print("\n✅ SUCCESS: All required IDs from both test sets are present in the distributions.")
        else:
            print("\n⚠️ Note: No test ID files were found, so validation was skipped.")
    
    # 5. Save out the final payload
    with open(args.output, 'w') as f:
        json.dump(distributions, f, indent=4)
        
    print(f"\nSaved {len(distributions)} final human distributions to '{args.output}'")
    print(f"Detected 6 Moral Foundations + None: {sorted(list(global_label_set))}")

if __name__ == '__main__':
    main()