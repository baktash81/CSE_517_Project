import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

def main():
    print("Downloading/Loading MFRC dataset from HuggingFace...")
    ds = load_dataset("USC-MOLA-Lab/MFRC", split="train")
    df = ds.to_pandas()

    # Standard paper mapping
    foundation_mapping = {
        'Care': 'care',
        'Equality': 'fairness',
        'Proportionality': 'fairness',
        'Loyalty': 'loyalty',
        'Authority': 'authority',
        'Purity': 'purity',
        'Thin Morality': 'none',
        'Non-Moral': 'none'
    }
    
    label_set = sorted(list(set(foundation_mapping.values())))
    
    # Structure: item_id -> annotator -> set of mapped labels
    annotations = {}
    grouped = df.groupby('text') # Grouping by text as a proxy for unique post ID
    
    for text, group in grouped:
        annotations[text] = {}
        for _, row in group.iterrows():
            annotator = row['annotator']
            raw_labels = str(row['annotation']).split(',')
            
            mapped_classes = set()
            for raw_label in raw_labels:
                clean_label = raw_label.strip()
                if clean_label in foundation_mapping:
                    mapped_classes.add(foundation_mapping[clean_label])
            
            # Fallback/Mutually exclusive rules
            if not mapped_classes:
                mapped_classes.add('none')
            if len(mapped_classes) > 1 and 'none' in mapped_classes:
                mapped_classes.remove('none')
                
            annotations[text][annotator] = mapped_classes

    # ==========================================
    # 1. Average Label-Wise Fleiss' Kappa
    # ==========================================
    # For Fleiss, we calculate agreement on a binary "Yes/No" for each foundation
    fleiss_kappas = {}
    
    for label in label_set:
        # N x 2 matrix: [Number of raters who said No, Number who said Yes]
        # We only use posts that have exactly 3 annotators to keep the matrix clean
        fleiss_table = []
        for text, raters in annotations.items():
            if len(raters) == 3:
                yes_votes = sum(1 for rater_labels in raters.values() if label in rater_labels)
                no_votes = 3 - yes_votes
                fleiss_table.append([no_votes, yes_votes])
                
        if fleiss_table:
            score = fleiss_kappa(np.array(fleiss_table))
            fleiss_kappas[label] = score
            
    avg_fleiss = np.mean(list(fleiss_kappas.values()))

    # ==========================================
    # 2. Average Label-Wise Pairwise Cohen's Kappa
    # ==========================================
    cohen_kappas = {label: [] for label in label_set}
    
    # Extract unique annotator IDs
    all_annotators = list(set(df['annotator'].unique()))
    
    for label in label_set:
        for i in range(len(all_annotators)):
            for j in range(i + 1, len(all_annotators)):
                ann1 = all_annotators[i]
                ann2 = all_annotators[j]
                
                labels1, labels2 = [], []
                for text, raters in annotations.items():
                    if ann1 in raters and ann2 in raters:
                        labels1.append(1 if label in raters[ann1] else 0)
                        labels2.append(1 if label in raters[ann2] else 0)
                
                # Only calculate if they share enough posts and have variance
                if len(labels1) >= 5 and (len(set(labels1)) > 1 or len(set(labels2)) > 1):
                    score = cohen_kappa_score(labels1, labels2)
                    if not np.isnan(score):
                        cohen_kappas[label].append(score)

    # Average the pairwise scores for each label, then average across all labels
    label_avg_cohen = {lbl: np.mean(scores) if scores else 0 for lbl, scores in cohen_kappas.items()}
    avg_cohen = np.mean(list(label_avg_cohen.values()))

    print("\n--- MFRC Inter-Annotator Agreement ---")
    print("Breakdown by Label (Fleiss' Kappa):")
    for lbl, score in fleiss_kappas.items():
        print(f"  {lbl.capitalize():<10}: {score:.4f}")
        
    print(f"\nOverall Average Fleiss' Kappa: {avg_fleiss:.4f}")
    print(f"Overall Average Cohen's Kappa: {avg_cohen:.4f}")

if __name__ == '__main__':
    main()