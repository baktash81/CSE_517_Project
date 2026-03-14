import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

def main():
    # Load the raw dataset using your exact path structure
    data_dir = 'datasets/hatexplain'
    
    print("Loading dataset...")
    with open(f"{data_dir}/dataset.json", "r") as f:
        raw_data = json.load(f)

    # Use the exact label mapping from your dataset class
    label_map = {"hatespeech": 0, "normal": 1, "offensive": 2}

    # ==========================================
    # 1. Average Pairwise Cohen's Kappa
    # ==========================================
    # Map annotator_id -> {post_id: label_index}
    annotator_data = defaultdict(dict)
    for post_id, post_info in raw_data.items():
        for ann in post_info["annotators"]:
            ann_id = ann["annotator_id"]
            label_idx = label_map[ann["label"]]
            annotator_data[ann_id][post_id] = label_idx

    annotator_ids = list(annotator_data.keys())
    pairwise_kappas = []

    print("Calculating Average Pairwise Cohen's Kappa...")
    # Compare every unique pair of annotators
    for i in range(len(annotator_ids)):
        for j in range(i + 1, len(annotator_ids)):
            ann1 = annotator_ids[i]
            ann2 = annotator_ids[j]

            # Find posts they both annotated
            common_posts = set(annotator_data[ann1].keys()).intersection(set(annotator_data[ann2].keys()))
            
            # Calculate Kappa if they share a meaningful number of posts (e.g., >= 5)
            # to prevent degenerate scores from isolated overlapping examples
            if len(common_posts) >= 5: 
                labels1 = [annotator_data[ann1][p] for p in common_posts]
                labels2 = [annotator_data[ann2][p] for p in common_posts]
                
                # Filter out perfectly uniform arrays (e.g., both raters only said "normal")
                # as Cohen's Kappa requires variance to calculate expected agreement
                if len(set(labels1)) > 1 or len(set(labels2)) > 1:
                    score = cohen_kappa_score(labels1, labels2)
                    if not np.isnan(score):
                        pairwise_kappas.append(score)

    avg_cohen = np.mean(pairwise_kappas)

    # ==========================================
    # 2. Fleiss' Kappa (The Recommended Metric)
    # ==========================================
    print("Calculating Fleiss' Kappa...")
    
    # Fleiss' Kappa requires an N x k matrix:
    # N = number of posts
    # k = number of categories (3)
    # Cell (i, j) = number of annotators who assigned category j to post i
    
    n_posts = len(raw_data)
    fleiss_table = np.zeros((n_posts, 3))

    for i, (post_id, post_info) in enumerate(raw_data.items()):
        for ann in post_info["annotators"]:
            label_idx = label_map[ann["label"]]
            fleiss_table[i, label_idx] += 1

    f_kappa = fleiss_kappa(fleiss_table)

    print("\n--- Inter-Annotator Agreement ---")
    print(f"Average Pairwise Cohen's Kappa: {avg_cohen:.4f}")
    print(f"Fleiss' Kappa:                  {f_kappa:.4f}")

if __name__ == "__main__":
    main()