"""
Create NDCG test data with graded relevance (rating-weighted ground truth).

Uses test.csv ratings directly as relevance scores:
- Rating 5 → relevance 2.0 (highly relevant)
- Rating 4 → relevance 1.0 (relevant)
- Rating 3 → relevance 0.5 (somewhat relevant)
- Rating ≤2 → relevance 0 (not relevant)

This legitimately increases items per user without data leakage.
NDCG naturally supports graded relevance (not just binary).
"""

import json
import pandas as pd
import numpy as np


def rating_to_relevance(rating):
    """Convert rating to graded relevance score"""
    if rating >= 4.5:
        return 2.0  # 5-star: highly relevant
    elif rating >= 3.5:
        return 1.0  # 4-star: relevant
    elif rating >= 2.5:
        return 0.5  # 3-star: somewhat relevant
    else:
        return 0.0  # 1-2 star: not relevant


def create_graded_ndcg_test_data(data_dir=".", min_relevant_items=3):
    """
    Create NDCG evaluation test data using graded relevance.
    
    Args:
        data_dir: directory containing test.csv
        min_relevant_items: minimum items with relevance > 0 per user
    
    Returns:
        dict with statistics
    """
    
    # Load test data
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    print(f"✓ Loaded test.csv: {len(test_df)} interactions")
    
    # Convert ratings to graded relevance
    test_df['relevance'] = test_df['rating'].apply(rating_to_relevance)
    
    # Create ground truth: user → [(item_id, relevance_score), ...]
    ground_truth = {}
    for user_id in test_df['user_id'].unique():
        user_data = test_df[test_df['user_id'] == user_id]
        # Filter to items with relevance > 0
        relevant_items = user_data[user_data['relevance'] > 0]
        
        if len(relevant_items) >= min_relevant_items:
            # Sort by relevance (highest first)
            items = relevant_items.sort_values('relevance', ascending=False)
            ground_truth[user_id] = [
                (row['item_id'], float(row['relevance']))
                for _, row in items.iterrows()
            ]
    
    print(f"\n📊 Graded Relevance Statistics (threshold: >={min_relevant_items} items/user):\n")
    
    # Statistics
    total_items_in_ground_truth = sum(len(items) for items in ground_truth.values())
    num_users = len(ground_truth)
    items_per_user = [len(items) for items in ground_truth.values()]
    
    print(f"  Users (≥{min_relevant_items} items): {num_users}")
    print(f"  Total items in ground truth: {total_items_in_ground_truth}")
    print(f"  Avg items/user: {np.mean(items_per_user):.2f}")
    print(f"  Min: {np.min(items_per_user)}, Max: {np.max(items_per_user)}")
    print(f"  Distribution:")
    
    for count in sorted(set(items_per_user)):
        n = sum(1 for x in items_per_user if x == count)
        print(f"    {count} items: {n} users")
    
    # Relevance score breakdown
    print(f"\n  Relevance score breakdown:")
    for relevance_level, label in [(2.0, "5-star (2.0)"), (1.0, "4-star (1.0)"), (0.5, "3-star (0.5)")]:
        count = sum(1 for items in ground_truth.values() 
                   for _, r in items if r == relevance_level)
        print(f"    {label}: {count} items")
    
    # Save ground_truth_graded.json
    # Format: {user_id: [(item_id, relevance_score), ...], ...}
    # For compatibility with ndcg_score, also save binary version
    
    output_file_graded = f"{data_dir}/ground_truth_graded.json"
    with open(output_file_graded, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\n✓ Saved ground_truth_graded.json ({num_users} users)")
    
    # Also save binary version (relevance > 0.5 is positive)
    ground_truth_binary = {
        user_id: [item for item, rel in items if rel > 0.5]
        for user_id, items in ground_truth.items()
    }
    
    output_file_binary = f"{data_dir}/ground_truth_binary.json"
    with open(output_file_binary, 'w') as f:
        json.dump(ground_truth_binary, f, indent=2)
    print(f"✓ Saved ground_truth_binary.json ({num_users} users, relevance > 0.5)")
    
    # Save queries
    queries_df = pd.DataFrame({
        'user_id': list(ground_truth.keys()),
        'num_relevant_items': [len(items) for items in ground_truth.values()]
    })
    queries_file = f"{data_dir}/test_queries_graded.csv"
    queries_df.to_csv(queries_file, index=False)
    print(f"✓ Saved test_queries_graded.csv")
    
    # Save summary
    summary = {
        'evaluation_type': 'NDCG with graded relevance',
        'num_test_users': num_users,
        'total_items_in_ground_truth': total_items_in_ground_truth,
        'avg_items_per_user': float(np.mean(items_per_user)),
        'min_items_per_user': int(np.min(items_per_user)),
        'max_items_per_user': int(np.max(items_per_user)),
        'relevance_scale': {
            '2.0': '5-star (highly relevant)',
            '1.0': '4-star (relevant)',
            '0.5': '3-star (somewhat relevant)',
            '0.0': '1-2 star (not relevant)'
        },
        'notes': 'Graded relevance allows NDCG to reward ranking of highly-relevant items higher'
    }
    
    summary_file = f"{data_dir}/ndcg_graded_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved ndcg_graded_summary.json")
    
    return summary


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    min_items = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print("=" * 70)
    print("Creating NDCG Test Data with Graded Relevance")
    print("=" * 70)
    print()
    
    summary = create_graded_ndcg_test_data(data_dir, min_relevant_items=min_items)
    
    print("\n" + "=" * 70)
    print("Example: How to use graded relevance in NDCG evaluation")
    print("=" * 70)
    print("""
import json
from sklearn.metrics import ndcg_score

# Load graded ground truth
with open('ground_truth_graded.json') as f:
    ground_truth = json.load(f)

# For each user:
for user_id in ground_truth:
    # Get model's ranked predictions
    y_pred = model.predict_scores(user_id, all_items)  # [n_items]
    
    # Create graded relevance vector
    y_true = []
    ground_truth_dict = {item: rel for item, rel in ground_truth[user_id]}
    for item in all_items:
        y_true.append(ground_truth_dict.get(item, 0.0))  # 0.0 if not in GT
    
    # Compute NDCG@10 (now accounts for graded relevance!)
    ndcg = ndcg_score([y_true], [y_pred], k=10)
    print(f'{user_id}: NDCG@10 = {ndcg:.4f}')

# NDCG will reward ranking 5-star items (rel=2.0) higher than 4-star (rel=1.0)
""")
    
    print("=" * 70)
