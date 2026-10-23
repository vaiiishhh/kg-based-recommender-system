"""
generate_ndcg_eval_set.py
=========================
Converts kg_data/ndgc_ground_truth.json into a flat eval_ndcg_set.json
suitable for both:

  - Hit Rate evaluation (binary: rating >= threshold is a positive)
  - Per-user NDCG evaluation (ranked list per user using graded relevance)

The resulting file is a flat list of {user, item, true_rating} records,
preserving per-user grouping order so that items from the same user are
contiguous (important for NDCG ranking checks).

Usage:
    python generate_ndcg_eval_set.py
    python generate_ndcg_eval_set.py \\
        --gt_json  kg_data/ndgc_ground_truth.json \\
        --output   eval_ndcg_set.json
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(
        description="Convert NDCG ground truth JSON to flat eval set."
    )
    parser.add_argument(
        "--gt_json", default="kg_data/ndgc_ground_truth.json",
        help="Path to ndgc_ground_truth.json (default: kg_data/ndgc_ground_truth.json)"
    )
    parser.add_argument(
        "--output", default="eval_ndcg_set.json",
        help="Output flat eval-set JSON (default: eval_ndcg_set.json)"
    )
    parser.add_argument(
        "--data_dir", default="kg_data/",
        help="KG data dir for filtering to valid pairs (default: kg_data/)"
    )
    parser.add_argument(
        "--skip_invalid", action="store_true", default=True,
        help="Skip (user, item) pairs not in the KG index (default: True)"
    )
    args = parser.parse_args()

    # Load KG ID maps for validity filtering
    with open(f"{args.data_dir}/user2id.json") as f:
        user2id = json.load(f)
    with open(f"{args.data_dir}/item2id.json") as f:
        item2id = json.load(f)

    # Load graded ground truth
    with open(args.gt_json) as f:
        gt = json.load(f)

    print(f"Loaded {len(gt)} users from {args.gt_json}")

    # Flatten: {user: [[item, rating], ...]} → [{user, item, true_rating}, ...]
    # Keep users grouped (contiguous) for NDCG grouping clarity
    records = []
    skipped_users = 0
    skipped_pairs = 0

    for user, item_list in gt.items():
        if args.skip_invalid and user not in user2id:
            skipped_users += 1
            continue

        user_records = []
        for entry in item_list:
            # entry is [item_asin, rating] or just item_asin (binary GT)
            if isinstance(entry, list):
                item_asin, rating = entry[0], float(entry[1])
            else:
                item_asin, rating = str(entry), 1.0  # binary GT → treat as positive

            if args.skip_invalid and item_asin not in item2id:
                skipped_pairs += 1
                continue

            user_records.append({
                "user":        user,
                "item":        item_asin,
                "true_rating": rating,
            })

        records.extend(user_records)

    total = len(records)
    n_users = len({r["user"] for r in records})

    if skipped_users:
        print(f"  Skipped {skipped_users} users not in KG.")
    if skipped_pairs:
        print(f"  Skipped {skipped_pairs} items not in KG.")

    print(f"Generated {total} pairs across {n_users} users.")

    # Print distribution
    from collections import Counter
    rating_dist = Counter(r["true_rating"] for r in records)
    print("Rating distribution:")
    for rating in sorted(rating_dist):
        print(f"  Rating {rating}: {rating_dist[rating]} pairs")

    # Per-user item count
    user_counts = Counter(r["user"] for r in records)
    item_count_dist = Counter(user_counts.values())
    print("Items per user:")
    for cnt in sorted(item_count_dist):
        print(f"  {cnt} items: {item_count_dist[cnt]} users")

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved → {args.output}")
    print("\nRun pipeline with:")
    print(f"  python predict.py --input_json {args.output} --output_json predictions_ndcg.json")
    print(f"  python evaluate_ndcg.py --input_json {args.output} --output_json predictions_ndcg.json")

if __name__ == "__main__":
    main()
