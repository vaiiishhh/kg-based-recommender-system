"""
generate_eval_set.py
====================
Generates a properly stratified evaluation set from the new kg_data test split.

Sampling strategy:
  - Uses kg_data/test.csv (pre-built by the KG pipeline, already unbiased)
  - Filters to pairs where BOTH user and item are in the KG
  - Stratifies across all 5 rating levels (equal counts per rating)
  - Ensures every (user, item) pair is unique
  - Outputs a JSON file with true_rating for NDCG relevance grading

Usage:
    python generate_eval_set.py --n_per_rating 40 --output eval_set.json
    (produces 200 pairs: 40 per rating level 1-5)
"""

import argparse
import json
import random
import pandas as pd
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Generate stratified evaluation set.")
    parser.add_argument("--data_dir",      default="kg_data/")
    parser.add_argument("--n_per_rating",  type=int, default=40,
                        help="Number of pairs per rating level (1-5). "
                             "Total = n_per_rating * 5.")
    parser.add_argument("--output",        default="eval_set.json")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load KG ID maps to filter to valid pairs
    with open(f"{args.data_dir}/user2id.json") as f:
        user2id = json.load(f)
    with open(f"{args.data_dir}/item2id.json") as f:
        item2id = json.load(f)

    # Load test.csv — columns: user_id, item_id, rating
    test = pd.read_csv(f"{args.data_dir}/test.csv")

    # Normalise column names to handle both old and new schemas
    col_map = {}
    for c in test.columns:
        if c.lower() in ("user", "user_id"):
            col_map[c] = "user"
        elif c.lower() in ("item", "item_id"):
            col_map[c] = "item"
        elif c.lower() in ("rating", "overall"):
            col_map[c] = "rating"
    test = test.rename(columns=col_map)

    # Filter to pairs where both user and item exist in the KG
    mask = test["user"].isin(user2id) & test["item"].isin(item2id)
    test = test[mask].drop_duplicates(subset=["user", "item"]).reset_index(drop=True)
    print(f"Valid (user, item) pairs in test.csv: {len(test):,}")

    # Stratify by rating
    buckets = defaultdict(list)
    for _, row in test.iterrows():
        r = float(row["rating"])
        # Bin to nearest integer rating
        r_int = round(r)
        r_int = max(1, min(5, r_int))
        buckets[r_int].append({
            "user":        row["user"],
            "item":        row["item"],
            "true_rating": r,
        })

    print("Available pairs per rating bucket:")
    for rating in sorted(buckets.keys()):
        print(f"  Rating {rating}: {len(buckets[rating]):,} pairs")

    # Sample n_per_rating from each bucket
    selected = []
    for rating in sorted(buckets.keys()):
        pool = buckets[rating]
        random.shuffle(pool)
        n = min(args.n_per_rating, len(pool))
        selected.extend(pool[:n])
        if n < args.n_per_rating:
            print(f"  WARNING: Only {n} pairs available for rating={rating} "
                  f"(requested {args.n_per_rating})")

    # Final dedup guard
    seen = set()
    final = []
    for rec in selected:
        key = (rec["user"], rec["item"])
        if key not in seen:
            seen.add(key)
            final.append(rec)

    # Deterministic ordering
    final.sort(key=lambda x: (x["true_rating"], x["user"], x["item"]))

    with open(args.output, "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nGenerated {len(final)} unique pairs → {args.output}")
    from collections import Counter
    dist = Counter(round(x["true_rating"]) for x in final)
    for r in sorted(dist):
        print(f"  Rating {r}: {dist[r]} pairs")


if __name__ == "__main__":
    main()
