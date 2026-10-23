"""
STEP 1: Load & Inspect Data
Paper reference: Section 4.1 - Data Description
Goal: Understand the structure of both files before building anything.
"""

import json
import pandas as pd
from collections import defaultdict

# ─── Paths (Kaggle) ───────────────────────────────────────────────────────────
META_PATH    = "/kaggle/input/datasets/vanshcantdoml/amazon-data/meta_subset.json"
REVIEWS_PATH = "/kaggle/input/datasets/vanshcantdoml/amazon-data/reviews_subset.json"

# ─── Helper: load JSON-lines file ─────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading files...")
meta_records    = load_jsonl(META_PATH)
review_records  = load_jsonl(REVIEWS_PATH)

print(f"  meta_subset    : {len(meta_records):,} products")
print(f"  reviews_subset : {len(review_records):,} reviews")

# ─── Reviews DataFrame ────────────────────────────────────────────────────────
reviews_df = pd.DataFrame(review_records)
print("\n── Reviews columns ──")
print(reviews_df.columns.tolist())
print(reviews_df.head(3))

print("\n── Rating distribution ──")
print(reviews_df["rating"].value_counts().sort_index())

# ─── Meta DataFrame ───────────────────────────────────────────────────────────
meta_df = pd.DataFrame(meta_records)
print("\n── Meta columns ──")
print(meta_df.columns.tolist())
print(meta_df.head(2))

# ─── Key field coverage ───────────────────────────────────────────────────────
print("\n── Meta field coverage (% non-null / non-empty) ──")
important_fields = ["asin", "title", "brand", "category", "feature",
                    "also_buy", "also_view", "description"]
for col in important_fields:
    if col in meta_df.columns:
        if meta_df[col].dtype == object:
            # count non-null AND non-empty-list
            non_empty = meta_df[col].apply(
                lambda x: x is not None and x != [] and x != ""
            ).sum()
        else:
            non_empty = meta_df[col].notna().sum()
        pct = 100 * non_empty / len(meta_df)
        print(f"  {col:<20}: {non_empty:>6,} / {len(meta_df):,}  ({pct:.1f}%)")

# ─── Overlap: reviews <-> meta ────────────────────────────────────────────────
review_asins = set(reviews_df["item"].unique())
meta_asins   = set(meta_df["asin"].unique())
overlap      = review_asins & meta_asins

print(f"\n── ASIN overlap ──")
print(f"  Items in reviews           : {len(review_asins):,}")
print(f"  Items in meta              : {len(meta_asins):,}")
print(f"  Items present in BOTH      : {len(overlap):,}")
print(f"  Reviews items missing meta : {len(review_asins - meta_asins):,}")
