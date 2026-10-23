"""
STEP 2: Binarize Ratings & Train/Test Split
Paper reference: Section 4.1
  - "randomly sampled 70% of user purchases as training data,
     took the rest 30% as test"
  - Ratings are binarized: rating >= 4 → positive interaction (kept)
    Ratings < 4 are discarded entirely (not used as negatives)
  - Only users with >= 5 positive interactions are kept (5-core filter)
    to ensure enough history for meaningful path reasoning
"""

import json
import pandas as pd
import numpy as np
import os

# ─── Config ───────────────────────────────────────────────────────────────────
META_PATH    = "/kaggle/input/datasets/vanshcantdoml/amazon-data/meta_subset.json"
REVIEWS_PATH = "/kaggle/input/datasets/vanshcantdoml/amazon-data/reviews_subset.json"
OUT_DIR      = "/kaggle/working/kg_data"
os.makedirs(OUT_DIR, exist_ok=True)

RATING_THRESHOLD  = 4.0   # >= this → positive interaction
MIN_USER_INTERACT = 5     # drop users with fewer positive reviews
MIN_ITEM_INTERACT = 5     # drop items with fewer positive reviews
TRAIN_RATIO       = 0.7
RANDOM_SEED       = 42

# ─── Load ─────────────────────────────────────────────────────────────────────
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

print("Loading reviews...")
reviews_df = pd.DataFrame(load_jsonl(REVIEWS_PATH))

# Normalise column names — handle both {'user','item','rating'} and
# {'reviewerID','asin','overall'} variants
col_map = {}
for c in reviews_df.columns:
    if c.lower() in ("user", "reviewerid"):
        col_map[c] = "user"
    elif c.lower() in ("item", "asin", "productid"):
        col_map[c] = "item"
    elif c.lower() in ("rating", "overall"):
        col_map[c] = "rating"
reviews_df = reviews_df.rename(columns=col_map)[["user", "item", "rating"]]
reviews_df["rating"] = pd.to_numeric(reviews_df["rating"], errors="coerce")

print(f"  Total reviews loaded : {len(reviews_df):,}")

# ─── Step 2a: Binarize ────────────────────────────────────────────────────────
# Keep ONLY positive interactions (rating >= threshold)
# This mirrors the paper's use of "purchase" — only confirmed positive signal
pos_df = reviews_df[reviews_df["rating"] >= RATING_THRESHOLD].copy()
pos_df["label"] = 1
print(f"  Positive interactions (rating >= {RATING_THRESHOLD}): {len(pos_df):,}")
print(f"  Discarded (rating < {RATING_THRESHOLD})             : {len(reviews_df) - len(pos_df):,}")

# Drop duplicate (user, item) pairs — keep the highest rating if duplicates exist
pos_df = pos_df.sort_values("rating", ascending=False).drop_duplicates(
    subset=["user", "item"]
)
print(f"  After dedup (user, item)                          : {len(pos_df):,}")

# ─── Step 2b: 5-core filtering (iterative) ────────────────────────────────────
# Iteratively remove users and items below the minimum interaction count
# until the dataset stabilises. Common practice in recommendation papers.
print("\nApplying 5-core filter (iterative)...")
prev_len = -1
iteration = 0
df = pos_df.copy()

while len(df) != prev_len:
    prev_len = len(df)
    iteration += 1
    user_counts = df["user"].value_counts()
    item_counts = df["item"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_INTERACT].index
    valid_items = item_counts[item_counts >= MIN_ITEM_INTERACT].index
    df = df[df["user"].isin(valid_users) & df["item"].isin(valid_items)]
    print(f"  Iteration {iteration}: {len(df):,} interactions, "
          f"{df['user'].nunique():,} users, {df['item'].nunique():,} items")

pos_filtered = df.reset_index(drop=True)
print(f"\n  Final after 5-core: {len(pos_filtered):,} interactions")
print(f"  Users : {pos_filtered['user'].nunique():,}")
print(f"  Items : {pos_filtered['item'].nunique():,}")

# ─── Step 2c: 70/30 train/test split ─────────────────────────────────────────
# Split PER USER so every user appears in both train and test
# (same method as paper — "randomly sampled 70% of user purchases")
np.random.seed(RANDOM_SEED)

train_rows = []
test_rows  = []

for user, group in pos_filtered.groupby("user"):
    items = group.sample(frac=1, random_state=RANDOM_SEED)  # shuffle
    n_train = max(1, int(len(items) * TRAIN_RATIO))         # at least 1 train
    train_rows.append(items.iloc[:n_train])
    test_rows.append(items.iloc[n_train:])

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

# Users with no test items (edge case when they had only 1 interaction)
# → move them fully to train only; they won't appear in evaluation
test_df = test_df[test_df["item"].isin(train_df["item"])]  # cold-item safety

print(f"\n  Train interactions : {len(train_df):,}")
print(f"  Test  interactions : {len(test_df):,}")
print(f"  Train/Test ratio   : {len(train_df)/len(pos_filtered):.2%} / "
      f"{len(test_df)/len(pos_filtered):.2%}")

# ─── Step 2d: Build entity ID maps ───────────────────────────────────────────
# Assign consecutive integer IDs to every user and item
# Paper requires integer IDs for embedding lookup tables
all_users = sorted(pos_filtered["user"].unique())
all_items = sorted(pos_filtered["item"].unique())

user2id = {u: i for i, u in enumerate(all_users)}
item2id = {it: i for i, it in enumerate(all_items)}

print(f"\n  user2id : {len(user2id):,} entries")
print(f"  item2id : {len(item2id):,} entries")

# ─── Save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv",  index=False)

with open(f"{OUT_DIR}/user2id.json", "w") as f:
    json.dump(user2id, f)
with open(f"{OUT_DIR}/item2id.json", "w") as f:
    json.dump(item2id, f)

# Also save full positive interaction list (needed for KG edge building)
pos_filtered.to_csv(f"{OUT_DIR}/positive_interactions.csv", index=False)

print(f"\n✓ Saved to {OUT_DIR}/")
print("  train.csv | test.csv | positive_interactions.csv")
print("  user2id.json | item2id.json")
