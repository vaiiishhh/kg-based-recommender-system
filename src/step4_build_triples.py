"""
STEP 4: Build KG Triples
Paper reference: Section 3.1, 3.2

This script:
  1. Assigns GLOBAL integer IDs to every entity across all types
  2. Builds all 7 forward relation triples
  3. Adds reverse edges for every triple  (Section 3.2: "if (e,r,e') ∈ G_R,
     then (e',r,e) ∈ G_R")
  4. Adds self-loop (NOOP) edges for every entity  (Section 3.2)
  5. Saves:
       - kg_triples.tsv          (head_id, relation_id, tail_id)
       - entity_index.json       (str → {global_id, type})
       - relation_index.json     (relation_name → relation_id)
       - kg_stats.md             (Table-1-style summary for your peer)

Relations (forward + reverse + noop):
  ID  0  purchase        User → Item
  ID  1  rev_purchase    Item → User
  ID  2  mention         User → Feature
  ID  3  rev_mention     Feature → User
  ID  4  described_by    Item → Feature
  ID  5  rev_described   Feature → Item
  ID  6  produced_by     Item → Brand
  ID  7  rev_produced    Brand → Item
  ID  8  belong_to       Item → Category
  ID  9  rev_belong      Category → Item
  ID  10 also_bought     Item → Item
  ID  11 rev_also_bought Item → Item  (symmetric, so same direction added both ways)
  ID  12 also_viewed     Item → Item
  ID  13 rev_also_viewed Item → Item
  ID  14 self_loop (NOOP) any → same
"""

import json
import os
import pandas as pd
from collections import defaultdict

OUT_DIR = "/kaggle/working/kg_data"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load all artefacts from Steps 2 & 3 ─────────────────────────────────────
def lj(fname):
    with open(f"{OUT_DIR}/{fname}") as f:
        return json.load(f)

user2id        = lj("user2id.json")
item2id        = lj("item2id.json")
brand2id       = lj("brand2id.json")
cat2id         = lj("cat2id.json")
feat2id        = lj("feat2id.json")

item_brand     = lj("item_brand.json")
item_cats      = lj("item_cats.json")
item_features  = lj("item_features.json")
item_also_bought = lj("item_also_bought.json")
item_also_viewed = lj("item_also_viewed.json")
user_mention   = lj("user_mention.json")

train_df = pd.read_csv(f"{OUT_DIR}/train.csv")

# ─── Step 4A: Build GLOBAL entity ID space ───────────────────────────────────
# Each entity type gets a non-overlapping integer range.
# This is required so that embedding matrices index every entity uniquely.

entity_index = {}   # "TYPE:str_id" → {"global_id": int, "type": str, "name": str}
type_offsets = {}   # entity_type → starting global ID

offset = 0
for etype, mapping in [
    ("user",     user2id),
    ("item",     item2id),
    ("feature",  feat2id),
    ("brand",    brand2id),
    ("category", cat2id),
]:
    type_offsets[etype] = offset
    for name, local_id in mapping.items():
        global_id = offset + local_id
        entity_index[f"{etype}:{name}"] = {
            "global_id": global_id,
            "type":      etype,
            "name":      name,
            "local_id":  local_id,
        }
    offset += len(mapping)
    print(f"  {etype:<10}: IDs {type_offsets[etype]:>7} – {offset-1:>7}  "
          f"({len(mapping):,} entities)")

NUM_ENTITIES = offset
print(f"\n  TOTAL entities : {NUM_ENTITIES:,}")

# ─── Lookup helpers ───────────────────────────────────────────────────────────
def uid(user_str):
    """Global ID for a user string."""
    return type_offsets["user"] + user2id[user_str]

def iid(item_str):
    """Global ID for an item ASIN string."""
    return type_offsets["item"] + item2id[item_str]

def fid(feat_str):
    """Global ID for a feature word."""
    return type_offsets["feature"] + feat2id[feat_str]

def bid(brand_str):
    """Global ID for a brand string."""
    return type_offsets["brand"] + brand2id[brand_str]

def cid(cat_str):
    """Global ID for a category string."""
    return type_offsets["category"] + cat2id[cat_str]

# ─── Step 4B: Relation index ─────────────────────────────────────────────────
RELATIONS = {
    "purchase":        0,
    "rev_purchase":    1,
    "mention":         2,
    "rev_mention":     3,
    "described_by":    4,
    "rev_described":   5,
    "produced_by":     6,
    "rev_produced":    7,
    "belong_to":       8,
    "rev_belong":      9,
    "also_bought":     10,
    "rev_also_bought": 11,
    "also_viewed":     12,
    "rev_also_viewed": 13,
    "self_loop":       14,
}
NUM_RELATIONS = len(RELATIONS)

# ─── Step 4C: Collect all triples ────────────────────────────────────────────
triples = []   # list of (head_global_id, relation_id, tail_global_id)

def add(h, r, t):
    triples.append((h, r, t))

print("\nBuilding triples...")

# 1. purchase: User → Item  (TRAIN ONLY — paper uses only train interactions in KG)
purchase_count = 0
for _, row in train_df.iterrows():
    u, it = row["user"], row["item"]
    if u in user2id and it in item2id:
        add(uid(u),  RELATIONS["purchase"],     iid(it))
        add(iid(it), RELATIONS["rev_purchase"], uid(u))   # reverse
        purchase_count += 1
print(f"  purchase edges (forward+rev) : {purchase_count*2:,}")

# 2. mention: User → Feature
mention_count = 0
for user_str, feats in user_mention.items():
    if user_str not in user2id:
        continue
    for feat_str in feats:
        if feat_str in feat2id:
            add(uid(user_str),  RELATIONS["mention"],     fid(feat_str))
            add(fid(feat_str),  RELATIONS["rev_mention"], uid(user_str))
            mention_count += 1
print(f"  mention edges (forward+rev)  : {mention_count*2:,}")

# 3. described_by: Item → Feature
desc_count = 0
for item_str, feats in item_features.items():
    if item_str not in item2id:
        continue
    for feat_str in feats:
        if feat_str in feat2id:
            add(iid(item_str),  RELATIONS["described_by"], fid(feat_str))
            add(fid(feat_str),  RELATIONS["rev_described"], iid(item_str))
            desc_count += 1
print(f"  described_by edges (fwd+rev) : {desc_count*2:,}")

# 4. produced_by: Item → Brand
brand_count = 0
for item_str, brand_str in item_brand.items():
    if item_str not in item2id or brand_str not in brand2id:
        continue
    add(iid(item_str),  RELATIONS["produced_by"], bid(brand_str))
    add(bid(brand_str), RELATIONS["rev_produced"], iid(item_str))
    brand_count += 1
print(f"  produced_by edges (fwd+rev)  : {brand_count*2:,}")

# 5. belong_to: Item → Category
cat_count = 0
for item_str, cats in item_cats.items():
    if item_str not in item2id:
        continue
    for cat_str in cats:
        if cat_str in cat2id:
            add(iid(item_str), RELATIONS["belong_to"], cid(cat_str))
            add(cid(cat_str),  RELATIONS["rev_belong"], iid(item_str))
            cat_count += 1
print(f"  belong_to edges (fwd+rev)    : {cat_count*2:,}")

# 6. also_bought: Item → Item
ab_count = 0
for item_str, neighbours in item_also_bought.items():
    if item_str not in item2id:
        continue
    for nb in neighbours:
        if nb in item2id:
            add(iid(item_str), RELATIONS["also_bought"],     iid(nb))
            add(iid(nb),       RELATIONS["rev_also_bought"], iid(item_str))
            ab_count += 1
print(f"  also_bought edges (fwd+rev)  : {ab_count*2:,}")

# 7. also_viewed: Item → Item
av_count = 0
for item_str, neighbours in item_also_viewed.items():
    if item_str not in item2id:
        continue
    for nb in neighbours:
        if nb in item2id:
            add(iid(item_str), RELATIONS["also_viewed"],     iid(nb))
            add(iid(nb),       RELATIONS["rev_also_viewed"], iid(item_str))
            av_count += 1
print(f"  also_viewed edges (fwd+rev)  : {av_count*2:,}")

# 8. NOOP self-loop: every entity → itself  (paper Section 3.2)
noop_count = 0
for global_id in range(NUM_ENTITIES):
    add(global_id, RELATIONS["self_loop"], global_id)
    noop_count += 1
print(f"  NOOP self-loops              : {noop_count:,}")

print(f"\n  TOTAL triples (all types)    : {len(triples):,}")

# ─── Deduplicate ─────────────────────────────────────────────────────────────
triples_set = list(set(triples))
print(f"  After dedup                  : {len(triples_set):,}")

# ─── Step 4D: Save ───────────────────────────────────────────────────────────
# kg_triples.tsv — the core file your peer's algorithm reads
triples_df = pd.DataFrame(triples_set, columns=["head_id", "relation_id", "tail_id"])
triples_df = triples_df.sort_values(["head_id", "relation_id"]).reset_index(drop=True)
triples_df.to_csv(f"{OUT_DIR}/kg_triples.tsv", sep="\t", index=False)

with open(f"{OUT_DIR}/entity_index.json", "w") as f:
    json.dump(entity_index, f, indent=2)

with open(f"{OUT_DIR}/relation_index.json", "w") as f:
    json.dump(RELATIONS, f, indent=2)

with open(f"{OUT_DIR}/type_offsets.json", "w") as f:
    json.dump(type_offsets, f, indent=2)

# ─── Step 4E: Stats file for peer (mirrors Table 1 of paper) ─────────────────
# Compute average out-degree per relation type
from collections import Counter
rel_counts = Counter(triples_df["relation_id"].values)

head_by_rel = defaultdict(set)
for _, row in triples_df.iterrows():
    head_by_rel[row["relation_id"]].add(row["head_id"])

rel_names_inv = {v: k for k, v in RELATIONS.items()}

stats_lines = [
    "# Knowledge Graph Statistics",
    "",
    "## Entity Counts",
    f"| Entity Type | Count |",
    f"|-------------|-------|",
    f"| User        | {len(user2id):,} |",
    f"| Item        | {len(item2id):,} |",
    f"| Feature     | {len(feat2id):,} |",
    f"| Brand       | {len(brand2id):,} |",
    f"| Category    | {len(cat2id):,} |",
    f"| **TOTAL**   | **{NUM_ENTITIES:,}** |",
    "",
    "## Relation Counts (forward relations only)",
    f"| Relation      | Head Type | Tail Type | Triple Count |",
    f"|---------------|-----------|-----------|-------------|",
    f"| purchase      | User      | Item      | {purchase_count:,} |",
    f"| mention       | User      | Feature   | {mention_count:,} |",
    f"| described_by  | Item      | Feature   | {desc_count:,} |",
    f"| produced_by   | Item      | Brand     | {brand_count:,} |",
    f"| belong_to     | Item      | Category  | {cat_count:,} |",
    f"| also_bought   | Item      | Item      | {ab_count:,} |",
    f"| also_viewed   | Item      | Item      | {av_count:,} |",
    "",
    "## Total Triples in KG (incl. reverse + NOOP)",
    f"| Type           | Count |",
    f"|----------------|-------|",
    f"| Forward        | {purchase_count+mention_count+desc_count+brand_count+cat_count+ab_count+av_count:,} |",
    f"| Reverse        | {purchase_count+mention_count+desc_count+brand_count+cat_count+ab_count+av_count:,} |",
    f"| NOOP self-loop | {noop_count:,} |",
    f"| **TOTAL**      | **{len(triples_set):,}** |",
    "",
    "## Notes for path reasoning implementation",
    f"- NUM_ENTITIES  = {NUM_ENTITIES}",
    f"- NUM_RELATIONS = {NUM_RELATIONS}",
    f"- NOOP relation ID = 14",
    f"- Relation IDs: {json.dumps(RELATIONS, indent=2)}",
]

with open(f"{OUT_DIR}/kg_stats.md", "w") as f:
    f.write("\n".join(stats_lines))

print(f"\n✓ Step 4 complete.")
print(f"  kg_triples.tsv      → {len(triples_set):,} triples")
print(f"  entity_index.json   → {NUM_ENTITIES:,} entities")
print(f"  relation_index.json → {NUM_RELATIONS} relations")
print(f"  kg_stats.md         → summary for your peer")
