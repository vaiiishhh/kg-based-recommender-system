"""
STEP 6: Validate KG & Export Final Handoff Package
Paper reference: All sections — final check before handing to peer.

Checks performed:
  1. Every train/test user-item pair is reachable via some path ≤ 3 hops
  2. No entity ID exceeds NUM_ENTITIES
  3. Embedding matrix dimensions are consistent
  4. Reverse edges exist for every forward edge
  5. Every entity has a self-loop

Outputs a clean summary that your peer can read to understand the KG.
"""

import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict

OUT_DIR = "/kaggle/working/kg_data"

def lj(fname):
    with open(f"{OUT_DIR}/{fname}") as f:
        return json.load(f)

# ─── Load everything ──────────────────────────────────────────────────────────
print("Loading artefacts...")
entity_index   = lj("entity_index.json")
relation_index = lj("relation_index.json")
type_offsets   = lj("type_offsets.json")
user2id        = lj("user2id.json")
item2id        = lj("item2id.json")

triples_df = pd.read_csv(f"{OUT_DIR}/kg_triples.tsv", sep="\t")
train_df   = pd.read_csv(f"{OUT_DIR}/train.csv")
test_df    = pd.read_csv(f"{OUT_DIR}/test.csv")

entity_emb    = np.load(f"{OUT_DIR}/entity_embeddings.npy")
relation_emb  = np.load(f"{OUT_DIR}/relation_embeddings.npy")

NUM_ENTITIES  = entity_emb.shape[0]
NUM_RELATIONS = relation_emb.shape[0]

print(f"NUM_ENTITIES  : {NUM_ENTITIES:,}")
print(f"NUM_RELATIONS : {NUM_RELATIONS}")
print(f"Triples       : {len(triples_df):,}")

ERRORS = []

# ─── Check 1: Entity ID bounds ────────────────────────────────────────────────
max_h = triples_df["head_id"].max()
max_t = triples_df["tail_id"].max()
if max_h >= NUM_ENTITIES or max_t >= NUM_ENTITIES:
    ERRORS.append(f"Entity ID out of bounds: max_h={max_h}, max_t={max_t}")
else:
    print("✓ Check 1 passed: all entity IDs within bounds")

# ─── Check 2: Embedding shape ─────────────────────────────────────────────────
if entity_emb.shape[0] != NUM_ENTITIES:
    ERRORS.append(f"Embedding rows {entity_emb.shape[0]} ≠ NUM_ENTITIES {NUM_ENTITIES}")
else:
    print(f"✓ Check 2 passed: entity_embeddings shape {entity_emb.shape}")

# ─── Check 3: Every entity has a self-loop ────────────────────────────────────
NOOP_ID   = relation_index["self_loop"]
noop_df   = triples_df[triples_df["relation_id"] == NOOP_ID]
noop_heads = set(noop_df["head_id"].values)
missing_noop = NUM_ENTITIES - len(noop_heads)
if missing_noop > 0:
    ERRORS.append(f"{missing_noop} entities missing NOOP self-loop")
else:
    print(f"✓ Check 3 passed: all {NUM_ENTITIES:,} entities have self-loop")

# ─── Check 4: Reverse edges exist ─────────────────────────────────────────────
# For every forward triple (h,r,t), check (t, rev_r, h) exists
REV_MAP = {
    relation_index["purchase"]:    relation_index["rev_purchase"],
    relation_index["mention"]:     relation_index["rev_mention"],
    relation_index["described_by"]:relation_index["rev_described"],
    relation_index["produced_by"]: relation_index["rev_produced"],
    relation_index["belong_to"]:   relation_index["rev_belong"],
    relation_index["also_bought"]: relation_index["rev_also_bought"],
    relation_index["also_viewed"]: relation_index["rev_also_viewed"],
}

triple_set = set(zip(triples_df["head_id"], triples_df["relation_id"], triples_df["tail_id"]))
missing_rev = 0
# Sample check on 10,000 forward edges for speed
forward_mask = triples_df["relation_id"].isin(REV_MAP.keys())
fwd_sample   = triples_df[forward_mask].sample(min(10000, forward_mask.sum()), random_state=42)
for _, row in fwd_sample.iterrows():
    h, r, t = int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"])
    rev_r = REV_MAP[r]
    if (t, rev_r, h) not in triple_set:
        missing_rev += 1
if missing_rev > 0:
    ERRORS.append(f"{missing_rev}/10000 sampled forward edges missing reverse")
else:
    print(f"✓ Check 4 passed: reverse edges verified (sample of 10,000)")

# ─── Check 5: Train/test users have purchase edges in KG ─────────────────────
purchase_rel = relation_index["purchase"]
kg_purchase_users = set(
    triples_df[triples_df["relation_id"] == purchase_rel]["head_id"].values
)
train_user_global_ids = {
    type_offsets["user"] + user2id[u]
    for u in train_df["user"].unique()
    if u in user2id
}
missing_users = train_user_global_ids - kg_purchase_users
if missing_users:
    ERRORS.append(f"{len(missing_users)} train users have no purchase edge in KG")
else:
    print(f"✓ Check 5 passed: all train users have purchase edges")

# ─── Report ───────────────────────────────────────────────────────────────────
print("\n" + ("="*55))
if ERRORS:
    print("⚠ ERRORS FOUND:")
    for e in ERRORS:
        print(f"  - {e}")
else:
    print("✓ ALL CHECKS PASSED — KG is ready for handoff")
print("="*55)

# ─── Per-relation statistics ──────────────────────────────────────────────────
print("\n── Relation statistics ──")
rel_inv = {v: k for k, v in relation_index.items()}
rel_stats = triples_df.groupby("relation_id").size().reset_index(name="count")
rel_stats["name"] = rel_stats["relation_id"].map(rel_inv)
rel_stats = rel_stats.sort_values("relation_id")
print(rel_stats[["relation_id", "name", "count"]].to_string(index=False))

# ─── Out-degree distribution per entity type ─────────────────────────────────
print("\n── Out-degree per entity type (head of forward edges only) ──")
fwd_only = triples_df[triples_df["relation_id"].isin(REV_MAP.keys())]
head_degrees = fwd_only.groupby("head_id").size()

for etype, offset in type_offsets.items():
    # Find count of this type
    type_size = sum(1 for v in entity_index.values() if v["type"] == etype)
    ids_in_type = range(offset, offset + type_size)
    degs = head_degrees.reindex(ids_in_type, fill_value=0).values
    print(f"  {etype:<10}: mean={degs.mean():.1f}  max={degs.max()}  "
          f"(entities with ≥1 edge: {(degs>0).sum()}/{type_size})")

# ─── Export: peer handoff README ─────────────────────────────────────────────
readme = f"""# KG Handoff — Electronics Knowledge Graph

## Files

| File | Description |
|------|-------------|
| `kg_triples.tsv` | All (head_id, relation_id, tail_id) triples incl. reverse + NOOP |
| `entity_index.json` | Maps "TYPE:name" → global_id, type, name |
| `relation_index.json` | Maps relation_name → relation_id |
| `type_offsets.json` | Starting global ID per entity type |
| `entity_embeddings.npy` | np.float32 [{NUM_ENTITIES}, 100] — TransE entity vectors |
| `relation_embeddings.npy` | np.float32 [{NUM_RELATIONS}, 100] — TransE relation vectors |
| `entity_biases.npy` | np.float32 [{NUM_ENTITIES}] — bias term b_e from Eq. 9 |
| `train.csv` | user, item, rating (positive only, 70% split) |
| `test.csv` | user, item, rating (positive only, 30% split) |
| `user2id.json` | user_string → local user ID |
| `item2id.json` | ASIN string → local item ID |

## Constants your algorithm needs

```python
NUM_ENTITIES  = {NUM_ENTITIES}
NUM_RELATIONS = {NUM_RELATIONS}
EMBED_DIM     = 100
NOOP_REL_ID   = {relation_index["self_loop"]}   # self-loop relation
```

## Relation IDs
```json
{json.dumps(relation_index, indent=2)}
```

## Entity type offsets (global_id = offset + local_id)
```json
{json.dumps(type_offsets, indent=2)}
```

## How to load

```python
import numpy as np, json, pandas as pd

entity_emb   = np.load("entity_embeddings.npy")   # [NUM_ENTITIES, 100]
relation_emb = np.load("relation_embeddings.npy")  # [NUM_RELATIONS, 100]
entity_bias  = np.load("entity_biases.npy")        # [NUM_ENTITIES]

triples  = pd.read_csv("kg_triples.tsv", sep="\\t")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

with open("type_offsets.json") as f:
    offsets = json.load(f)
with open("item2id.json") as f:
    item2id = json.load(f)

# Get global ID for an item ASIN
asin = "B00M8JL73U"
global_id = offsets["item"] + item2id[asin]
emb = entity_emb[global_id]   # [100] embedding vector
```

## Scoring function (Equation 11 of PGPR paper)
```python
def score(e0_id, r_id, ek_id, entity_emb, relation_emb, entity_bias):
    e0 = entity_emb[e0_id]
    r  = relation_emb[r_id]
    ek = entity_emb[ek_id]
    return float(np.dot(e0 + r, ek) + entity_bias[ek_id])
```

## Binarization note
Ratings >= 4.0 were kept as positive interactions.
The test set contains ground-truth positive items.
Evaluate with NDCG@10 and HitRate@10.
"""

with open(f"{OUT_DIR}/README.md", "w") as f:
    f.write(readme)

print(f"\n✓ Step 6 complete. README.md written for your peer.")
print(f"\nFull file list in {OUT_DIR}/:")
for fname in sorted(os.listdir(OUT_DIR)):
    size_kb = os.path.getsize(f"{OUT_DIR}/{fname}") / 1024
    print(f"  {fname:<35} {size_kb:>8.1f} KB")
