# KG Handoff — Electronics Knowledge Graph

## Files

| File | Description |
|------|-------------|
| `kg_triples.tsv` | All (head_id, relation_id, tail_id) triples incl. reverse + NOOP |
| `entity_index.json` | Maps "TYPE:name" → global_id, type, name |
| `relation_index.json` | Maps relation_name → relation_id |
| `type_offsets.json` | Starting global ID per entity type |
| `entity_embeddings.npy` | np.float32 [53956, 100] — TransE entity vectors |
| `relation_embeddings.npy` | np.float32 [15, 100] — TransE relation vectors |
| `entity_biases.npy` | np.float32 [53956] — bias term b_e from Eq. 9 |
| `train.csv` | user, item, rating (positive only, 70% split) |
| `test.csv` | user, item, rating (positive only, 30% split) |
| `user2id.json` | user_string → local user ID |
| `item2id.json` | ASIN string → local item ID |

## Constants your algorithm needs

```python
NUM_ENTITIES  = 53956
NUM_RELATIONS = 15
EMBED_DIM     = 100
NOOP_REL_ID   = 14   # self-loop relation
```

## Relation IDs
```json
{
  "purchase": 0,
  "rev_purchase": 1,
  "mention": 2,
  "rev_mention": 3,
  "described_by": 4,
  "rev_described": 5,
  "produced_by": 6,
  "rev_produced": 7,
  "belong_to": 8,
  "rev_belong": 9,
  "also_bought": 10,
  "rev_also_bought": 11,
  "also_viewed": 12,
  "rev_also_viewed": 13,
  "self_loop": 14
}
```

## Entity type offsets (global_id = offset + local_id)
```json
{
  "user": 0,
  "item": 19111,
  "feature": 30341,
  "brand": 50830,
  "category": 53249
}
```

## How to load

```python
import numpy as np, json, pandas as pd

entity_emb   = np.load("entity_embeddings.npy")   # [NUM_ENTITIES, 100]
relation_emb = np.load("relation_embeddings.npy")  # [NUM_RELATIONS, 100]
entity_bias  = np.load("entity_biases.npy")        # [NUM_ENTITIES]

triples  = pd.read_csv("kg_triples.tsv", sep="\t")
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
