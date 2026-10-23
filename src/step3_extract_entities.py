"""
STEP 3: Extract KG Entities from Metadata
Paper reference: Section 4.1, Table 1

Entity types we build (mapping to paper's Amazon setup):
  User     → from reviews (already done in step 2)
  Item     → products in our filtered interaction set
  Feature  → keyword/feature words from product 'feature' field
               filtered with TF-IDF (paper: keep freq < 5000, TF-IDF > 0.1)
  Brand    → product brand field
  Category → last (most specific) level of category list

Relations we extract here (structural, not user-item):
  Item --described_by--> Feature    (paper: described_by)
  Item --produced_by-->  Brand      (paper: produced_by)
  Item --belong_to-->    Category   (paper: belong_to)
  Item --also_bought-->  Item       (paper: also_bought)  ← from also_buy field
  Item --also_viewed-->  Item       (paper: also_viewed)  ← from also_view field
  User --purchase-->     Item       (paper: purchase)     ← from train.csv
  User --mention-->      Feature    (paper: mention)      ← derived from items user bought
"""

import json
import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Paths ────────────────────────────────────────────────────────────────────
META_PATH = "/kaggle/input/datasets/vanshcantdoml/amazon-data/meta_subset.json"
OUT_DIR   = "/kaggle/working/kg_data"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Load already-computed artefacts from Step 2 ──────────────────────────────
with open(f"{OUT_DIR}/item2id.json") as f:
    item2id = json.load(f)
with open(f"{OUT_DIR}/user2id.json") as f:
    user2id = json.load(f)

train_df = pd.read_csv(f"{OUT_DIR}/train.csv")
valid_items = set(item2id.keys())   # only items in our filtered set

print(f"Valid items (from step 2): {len(valid_items):,}")

# ─── Load meta ────────────────────────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    continue
    return records

print("Loading meta...")
meta_list = load_jsonl(META_PATH)

# Filter to only valid items
meta = {r["asin"]: r for r in meta_list if r.get("asin") in valid_items}
print(f"Meta records for valid items: {len(meta):,}")

# ─── Helper: clean text ───────────────────────────────────────────────────────
def clean_text(s):
    s = re.sub(r"<[^>]+>", " ", str(s))        # strip HTML
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)      # keep alphanum + space
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def extract_words(text, min_len=3, max_len=25):
    """Split cleaned text into valid words."""
    return [w for w in clean_text(text).split()
            if min_len <= len(w) <= max_len]

# ─── 3A: Brand extraction ─────────────────────────────────────────────────────
brand2id   = {}
item_brand = {}   # asin → brand_str

for asin, r in meta.items():
    b = r.get("brand", "")
    if isinstance(b, str):
        b = clean_text(b)
    else:
        b = ""
    if b and b not in ("", "unknown", "na", "n/a"):
        if b not in brand2id:
            brand2id[b] = len(brand2id)
        item_brand[asin] = b

print(f"\nBrands found : {len(brand2id):,}")

# ─── 3B: Category extraction ─────────────────────────────────────────────────
# Use the LAST (most specific) category in the list as the entity
# e.g. ["Electronics", "Headphones", "On-Ear Headphones"] → "On-Ear Headphones"
cat2id      = {}
item_cats   = defaultdict(list)   # asin → [cat_str, ...]

for asin, r in meta.items():
    cats = r.get("category", [])
    if not isinstance(cats, list):
        cats = []
    # Take last 2 levels for richer coverage
    for c in cats[-2:]:
        c = clean_text(c)
        if c and len(c) > 2:
            if c not in cat2id:
                cat2id[c] = len(cat2id)
            item_cats[asin].append(c)

print(f"Categories found : {len(cat2id):,}")

# ─── 3C: Feature words extraction with TF-IDF filtering ──────────────────────
# Paper: "keep frequency of feature words < 5000 with TF-IDF score > 0.1"
# We collect all text from the 'feature' field (bullet points) of each item.

print("\nExtracting feature words...")
item_raw_text = {}   # asin → joined feature text

for asin, r in meta.items():
    parts = []
    for field in ("feature", "description", "title"):
        val = r.get(field, [])
        if isinstance(val, list):
            parts.extend(val)
        elif isinstance(val, str) and val:
            parts.append(val)
    item_raw_text[asin] = " ".join(parts)

# Build corpus for TF-IDF
asins_with_text = [a for a in valid_items if a in item_raw_text and item_raw_text[a].strip()]
corpus          = [item_raw_text[a] for a in asins_with_text]

print(f"  Items with text for TF-IDF: {len(corpus):,}")

# TF-IDF — word-level, ignore very common and very rare words
# min_df: word must appear in >= 2 docs to count
# max_df: paper threshold maps to ignoring top-frequency words
tfidf = TfidfVectorizer(
    analyzer        = "word",
    token_pattern   = r"(?u)\b[a-zA-Z][a-zA-Z0-9]{2,24}\b",
    min_df          = 2,
    max_df          = 0.95,           # drop words in >95% of docs (too common)
    max_features    = 50_000,
    sublinear_tf    = True,
    stop_words      = "english",
)
tfidf_matrix = tfidf.fit_transform(corpus)
vocab        = tfidf.get_feature_names_out()
print(f"  TF-IDF vocabulary size      : {len(vocab):,}")

# Build per-item feature word list:
# For each item, keep words whose TF-IDF score > 0.1 (paper threshold)
TFIDF_THRESHOLD = 0.1
item_features   = defaultdict(list)   # asin → [word, ...]
word_freq       = Counter()           # global doc frequency for pruning

for idx, asin in enumerate(asins_with_text):
    row    = tfidf_matrix[idx]
    cx     = row.tocoo()
    for col, score in zip(cx.col, cx.data):
        if score >= TFIDF_THRESHOLD:
            word = vocab[col]
            item_features[asin].append(word)
            word_freq[word] += 1

# Paper: drop words appearing in more than 5000 items  
MAX_WORD_DOC_FREQ = 5000
item_features = {
    asin: [w for w in words if word_freq[w] <= MAX_WORD_DOC_FREQ]
    for asin, words in item_features.items()
}

# Build feature vocab from remaining words
all_feature_words = sorted({w for words in item_features.values() for w in words})
feat2id = {w: i for i, w in enumerate(all_feature_words)}

print(f"  Feature words after TF-IDF filter : {len(feat2id):,}")
avg_feats = np.mean([len(v) for v in item_features.values()])
print(f"  Avg feature words per item        : {avg_feats:.1f}")

# ─── 3D: Also-bought / Also-viewed (Item-Item edges) ─────────────────────────
item_also_bought  = defaultdict(list)   # asin → [asin, ...]
item_also_viewed  = defaultdict(list)

for asin, r in meta.items():
    for nb in r.get("also_buy", []):
        if nb in valid_items and nb != asin:
            item_also_bought[asin].append(nb)
    for nb in r.get("also_view", []):
        if nb in valid_items and nb != asin:
            item_also_viewed[asin].append(nb)

# Cap at top-50 neighbours per item (paper avg ~57 for also_bought)
MAX_NEIGHBOURS = 50
item_also_bought = {a: v[:MAX_NEIGHBOURS] for a, v in item_also_bought.items()}
item_also_viewed = {a: v[:MAX_NEIGHBOURS] for a, v in item_also_viewed.items()}

print(f"\nAlso-bought edges (before cap) in valid items: "
      f"{sum(len(v) for v in item_also_bought.values()):,}")
print(f"Also-viewed edges (before cap) in valid items: "
      f"{sum(len(v) for v in item_also_viewed.values()):,}")

# ─── 3E: User-Feature 'mention' edges ────────────────────────────────────────
# Paper: User --mention--> Feature
# Derived: for each user, collect features of items they bought in TRAIN split
user_item_train = defaultdict(set)
for _, row in train_df.iterrows():
    user_item_train[row["user"]].add(row["item"])

user_mention = defaultdict(set)   # user → {feature_word, ...}
for user, items in user_item_train.items():
    for item in items:
        for word in item_features.get(item, []):
            user_mention[user].add(word)

print(f"\nUsers with mention edges: {len(user_mention):,}")
avg_mentions = np.mean([len(v) for v in user_mention.values()])
print(f"Avg features mentioned per user: {avg_mentions:.1f}")

# ─── Save all entity maps ─────────────────────────────────────────────────────
with open(f"{OUT_DIR}/brand2id.json",        "w") as f: json.dump(brand2id, f)
with open(f"{OUT_DIR}/cat2id.json",          "w") as f: json.dump(cat2id, f)
with open(f"{OUT_DIR}/feat2id.json",         "w") as f: json.dump(feat2id, f)

# Save relation dicts as JSON-serialisable (convert defaultdict → dict)
with open(f"{OUT_DIR}/item_brand.json",       "w") as f: json.dump(item_brand, f)
with open(f"{OUT_DIR}/item_cats.json",        "w") as f: json.dump(dict(item_cats), f)
with open(f"{OUT_DIR}/item_features.json",    "w") as f: json.dump(dict(item_features), f)
with open(f"{OUT_DIR}/item_also_bought.json", "w") as f: json.dump(dict(item_also_bought), f)
with open(f"{OUT_DIR}/item_also_viewed.json", "w") as f: json.dump(dict(item_also_viewed), f)
with open(f"{OUT_DIR}/user_mention.json",     "w") as f:
    json.dump({u: list(v) for u, v in user_mention.items()}, f)

# ─── Summary table (mirrors Table 1 of paper) ────────────────────────────────
print("\n" + "="*55)
print("ENTITY SUMMARY (mirrors Table 1 of paper)")
print("="*55)
print(f"{'Entity':<15} {'Count':>10}")
print("-"*30)
print(f"{'User':<15} {len(user2id):>10,}")
print(f"{'Item':<15} {len(item2id):>10,}")
print(f"{'Feature':<15} {len(feat2id):>10,}")
print(f"{'Brand':<15} {len(brand2id):>10,}")
print(f"{'Category':<15} {len(cat2id):>10,}")
print("="*55)

print("\n✓ Step 3 complete. All entity maps saved.")
