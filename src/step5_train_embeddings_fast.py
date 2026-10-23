"""
STEP 5 (GPU-NATIVE): Pre-train KG Embeddings
═══════════════════════════════════════════════════════════════
Why the previous version was still slow:
  ✗ DataParallel  — scatter/gather overhead each batch kills small models
  ✗ DataLoader    — Python-side IPC overhead even with num_workers=0
  ✗ CPU→GPU xfer  — every batch transferred across PCIe bus

This version:
  ✓ ALL tensors pre-loaded to GPU once at startup
  ✓ Manual torch.randperm batching — zero Python/IPC overhead
  ✓ Single GPU only (DataParallel hurts throughput here)
  ✓ BATCH_SIZE 32768 — 4× fewer iterations
  ✓ Fixed deprecated AMP API (torch.amp.autocast / GradScaler)

Memory estimate: ~432 MB GPU RAM (fits T4 16 GB easily)
Expected epoch time: ~20–60 s  (was 9 min)
═══════════════════════════════════════════════════════════════
"""

import json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

OUT_DIR = "/kaggle/working/kg_data"

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM   = 100
NEG_SAMPLES = 5
BATCH_SIZE  = 32_768    # 4× bigger → 4× fewer iterations
EPOCHS      = 20
LR          = 0.001
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Device: ALWAYS single GPU — DataParallel is slower for this model ─────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"Device : {DEVICE}  |  AMP : {USE_AMP}")
print("NOTE: Using single GPU — DataParallel adds overhead for this model size.")

# ── Load artefacts ────────────────────────────────────────────────────────────
def lj(f):
    with open(f"{OUT_DIR}/{f}") as fh:
        return json.load(fh)

relation_index = lj("relation_index.json")
entity_index   = lj("entity_index.json")
type_offsets   = lj("type_offsets.json")

NOOP_ID       = relation_index["self_loop"]
NUM_ENTITIES  = max(v["global_id"] for v in entity_index.values()) + 1
NUM_RELATIONS = len(relation_index)
print(f"Entities : {NUM_ENTITIES:,}  |  Relations : {NUM_RELATIONS}")

triples_df = pd.read_csv(f"{OUT_DIR}/kg_triples.tsv", sep="\t")
mask    = triples_df["relation_id"] != NOOP_ID
triples = triples_df[mask][["head_id", "relation_id", "tail_id"]].values.astype(np.int32)
N = len(triples)
print(f"Training triples : {N:,}")

# ── Pre-sample negatives once (CPU) ──────────────────────────────────────────
print("Pre-sampling negatives...", end=" ", flush=True)
t0 = time.time()
neg_tails = np.random.randint(0, NUM_ENTITIES, size=(N, NEG_SAMPLES), dtype=np.int32)
print(f"done in {time.time()-t0:.1f}s")

# ── Move ALL data to GPU once ─────────────────────────────────────────────────
# This is the key change: zero CPU->GPU transfer during training
print("Loading all tensors to GPU...", end=" ", flush=True)
t0 = time.time()
H   = torch.from_numpy(triples[:, 0].astype(np.int64)).to(DEVICE)
R   = torch.from_numpy(triples[:, 1].astype(np.int64)).to(DEVICE)
T   = torch.from_numpy(triples[:, 2].astype(np.int64)).to(DEVICE)
NEG = torch.from_numpy(neg_tails.astype(np.int64)).to(DEVICE)
del triples, neg_tails   # free CPU RAM
mem_mb = torch.cuda.memory_allocated(DEVICE) / 1024**2
print(f"done in {time.time()-t0:.1f}s  |  GPU RAM used: {mem_mb:.0f} MB")

steps_per_epoch = (N + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Steps per epoch  : {steps_per_epoch:,}  (batch={BATCH_SIZE:,})")

# ── Model ─────────────────────────────────────────────────────────────────────
class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.ent_emb  = nn.Embedding(n_ent, dim)
        self.rel_emb  = nn.Embedding(n_rel, dim)
        self.ent_bias = nn.Embedding(n_ent, 1)

        bound = 6.0 / dim**0.5
        nn.init.uniform_(self.ent_emb.weight, -bound, bound)
        nn.init.uniform_(self.rel_emb.weight, -bound, bound)
        nn.init.zeros_(self.ent_bias.weight)

    def forward(self, h, r, t_pos, t_neg):
        """
        h, r, t_pos : [B]
        t_neg       : [B, K]
        """
        B, K = t_neg.shape

        e_h = self.ent_emb(h)                          # [B, D]
        e_r = self.rel_emb(r)                          # [B, D]
        e_tp = self.ent_emb(t_pos)                     # [B, D]
        b_tp = self.ent_bias(t_pos).squeeze(-1)        # [B]

        # Positive: <e_h + e_r, e_t> + b_t  (Eq. 11)
        pos_score = ((e_h + e_r) * e_tp).sum(-1) + b_tp
        pos_loss  = -torch.log(torch.sigmoid(pos_score) + 1e-9)

        # Negative: expand h,r across K samples
        e_hn  = e_h.unsqueeze(1).expand(B, K, -1)     # [B, K, D]
        e_rn  = e_r.unsqueeze(1).expand(B, K, -1)     # [B, K, D]
        e_tn  = self.ent_emb(t_neg)                    # [B, K, D]
        b_tn  = self.ent_bias(t_neg).squeeze(-1)       # [B, K]

        neg_score = ((e_hn + e_rn) * e_tn).sum(-1) + b_tn
        neg_loss  = -torch.log(torch.sigmoid(-neg_score) + 1e-9).mean(-1)

        return (pos_loss + neg_loss).mean()

    @torch.no_grad()
    def normalize(self):
        norms = self.ent_emb.weight.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.ent_emb.weight.div_(norms)

model = TransE(NUM_ENTITIES, NUM_RELATIONS, EMBED_DIM).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
scaler    = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ── Training loop — pure GPU, no DataLoader ───────────────────────────────────
print(f"\nTraining {EPOCHS} epochs | batch={BATCH_SIZE:,} | neg={NEG_SAMPLES} | AMP={USE_AMP}")
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    ep_t    = time.time()

    # Shuffle indices on GPU — no CPU involvement
    perm = torch.randperm(N, device=DEVICE)

    for start in range(0, N, BATCH_SIZE):
        idx   = perm[start: start + BATCH_SIZE]
        h     = H[idx]
        r     = R[idx]
        t_pos = T[idx]
        t_neg = NEG[idx]                     # already on GPU

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = model(h, r, t_pos, t_neg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ep_loss += loss.item()

    # Normalize once per epoch (not per batch!)
    model.normalize()
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - ep_t
        total   = time.time() - t_start
        avg     = ep_loss / steps_per_epoch
        print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={avg:.4f}"
              f"  epoch_time={elapsed:.1f}s  total={total/60:.1f}min")

print(f"\nTotal training time: {(time.time()-t_start)/60:.1f} min")

# ── Save embeddings ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    ent_emb  = model.ent_emb.weight.cpu().float().numpy()
    rel_emb  = model.rel_emb.weight.cpu().float().numpy()
    ent_bias = model.ent_bias.weight.cpu().float().numpy().squeeze(-1)

np.save(f"{OUT_DIR}/entity_embeddings.npy",  ent_emb)
np.save(f"{OUT_DIR}/relation_embeddings.npy", rel_emb)
np.save(f"{OUT_DIR}/entity_biases.npy",      ent_bias)

print(f"Saved: entity_emb {ent_emb.shape}, relation_emb {rel_emb.shape}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\n── Nearest-neighbour sanity check (items) ──")
item_offset = type_offsets["item"]
item2id     = lj("item2id.json")
id2item     = {v: k for k, v in item2id.items()}
n_items     = len(item2id)
item_embs   = ent_emb[item_offset: item_offset + n_items]

for local_id in list(item2id.values())[:3]:
    q    = item_embs[local_id]
    sims = item_embs @ q
    sims[local_id] = -1e9
    top5 = np.argsort(sims)[::-1][:5]
    print(f"  {id2item[local_id]} -> {[id2item[i] for i in top5]}")
