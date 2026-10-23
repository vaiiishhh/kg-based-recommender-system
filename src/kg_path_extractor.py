"""
KG Reasoning Path Extractor
============================
Extracts top-K reasoning paths between a user and a target item
using a Beam Search + A* hybrid, guided by TransE embeddings.

Based on:
  Xian et al. "Reinforcement Knowledge Graph Reasoning for Explainable
  Recommendation." SIGIR 2019. (PGPR)
  + Target-Aware Beam Search roadmap (A* heuristic for goal-directed search)

Usage:
  python kg_path_extractor.py \
      --user  A0220159ZRNBTRKLG08H \
      --item  B001TH7GUA \
      --top_k 5

Or import:
  from kg_path_extractor import KGPathExtractor
  extractor = KGPathExtractor("kg_data/")
  results   = extractor.find_paths(user_str, item_asin, top_k=5)
"""

import os
import json
import argparse
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants / defaults  (all overridable via Config)
# ---------------------------------------------------------------------------

RELATION_NAMES = {
    0: "purchase",
    1: "rev_purchase",
    2: "mention",
    3: "rev_mention",
    4: "described_by",
    5: "rev_described",
    6: "produced_by",
    7: "rev_produced",
    8: "belong_to",
    9: "rev_belong",
    10: "also_bought",
    11: "rev_also_bought",
    12: "also_viewed",
    13: "rev_also_viewed",
    14: "self_loop",
}

ENTITY_TYPES = ["user", "item", "feature", "brand", "category"]


# ---------------------------------------------------------------------------
# Config dataclass — all tunable parameters in one place
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── Path search ──────────────────────────────────────────────────────────
    max_hops: int = 3
    """Maximum path length (hops).  Paper uses 3."""

    top_k_paths: int = 5
    """Number of top paths to return."""

    beam_width: int = 64
    """Beam width at every hop (active frontier size)."""

    max_actions_per_node: int = 250
    """Max neighbour actions kept after embedding-based pruning (PGPR α)."""

    # ── A* heuristic ─────────────────────────────────────────────────────────
    alpha: float = 0.5
    """Weight of the A* heuristic: f = g + alpha * h.
    0 → pure beam search (no goal awareness)
    1 → equal weight to past and future estimate"""

    # ── Scoring ──────────────────────────────────────────────────────────────
    relation_weights: Dict[int, float] = field(default_factory=lambda: {
        0:  1.00,   # purchase          (strongest signal)
        1:  1.00,   # rev_purchase
        2:  0.60,   # mention
        3:  0.60,   # rev_mention
        4:  0.75,   # described_by
        5:  0.75,   # rev_described
        6:  0.85,   # produced_by
        7:  0.85,   # rev_produced
        8:  0.70,   # belong_to
        9:  0.70,   # rev_belong
        10: 0.90,   # also_bought 
        11: 0.90,   # rev_also_bought
        12: 0.80,   # also_viewed
        13: 0.80,   # rev_also_viewed
        14: 0.00,   # self_loop — penalise staying
    })
    """Per-relation weights used in the path g-score."""
# a->c at b h(n)-> possible cost ->cosine(b,c)
    use_embedding_score: bool = True
    """Include TransE embedding score in node expansion scoring.
    If False, only relation_weights are used (faster but less accurate)."""

    embedding_score_weight: float = 0.3
    """How much the TransE dot-product score contributes alongside
    the relation weight when ranking neighbours.  0 = ignore embeddings."""

    # ── Filtering ─────────────────────────────────────────────────────────────
    exclude_relations: List[int] = field(default_factory=lambda: [14])
    """Relation IDs to skip entirely during expansion (default: self-loop)."""

    require_item_at_end: bool = True
    """Only return paths that end at an item entity."""

    diverse_patterns: bool = True
    """When True, deduplicate results so each unique relation-type pattern
    appears at most once (best-scoring representative kept).
    Set to False to allow the beam to return many paths of the same type."""

    max_same_pattern: int = 2
    """When diverse_patterns=True, allow at most this many paths per
    relation pattern before forcing the next unique one."""


# ---------------------------------------------------------------------------
# Path result object
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    nodes: List[int]          # global entity IDs  [u, e1, e2, …, eT]
    relations: List[int]      # relation IDs        [r1, r2, …, rT]
    node_names: List[str]     # human-readable node names
    relation_names: List[str] # human-readable relation names
    g_score: float            # average relation-weight along path
    h_score: float            # heuristic distance to target at end node
    f_score: float            # g + alpha*h  (ranking score)
    embed_score: float        # TransE score at final (user, item) pair

    def pretty(self) -> str:
        """One-line human-readable path with score."""
        arrow = " → ".join(
            f"[{n}] --{r}--> " if i < len(self.relations)
            else f"[{n}]"
            for i, (n, r) in enumerate(
                zip(self.node_names, self.relation_names + [""])
            )
        )
        return f"f={self.f_score:.4f}  g={self.g_score:.4f}  emb={self.embed_score:.4f}  |  {arrow}"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class KGPathExtractor:
    """
    Loads the KG artefacts from `data_dir` once, then answers path queries
    efficiently by searching the adjacency list with Beam + A*.
    """

    def __init__(self, data_dir: str = "kg_data/", config: Optional[Config] = None):
        self.cfg = config or Config()
        self._load(data_dir)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self, data_dir: str):
        print(f"[KGPathExtractor] Loading data from {data_dir!r} …")

        def p(fname): return os.path.join(data_dir, fname)

        # Embeddings
        self.entity_emb    = np.load(p("entity_embeddings.npy"))   # [E, D]
        self.relation_emb  = np.load(p("relation_embeddings.npy")) # [R, D]
        self.entity_bias   = np.load(p("entity_biases.npy"))       # [E]

        # ID mappings
        with open(p("type_offsets.json"))  as f: self.offsets  = json.load(f)
        with open(p("user2id.json"))       as f: self.user2id  = json.load(f)
        with open(p("item2id.json"))       as f: self.item2id  = json.load(f)
        with open(p("entity_index.json"))  as f: raw_idx       = json.load(f)

        # entity_index maps "TYPE:name" → {global_id, type, name}
        self.id2name: Dict[int, str] = {}
        for key, val in raw_idx.items():
            gid = int(val["global_id"])
            self.id2name[gid] = val.get("name", key)

        # Pre-computed type ranges for fast entity-type lookup
        sorted_types = sorted(self.offsets.items(), key=lambda x: x[1])
        self._type_ranges: List[Tuple[str, int, int]] = []
        for i, (etype, start) in enumerate(sorted_types):
            end = sorted_types[i + 1][1] if i + 1 < len(sorted_types) else len(self.entity_emb)
            self._type_ranges.append((etype, start, end))

        self.item_start = self.offsets["item"]
        self.item_end   = self.item_start + len(self.item2id)
        self.n_items    = len(self.item2id)

        # Build adjacency index
        print("[KGPathExtractor] Building adjacency index …")
        triples = pd.read_csv(p("kg_triples.tsv"), sep="\t",
                               dtype={"head_id": int, "relation_id": int, "tail_id": int})
        self.adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for row in triples.itertuples(index=False):
            self.adj[row.head_id].append((row.relation_id, row.tail_id))

        print(f"[KGPathExtractor] Ready.  "
              f"{len(self.entity_emb)} entities, "
              f"{len(self.relation_emb)} relations, "
              f"{sum(len(v) for v in self.adj.values()):,} directed edges.")

    # ── Public API ───────────────────────────────────────────────────────────

    def find_paths(
        self,
        user_str: str,
        item_asin: str,
        top_k: Optional[int] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[PathResult]:
        """
        Find the top-K reasoning paths from `user_str` to `item_asin`.

        Parameters
        ----------
        user_str         : raw user ID string (key in user2id.json)
        item_asin        : ASIN string (key in item2id.json)
        top_k            : number of paths to return (overrides Config.top_k_paths)
        config_overrides : dict of Config field overrides for this call only

        Returns
        -------
        List[PathResult] sorted by f_score descending.
        """
        cfg = self._apply_overrides(config_overrides)
        top_k = top_k or cfg.top_k_paths

        # Resolve IDs
        if user_str not in self.user2id:
            raise KeyError(f"Unknown user: {user_str!r}")
        if item_asin not in self.item2id:
            raise KeyError(f"Unknown item ASIN: {item_asin!r}")

        user_gid   = self.offsets["user"] + self.user2id[user_str]
        target_gid = self.offsets["item"] + self.item2id[item_asin]

        # Pre-compute normalisation denominator for reward (Eq. 2)
        max_user_item_score = self._max_item_score(user_gid)

        # Run hybrid Beam + A* search
        paths = self._beam_astar(user_gid, target_gid, cfg)

        # Score, rank, trim
        results = []
        for nodes, rels, g in paths:
            terminal = nodes[-1]
            h = self._heuristic(terminal, target_gid, cfg)
            f = g + cfg.alpha * h
            emb_s = self._embed_score(user_gid, terminal, max_user_item_score)
            node_names = [self._name(n) for n in nodes]
            rel_names  = [RELATION_NAMES.get(r, str(r)) for r in rels]
            results.append(PathResult(
                nodes=nodes, relations=rels,
                node_names=node_names, relation_names=rel_names,
                g_score=g, h_score=h, f_score=f, embed_score=emb_s,
            ))

        results.sort(key=lambda r: r.f_score, reverse=True)

        if cfg.diverse_patterns:
            results = self._diversity_filter(results, top_k, cfg.max_same_pattern)

        return results[:top_k]

    def _diversity_filter(
        self,
        results: List[PathResult],
        top_k: int,
        max_per_pattern: int,
    ) -> List[PathResult]:
        """
        Greedily keep results so no relation-pattern appears more than
        max_per_pattern times, until we have top_k or exhaust the list.
        """
        pattern_count: Dict[tuple, int] = {}
        kept = []
        for r in results:
            pat = tuple(r.relations)
            cnt = pattern_count.get(pat, 0)
            if cnt < max_per_pattern:
                kept.append(r)
                pattern_count[pat] = cnt + 1
            if len(kept) >= top_k:
                break
        return kept

    # ── Beam + A* core ────────────────────────────────────────────────────────

    def _beam_astar(
        self,
        user_gid: int,
        target_gid: int,
        cfg: Config,
    ) -> List[Tuple[List[int], List[int], float]]:
        """
        Pattern-diverse Beam Search + A* hybrid.

        Key idea: alongside the top-B paths by f-score, we also keep the
        best path for each *relation pattern* seen so far.  This ensures
        the beam never starves out minority-pattern paths even when the
        dominant pattern (purchase→rev_purchase→purchase) scores much
        higher overall.

        Each beam entry: dict with keys
          nodes   : List[int]
          rels    : List[int]
          g       : float
          f       : float
          pattern : tuple  (relation IDs so far)
        """
        # Initialise
        initial_h = self._heuristic(user_gid, target_gid, cfg)
        initial_f = cfg.alpha * initial_h

        beam = [{
            "nodes": [user_gid],
            "rels":  [],
            "g":     0.0,
            "f":     initial_f,
            "pattern": (),
        }]

        completed: List[Tuple[List[int], List[int], float]] = []

        for hop in range(cfg.max_hops):
            candidates = []

            for entry in beam:
                current = entry["nodes"][-1]
                visited = set(entry["nodes"])

                neighbours = self._get_actions(current, user_gid, visited, cfg)

                for rel_id, next_eid in neighbours:
                    step_weight = cfg.relation_weights.get(rel_id, 0.5)
                    new_rels    = entry["rels"] + [rel_id]
                    n           = len(new_rels)
                    new_g       = (entry["g"] * (n - 1) + step_weight) / n
                    h           = self._heuristic(next_eid, target_gid, cfg)
                    new_f       = new_g + cfg.alpha * h
                    new_pattern = entry["pattern"] + (rel_id,)

                    candidates.append({
                        "nodes":   entry["nodes"] + [next_eid],
                        "rels":    new_rels,
                        "g":       new_g,
                        "f":       new_f,
                        "pattern": new_pattern,
                    })

            if not candidates:
                break

            # ── Pattern-aware beam selection ────────────────────────────────
            # 1. Among all candidates, keep the top-1 per unique pattern
            #    (guarantees diverse patterns are represented)
            # 2. Fill remaining slots up to beam_width with highest-f entries
            candidates.sort(key=lambda x: -x["f"])

            pattern_best: Dict[tuple, dict] = {}
            for c in candidates:
                p = c["pattern"]
                if p not in pattern_best:
                    pattern_best[p] = c

            diverse_entries = list(pattern_best.values())
            diverse_entries.sort(key=lambda x: -x["f"])

            # Fill to beam_width with top-f (may overlap with diverse set)
            top_by_f = candidates[: cfg.beam_width]

            # Merge: diverse first, then fill with top_by_f
            seen_nodes = {tuple(e["nodes"]) for e in diverse_entries}
            merged = list(diverse_entries)
            for c in top_by_f:
                key = tuple(c["nodes"])
                if key not in seen_nodes:
                    merged.append(c)
                    seen_nodes.add(key)
                if len(merged) >= cfg.beam_width:
                    break

            beam = merged

            # ── Collect hits at any hop ─────────────────────────────────────
            for entry in beam:
                terminal    = entry["nodes"][-1]
                is_item     = self.item_start <= terminal < self.item_end
                hits_target = terminal == target_gid
                if len(entry["nodes"]) < 2:
                    continue

                keep = hits_target or (
                    is_item and (
                        not cfg.require_item_at_end
                        or hop == cfg.max_hops - 1
                    )
                )
                if keep:
                    completed.append((entry["nodes"], entry["rels"], entry["g"]))

        # De-duplicate by node sequence
        seen: set = set()
        deduped = []
        for nodes, rels, g in completed:
            key = tuple(nodes)
            if key not in seen:
                seen.add(key)
                deduped.append((nodes, rels, g))

        return deduped

    # ── Action expansion ──────────────────────────────────────────────────────

    def _get_actions(
        self,
        entity_id: int,
        user_id: int,
        visited: set,
        cfg: Config,
    ) -> List[Tuple[int, int]]:
        """
        Return a pruned list of (relation_id, next_entity_id) actions
        from entity_id, filtered by visited set and cfg.exclude_relations,
        ranked by a combined embedding + relation-weight score, capped at
        cfg.max_actions_per_node.
        """
        raw = [
            (r, e) for (r, e) in self.adj.get(entity_id, [])
            if e not in visited and r not in cfg.exclude_relations
        ]
        if not raw:
            return []

        if len(raw) <= cfg.max_actions_per_node and not cfg.use_embedding_score:
            return raw

        # Score each candidate
        rel_ids  = np.array([r for r, e in raw])
        ent_ids  = np.array([e for r, e in raw])

        # Relation weight component
        rel_weights = np.array([cfg.relation_weights.get(r, 0.5) for r in rel_ids])

        if cfg.use_embedding_score and cfg.embedding_score_weight > 0:
            # TransE: <user_emb + rel_emb, neighbour_emb> + bias
            u_emb  = self.entity_emb[user_id]          # [D]
            r_emb  = self.relation_emb[rel_ids]         # [N, D]
            e_emb  = self.entity_emb[ent_ids]           # [N, D]
            b      = self.entity_bias[ent_ids]          # [N]
            query  = u_emb + r_emb                      # broadcast: [N, D]
            emb_sc = (query * e_emb).sum(axis=1) + b   # [N]

            # Normalise embedding scores to [0, 1]
            sc_min, sc_max = emb_sc.min(), emb_sc.max()
            if sc_max > sc_min:
                emb_sc = (emb_sc - sc_min) / (sc_max - sc_min)

            scores = (1 - cfg.embedding_score_weight) * rel_weights \
                   + cfg.embedding_score_weight * emb_sc
        else:
            scores = rel_weights

        top_idx = np.argsort(scores)[::-1][: cfg.max_actions_per_node]
        return [raw[i] for i in top_idx]

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _heuristic(self, node_id: int, target_id: int, cfg: Config) -> float:
        """
        A* heuristic h(node, target):
        Uses cosine similarity between entity embeddings as a proxy for
        'how close' node is to target.  Returns value in [0, 1].

        Higher = closer to target = better.
        """
        n_emb = self.entity_emb[node_id]
        t_emb = self.entity_emb[target_id]
        norm_n = np.linalg.norm(n_emb) + 1e-9
        norm_t = np.linalg.norm(t_emb) + 1e-9
        cosine = float(np.dot(n_emb, t_emb) / (norm_n * norm_t))
        return (cosine + 1.0) / 2.0   # shift to [0, 1]

    def _embed_score(self, user_id: int, item_id: int, max_score: float) -> float:
        """
        Soft reward per Eq. 2:  f(u, i) = <u + r_purchase, i> + bias_i,
        normalised by the max item score for this user.
        """
        if not (self.item_start <= item_id < self.item_end):
            return 0.0
        u  = self.entity_emb[user_id]
        r  = self.relation_emb[0]          # relation 0 = purchase
        it = self.entity_emb[item_id]
        b  = float(self.entity_bias[item_id])
        raw = float(np.dot(u + r, it) + b)
        return max(0.0, raw / (max_score + 1e-9))

    def _max_item_score(self, user_id: int) -> float:
        """Pre-compute denominator for reward normalisation (once per user)."""
        u = self.entity_emb[user_id]            # [D]
        r = self.relation_emb[0]               # purchase relation
        all_items = self.entity_emb[self.item_start: self.item_end]   # [I, D]
        all_bias  = self.entity_bias[self.item_start: self.item_end]  # [I]
        scores    = all_items @ (u + r) + all_bias                    # [I]
        return float(scores.max())

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _entity_type(self, gid: int) -> str:
        for etype, start, end in self._type_ranges:
            if start <= gid < end:
                return etype
        return "unknown"

    def _name(self, gid: int) -> str:
        raw = self.id2name.get(gid, str(gid))
        etype = self._entity_type(gid)
        return f"{raw} ({etype})"

    def _apply_overrides(self, overrides: Optional[Dict[str, Any]]) -> Config:
        if not overrides:
            return self.cfg
        import copy
        cfg = copy.copy(self.cfg)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown Config field: {k!r}")
        return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(results: List[PathResult], user_str: str, item_asin: str):
    print(f"\n{'='*70}")
    print(f"  Top-{len(results)} Reasoning Paths")
    print(f"  User : {user_str}")
    print(f"  Item : {item_asin}")
    print(f"{'='*70}\n")
    if not results:
        print("  ⚠  No paths found.  Try increasing --beam_width or --max_hops.")
        return
    for rank, r in enumerate(results, 1):
        print(f"  Rank #{rank}  f={r.f_score:.4f}  g={r.g_score:.4f}  "
              f"h={r.h_score:.4f}  emb={r.embed_score:.4f}")
        for i, (n, rel) in enumerate(zip(r.node_names, r.relation_names)):
            prefix = "  " + "    " * i
            print(f"{prefix}[{n}]")
            if i < len(r.relation_names):
                arrow_rel = r.relation_names[i]
                print(f"{prefix}  --{arrow_rel}-->")
        print(f"  " + "    " * len(r.node_names) + f"[{r.node_names[-1]}]")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-K KG reasoning paths (Beam + A*) for a user-item pair."
    )
    parser.add_argument("--data_dir",    default="kg_data/", help="Path to kg_data/ folder")
    parser.add_argument("--user",        required=True,      help="User ID string")
    parser.add_argument("--item",        required=True,      help="Item ASIN string")
    parser.add_argument("--top_k",       type=int, default=5, help="Number of paths to return")

    # Config overrides
    parser.add_argument("--max_hops",   type=int,   default=3,    help="Max path length in hops")
    parser.add_argument("--beam_width", type=int,   default=64,   help="Beam width")
    parser.add_argument("--alpha",      type=float, default=0.5,  help="A* heuristic weight")
    parser.add_argument("--max_actions",type=int,   default=250,  help="Max actions per node")
    parser.add_argument("--no_emb",     action="store_true",      help="Disable embedding scoring")
    parser.add_argument("--emb_weight", type=float, default=0.3,  help="Embedding score weight")

    args = parser.parse_args()

    cfg = Config(
        max_hops              = args.max_hops,
        beam_width            = args.beam_width,
        alpha                 = args.alpha,
        max_actions_per_node  = args.max_actions,
        use_embedding_score   = not args.no_emb,
        embedding_score_weight= args.emb_weight,
        top_k_paths           = args.top_k,
    )

    extractor = KGPathExtractor(data_dir=args.data_dir, config=cfg)
    results   = extractor.find_paths(args.user, args.item, top_k=args.top_k)
    _print_results(results, args.user, args.item)


if __name__ == "__main__":
    main()
