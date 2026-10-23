"""
kg_run_demo.py
==============
Quick demo: picks a few user-item pairs from test.csv and extracts paths.
Also shows how to use the Python API directly.

Usage:
    python kg_run_demo.py --data_dir kg_data/ --n_examples 3 --top_k 5
"""

import argparse
import json
import pandas as pd
from kg_path_extractor import KGPathExtractor, Config


def demo(data_dir: str, n_examples: int, top_k: int, cfg: Config):
    # Load test set to pick real user-item pairs
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    # Filter to positive interactions
    pos = test_df[test_df["label"] == 1].head(n_examples)

    extractor = KGPathExtractor(data_dir=data_dir, config=cfg)

    for _, row in pos.iterrows():
        user_str  = row["user"]
        item_asin = row["item"]

        print(f"\n{'#'*70}")
        print(f"# User: {user_str}")
        print(f"# Item: {item_asin}  (rating={row['rating']})")
        print(f"{'#'*70}")

        try:
            results = extractor.find_paths(user_str, item_asin, top_k=top_k)
        except KeyError as e:
            print(f"  Skipped: {e}")
            continue

        if not results:
            print("  No paths found.")
            continue

        for rank, r in enumerate(results, 1):
            print(f"\n  ── Path #{rank}  "
                  f"f={r.f_score:.4f}  g={r.g_score:.4f}  "
                  f"h={r.h_score:.4f}  emb={r.embed_score:.4f}")
            # Build one-line chain  A --rel--> B --rel--> C
            chain = r.node_names[0]
            for n, rel in zip(r.node_names[1:], r.relation_names):
                chain += f"  --{rel}-->  {n}"
            print(f"  {chain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="kg_data/")
    parser.add_argument("--n_examples",  type=int, default=3)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--max_hops",    type=int, default=3)
    parser.add_argument("--beam_width",  type=int, default=64)
    parser.add_argument("--alpha",       type=float, default=0.5)
    args = parser.parse_args()

    cfg = Config(
        max_hops   = args.max_hops,
        beam_width = args.beam_width,
        alpha      = args.alpha,
        top_k_paths= args.top_k,
    )
    demo(args.data_dir, args.n_examples, args.top_k, cfg)
