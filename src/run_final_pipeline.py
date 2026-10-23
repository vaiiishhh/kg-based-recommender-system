"""
run_final_pipeline.py
=====================
This script executes the BEST configuration of the KG-RAG Recommendation system
on the complete evaluation dataset (eval_set.json) containing all ground-truth pairs.

It performs the entire pipeline from end-to-end:
1. Loads the Evaluation Set queries.
2. Extracts KG Paths using the optimal parameters (top_k=3, beam=128).
3. Synthesizes Prompts.
4. Queries the LLM.
5. Saves all predictions to 'final_predictions.json'.
6. Evaluates the performance (Hit Rate, NDCG, F1) against the ground truth.

Usage:
    python run_final_pipeline.py
"""

import json
import argparse
import sys
import time

from kg_path_extractor import KGPathExtractor, Config
from kg_entity_context import KGEntityContextManager
from prompt_variants import build_variant_prompt_builder
from predict import predict_single
from evaluate import compute_binary_metrics, compute_ndcg

def main():
    parser = argparse.ArgumentParser(description="Run Final Evaluation Pipeline Demo")
    parser.add_argument("--eval_set", default="eval_set.json", help="Input query set (must contain true_rating)")
    parser.add_argument("--data_dir", default="kg_data/", help="KG features directory")
    parser.add_argument("--output", default="final_predictions.json", help="Path to save predictions")
    args = parser.parse_args()

    # Best overall configuration determined by the parameter sweep
    best_config = {
        "top_k": 3,
        "beam_width": 128,
        "max_hops": 3,
        "alpha": 0.5,
        "embedding_score_weight": 0.3,
        "prompt_variant": "sys_holistic"
    }

    print("=" * 60)
    print("🚀 KG-RAG FINAL PIPELINE DEMONSTRATION")
    print("=" * 60)
    print(f"Optimal Configuration: {best_config}")
    print("-" * 60)

    # 1. Load Data
    print(f"Loading queries from {args.eval_set}...")
    with open(args.eval_set, "r") as f:
        queries = json.load(f)
    print(f"Found {len(queries)} evaluation pairs.")

    # 2. Initialize Components
    print("\nInitializing KG Extractor...")
    cfg = Config(
        max_hops=best_config["max_hops"],
        beam_width=best_config["beam_width"],
        top_k_paths=best_config["top_k"]
    )
    extractor = KGPathExtractor(data_dir=args.data_dir, config=cfg)
    # Apply optimal alpha and embed weights
    cfg.alpha = best_config["alpha"]
    cfg.embedding_score_weight = best_config["embedding_score_weight"]

    print("\nInitializing Entity Context Manager (no leakage)...")
    context_mgr = KGEntityContextManager(
        data_dir=args.data_dir,
        user2id=extractor.user2id,
        item2id=extractor.item2id,
    )
    
    print(f"\nBuilding Prompt Engine (Variant: {best_config['prompt_variant']})...")
    prompt_builder = build_variant_prompt_builder(context_mgr, variant_key=best_config["prompt_variant"])

    # 3. Execution
    print("\n" + "=" * 60)
    print(f"Executing Inference Engine on {len(queries)} pairs...")
    print("=" * 60)
    
    predictions = []
    start_time = time.time()
    
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Thread-safe lock for printing
    import threading
    print_lock = threading.Lock()

    def process_pair(i, pair):
        user = pair.get("user") or pair.get("user_id")
        item = pair.get("item") or pair.get("item_id")
        true_rating = pair.get("true_rating", 0.0)
        
        try:
            res = predict_single(
                user, item, 
                extractor, context_mgr, prompt_builder, 
                cfg, top_k=best_config["top_k"]
            )
            res["true_rating"] = true_rating
            with print_lock:
                print(f"[{i:03d}/{len(queries)}] {user[:15]:<16} → {item:<12} pred={res['prediction']} score={res['ranking_score']:.3f}")
            return res
        except Exception as e:
            with print_lock:
                print(f"[{i:03d}/{len(queries)}] {user[:15]:<16} → {item:<12} ERROR: {str(e)}")
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        future_to_i = {pool.submit(process_pair, i, pair): i for i, pair in enumerate(queries, 1)}
        for future in as_completed(future_to_i):
            res = future.result()
            if res is not None:
                predictions.append(res)

    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed:.1f}s")
    
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {args.output}")

    # 4. Evaluation
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION REPORT")
    print("=" * 60)
    
    true_labels = {}
    threshold = 1.0  # Threshold for graded relevance (> 0 is positive)
    for rec in queries:
        u = rec.get("user") or rec.get("user_id")
        i = rec.get("item") or rec.get("item_id")
        r = float(rec.get("true_rating", 0.0))
        true_labels[(u, i)] = {
            "rating": r,
            "is_positive": 1 if r >= threshold else 0,
        }

    metrics = compute_binary_metrics(predictions, true_labels)
    ndcg_10 = compute_ndcg(predictions, true_labels, 10)
    
    print(f"  Hit Rate (Recall): {metrics['hit_rate']:.4f} ({metrics['hit_rate']*100:.1f}%)")
    print(f"  NDCG@10          : {ndcg_10:.4f} ({ndcg_10*100:.1f}%)")
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  Precision        : {metrics['precision']:.4f}")
    print(f"  F1 Score         : {metrics['f1']:.4f}")
    print("\n  Confusion Matrix:")
    print(f"                  Pred=1   Pred=0")
    print(f"  Actual=1 (pos)  {metrics['tp']:<7}  {metrics['fn']}")
    print(f"  Actual=0 (neg)  {metrics['fp']:<7}  {metrics['tn']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
