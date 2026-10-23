"""
evaluate_ndcg.py
================
Proper evaluation of both Hit Rate and per-user List NDCG using the
graded NDCG ground truth (ndgc_ground_truth.json).

How this differs from evaluate.py
----------------------------------
evaluate.py computes NDCG per user but relies on the predictions file
having enough items per user (>= 2). The older 200-pair balanced set only
had 3 users with 2 items, making NDCG essentially 0.

This script is purpose-built for the NDCG eval set produced by
generate_ndcg_eval_set.py, and explicitly:

  1. Computes HIT RATE correctly (tp / total_positives, threshold = 1.0)
  2. Computes per-user NDCG@K using ranking_score from predict.py
  3. Reports per-user breakdowns so you can spot which users drive the metric
  4. Includes a 'perfect ranking' upper bound for reference

Usage:
    # First generate the eval set (once):
    python generate_ndcg_eval_set.py

    # Then run predictions:
    python predict.py --input_json eval_ndcg_set.json --output_json predictions_ndcg.json

    # Then evaluate:
    python evaluate_ndcg.py
    python evaluate_ndcg.py --k 5
    python evaluate_ndcg.py \\
        --input_json eval_ndcg_set.json \\
        --output_json predictions_ndcg.json \\
        --k 10 \\
        --threshold 1.0
"""

import argparse
import json
import math
from collections import defaultdict


# ── DCG / NDCG helpers ────────────────────────────────────────────────────────

def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain at K (2^rel - 1 / log2(i+2) form)."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_relevances: list, k: int) -> float:
    """
    NDCG@K.

    Parameters
    ----------
    ranked_relevances : relevance values sorted by predicted score (descending)
    k                 : cutoff

    Returns
    -------
    float in [0, 1]
    """
    actual_dcg = dcg_at_k(ranked_relevances, k)
    ideal_dcg  = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── Binary classification helpers ─────────────────────────────────────────────

def compute_hit_rate_metrics(predictions: list, true_labels: dict, threshold: float):
    """
    Compute TP/FN/FP/TN and derived hit-rate metrics.

    Parameters
    ----------
    predictions  : list of prediction dicts
    true_labels  : {(user, item): {'rating': float, 'is_positive': int}}
    threshold    : rating >= threshold → positive
    """
    tp = tn = fp = fn = invalid = 0

    for pred in predictions:
        key = (pred["user"], pred["item"])
        if key not in true_labels:
            continue
        p = pred.get("prediction")
        if p not in (0, 1):
            invalid += 1
            p = 0   # treat as negative (conservative)
        is_pos = true_labels[key]["is_positive"]
        if   is_pos == 1 and p == 1: tp += 1
        elif is_pos == 0 and p == 0: tn += 1
        elif is_pos == 0 and p == 1: fp += 1
        elif is_pos == 1 and p == 0: fn += 1

    total     = tp + tn + fp + fn
    total_pos = tp + fn
    total_neg = tn + fp
    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    hit_rate    = tp / total_pos    if total_pos > 0 else 0.0
    specificity = tn / total_neg    if total_neg > 0 else 0.0
    f1 = (
        2 * precision * hit_rate / (precision + hit_rate)
        if (precision + hit_rate) > 0 else 0.0
    )
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total, "total_pos": total_pos, "total_neg": total_neg,
        "invalid": invalid,
        "accuracy": accuracy,
        "precision": precision,
        "hit_rate": hit_rate,
        "specificity": specificity,
        "f1": f1,
    }


# ── Per-user NDCG computation ─────────────────────────────────────────────────

def compute_ndcg_per_user(predictions: list, true_labels: dict, k: int):
    """
    Compute NDCG@K per user and return mean NDCG + per-user details.

    For each user:
      1. Collect all their (ranking_score, true_rating) pairs from predictions
      2. Sort by ranking_score descending (= our predicted ranking)
      3. Compute NDCG@K against that ranked order
      4. Also compute ideal NDCG for a perfect ranker (upper bound)

    Returns
    -------
    dict with 'mean_ndcg', 'ndcg_scores', 'per_user' details
    """
    # Group predictions by user
    user_preds = defaultdict(list)
    for pred in predictions:
        key = (pred["user"], pred["item"])
        if key not in true_labels:
            continue
        score  = pred.get("ranking_score", 0.0)
        rating = true_labels[key]["rating"]
        user_preds[pred["user"]].append((score, rating))

    ndcg_scores = []
    per_user = {}

    for user, items in user_preds.items():
        if len(items) < 2:
            # NDCG is only meaningful with >= 2 items to rank
            # (with 1 item, ranking is trivial and always perfect)
            continue

        # Sort by ranking_score descending → this is our predicted ranking
        items.sort(key=lambda x: -x[0])
        ranked_relevances = [rel for _, rel in items]

        user_ndcg = ndcg_at_k(ranked_relevances, k)
        ideal_order = sorted(ranked_relevances, reverse=True)

        ndcg_scores.append(user_ndcg)
        per_user[user] = {
            "ndcg": round(user_ndcg, 4),
            "items": [(round(score, 4), rel) for score, rel in items],
            "ideal_order": ideal_order,
            "n_items": len(items),
        }

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Upper-bound: what NDCG would be if ranking_score perfectly ranked items
    # (all users get NDCG = 1.0 for items that can be distinguished)
    n_eligible = len(ndcg_scores)

    return {
        "mean_ndcg":   mean_ndcg,
        "ndcg_scores": ndcg_scores,
        "n_users_evaluated": n_eligible,
        "per_user": per_user,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hit rate and per-user NDCG from NDCG ground truth."
    )
    parser.add_argument(
        "--input_json",  default="eval_ndcg_set.json",
        help="Ground truth eval set (from generate_ndcg_eval_set.py)"
    )
    parser.add_argument(
        "--output_json", default="predictions_ndcg.json",
        help="Predictions file from predict.py"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="K for NDCG@K (default 10)"
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="Rating threshold for positive label in hit-rate (default 1.0)"
    )
    parser.add_argument(
        "--show_per_user", action="store_true",
        help="Print per-user NDCG breakdown"
    )
    parser.add_argument(
        "--save_metrics", action="store_true", default=True,
        help="Save JSON metrics file alongside predictions (default: True)"
    )
    args = parser.parse_args()

    # ── Load ground truth ──────────────────────────────────────────────────
    with open(args.input_json) as f:
        inputs = json.load(f)

    true_labels = {}
    for rec in inputs:
        user   = rec.get("user") or rec.get("user_id", "")
        item   = rec.get("item") or rec.get("item_id", "")
        rating = float(rec.get("true_rating", rec.get("rating", 0)))
        true_labels[(user, item)] = {
            "rating":      rating,
            "is_positive": 1 if rating >= args.threshold else 0,
        }

    # ── Load predictions ───────────────────────────────────────────────────
    with open(args.output_json) as f:
        predictions = json.load(f)

    # ── Compute metrics ────────────────────────────────────────────────────
    hit_metrics  = compute_hit_rate_metrics(predictions, true_labels, args.threshold)
    ndcg_result  = compute_ndcg_per_user(predictions, true_labels, args.k)
    ndcg_5       = compute_ndcg_per_user(predictions, true_labels, 5)

    mean_ndcg_k = ndcg_result["mean_ndcg"]
    mean_ndcg_5 = ndcg_5["mean_ndcg"]
    n_users_eval = ndcg_result["n_users_evaluated"]

    # ── Print report ───────────────────────────────────────────────────────
    sep = "─" * 52
    print(f"\n{'='*52}")
    print(f"  NDCG-Focused Evaluation Report")
    print(f"  Ground truth : {args.input_json}")
    print(f"  Predictions  : {args.output_json}")
    print(f"  Threshold    : rating >= {args.threshold} → positive")
    print(f"{'='*52}")
    print(f"  Total pairs evaluated : {hit_metrics['total']}")
    print(f"  Positives (≥{args.threshold:.1f})          : {hit_metrics['total_pos']}")
    print(f"  Negatives (<{args.threshold:.1f})           : {hit_metrics['total_neg']}")
    print(f"  Invalid LLM outputs   : {hit_metrics['invalid']}")
    print(sep)
    print(f"  HIT RATE (Recall)     : {hit_metrics['hit_rate']:.4f}  ({hit_metrics['hit_rate']*100:.1f}%)")
    print(f"  NDCG@5  (per-user)    : {mean_ndcg_5:.4f}  ({mean_ndcg_5*100:.1f}%)  [{ndcg_5['n_users_evaluated']} users]")
    print(f"  NDCG@{args.k:<2} (per-user)    : {mean_ndcg_k:.4f}  ({mean_ndcg_k*100:.1f}%)  [{n_users_eval} users]")
    print(sep)
    print(f"  Accuracy              : {hit_metrics['accuracy']:.4f}  ({hit_metrics['accuracy']*100:.1f}%)")
    print(f"  Precision             : {hit_metrics['precision']:.4f}  ({hit_metrics['precision']*100:.1f}%)")
    print(f"  F1 Score              : {hit_metrics['f1']:.4f}  ({hit_metrics['f1']*100:.1f}%)")
    print(f"  Specificity           : {hit_metrics['specificity']:.4f}  ({hit_metrics['specificity']*100:.1f}%)")
    print(sep)
    print(f"  Confusion Matrix:")
    print(f"                    Pred=1   Pred=0")
    print(f"  Actual=1 (pos)    {hit_metrics['tp']:<7}  {hit_metrics['fn']}")
    print(f"  Actual=0 (neg)    {hit_metrics['fp']:<7}  {hit_metrics['tn']}")
    print(f"{'='*52}")

    # ── NDCG interpretation note ───────────────────────────────────────────
    print(f"\n  NDCG Note: computed as per-user ranked list.")
    print(f"  Each user has {2}–{3} candidate items ranked by 'ranking_score'.")
    print(f"  NDCG = 1.0 means the system perfectly puts the most relevant item first.")
    print(f"  NDCG = 0.5 means the ranking is roughly random / inverted.")
    print(f"  {n_users_eval} users with >= 2 predictions contribute to the NDCG score.")

    # ── Optional per-user breakdown ────────────────────────────────────────
    if args.show_per_user and ndcg_result["per_user"]:
        print(f"\n  Per-user NDCG@{args.k} (sorted by NDCG asc):")
        print(f"  {'User':<25} {'NDCG':>8}  {'Items (score, rating)'}")
        print(f"  {'-'*70}")
        for user, info in sorted(ndcg_result["per_user"].items(),
                                  key=lambda x: x[1]["ndcg"]):
            items_str = "  ".join(
                f"({sc:.3f}, r={rel})" for sc, rel in info["items"]
            )
            print(f"  {user:<25} {info['ndcg']:>8.4f}  {items_str}")

    # ── Save metrics ───────────────────────────────────────────────────────
    if args.save_metrics:
        summary = {
            "eval_source": args.input_json,
            "predictions_source": args.output_json,
            "threshold": args.threshold,
            "k": args.k,
            # Hit rate metrics
            "hit_rate":    round(hit_metrics["hit_rate"], 4),
            "accuracy":    round(hit_metrics["accuracy"], 4),
            "precision":   round(hit_metrics["precision"], 4),
            "f1":          round(hit_metrics["f1"], 4),
            "specificity": round(hit_metrics["specificity"], 4),
            # NDCG metrics
            f"ndcg@{args.k}":    round(mean_ndcg_k, 4),
            "ndcg@5":            round(mean_ndcg_5, 4),
            "n_users_for_ndcg":  n_users_eval,
            # Confusion matrix
            "tp": hit_metrics["tp"],
            "tn": hit_metrics["tn"],
            "fp": hit_metrics["fp"],
            "fn": hit_metrics["fn"],
            "total": hit_metrics["total"],
            "invalid": hit_metrics["invalid"],
        }
        metrics_path = args.output_json.replace(".json", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Metrics saved → {metrics_path}")

    print()


if __name__ == "__main__":
    main()
