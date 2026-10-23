"""
evaluate.py
===========
Computes recommendation metrics from predictions vs. ground truth.

Metrics computed:
  - Hit Rate (Recall for positives, rating >= 4)
  - NDCG@K  (per-user ranking evaluation using ranking_score)
  - Accuracy, Precision, F1, Specificity
  - Confusion matrix

Usage:
    python evaluate.py --input_json eval_set.json --output_json predictions.json
    python evaluate.py --input_json eval_set.json --output_json predictions.json --k 10
"""

import json
import argparse
import math
from collections import defaultdict


# ── NDCG helpers ──────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)   # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(relevances: list, k: int) -> float:
    """Normalised DCG at K. relevances is already sorted by predicted score desc."""
    actual_dcg  = dcg_at_k(relevances, k)
    ideal_dcg   = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_ndcg(predictions: list, true_labels: dict, k: int) -> float:
    """
    Per-user NDCG@K.
    For each user, rank their test items by ranking_score (descending),
    then compute NDCG using true_rating as relevance grade.
    """
    # Group predictions by user
    user_preds = defaultdict(list)
    for pred in predictions:
        key = (pred["user"], pred["item"])
        if key not in true_labels:
            continue
        score = pred.get("ranking_score", 0.0)
        # Use raw rating as relevance grade (1-5 scale)
        relevance = true_labels[key]["rating"]
        user_preds[pred["user"]].append((score, relevance))

    if not user_preds:
        return 0.0

    ndcg_scores = []
    for user, items in user_preds.items():
        if len(items) < 2:
            # NDCG is only meaningful when there's more than one item to rank
            continue
        # Sort by ranking_score descending → ranked list
        items.sort(key=lambda x: -x[0])
        relevances = [rel for _, rel in items]
        ndcg_scores.append(ndcg_at_k(relevances, k))

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


# ── Binary classification metrics ─────────────────────────────────────────────

def compute_binary_metrics(predictions: list, true_labels: dict):
    """Compute TP/TN/FP/FN and derived metrics. Threshold: rating >= 4 = positive."""
    tp = tn = fp = fn = invalid = 0

    for pred in predictions:
        key = (pred["user"], pred["item"])
        if key not in true_labels:
            continue
        p = pred.get("prediction")
        if p not in (0, 1):
            invalid += 1
            p = 0   # treat as negative
        is_pos = true_labels[key]["is_positive"]
        if   is_pos == 1 and p == 1: tp += 1
        elif is_pos == 0 and p == 0: tn += 1
        elif is_pos == 0 and p == 1: fp += 1
        elif is_pos == 1 and p == 0: fn += 1

    total       = tp + tn + fp + fn
    total_pos   = tp + fn
    total_neg   = tn + fp
    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    hit_rate    = tp / total_pos    if total_pos > 0 else 0.0   # recall
    specificity = tn / total_neg    if total_neg > 0 else 0.0
    f1          = (
        2 * precision * hit_rate / (precision + hit_rate)
        if (precision + hit_rate) > 0 else 0.0
    )
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total, "total_pos": total_pos, "total_neg": total_neg,
        "invalid": invalid,
        "accuracy": accuracy, "precision": precision,
        "hit_rate": hit_rate, "specificity": specificity, "f1": f1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation predictions.")
    parser.add_argument("--input_json",  default="eval_set.json",
                        help="Ground truth file (must contain true_rating per pair)")
    parser.add_argument("--output_json", default="predictions.json",
                        help="Predictions file from predict.py")
    parser.add_argument("--k",           type=int, default=10,
                        help="K for NDCG@K (default 10)")
    parser.add_argument("--threshold",   type=float, default=1.0,
                        help="Rating threshold for positive label (default 1.0 for graded relevance)")
    args = parser.parse_args()

    # Load ground truth
    with open(args.input_json, "r") as f:
        inputs = json.load(f)

    true_labels = {}
    for rec in inputs:
        user = rec.get("user") or rec.get("user_id", "")
        item = rec.get("item") or rec.get("item_id", "")
        rating = float(rec.get("true_rating", rec.get("rating", 0)))
        key = (user, item)
        true_labels[key] = {
            "rating": rating,
            "is_positive": 1 if rating >= args.threshold else 0,
        }

    # Load predictions
    with open(args.output_json, "r") as f:
        predictions = json.load(f)

    # Binary metrics
    m = compute_binary_metrics(predictions, true_labels)

    # NDCG@K
    ndcg_k = compute_ndcg(predictions, true_labels, args.k)
    ndcg_5 = compute_ndcg(predictions, true_labels, 5)

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "-" * 42
    print(f"\n{'='*42}")
    print(f"  Evaluation Report")
    print(f"  Ground truth: {args.input_json}")
    print(f"  Predictions:  {args.output_json}")
    print(f"{'='*42}")
    print(f"  Total evaluated  : {m['total']} pairs  (invalid: {m['invalid']})")
    print(f"  Positive (≥{args.threshold:.0f})  : {m['total_pos']}  |  Negative: {m['total_neg']}")
    print(sep)
    print(f"  HIT RATE (Recall): {m['hit_rate']:.4f}  ({m['hit_rate']*100:.1f}%)")
    print(f"  NDCG@5           : {ndcg_5:.4f}  ({ndcg_5*100:.1f}%)")
    print(f"  NDCG@{args.k:<2}          : {ndcg_k:.4f}  ({ndcg_k*100:.1f}%)")
    print(sep)
    print(f"  Accuracy         : {m['accuracy']:.4f}  ({m['accuracy']*100:.1f}%)")
    print(f"  Precision        : {m['precision']:.4f}  ({m['precision']*100:.1f}%)")
    print(f"  F1 Score         : {m['f1']:.4f}  ({m['f1']*100:.1f}%)")
    print(f"  Specificity      : {m['specificity']:.4f}  ({m['specificity']*100:.1f}%)")
    print(sep)
    print(f"  Confusion Matrix:")
    print(f"                  Pred=1   Pred=0")
    print(f"  Actual=1 (pos)  {m['tp']:<7}  {m['fn']}")
    print(f"  Actual=0 (neg)  {m['fp']:<7}  {m['tn']}")
    print(f"{'='*42}\n")

    # Save summary to JSON for logging / comparison across runs
    summary = {
        "hit_rate": round(m["hit_rate"], 4),
        f"ndcg@{args.k}": round(ndcg_k, 4),
        "ndcg@5": round(ndcg_5, 4),
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "f1": round(m["f1"], 4),
        "specificity": round(m["specificity"], 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "total": m["total"], "invalid": m["invalid"],
    }
    summary_path = args.output_json.replace(".json", "_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved → {summary_path}")


if __name__ == "__main__":
    main()
