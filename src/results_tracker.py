"""
results_tracker.py
==================
Persists and compares sweep results across all experiments.

Schema of sweep_results.json:
[
    {
        "run_id":            "exp01_top_k__top_k=5",
        "experiment":        "exp01_top_k",
        "sweep_axis":        "top_k",
        "sweep_value":       5,
        "kg_params":         {...},
        "prompt_variant":    "sys_strict",
        "n_pairs":           200,
        "hit_rate":          0.72,
        "ndcg@5":            0.68,
        "ndcg@10":           0.71,
        "accuracy":          0.70,
        "precision":         0.73,
        "f1":                0.69,
        "tp": 36, "tn": 34, "fp": 14, "fn": 16,
        "invalid":           0,
        "timestamp":         "2026-04-22T10:05:00"
    },
    ...
]

Usage:
    from results_tracker import ResultsTracker
    tracker = ResultsTracker()
    tracker.add(run_id, experiment, sweep_axis, sweep_value, kg_params,
                prompt_variant, metrics_dict)
    tracker.print_leaderboard("exp01_top_k")
    tracker.export_csv("exp01_top_k")
"""

from __future__ import annotations

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


RESULTS_FILE = "sweep_results.json"


class ResultsTracker:
    """
    Append-only tracker for sweep run metrics.
    Loads existing results on construction, appends new ones, and supports
    leaderboard printing and CSV export.
    """

    def __init__(self, results_file: str = RESULTS_FILE):
        self.results_file = results_file
        self.runs: List[Dict[str, Any]] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def save(self):
        with open(self.results_file, "w") as f:
            json.dump(self.runs, f, indent=2)

    # ── Add a result ─────────────────────────────────────────────────────────

    def add(
        self,
        run_id:         str,
        experiment:     str,
        sweep_axis:     Optional[str],
        sweep_value:    Any,
        kg_params:      Dict[str, Any],
        prompt_variant: str,
        metrics:        Dict[str, Any],
        overwrite:      bool = True,
    ):
        """
        Add or update a run result.

        Parameters
        ----------
        run_id         : unique identifier for this run  (e.g. "exp01_top_k__top_k=5")
        experiment     : experiment name from EXPERIMENTS dict
        sweep_axis     : which config key is being varied  (or None for fixed configs)
        sweep_value    : the value used for sweep_axis in this run
        kg_params      : dict of KG extractor params used
        prompt_variant : prompt variant key used
        metrics        : dict returned by evaluate.py (hit_rate, ndcg@k, …)
        overwrite      : if True, replace existing entry with same run_id
        """
        record = {
            "run_id":         run_id,
            "experiment":     experiment,
            "sweep_axis":     sweep_axis,
            "sweep_value":    str(sweep_value),
            "kg_params":      kg_params,
            "prompt_variant": prompt_variant,
            "n_pairs":        metrics.get("total", 0),
            "hit_rate":       metrics.get("hit_rate", 0.0),
            "ndcg@5":         metrics.get("ndcg@5", 0.0),
            "ndcg@10":        metrics.get("ndcg@10", metrics.get(f"ndcg@10", 0.0)),
            "accuracy":       metrics.get("accuracy", 0.0),
            "precision":      metrics.get("precision", 0.0),
            "f1":             metrics.get("f1", 0.0),
            "tp":             metrics.get("tp", 0),
            "tn":             metrics.get("tn", 0),
            "fp":             metrics.get("fp", 0),
            "fn":             metrics.get("fn", 0),
            "invalid":        metrics.get("invalid", 0),
            "timestamp":      datetime.utcnow().isoformat(timespec="seconds"),
        }

        if overwrite:
            # Replace existing run with same run_id
            self.runs = [r for r in self.runs if r["run_id"] != run_id]
        self.runs.append(record)
        self.save()
        print(f"[tracker] Saved: {run_id}  "
              f"hit_rate={record['hit_rate']:.4f}  "
              f"ndcg@10={record['ndcg@10']:.4f}  "
              f"f1={record['f1']:.4f}")

    # ── Analysis / display ────────────────────────────────────────────────────

    def leaderboard(
        self,
        experiment: Optional[str] = None,
        sort_by: str = "ndcg@10",
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return top_n runs, optionally filtered to a single experiment,
        sorted by sort_by descending.
        """
        runs = self.runs
        if experiment:
            runs = [r for r in runs if r["experiment"] == experiment]
        return sorted(runs, key=lambda r: r.get(sort_by, 0.0), reverse=True)[:top_n]

    def print_leaderboard(
        self,
        experiment: Optional[str] = None,
        sort_by: str = "ndcg@10",
        top_n: int = 20,
    ):
        """Print a formatted leaderboard table to stdout."""
        board = self.leaderboard(experiment, sort_by, top_n)
        if not board:
            print("No results found.")
            return

        exp_label = experiment or "ALL EXPERIMENTS"
        print(f"\n{'='*88}")
        print(f"  Leaderboard — {exp_label}  (sorted by {sort_by})")
        print(f"{'='*88}")
        header = (
            f"  {'Run ID':<40} {'HR':>6} {'N@5':>6} {'N@10':>6} "
            f"{'Acc':>6} {'F1':>6} {'n':>5}"
        )
        print(header)
        print(f"  {'-'*83}")
        for r in board:
            print(
                f"  {r['run_id']:<40} "
                f"{r['hit_rate']:>6.4f} "
                f"{r['ndcg@5']:>6.4f} "
                f"{r['ndcg@10']:>6.4f} "
                f"{r['accuracy']:>6.4f} "
                f"{r['f1']:>6.4f} "
                f"{r['n_pairs']:>5}"
            )
        print(f"{'='*88}\n")

    def print_improvement_over_baseline(
        self,
        baseline_run_id: str,
        experiment: Optional[str] = None,
        sort_by: str = "ndcg@10",
    ):
        """Show delta metrics vs. a named baseline run."""
        baseline = next((r for r in self.runs if r["run_id"] == baseline_run_id), None)
        if not baseline:
            print(f"[tracker] Baseline run_id not found: {baseline_run_id!r}")
            return

        board = self.leaderboard(experiment, sort_by)
        print(f"\n  Delta vs. baseline ({baseline_run_id})")
        print(f"  {'Run ID':<40} {'ΔHR':>7} {'ΔN@10':>7} {'ΔF1':>7}")
        print(f"  {'-'*60}")
        for r in board:
            if r["run_id"] == baseline_run_id:
                continue
            dhr  = r["hit_rate"] - baseline["hit_rate"]
            dn10 = r["ndcg@10"] - baseline["ndcg@10"]
            df1  = r["f1"]       - baseline["f1"]
            sign = lambda v: ("+" if v >= 0 else "") + f"{v:.4f}"
            print(
                f"  {r['run_id']:<40} "
                f"{sign(dhr):>7} {sign(dn10):>7} {sign(df1):>7}"
            )
        print()

    def export_csv(self, experiment: Optional[str] = None, output_file: Optional[str] = None):
        """Export results to CSV for spreadsheet analysis."""
        runs = self.runs
        if experiment:
            runs = [r for r in runs if r["experiment"] == experiment]
        if not runs:
            print("[tracker] No results to export.")
            return

        fname = output_file or (
            f"sweep_{experiment}.csv" if experiment else "sweep_all.csv"
        )
        fields = list(runs[0].keys())
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in runs:
                flat_r = {k: (json.dumps(v) if isinstance(v, dict) else v)
                           for k, v in r.items()}
                writer.writerow(flat_r)
        print(f"[tracker] Exported {len(runs)} rows → {fname}")
        return fname

    def best_config(
        self,
        experiment: Optional[str] = None,
        sort_by: str = "ndcg@10",
    ) -> Optional[Dict[str, Any]]:
        """Return the single best run record."""
        board = self.leaderboard(experiment, sort_by, top_n=1)
        return board[0] if board else None


# ---------------------------------------------------------------------------
# CLI: python results_tracker.py [--experiment EXP] [--sort hit_rate] [--csv]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Display sweep leaderboard.")
    p.add_argument("--experiment", type=str, default=None,
                   help="Filter to a specific experiment name")
    p.add_argument("--sort",       type=str, default="ndcg@10",
                   choices=["ndcg@10", "ndcg@5", "hit_rate", "f1", "accuracy"],
                   help="Metric to sort by (default: ndcg@10)")
    p.add_argument("--top",        type=int, default=20,
                   help="How many rows to display")
    p.add_argument("--csv",        action="store_true",
                   help="Also export matching results to CSV")
    p.add_argument("--baseline",   type=str, default=None,
                   help="Run ID to compare all others against")
    p.add_argument("--results_file", type=str, default=RESULTS_FILE)
    args = p.parse_args()

    tracker = ResultsTracker(results_file=args.results_file)
    tracker.print_leaderboard(args.experiment, args.sort, args.top)

    if args.baseline:
        tracker.print_improvement_over_baseline(
            args.baseline, args.experiment, args.sort
        )

    if args.csv:
        tracker.export_csv(args.experiment)
