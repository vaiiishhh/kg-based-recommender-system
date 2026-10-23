"""
param_sweep.py
==============
Automated parameter-tuning sweep for the KG-RAG recommendation pipeline.

Uses TWO-PHASE execution per config point for maximum throughput:
  Phase 1 (sequential, CPU): KG path extraction + prompt building for all pairs
  Phase 2 (parallel, I/O):   Concurrent LLM requests via ThreadPoolExecutor
  Phase 3 (sequential):      Collect results, evaluate, log to tracker

Usage:
    python param_sweep.py --experiment exp01_top_k
    python param_sweep.py --all
    python param_sweep.py --experiment exp01_top_k --dry_run
    python param_sweep.py --experiment exp01_top_k --resume
    python param_sweep.py --leaderboard --experiment exp01_top_k

Options:
    --workers N     Concurrent LLM requests (default: 16)
    --eval_set      Path to eval_set.json  (default: eval_set.json)
    --data_dir      Path to kg_data/       (default: kg_data/)
    --k             NDCG@K value           (default: 10)
    --n_pairs       Limit pairs to first N for quick tests (default: all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# ── Pipeline components ───────────────────────────────────────────────────────
from kg_path_extractor import KGPathExtractor, Config
from kg_path_to_json import paths_to_json as _paths_to_json
from kg_entity_context import KGEntityContextManager
from evaluate import compute_binary_metrics, compute_ndcg
from results_tracker import ResultsTracker
from prompt_variants import build_variant_prompt_builder
from sweep_config import EXPERIMENTS, BASELINE_KG_PARAMS, BASELINE_PROMPT_VARIANT
# ── Persistent Cache for KG Results ──────────────────────────────────────────
_KG_CACHE_FILE = "kg_path_cache.json"

def _get_cache_key(user, item, config_overrides):
    # KG-relevant parameters that affect path extraction
    kg_config = {
        "top_k_paths":            config_overrides.get("top_k", config_overrides.get("top_k_paths", 5)),
        "beam_width":             config_overrides.get("beam_width", 128),
        "max_hops":               config_overrides.get("max_hops", 3),
        "alpha":                  config_overrides.get("alpha", 0.5),
        "embedding_score_weight": config_overrides.get("embedding_score_weight", 0.3),
    }
    kg_config_tuple = tuple(sorted(kg_config.items()))
    return f"{user}|{item}|{str(kg_config_tuple)}"

def _load_kg_cache() -> Dict[str, Any]:
    if os.path.exists(_KG_CACHE_FILE):
        try:
            with open(_KG_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_kg_cache(cache: Dict[str, Any]):
    with open(_KG_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# ── Global instances for worker processes ─────────────────────────────────────
_worker_extractor = None
_worker_prompt_builder = None

def _init_worker(data_dir, prompt_variant_key):
    """Load KG and Context once per worker process."""
    global _worker_extractor, _worker_prompt_builder
    from kg_path_extractor import KGPathExtractor, Config
    from kg_entity_context import KGEntityContextManager
    from prompt_variants import build_variant_prompt_builder
    
    # Minimal config for init
    extractor = KGPathExtractor(data_dir=data_dir)
    context_mgr = KGEntityContextManager(
        data_dir=data_dir,
        user2id=extractor.user2id,
        item2id=extractor.item2id
    )
    _worker_extractor = extractor
    _worker_prompt_builder = build_variant_prompt_builder(context_mgr, prompt_variant_key)


# ---------------------------------------------------------------------------
# Helper: evaluate predictions against the eval set
# ---------------------------------------------------------------------------

def _evaluate(
    predictions: List[Dict[str, Any]],
    eval_set:    List[Dict[str, Any]],
    k:           int = 10,
) -> Dict[str, Any]:
    """Run binary metrics + NDCG inline. Returns metrics dict."""
    true_labels = {}
    threshold = 1.0
    for rec in eval_set:
        user   = rec.get("user") or rec.get("user_id", "")
        item   = rec.get("item") or rec.get("item_id", "")
        rating = float(rec.get("true_rating", rec.get("rating", 0)))
        true_labels[(user, item)] = {
            "rating":      rating,
            "is_positive": 1 if rating >= threshold else 0,
        }

    m      = compute_binary_metrics(predictions, true_labels)
    ndcg_k = compute_ndcg(predictions, true_labels, k)
    ndcg_5 = compute_ndcg(predictions, true_labels, 5)
    ndcg_10 = ndcg_k if k == 10 else compute_ndcg(predictions, true_labels, 10)

    return {
        **m,
        "ndcg@5":    round(ndcg_5, 4),
        f"ndcg@{k}": round(ndcg_k, 4),
        "ndcg@10":   round(ndcg_10, 4),
    }


# ---------------------------------------------------------------------------
# Phase 1: KG extraction + prompt building (sequential, CPU-bound)
# ---------------------------------------------------------------------------

def _process_single_pair(rec, config_overrides, top_k):
    """Worker task function."""
    global _worker_extractor, _worker_prompt_builder
    
    user = rec.get("user") or rec.get("user_id", "")
    item = rec.get("item") or rec.get("item_id", "")

    # Ensure config mapping is correct for the extractor
    safe_overrides = {
        "top_k_paths":            config_overrides.get("top_k", config_overrides.get("top_k_paths", 5)),
        "beam_width":             config_overrides.get("beam_width", 128),
        "max_hops":               config_overrides.get("max_hops", 3),
        "alpha":                  config_overrides.get("alpha", 0.5),
        "embedding_score_weight": config_overrides.get("embedding_score_weight", 0.3),
    }

    try:
        results = _worker_extractor.find_paths(
            user, item,
            top_k=top_k,
            config_overrides=safe_overrides,
        )
    except KeyError as e:
        return {
            "error": str(e),
            "results_data": None
        }

    # Extract the data we need to cache (can't pickle PathResult easily, so simplify)
    results_data = []
    for r in results:
        results_data.append({
            "nodes": r.nodes,
            "rels": r.relations,
            "node_names": r.node_names,
            "relation_names": r.relation_names,
            "g_score": r.g_score,
            "h_score": r.h_score,
            "f_score": r.f_score,
            "embed_score": r.embed_score
        })
    
    return {
        "error": None,
        "results_data": results_data
    }

def _process_single_pair_thread(rec, config_overrides, top_k, extractor, prompt_builder):
    """Thread task function."""
    user = rec.get("user") or rec.get("user_id", "")
    item = rec.get("item") or rec.get("item_id", "")

    # Ensure config mapping is correct for the extractor
    safe_overrides = {
        "top_k_paths":            config_overrides.get("top_k", config_overrides.get("top_k_paths", 5)),
        "beam_width":             config_overrides.get("beam_width", 128),
        "max_hops":               config_overrides.get("max_hops", 3),
        "alpha":                  config_overrides.get("alpha", 0.5),
        "embedding_score_weight": config_overrides.get("embedding_score_weight", 0.3),
    }

    try:
        results = extractor.find_paths(
            user, item,
            top_k=top_k,
            config_overrides=safe_overrides,
        )
    except KeyError as e:
        return {
            "error": str(e),
            "results_data": None
        }

    results_data = []
    for r in results:
        results_data.append({
            "nodes": r.nodes,
            "rels": r.relations,
            "node_names": r.node_names,
            "relation_names": r.relation_names,
            "g_score": r.g_score,
            "h_score": r.h_score,
            "f_score": r.f_score,
            "embed_score": r.embed_score
        })
    
    return {
        "error": None,
        "results_data": results_data
    }

def _build_jobs(
    pairs:          List[Dict[str, Any]],
    extractor:      KGPathExtractor,
    prompt_builder,
    config_overrides: Dict[str, Any],
    data_dir: str,
    prompt_variant: str,
) -> List[Dict[str, Any]]:
    """
    Parallel KG extraction with persistent caching.
    """
    top_k = config_overrides.get("top_k", config_overrides.get("top_k_paths", 5))
    total = len(pairs)
    
    full_cache = _load_kg_cache()
    
    jobs = []
    to_compute_indices = []
    
    for idx, pair in enumerate(pairs):
        user = pair.get("user") or pair.get("user_id", "")
        item = pair.get("item") or pair.get("item_id", "")
        key = _get_cache_key(user, item, config_overrides)
        
        if key in full_cache:
            res_data = full_cache[key]
            # Build prompt from cached results
            prompt = prompt_builder.build_prompt(res_data)
            ranking_score = max((r["f_score"] for r in res_data), default=0.0)
            jobs.append({
                "user": user, "item": item, "error": None,
                "true_rating": pair.get("true_rating", pair.get("rating", None)),
                "ranking_score": ranking_score, "num_paths": len(res_data),
                "prompt": prompt, "status": " CACHED"
            })
        else:
            to_compute_indices.append(idx)

    if to_compute_indices:
        print(f"  [kg] {total - len(to_compute_indices)} CACHED, {len(to_compute_indices)} to compute.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor_pool:
            # Note: ThreadPool doesn't need initializer/initargs like ProcessPool
            future_to_idx = {
                executor_pool.submit(_process_single_pair_thread, pairs[idx], config_overrides, top_k, extractor, prompt_builder): idx 
                for idx in to_compute_indices
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                worker_res = future.result()
                
                user = pairs[idx].get("user") or pairs[idx].get("user_id", "")
                item = pairs[idx].get("item") or pairs[idx].get("item_id", "")
                
                if worker_res["error"]:
                    print(f"  [kg {idx+1:>4}/{total}] {user[:20]:<22} → {item:<14} SKIP ({worker_res['error']})", flush=True)
                    jobs.append({
                        "user": user, "item": item, "error": worker_res["error"],
                        "true_rating": pairs[idx].get("true_rating", pairs[idx].get("rating", None)),
                        "ranking_score": 0.0, "num_paths": 0, "prompt": None, "status": " SKIP"
                    })
                else:
                    res_data = worker_res["results_data"]
                    # Update cache
                    key = _get_cache_key(user, item, config_overrides)
                    full_cache[key] = res_data
                    
                    prompt = prompt_builder.build_prompt(res_data)
                    ranking_score = max((r["f_score"] for r in res_data), default=0.0)
                    print(f"  [kg {idx+1:>4}/{total}] {user[:20]:<22} → {item:<14} paths={len(res_data)} score={ranking_score:.3f}", flush=True)
                    jobs.append({
                        "user": user, "item": item, "error": None,
                        "true_rating": pairs[idx].get("true_rating", pairs[idx].get("rating", None)),
                        "ranking_score": ranking_score, "num_paths": len(res_data),
                        "prompt": prompt, "status": " DONE"
                    })
        
        # Save updated cache after compute batch
        _save_kg_cache(full_cache)
    else:
        print(f"  [kg] All {total} pairs loaded from cache.")

    return jobs




# ---------------------------------------------------------------------------
# Phase 2: Concurrent LLM calls (I/O-bound, ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _run_llm_job(job: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    """
    Execute one LLM request for a prepared job dict.
    Called concurrently by ThreadPoolExecutor workers.
    """
    if job["error"] is not None:
        with _progress_lock:
            print(f"  [{idx:>4}/{total}] {job['user'][:20]:<22} → {job['item']:<14} SKIP ({job['error']})")
        return {
            "user":           job["user"],
            "item":           job["item"],
            "prediction":     None,
            "raw_llm_output": "",
            "num_paths":      0,
            "ranking_score":  0.0,
            "true_rating":    job["true_rating"],
            "error":          job["error"],
        }

    prompt = job["prompt"]
    raw_output = query_llm(prompt["system"], prompt["user"])

    prediction = None
    for ch in raw_output:
        if ch in ("0", "1"):
            prediction = int(ch)
            break

    with _progress_lock:
        print(
            f"  [{idx:>4}/{total}] {job['user'][:20]:<22} → {job['item']:<14} "
            f"pred={prediction}  score={job['ranking_score']:.3f}  raw={raw_output[:6]!r}"
        )

    rec = {
        "user":           job["user"],
        "item":           job["item"],
        "prediction":     prediction,
        "raw_llm_output": raw_output,
        "num_paths":      job["num_paths"],
        "ranking_score":  job["ranking_score"],
    }
    if job["true_rating"] is not None:
        rec["true_rating"] = job["true_rating"]
    return rec


# ---------------------------------------------------------------------------
# Core: run one sweep configuration point (both phases)
# ---------------------------------------------------------------------------

def run_config_point(
    run_id:           str,
    experiment_name:  str,
    sweep_axis:       str,
    sweep_value:      Any,
    pairs:            List[Dict[str, Any]],
    extractor:        KGPathExtractor,
    context_mgr:      KGEntityContextManager,
    config_overrides: Dict[str, Any],
    prompt_variant:   str,
    data_dir:         str,
    n_pairs:          int,
    workers:          int = 16,
) -> Dict[str, Any]:
    """
    Two-phase execution:
      Phase 1: build all jobs (KG extraction + prompt building, sequential)
      Phase 2: fire all LLM requests concurrently (ThreadPoolExecutor)
    Returns metrics dict, or None if dry_run=True.
    """
    print(f"\n{'─'*70}")
    print(f"  Run: {run_id}")
    print(f"  KG params:      {config_overrides}")
    print(f"  Prompt variant: {prompt_variant}  |  Workers: {workers}")
    print(f"{'─'*70}")

    # Build variant prompt builder
    prompt_builder = build_variant_prompt_builder(context_mgr, variant_key=prompt_variant)

    # ── Phase 1: Build all prompts ───────────────────────────────────────────
    print(f"  [phase 1] Extracting KG paths + building {len(pairs)} prompts ...")
    jobs = _build_jobs(pairs, extractor, prompt_builder, config_overrides, data_dir, prompt_variant)
    n_skip = sum(1 for j in jobs if j["error"] is not None)
    n_llm  = len(pairs) - n_skip
    print(f"  [phase 1] Done. {n_llm} prompts built, {n_skip} pairs skipped.")

    # ── Phase 2: Concurrent LLM requests ────────────────────────────────────
    print(f"  [phase 2] Firing {n_llm} LLM requests ({workers} workers) ...")
    predictions_unordered: Dict[int, Dict] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_run_llm_job, job, i + 1, len(pairs)): i
            for i, job in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                predictions_unordered[idx] = future.result()
            except Exception as exc:
                job = jobs[idx]
                with _progress_lock:
                    print(f"  [ERROR] {job['user']} → {job['item']}: {exc}")
                predictions_unordered[idx] = {
                    "user": job["user"], "item": job["item"],
                    "prediction": None, "raw_llm_output": "ERROR",
                    "num_paths": 0, "ranking_score": 0.0,
                    "error": str(exc),
                }

    # Restore original order
    predictions = [predictions_unordered[i] for i in range(len(pairs))]

    # ── Phase 3: Evaluate ────────────────────────────────────────────────────
    metrics = _evaluate(predictions, pairs, k=10)
    print(
        f"\n  ★ {run_id}\n"
        f"    hit_rate={metrics['hit_rate']:.4f}  "
        f"ndcg@5={metrics['ndcg@5']:.4f}  "
        f"ndcg@10={metrics['ndcg@10']:.4f}  "
        f"f1={metrics['f1']:.4f}  "
        f"accuracy={metrics['accuracy']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Sweep runner — iterate over one experiment's values
# ---------------------------------------------------------------------------

def run_experiment(
    exp_name:   str,
    exp_info:   Dict[str, Any],
    pairs:      List[Dict[str, Any]],
    extractor:  KGPathExtractor,
    context_mgr: KGEntityContextManager,
    data_dir:   str,
    tracker:    ResultsTracker,
    workers:    int,
    resume:     bool = False,
):
    """Run all configuration points for a single experiment definition."""
    sweep_axis    = exp_info.get("sweep_axis")
    values        = exp_info.get("values", [None])
    fixed_kg      = exp_info.get("fixed_kg_params", BASELINE_KG_PARAMS)
    fixed_variant = exp_info.get("fixed_prompt_variant", BASELINE_PROMPT_VARIANT)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {exp_info.get('description', '')}")
    print(f"  Sweep axis: {sweep_axis}  |  Values: {values}")
    print(f"{'='*70}")

    existing_ids = {r["run_id"] for r in tracker.runs}

    for val in values:
        # Resolve run_id, kg_params, prompt_variant
        if sweep_axis is None:
            run_id = f"{exp_name}__best_joint"
            kg_params = fixed_kg
            variant = fixed_variant or BASELINE_PROMPT_VARIANT
        elif sweep_axis in ("system_variant", "entity_description_length",
                            "include_embed_score", "annotate_path_types"):
            run_id = f"{exp_name}__{sweep_axis}={val}"
            kg_params = fixed_kg
            variant = val
        else:
            run_id = f"{exp_name}__{sweep_axis}={val}"
            kg_params = {**fixed_kg, sweep_axis: val}
            variant = fixed_variant or BASELINE_PROMPT_VARIANT

        if resume and run_id in existing_ids:
            print(f"  [resume] Skipping already-logged: {run_id}")
            continue

        metrics = run_config_point(
            run_id=run_id,
            experiment_name=exp_name,
            sweep_axis=sweep_axis,
            sweep_value=val,
            pairs=pairs,
            extractor=extractor,
            context_mgr=context_mgr,
            config_overrides=kg_params,
            prompt_variant=variant,
            data_dir=data_dir,
            n_pairs=len(pairs),
            workers=workers
        )

        tracker.add(
            run_id         = run_id,
            experiment     = exp_name,
            sweep_axis     = sweep_axis,
            sweep_value    = val,
            kg_params      = kg_params,
            prompt_variant = variant,
            metrics        = metrics,
        )

    tracker.print_leaderboard(experiment=exp_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep for KG-RAG recommendation pipeline."
    )

    # What to run
    parser.add_argument("--experiment",  type=str,  default=None)
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--list",        action="store_true")
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--baseline",    type=str,  default=None)
    parser.add_argument("--csv",         action="store_true")

    # Execution options
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip run_ids already in sweep_results.json")
    parser.add_argument("--n_pairs",     type=int,  default=None,
                        help="Limit to first N pairs (quick sanity check)")
    parser.add_argument("--workers",     type=int,  default=16,
                        help="Concurrent LLM request threads (default: 16)")

    # Data paths
    parser.add_argument("--eval_set",     default="eval_set.json")
    parser.add_argument("--data_dir",     default="kg_data/")
    parser.add_argument("--k",            type=int, default=10)
    parser.add_argument("--results_file", default="sweep_results.json")

    args = parser.parse_args()
    tracker = ResultsTracker(results_file=args.results_file)

    # ── Information-only modes ───────────────────────────────────────────────

    if args.list:
        print("\nAvailable experiments:")
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:<30}  {exp.get('description', '')}")
        return

    if args.leaderboard:
        tracker.print_leaderboard(args.experiment, top_n=30)
        if args.baseline:
            tracker.print_improvement_over_baseline(args.baseline, args.experiment)
        if args.csv:
            tracker.export_csv(args.experiment)
        return

    if args.csv and not args.leaderboard:
        tracker.export_csv(args.experiment)
        return

    # ── Determine which experiments to run ───────────────────────────────────

    if args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
    elif args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"[ERROR] Unknown experiment: {args.experiment!r}")
            print("Available:", list(EXPERIMENTS.keys()))
            sys.exit(1)
        experiments_to_run = [args.experiment]
    else:
        parser.print_help()
        sys.exit(0)

    # ── Load shared components (once) ────────────────────────────────────────

    print(f"\n[sweep] Loading eval set from {args.eval_set!r} ...")
    with open(args.eval_set) as f:
        full_eval_set = json.load(f)
    pairs = full_eval_set[:args.n_pairs] if args.n_pairs else full_eval_set
    print(f"[sweep] {len(pairs)} pairs loaded.")

    if not args.dry_run:
        print("[sweep] Loading KG Path Extractor ...")
        extractor = KGPathExtractor(data_dir=args.data_dir)

        print("[sweep] Loading Entity Context Manager ...")
        context_mgr = KGEntityContextManager(
            data_dir     = args.data_dir,
            user2id      = extractor.user2id,
            item2id      = extractor.item2id,
        )
        print(f"[sweep] Ready. Workers per run: {args.workers}\n")
    else:
        extractor   = None  # type: ignore
        context_mgr = None  # type: ignore

    # ── Run experiments ──────────────────────────────────────────────────────

    for exp_name in experiments_to_run:
        run_experiment(
            exp_name=exp_name,
            exp_info=EXPERIMENTS[exp_name],
            pairs=pairs,
            extractor=extractor,
            context_mgr=context_mgr,
            data_dir=args.data_dir,
            tracker=tracker,
            workers=args.workers,
            resume=args.resume
        )

    # ── Final summary ────────────────────────────────────────────────────────
    if not args.dry_run:
        print("\n" + "="*70)
        print("  OVERALL LEADERBOARD (sorted by NDCG@10)")
        print("="*70)
        tracker.print_leaderboard(sort_by="ndcg@10", top_n=20)

        best = tracker.best_config(sort_by="ndcg@10")
        if best:
            print(f"  ★ Best config overall:")
            print(f"    Run ID    : {best['run_id']}")
            print(f"    KG params : {best['kg_params']}")
            print(f"    Prompt    : {best['prompt_variant']}")
            print(f"    NDCG@10   : {best['ndcg@10']:.4f}")
            print(f"    Hit Rate  : {best['hit_rate']:.4f}")
            print(f"    F1        : {best['f1']:.4f}")
        print()


if __name__ == "__main__":
    main()
