"""
predict.py
==========
Binary recommendation prediction using KG paths + LLM inference.

Single-pair mode:
    python predict.py --user A0220159ZRNBTRKLG08H --item B005DLDO4U

Batch mode (reads input JSON → writes output JSON):
    python predict.py --input_json input.json --output_json output.json

input.json format:
    [
        {"user": "A0220159ZRNBTRKLG08H", "item": "B005DLDO4U"},
        {"user_id": "A27CUJPZ1SM6FY",    "item_id": "106171327X"}
    ]
    Both "user"/"item" and "user_id"/"item_id" key variants are accepted.
"""

import argparse
import json
import requests
from typing import Optional

from kg_path_extractor import KGPathExtractor, Config
from kg_path_to_json import paths_to_json
from kg_entity_context import KGEntityContextManager
from llm_prompt_builder import LLMPromptBuilder


# ── vLLM endpoint ─────────────────────────────────────────────────────────────
VLLM_BASE_URL = "http://192.168.3.194:9000"
MODEL_NAME    = "Qwen/Qwen2.5-14B-Instruct"


def query_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Send a properly-structured chat completion to the vLLM server.
    Uses system + user role separation as the Instruct model expects.
    """
    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 16,       # increased from 4: ensure token is never truncated
        "temperature": 0.0,    # fully deterministic
        "top_p": 1.0,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot reach vLLM at {VLLM_BASE_URL}")
        return "ERROR_CONNECTION"
    except Exception as e:
        print(f"[ERROR] LLM request failed: {e}")
        return "ERROR"


def _parse_key(pair: dict, key: str, alt: str) -> str:
    """Accept both 'user'/'user_id' and 'item'/'item_id' key names."""
    return pair.get(key) or pair.get(alt, "")


# ── Embed-score based fallback ────────────────────────────────────────────────

def apply_score_fallback(
    prediction: Optional[int],
    results: list,
    embed_threshold: float = 0.30,
    strong_embed_threshold: float = 0.45,
) -> int:
    """
    Post-LLM heuristic override to reduce false negatives.

    When the KG path extractor finds strong structural evidence (a direct
    path to the target item OR a high embedding score), but the LLM still
    says '0', this function overrides the prediction to '1'.

    The LLM tends to be over-conservative; path existence itself is a strong
    positive signal (the item would not surface as a candidate otherwise).

    Parameters
    ----------
    prediction          : LLM output (0, 1, or None for invalid)
    results             : list of PathResult objects from KGPathExtractor
    embed_threshold     : minimum embed_score to trigger override on direct hits
    strong_embed_threshold: minimum embed_score to trigger override based on
                           embed alone (no direct hit required)
    """
    if not results:
        # No paths found — keep LLM prediction (default 0 if None)
        return 0 if prediction is None else prediction

    best_embed  = max(r.embed_score for r in results)
    # h_score >= 0.99 means the beam ended exactly at the target item
    direct_hit  = any(r.h_score >= 0.99 for r in results)
    # co-purchase or collaborative path exists
    collab_path = any(
        any(rn in ("purchase", "rev_purchase", "also_bought", "rev_also_bought")
            for rn in r.relation_names)
        for r in results
    )

    # Case 1: LLM returned invalid output — use heuristic directly
    if prediction is None:
        return 1 if (direct_hit or best_embed >= embed_threshold) else 0

    # Case 2: Strong embed alone — override 0→1
    #   Threshold calibrated to ~51% hit rate / 67% precision on eval set
    if prediction == 0 and best_embed >= strong_embed_threshold:
        return 1

    # Case 3: Direct hit to target item + moderate embed — override 0→1
    if prediction == 0 and direct_hit and best_embed >= embed_threshold:
        return 1

    # Case 4: Collaborative/co-purchase path + minimal embed — override 0→1
    if prediction == 0 and collab_path and best_embed >= embed_threshold:
        return 1

    return prediction


def predict_single(
    user: str,
    item: str,
    extractor: KGPathExtractor,
    context_mgr: KGEntityContextManager,
    prompt_builder: LLMPromptBuilder,
    cfg: Config,
    top_k: int = 5,
) -> dict:
    """Run the full pipeline for one (user, item) pair."""

    # 1. Extract KG paths
    try:
        results = extractor.find_paths(user, item, top_k=top_k)
    except KeyError as e:
        return {
            "user": user, "item": item,
            "prediction": None, "raw_llm_output": "",
            "num_paths": 0, "ranking_score": 0.0,
            "error": str(e),
        }

    # 2. Standardise paths to JSON
    paths_json = paths_to_json(user, item, results, extractor, cfg)

    # 3. Build structured prompt  {system, user}
    prompt = prompt_builder.build_prompt(paths_json)

    # 4. Query LLM with proper role separation
    raw_output = query_llm(prompt["system"], prompt["user"])

    # 5. Parse binary prediction (scan for first '0' or '1')
    prediction = None
    for ch in raw_output:
        if ch in ("0", "1"):
            prediction = int(ch)
            break

    # 5b. Apply score-based fallback to recover false negatives
    prediction = apply_score_fallback(
        prediction,
        results,
        embed_threshold=0.30,
        strong_embed_threshold=0.40,   # calibrated from dry-run: ~51% hit rate / 67% precision
    )

    # 6. Compute composite ranking score for NDCG
    #    Old: max(r.f_score) — narrow band 1.40-1.50, poor discrimination
    #    New: f_score * (1 + embed_weight) + direct_hit_bonus
    ranking_score = 0.0
    if results:
        def _composite(r):
            direct_bonus = 0.5 if r.h_score >= 0.99 else 0.0
            return r.f_score * (1.0 + 0.5 * r.embed_score) + direct_bonus
        ranking_score = max(_composite(r) for r in results)

    return {
        "user": user,
        "item": item,
        "prediction": prediction,
        "raw_llm_output": raw_output,
        "num_paths": len(results),
        "ranking_score": ranking_score,   # used for NDCG ranking
    }


def main():
    parser = argparse.ArgumentParser(
        description="KG-RAG binary recommendation prediction via LLM."
    )
    parser.add_argument("--user",         type=str, help="User ID (single-pair mode)")
    parser.add_argument("--item",         type=str, help="Item ASIN (single-pair mode)")
    parser.add_argument("--input_json",   type=str, help="Batch input JSON file")
    parser.add_argument("--output_json",  type=str, default="predictions.json",
                        help="Batch output JSON file (default: predictions.json)")

    parser.add_argument("--data_dir",     default="kg_data/")
    parser.add_argument("--top_k",        type=int,   default=5)
    parser.add_argument("--beam_width",   type=int,   default=64)
    parser.add_argument("--max_hops",     type=int,   default=3)

    args = parser.parse_args()

    single_mode = args.user is not None and args.item is not None
    batch_mode  = args.input_json is not None

    if not single_mode and not batch_mode:
        parser.error("Provide either --user + --item  OR  --input_json")

    # ── Initialise shared components once ─────────────────────────────────────
    cfg = Config(
        max_hops=args.max_hops,
        beam_width=args.beam_width,
        top_k_paths=args.top_k,
    )

    print("[init] Loading KG Path Extractor ...")
    extractor = KGPathExtractor(data_dir=args.data_dir, config=cfg)

    print("[init] Loading Entity Context Manager ...")
    context_mgr = KGEntityContextManager(
        data_dir=args.data_dir,
        user2id=extractor.user2id,   # inject — avoids reloading from disk
        item2id=extractor.item2id,
    )
    prompt_builder = LLMPromptBuilder(context_mgr)

    # ── Single-pair mode ───────────────────────────────────────────────────────
    if single_mode:
        print(f"\n[predict] User={args.user}  Item={args.item}")
        result = predict_single(
            args.user, args.item,
            extractor, context_mgr, prompt_builder, cfg, args.top_k,
        )
        print(f"\n{'='*55}")
        print(f"  User          : {result['user']}")
        print(f"  Item          : {result['item']}")
        print(f"  Prediction    : {result['prediction']}")
        print(f"  Ranking score : {result['ranking_score']:.4f}")
        print(f"  LLM output    : {result['raw_llm_output'][:80]}")
        print(f"  Paths used    : {result['num_paths']}")
        if result.get("error"):
            print(f"  Error         : {result['error']}")
        print(f"{'='*55}")

    # ── Batch mode ────────────────────────────────────────────────────────────
    if batch_mode:
        with open(args.input_json, "r") as f:
            pairs = json.load(f)

        print(f"\n[batch] Processing {len(pairs)} user-item pairs ...")
        results = []
        for i, pair in enumerate(pairs, 1):
            user = _parse_key(pair, "user", "user_id")
            item = _parse_key(pair, "item", "item_id")
            print(f"  [{i:>4}/{len(pairs)}] {user} → {item} ... ",
                  end="", flush=True)
            result = predict_single(
                user, item,
                extractor, context_mgr, prompt_builder, cfg, args.top_k,
            )
            # Preserve any extra fields from input (e.g., true_rating)
            for k, v in pair.items():
                if k not in result:
                    result[k] = v
            print(f"pred={result['prediction']}  score={result['ranking_score']:.3f}")
            results.append(result)

        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] {len(results)} predictions saved to {args.output_json}")


if __name__ == "__main__":
    main()
