"""
prompt_variants.py
==================
Concrete implementations of every prompt variant defined in sweep_config.py.

This module extends LLMPromptBuilder with support for:
  - 4 system instruction styles (strict / balanced / permissive / chain-of-thought)
  - 3 entity description verbosity levels (short / medium / full)
  - KG embed-score inclusion toggle
  - Path type label annotation toggle

Usage (internal — called by param_sweep.py):
    from prompt_variants import build_variant_prompt_builder
    builder = build_variant_prompt_builder(context_mgr, variant_key="sys_balanced")
    prompt = builder.build_prompt(paths_json)  # → {"system": ..., "user": ...}
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional

from kg_entity_context import KGEntityContextManager
from llm_prompt_builder import LLMPromptBuilder


# ---------------------------------------------------------------------------
# System prompt texts (all four variants)
# ---------------------------------------------------------------------------

_SYSTEM_HOLISTIC = (
    "You are an expert recommendation analyst. "
    "Your job is to predict whether a specific item would be a good recommendation "
    "for a specific user, given evidence from a Knowledge Graph (KG).\n\n"

    "The KG evidence you will see includes:\n"
    "- A profile of the target user (their purchase history, preferred categories, brands, "
    "average rating behaviour)\n"
    "- A description of the target item (title, brand, category, key features)\n"
    "- KG compatibility scores: f_score (path quality) and embed_score (learned "
    "user-item affinity — higher means the KG's embedding model considers them compatible)\n"
    "- Reasoning paths through the KG connecting the user to the item via shared "
    "purchases, brands, categories, features, and co-purchase/co-view relationships\n\n"

    "How to reason about the evidence:\n"
    "- A user who buys many items from the same category or brand as the target item "
    "is likely to enjoy it. This is the strongest signal.\n"
    "- Collaborative paths (other users who bought what this user bought also bought "
    "the target item) are strong positive evidence.\n"
    "- A high embed_score (above 0.5) indicates the KG's embedding model predicts "
    "affinity between this user and item — treat it as a supporting signal.\n"
    "- Multiple diverse path types converging on the target item (e.g., both a "
    "collaborative path and a brand/category path) strengthen the case for recommendation.\n"
    "- If the item's category is completely outside the user's demonstrated interests "
    "and no strong path evidence exists, that is a negative signal.\n\n"

    "Make your best expert judgment based on all the evidence together. "
    "You are NOT applying a fixed rule — you are making a nuanced recommendation decision "
    "the way a skilled analyst would.\n\n"

    "Output exactly '1' if you would recommend this item to this user, "
    "or '0' if you would not. "
    "Output only that single character — no punctuation, no explanation."
)

_SYSTEM_NUANCED = (
    "You are a senior product recommendation specialist. "
    "Given structured Knowledge Graph evidence about a user and an item, "
    "decide whether the item should be recommended.\n\n"

    "Evidence you will receive:\n"
    "USER PROFILE: purchase history, categories, brands, and rating patterns.\n"
    "ITEM PROFILE: title, brand, category, features.\n"
    "KG SCORES: f_score (path traversal quality) and embed_score (latent affinity, 0–1).\n"
    "REASONING PATHS: graph-traversal evidence connecting user to item.\n\n"

    "Weigh these signals with the following intuition:\n"
    "  Strong FOR recommendation: user has bought items from the same category or brand; "
    "collaborative filtering paths (shared co-purchasers); embed_score ≥ 0.6; "
    "multiple path types all pointing to the same item.\n"
    "  Moderate FOR: co-purchase/co-view links; single collaborative path; "
    "embed_score 0.4–0.6; feature overlap.\n"
    "  Against recommendation: item category has zero overlap with user history; "
    "only weak paths exist; embed_score below 0.3.\n\n"

    "Think like a human expert: an item does not need to be perfect on every axis to be "
    "worth recommending. A strong signal on one axis (e.g., same category, high embed_score) "
    "can outweigh weak evidence on others.\n\n"

    "Output '1' to recommend, '0' to not recommend. Single character only."
)

_SYSTEM_CONVERSATIONAL = (
    "You are helping decide whether to recommend a product to a customer. "
    "You have access to data from a product Knowledge Graph that shows how this customer "
    "is connected to the product through their purchase history, shared buyers, brand "
    "relationships, and category links.\n\n"

    "Think about it this way: if a knowledgeable friend who knew this customer's tastes "
    "saw these connections, would they say 'yes, recommend it'?\n\n"

    "Key things to consider:\n"
    "1. Does the item match what the customer typically buys (same categories, brands)?\n"
    "2. Have other customers with similar taste bought this item?\n"
    "3. Does the KG embedding score suggest the customer and item are a natural match?\n"
    "4. How many and how varied are the graph paths connecting them?\n\n"

    "If the answer is clearly yes based on the evidence, output '1'. "
    "If the evidence does not support a recommendation, output '0'. "
    "Give only the single digit — no other text."
)

_SYSTEM_COT = (
    "You are a recommendation system analyst. "
    "Your task: predict whether an item should be recommended to a user, "
    "using Knowledge Graph evidence.\n\n"

    "Before giving your answer, briefly reason (internally) through these questions:\n"
    "1. Does the item fit within the user's established taste profile "
    "(categories, brands, feature preferences)?\n"
    "2. What do the KG reasoning paths tell you — are there direct collaborative "
    "signals (shared co-purchasers), brand/category overlap, or feature matches?\n"
    "3. What does the embed_score indicate about inherent user-item affinity?\n"
    "4. Taking all signals together, does the evidence lean toward recommendation or not?\n\n"

    "After reasoning through the above, output ONLY '1' (recommend) or '0' (do not recommend). "
    "No reasoning text in your output — just the final digit."
)

SYSTEM_PROMPTS = {
    "holistic":      _SYSTEM_HOLISTIC,
    "nuanced":       _SYSTEM_NUANCED,
    "conversational":_SYSTEM_CONVERSATIONAL,
    "cot":           _SYSTEM_COT,
}

DEFAULT_SYSTEM_VARIANT = "holistic"


# ---------------------------------------------------------------------------
# VariantPromptBuilder — extends LLMPromptBuilder with per-variant config
# ---------------------------------------------------------------------------

class VariantPromptBuilder(LLMPromptBuilder):
    """
    Drop-in replacement for LLMPromptBuilder that supports all prompt variant
    dimensions: system variant, entity description length, embed score toggle,
    and path type label toggle.

    Parameters
    ----------
    context_manager          : KGEntityContextManager instance
    system_variant           : one of "strict" / "balanced" / "permissive" / "cot"
    entity_description_length: "short" / "medium" / "full"
    include_embed_score      : bool — whether to show embed_score in the prompt
    annotate_path_types      : bool — whether to label each path's structural type
    """

    def __init__(
        self,
        context_manager: KGEntityContextManager,
        system_variant:            str  = DEFAULT_SYSTEM_VARIANT,
        entity_description_length: str  = "medium",
        include_embed_score:       bool = True,
        annotate_path_types:       bool = True,
    ):
        super().__init__(context_manager)
        self._system_text = SYSTEM_PROMPTS.get(system_variant, _SYSTEM_HOLISTIC)
        self._entity_length = entity_description_length
        self._include_embed = include_embed_score
        self._annotate_types = annotate_path_types

    # ── Override: system prompt ───────────────────────────────────────────────

    @property
    def SYSTEM_PROMPT(self) -> str:            # type: ignore[override]
        return self._system_text

    # ── Override: entity descriptions ────────────────────────────────────────

    def _build_user_section(self, user_id: str) -> str:
        full = self.context_manager.get_user_description(user_id)
        return self._trim_description(full, "user")

    def _build_item_section(self, item_id: str) -> str:
        full = self.context_manager.get_item_description(item_id)
        return self._trim_description(full, "item")

    def _trim_description(self, desc: str, entity_type: str) -> str:
        """
        Truncate / filter entity descriptions according to verbosity level.
        - "short" : title + first category only
        - "medium": title + brand + category + first 2 features
        - "full"  : everything (unchanged)
        """
        if self._entity_length == "full":
            return desc

        if self._entity_length == "short":
            # Keep only up to the first period (title sentence)
            parts = desc.split(".")
            return parts[0].strip() + "." if parts else desc

        # "medium" — keep title, brand, categories, first 2 features, omit
        # interaction count and extra metadata sentences beyond feature[2]
        # Since descriptions are free-form text, we truncate by sentence count.
        sentences = [s.strip() for s in desc.split(".") if s.strip()]
        # Keep first 4 sentences (title/brand, price, category, features)
        return ". ".join(sentences[:4]) + "."

    # ── Override: KG compatibility section ───────────────────────────────────

    def _build_kg_score_section(self, paths_json: Dict) -> str:
        if not self._include_embed:
            # Embed score suppressed — show only f_score
            paths = paths_json.get("paths", [])
            if not paths:
                return "=== KG Compatibility ===\nNo reasoning paths found."
            best_f = max(
                (p.get("scores", {}).get("f_score", 0.0) for p in paths),
                default=0.0,
            )
            return (
                f"=== KG Compatibility Signal ===\n"
                f"Best path f_score: {best_f:.4f}"
            )
        # Default — both scores shown
        return super()._build_kg_score_section(paths_json)

    # ── Override: path section (type labels toggle) ───────────────────────────

    def _build_paths_section(
        self, paths_json: Dict, query_user: str, query_item: str
    ) -> str:
        if self._annotate_types:
            # Use parent implementation which already annotates types
            return super()._build_paths_section(paths_json, query_user, query_item)

        # Suppress type labels — paths without [type:…] annotation
        paths = sorted(paths_json.get("paths", []), key=lambda x: x.get("rank", 0))
        lines = ["=== Reasoning Paths ==="]

        if not paths:
            lines.append("No paths found between this user and item in the KG.")
            return "\n".join(lines)

        for path in paths:
            nodes  = path.get("nodes", [])
            edges  = path.get("edges", [])
            scores = path.get("scores", {})

            f_score = scores.get("f_score", 0.0)
            lines.append(f"Path {path['rank']} (f={f_score:.4f}):")

            sentences = []
            for i, edge in enumerate(edges):
                src = nodes[i]["name"]
                tgt = nodes[i + 1]["name"]
                sentences.append(self._edge_to_text(edge, src, tgt))
            lines.append("  " + " ".join(sentences))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory function — resolves a variant key string to a configured builder
# ---------------------------------------------------------------------------

_VARIANT_KEY_MAP: Dict[str, Dict[str, Any]] = {
    # System variants
    "sys_holistic":       {"system_variant": "holistic"},
    "sys_nuanced":        {"system_variant": "nuanced"},
    "sys_conversational": {"system_variant": "conversational"},
    "sys_cot":            {"system_variant": "cot"},
    # Entity lengths
    "entity_short":       {"entity_description_length": "short"},
    "entity_medium":      {"entity_description_length": "medium"},
    "entity_full":        {"entity_description_length": "full"},
    # Embed signal
    "signal_with_embed":    {"include_embed_score": True},
    "signal_without_embed": {"include_embed_score": False},
    # Path type labels
    "labels_on":            {"annotate_path_types": True},
    "labels_off":           {"annotate_path_types": False},
}

_DEFAULTS: Dict[str, Any] = {
    "system_variant":            "holistic",
    "entity_description_length": "medium",
    "include_embed_score":        True,
    "annotate_path_types":        True,
}


def build_variant_prompt_builder(
    context_mgr: KGEntityContextManager,
    variant_key: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> VariantPromptBuilder:
    """
    Create a VariantPromptBuilder from a variant key name (from sweep_config.py)
    plus optional fine-grained overrides.

    Parameters
    ----------
    context_mgr  : KGEntityContextManager
    variant_key  : name from PROMPT_VARIANTS in sweep_config.py
                   (e.g. "sys_balanced", "entity_short")
    overrides    : dict of constructor kwargs to override after applying variant_key

    Returns
    -------
    VariantPromptBuilder ready for use in predict_single()
    """
    kwargs = dict(_DEFAULTS)

    # Apply the variant key's settings
    if variant_key and variant_key in _VARIANT_KEY_MAP:
        kwargs.update(_VARIANT_KEY_MAP[variant_key])

    # Apply any additional overrides
    if overrides:
        kwargs.update(overrides)

    return VariantPromptBuilder(context_mgr, **kwargs)
