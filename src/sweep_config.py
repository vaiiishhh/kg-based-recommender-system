"""
sweep_config.py
===============
Single source of truth for all tunable parameters and prompt variants.

Edit this file to add/remove sweep dimensions or change experiment scope.
param_sweep.py reads this file to drive automated runs.
"""

from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# 1.  KG Path Extractor parameters  (Config fields in kg_path_extractor.py)
# ---------------------------------------------------------------------------

KG_PARAM_GRID: Dict[str, List[Any]] = {
    # Number of top paths shown to the LLM
    "top_k": [3, 5, 8, 10],

    # Beam width during path search
    "beam_width": [64, 128, 256],

    # Maximum hop depth
    "max_hops": [2, 3],

    # A* heuristic weight (0 = pure beam, 1 = equal past/future)
    "alpha": [0.3, 0.5, 0.7],

    # Embedding score contribution when ranking neighbours
    "embedding_score_weight": [0.1, 0.3, 0.5],
}

# ---------------------------------------------------------------------------
# 2.  Prompt configuration variants
#     Each variant is a dict of keyword args passed to LLMPromptBuilder.
#     The builder must support these as constructor or call-time kwargs.
# ---------------------------------------------------------------------------

PROMPT_VARIANTS: Dict[str, Dict[str, Any]] = {
    # ── System instruction style ─────────────────────────────────────────────

    "sys_holistic": {
        "system_variant": "holistic",
        "description": (
            "Expert analyst — holistic judgment. Explains what each signal means and "
            "asks the LLM to reason like a skilled analyst, no fixed rules."
        ),
    },
    "sys_nuanced": {
        "system_variant": "nuanced",
        "description": (
            "Senior specialist — signal-weighted judgment. Provides explicit strength "
            "tiers (strong/moderate/against) without hard boolean gates."
        ),
    },
    "sys_conversational": {
        "system_variant": "conversational",
        "description": (
            "Knowledgeable-friend framing — accessible tone. Asks would a friend "
            "who knows this customer's taste recommend it?"
        ),
    },
    "sys_cot": {
        "system_variant": "cot",
        "description": (
            "Chain-of-thought guided — asks model to reason through 4 questions "
            "internally before giving the final digit."
        ),
    },

    # ── Entity description verbosity ─────────────────────────────────────────

    "entity_short": {
        "entity_description_length": "short",
        "description": "Short entity descriptions (title + category only, no features).",
    },
    "entity_medium": {
        "entity_description_length": "medium",
        "description": "Medium entity descriptions (title + brand + category + top 2 features).",
    },
    "entity_full": {
        "entity_description_length": "full",
        "description": "Full entity descriptions (all available metadata fields).",
    },

    # ── KG signal inclusion ──────────────────────────────────────────────────

    "signal_with_embed": {
        "include_embed_score": True,
        "description": "Include KG embedding compatibility score in prompt.",
    },
    "signal_without_embed": {
        "include_embed_score": False,
        "description": "Suppress KG embedding score (path structure only).",
    },

    # ── Path type labels ─────────────────────────────────────────────────────

    "labels_on": {
        "annotate_path_types": True,
        "description": "Annotate each path with its structural type (collaborative/brand/category/…).",
    },
    "labels_off": {
        "annotate_path_types": False,
        "description": "No path type labels — raw path text only.",
    },
}

# ---------------------------------------------------------------------------
# 3.  Predefined experiment configurations
#     Each experiment specifies one focused axis of variation,
#     keeping everything else at the default (baseline) values.
# ---------------------------------------------------------------------------

BASELINE_KG_PARAMS: Dict[str, Any] = {
    "top_k":                   5,
    "beam_width":              64,
    "max_hops":                3,
    "alpha":                   0.5,
    "embedding_score_weight":  0.3,
}

BASELINE_PROMPT_VARIANT: str = "sys_holistic"
BASELINE_ENTITY_LENGTH: str = "medium"

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # ── Exp 1: sweep top_k ───────────────────────────────────────────────────
    "exp01_top_k": {
        "description": "Sweep number of paths shown to the LLM (beam_width=128 fixed).",
        "sweep_axis": "top_k",
        "values": KG_PARAM_GRID["top_k"],
        "fixed_kg_params": {**BASELINE_KG_PARAMS, "beam_width": 128},
        "fixed_prompt_variant": BASELINE_PROMPT_VARIANT,
    },

    # ── Exp 2: sweep beam_width ──────────────────────────────────────────────
    "exp02_beam_width": {
        "description": "Sweep beam width (controls path diversity from search).",
        "sweep_axis": "beam_width",
        "values": KG_PARAM_GRID["beam_width"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": BASELINE_PROMPT_VARIANT,
    },

    # ── Exp 3: sweep max_hops ────────────────────────────────────────────────
    "exp03_max_hops": {
        "description": "Sweep maximum hop depth (2 vs 3).",
        "sweep_axis": "max_hops",
        "values": KG_PARAM_GRID["max_hops"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": BASELINE_PROMPT_VARIANT,
    },

    # ── Exp 4: sweep alpha ───────────────────────────────────────────────────
    "exp04_alpha": {
        "description": "Sweep A* heuristic weight (goal-directedness of search).",
        "sweep_axis": "alpha",
        "values": KG_PARAM_GRID["alpha"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": BASELINE_PROMPT_VARIANT,
    },

    # ── Exp 5: sweep embedding_score_weight ─────────────────────────────────
    "exp05_emb_weight": {
        "description": "Sweep embedding score contribution during neighbour ranking.",
        "sweep_axis": "embedding_score_weight",
        "values": KG_PARAM_GRID["embedding_score_weight"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": BASELINE_PROMPT_VARIANT,
    },

    # ── Exp 6: system prompt variants ────────────────────────────────────────
    "exp06_system_prompt": {
        "description": "Compare 4 system prompt styles (holistic/nuanced/conversational/cot).",
        "sweep_axis": "system_variant",
        "values": ["sys_holistic", "sys_nuanced", "sys_conversational", "sys_cot"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": None,   # values IS the variant key
    },

    # ── Exp 7: entity description verbosity ─────────────────────────────────
    "exp07_entity_length": {
        "description": "Compare short / medium / full entity description length.",
        "sweep_axis": "entity_description_length",
        "values": ["entity_short", "entity_medium", "entity_full"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": None,   # values IS the variant key
    },

    # ── Exp 8: KG embed signal inclusion ────────────────────────────────────
    "exp08_embed_signal": {
        "description": "Does surfacing the KG embedding score in the prompt help?",
        "sweep_axis": "include_embed_score",
        "values": ["signal_with_embed", "signal_without_embed"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": None,   # values IS the variant key
    },

    # ── Exp 9: path type labels ──────────────────────────────────────────────
    "exp09_path_labels": {
        "description": "Does annotating path structural types help the LLM?",
        "sweep_axis": "annotate_path_types",
        "values": ["labels_on", "labels_off"],
        "fixed_kg_params": BASELINE_KG_PARAMS,
        "fixed_prompt_variant": None,   # values IS the variant key
    },

    # ── Exp 10: joint best-known config ─────────────────────────────────────
    # Fill in after reading Exp 1-9 results.
    "exp10_best_joint": {
        "description": "Best-known combination from single-axis sweeps.",
        "sweep_axis": None,   # single config, no sweep
        "values": [None],
        "fixed_kg_params": {
            # ← Update these after sweeps 1-9
            "top_k":                   8,
            "beam_width":              128,
            "max_hops":                3,
            "alpha":                   0.5,
            "embedding_score_weight":  0.3,
        },
        "fixed_prompt_variant": "sys_balanced",
    },
}
