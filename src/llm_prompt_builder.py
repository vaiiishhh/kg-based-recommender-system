"""
llm_prompt_builder.py
=====================
Builds a deterministic, structured LLM prompt from standardised JSON paths
and entity context. Returns a {system, user} dict for proper role separation.
"""

import json
from typing import Dict, Any, List, Tuple
from kg_entity_context import KGEntityContextManager


class LLMPromptBuilder:
    """
    Builds a deterministic LLM prompt split into system and user messages.

    System message: fixed behavioral instructions (role, output format).
    User message:   query-specific content (user/item profiles, KG paths,
                    entity context).
    """

    # ── Fixed system prompt ─────────────────────────────────────────────
    SYSTEM_PROMPT = (
        "You are a product recommendation expert. "
        "Your task is to decide whether a given item should be recommended to a user, "
        "based on structured Knowledge Graph (KG) evidence.\n\n"

        "IMPORTANT BASELINE ASSUMPTION: "
        "The KG search algorithm only surfaces this item as a candidate because it already found "
        "structural connections between the user and the item. "
        "Therefore, when you see KG paths, your default stance should lean TOWARD recommending "
        "unless there is clear evidence against it.\n\n"

        "Evidence you will receive:\n"
        "- USER PROFILE: features/keywords associated with this user's purchase history\n"
        "- ITEM PROFILE: brand, category, and key features of the target item\n"
        "- RECOMMENDATION STRENGTH: a pre-computed signal (Strong / Moderate / Weak) based on "
        "path quality and embedding affinity — trust this signal heavily\n"
        "- KG SCORES: f_score (path traversal quality) and embed_score (latent user-item affinity, 0-1)\n"
        "- REASONING PATHS: graph evidence connecting user to item\n\n"

        "Decision rules:\n"
        "1. If Recommendation Strength is STRONG → output '1' unless the item category is "
        "completely unrelated to the user's history.\n"
        "2. If Recommendation Strength is MODERATE → output '1' if there is any category or "
        "brand overlap, or if co-purchase paths exist.\n"
        "3. If Recommendation Strength is WEAK → be more selective, but still lean '1' if the "
        "user profile shows some relevant interest.\n"
        "4. embed_score interpretation: above 0.45 = strong affinity, 0.3-0.45 = moderate, "
        "below 0.3 = weak.\n\n"

        "Remember: the cost of MISSING a good recommendation (false negative) is just as bad as "
        "making a wrong recommendation (false positive). Do not be overly conservative.\n\n"

        "Output exactly '1' to recommend or '0' to not recommend. "
        "Single character only — no spaces, punctuation, or explanation."
    )

    # ── Relation → natural language templates ─────────────────────────────────
    RELATION_TEMPLATES = {
        "purchase":        "User '{source}' has purchased Item '{target}'.",
        "rev_purchase":    "Item '{source}' was purchased by User '{target}'.",
        "mention":         "User '{source}' is associated with feature keyword '{target}'.",
        "rev_mention":     "Feature keyword '{source}' is associated with User '{target}'.",
        "described_by":    "Item '{source}' is described by feature '{target}'.",
        "rev_described":   "Feature '{source}' appears in the description of Item '{target}'.",
        "produced_by":     "Item '{source}' is manufactured by Brand '{target}'.",
        "rev_produced":    "Brand '{source}' manufactures Item '{target}'.",
        "belong_to":       "Item '{source}' belongs to Category '{target}'.",
        "rev_belong":      "Category '{source}' contains Item '{target}'.",
        "also_bought":     "Item '{source}' is frequently co-purchased with Item '{target}'.",
        "rev_also_bought": "Item '{source}' is frequently co-purchased with Item '{target}'.",
        "also_viewed":     "Item '{source}' is frequently co-viewed with Item '{target}'.",
        "rev_also_viewed": "Item '{source}' is frequently co-viewed with Item '{target}'.",
    }

    # Path structural types for annotation
    COLLABORATIVE_RELS = {"purchase", "rev_purchase"}
    BRAND_RELS = {"produced_by", "rev_produced"}
    CATEGORY_RELS = {"belong_to", "rev_belong"}
    FEATURE_RELS = {"described_by", "rev_described", "mention", "rev_mention"}
    COPURCHASE_RELS = {"also_bought", "rev_also_bought", "also_viewed", "rev_also_viewed"}

    def __init__(self, context_manager: KGEntityContextManager):
        self.context_manager = context_manager

    # ── Public API ────────────────────────────────────────────────────────────

    def build_prompt(self, paths_json: Dict[str, Any]) -> Dict[str, str]:
        """
        Builds the full prompt.

        Returns
        -------
        dict with keys:
            "system": str  — fixed behavioral instructions
            "user"  : str  — query-specific content
        """
        query = paths_json.get("query", {})
        query_user = query.get("user_id", "Unknown")
        query_item = query.get("item_id", "Unknown")

        user_section = self._build_user_section(query_user)
        item_section = self._build_item_section(query_item)
        strength_section = self._build_strength_section(paths_json)
        kg_score_section = self._build_kg_score_section(paths_json)
        paths_section = self._build_paths_section(paths_json, query_user, query_item)
        context_section = self._build_context_section(paths_json, query_user, query_item)

        user_message = "\n\n".join([
            f"=== Target User: {query_user} ===\n{user_section}",
            f"=== Target Item: {query_item} ===\n{item_section}",
            strength_section,
            kg_score_section,
            paths_section,
            context_section,
            "Based on all evidence above, should this item be recommended? Output only '1' (yes) or '0' (no):",
        ])

        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_message,
        }

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_user_section(self, user_id: str) -> str:
        return self.context_manager.get_entity_description("user", user_id)

    def _build_item_section(self, item_id: str) -> str:
        return self.context_manager.get_entity_description("item", item_id)

    def _build_strength_section(self, paths_json: Dict[str, Any]) -> str:
        """Computes a pre-digested Recommendation Strength label to guide LLM."""
        paths = paths_json.get("paths", [])
        if not paths:
            return "=== Recommendation Strength ===\nWEAK — No KG paths found."

        best_embed = max(
            (p.get("scores", {}).get("embed_score", 0.0) for p in paths),
            default=0.0,
        )
        # Check if any path reaches the target item directly (h_score ≈ 1.0)
        direct_hit = any(
            p.get("scores", {}).get("h_score", 0.0) >= 0.95 for p in paths
        )
        # Check for collaborative paths (purchase → rev_purchase pattern)
        has_collaborative = any(
            any(e.get("relation_name") in ("purchase", "rev_purchase")
                for e in p.get("edges", []))
            for p in paths
        )
        # Check for co-purchase paths
        has_copurchase = any(
            any(e.get("relation_name") in ("also_bought", "rev_also_bought")
                for e in p.get("edges", []))
            for p in paths
        )

        # Determine strength tier
        if (direct_hit and best_embed >= 0.45) or best_embed >= 0.60:
            strength = "STRONG"
            rationale = "direct KG path to target item with high embedding affinity"
        elif (direct_hit and best_embed >= 0.30) or (has_collaborative and best_embed >= 0.35) or best_embed >= 0.45:
            strength = "MODERATE"
            rationale = f"good path evidence (direct={direct_hit}, collaborative={has_collaborative}, embed={best_embed:.2f})"
        elif has_copurchase or has_collaborative or best_embed >= 0.25:
            strength = "MODERATE"
            rationale = f"co-purchase or collaborative signals present (embed={best_embed:.2f})"
        else:
            strength = "WEAK"
            rationale = f"limited path evidence (embed={best_embed:.2f})"

        return (
            f"=== Recommendation Strength ===\n"
            f"{strength} — {rationale}"
        )

    def _build_kg_score_section(self, paths_json: Dict[str, Any]) -> str:
        paths = paths_json.get("paths", [])
        if not paths:
            return "=== KG Compatibility ===\nNo reasoning paths found."

        # Use the best embed_score across all paths
        best_embed = max(
            (p.get("scores", {}).get("embed_score", 0.0) for p in paths),
            default=0.0,
        )
        best_f = max(
            (p.get("scores", {}).get("f_score", 0.0) for p in paths),
            default=0.0,
        )
        level = (
            "strong" if best_embed >= 0.45
            else "moderate" if best_embed >= 0.30
            else "weak"
        )
        return (
            f"=== KG Compatibility Signal ===\n"
            f"Best path f_score: {best_f:.4f}  |  "
            f"KG embed_score: {best_embed:.4f}/1.00 ({level})"
        )

    def _path_structural_type(self, edges: List[Dict]) -> str:
        """Classify the structural type of a path."""
        rel_names = {e.get("relation_name", "") for e in edges}
        types = []
        if rel_names & self.COLLABORATIVE_RELS:
            types.append("collaborative")
        if rel_names & self.BRAND_RELS:
            types.append("brand")
        if rel_names & self.CATEGORY_RELS:
            types.append("category")
        if rel_names & self.FEATURE_RELS:
            types.append("feature")
        if rel_names & self.COPURCHASE_RELS:
            types.append("co-purchase")
        return "+".join(types) if types else "unknown"

    def _edge_to_text(self, edge: Dict, source_name: str, target_name: str) -> str:
        rel_name = edge.get("relation_name", "")
        template = self.RELATION_TEMPLATES.get(
            rel_name, "{source} is connected to {target} via " + rel_name + "."
        )
        return template.format(source=source_name, target=target_name)

    def _build_paths_section(
        self, paths_json: Dict[str, Any], query_user: str, query_item: str
    ) -> str:
        paths = sorted(paths_json.get("paths", []), key=lambda x: x.get("rank", 0))
        lines = ["=== Reasoning Paths ==="]

        if not paths:
            lines.append("No paths found between this user and item in the KG.")
            return "\n".join(lines)

        for path in paths:
            nodes = path.get("nodes", [])
            edges = path.get("edges", [])
            scores = path.get("scores", {})
            path_type = self._path_structural_type(edges)

            f_score = scores.get("f_score", 0.0)
            embed_score = scores.get("embed_score", 0.0)

            lines.append(
                f"Path {path['rank']} "
                f"[type: {path_type}] "
                f"(f={f_score:.4f}, embed={embed_score:.4f}):"
            )

            sentences = []
            for i, edge in enumerate(edges):
                src = nodes[i]["name"]
                tgt = nodes[i + 1]["name"]
                sentences.append(self._edge_to_text(edge, src, tgt))
            lines.append("  " + " ".join(sentences))

        return "\n".join(lines)

    def _build_context_section(
        self, paths_json: Dict[str, Any], query_user: str, query_item: str
    ) -> str:
        """Describes intermediate entities only (excludes target user/item)."""
        entities: Dict = {}
        for path in paths_json.get("paths", []):
            for node in path.get("nodes", []):
                name = node["name"]
                # Skip the target user and target item — they have dedicated sections
                if name in (query_user, query_item):
                    continue
                key = (node["type"], node["global_id"], name)
                entities[key] = node

        if not entities:
            return "=== Supporting Entity Context ===\nNo intermediate entities."

        sorted_keys = sorted(entities.keys())
        lines = ["=== Supporting Entity Context ==="]
        for key in sorted_keys:
            node = entities[key]
            desc = self.context_manager.get_entity_description(node["type"], node["name"])
            lines.append(f"[{node['type'].capitalize()}] {node['name']}: {desc}")

        return "\n".join(lines)
