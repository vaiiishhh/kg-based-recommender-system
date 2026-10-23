"""
kg_entity_context.py
====================
Generates natural language descriptions for every entity type in the KG.
Loads extracted metadata mappings directly from kg_data/ directory.
"""

import os
import json
from collections import defaultdict
from typing import Dict, Any, List, Optional


class KGEntityContextManager:
    """
    Extracts and caches entity metadata from the KG data directory.
    Provides deterministic natural language descriptions for all 5 entity types.

    Parameters
    ----------
    data_dir    : path to kg_data/ (contains user2id.json, item_cats.json, etc.)
    user2id     : optional pre-loaded user2id dict (injected from extractor)
    item2id     : optional pre-loaded item2id dict (injected from extractor)
    """

    def __init__(
        self,
        data_dir: str = "kg_data/",
        raw_data_dir: str = None, # Left for backward compatibility, ignored
        user2id: Optional[Dict[str, int]] = None,
        item2id: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = data_dir

        # ID mappings — accept injected or load from disk
        self.user2id: Dict[str, int] = user2id or {}
        self.item2id: Dict[str, int] = item2id or {}

        # Dictionaries to hold metadata
        self.user_mention: Dict[str, List[str]] = {}
        self.item_cats: Dict[str, List[str]] = {}
        self.item_features: Dict[str, List[str]] = {}
        self.item_brand: Dict[str, str] = {}
        self.item_interactions: Dict[str, int] = defaultdict(int)

        self._load_kg_artefacts()
        self._load_metadata()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_kg_artefacts(self):
        """Load id mappings from kg_data/ if not already injected."""
        print("[EntityContext] Loading KG artefacts...")

        def _lj(fname):
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return {}

        if not self.user2id:
            self.user2id = _lj("user2id.json")
        if not self.item2id:
            self.item2id = _lj("item2id.json")

    def _load_metadata(self):
        """Load entity metadata (features, categories, brands) from kg_data/."""
        print(f"[EntityContext] Loading metadata from {self.data_dir!r}...")

        def _lj(fname):
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    return {}
            return {}

        self.user_mention = _lj("user_mention.json")
        self.item_cats = _lj("item_cats.json")
        self.item_features = _lj("item_features.json")
        self.item_brand = _lj("item_brand.json")
        
        # Optionally, load interactions to compute popularity if needed
        # Just simple counting from pos_test or train is omitted since we can't reliably load csv here without pandas dependency or logic.
        # So we leave item_interactions empty unless explicitly set.

        print(f"[EntityContext] Ready. "
              f"{len(self.item_cats)} items with categories, "
              f"{len(self.user_mention)} users with feature contexts.")

    # ── Entity description generators ─────────────────────────────────────────

    def get_user_description(self, user_id: str) -> str:
        """
        Describes a user based on the feature words they frequently mention or associate with.
        """
        mentions = self.user_mention.get(user_id, [])
        if not mentions:
            return "A user profile with no specific feature preferences recorded."

        # Keep top 10 feature mentions for context
        top_mentions = mentions[:10]
        feat_str = ", ".join(f"'{m}'" for m in top_mentions)
        
        return f"This user is frequently associated with the following product features/keywords: {feat_str}."

    def get_item_description(self, item_asin: str) -> str:
        """
        Describes an item based on its categories, brand, and feature keywords.
        """
        cats = self.item_cats.get(item_asin, [])
        brand = self.item_brand.get(item_asin, "Unknown brand")
        features = self.item_features.get(item_asin, [])

        cat_str = ""
        if cats:
            # Join top categories with ' > '
            cat_str = f" Categories: {' > '.join(cats[:3])}."

        feat_str = ""
        if features:
            # Use top 5 features
            feat_parts = [f for f in features[:5] if isinstance(f, str) and f.strip()]
            if feat_parts:
                feat_str = f" Features: {', '.join(feat_parts)}."

        return f"Item '{item_asin}' by {brand}.{cat_str}{feat_str}"

    def get_brand_description(self, brand_name: str) -> str:
        return f"Manufacturer/brand: '{brand_name}'."

    def get_category_description(self, category_name: str) -> str:
        return f"Product category: '{category_name}'."

    def get_feature_description(self, feature_name: str) -> str:
        return f"Product attribute/feature keyword: '{feature_name}'."

    def get_entity_description(self, entity_type: str, entity_name: str) -> str:
        """Deterministic dispatcher for all entity types."""
        if entity_type == "user":
            return self.get_user_description(entity_name)
        elif entity_type == "item":
            return self.get_item_description(entity_name)
        elif entity_type == "brand":
            return self.get_brand_description(entity_name)
        elif entity_type == "category":
            return self.get_category_description(entity_name)
        elif entity_type == "feature":
            return self.get_feature_description(entity_name)
        else:
            return f"Unknown entity type '{entity_type}'."

