"""Semantic taxonomy: deterministic curated mapping from iGibson classes
to L3MVN semantic IDs.

Final responsibility split (L3MVN-compatible)
---------------------------------------------
- Adapter layer (EnvWrapper/ObsAdapter) outputs Stage-1 semantic *id*
  single-channel.
- Stage-2 one-hot(16ch) generation belongs to
  ``Sem_Exp_Env_Agent._preprocess_obs()``, not this module.

This module focuses on:
1) class-name -> L3MVN semantic id mapping
2) iGibson semantic-id map -> L3MVN semantic-id map remapping
"""

from __future__ import annotations

import re

import numpy as np

# L3MVN semantic id space used before Stage-2 one-hot conversion:
# 0: unknown/background, 1..15: named categories, 16: optional extra.
NUM_SEMANTIC_CHANNELS: int = 16
BACKGROUND_SEMANTIC_ID: int = 0

L3MVN_CATEGORY_TO_ID: dict[str, int] = {
    "chair": 1,
    "sofa": 2,
    "plant": 3,
    "bed": 4,
    "toilet": 5,
    "tv_monitor": 6,
    "bathtub": 7,
    "shower": 8,
    "fireplace": 9,
    "appliances": 10,
    "towel": 11,
    "sink": 12,
    "chest_of_drawers": 13,
    "table": 14,
    "stairs": 15,
}

# Curated alias table (deterministic, manual mapping).
# Ambiguous classes are intentionally omitted and remain unmapped (0).
L3MVN_CATEGORY_TO_ALIASES: dict[str, tuple[str, ...]] = {
    "chair": (
        "chair",
        "armchair",
        "office_chair",
        "folding_chair",
        "straight_chair",
        "swivel_chair",
        "rocking_chair",
        "stool",
        "bench",
        "highchair",
    ),
    "sofa": (
        "sofa",
        "chaise_longue",
        "couch",
        "loveseat",
        "sectional",
        "futon",
    ),
    "plant": (
        "plant",
        "potted_plant",
        "pot_plant",
        "flower",
        "vase_plant",
    ),
    "bed": (
        "bed",
        "bunk_bed",
        "crib",
    ),
    "toilet": ("toilet",),
    "tv_monitor": (
        "tv",
        "television",
        "monitor",
        "standing_tv",
        "wall_mounted_tv",
        "screen",
    ),
    "bathtub": (
        "bathtub",
        "tub",
    ),
    "shower": ("shower",),
    "fireplace": ("fireplace",),
    "appliances": (
        "refrigerator",
        "fridge",
        "microwave",
        "oven",
        "dishwasher",
        "washer",
        "dryer",
        "stove",
        "griddle",
        "grill",
        "heater",
        "iron",
        "kettle",
        "range_hood",
        "toaster",
        "vacuum",
        "burner",
        "blender",
        "coffee_maker",
        "cooktop",
    ),
    "towel": (
        "towel",
        "bath_towel",
        "hand_towel",
        "dishtowel",
        "rag",
    ),
    "sink": (
        "sink",
        "basin",
        "bathroom_sink",
        "kitchen_sink",
    ),
    "chest_of_drawers": (
        "dresser",
        "drawer",
        "chest_of_drawers",
        "bureau",
        "cabinet_dresser",
    ),
    "table": (
        "table",
        "breakfast_table",
        "dining_table",
        "coffee_table",
        "side_table",
        "console_table",
        "desk",
        "gaming_table",
        "pedestal_table",
        "pool_table",
        "counter",
        "countertop",
        "kitchen_counter",
    ),
    "stairs": (
        "stair",
        "stairs",
    ),
}

# Explicitly conservative classes intentionally left unmapped.
_EXPLICIT_UNMAPPED_ALIASES: set[str] = {
    "cabinet",
    "bookshelf",
    "bookcase",
    "shelf",
    "rack",
    "tv_stand",
    "ottoman",
}


def _normalize_class_name(class_name: str | None) -> str:
    """Normalize iGibson class name for deterministic matching.

    Examples
    --------
    - "armchair.n.01" -> "armchair"
    - "Dining Table" -> "dining_table"
    - "kitchen-counter-01" -> "kitchen_counter"
    """
    if not class_name:
        return ""
    name = class_name.strip().lower()
    # Strip WordNet style suffixes used in some datasets (e.g. ".n.01").
    name = re.sub(r"\.[a-z]\.\d+$", "", name)
    # Keep only alnum separators as underscores.
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    # Drop common numeric postfixes (e.g. "_01", "_2").
    name = re.sub(r"(?:_\d+)+$", "", name)
    return name


def _build_alias_to_semantic_id() -> dict[str, int]:
    alias_to_sem_id: dict[str, int] = {}
    for category, aliases in L3MVN_CATEGORY_TO_ALIASES.items():
        sem_id = L3MVN_CATEGORY_TO_ID[category]
        for alias in aliases:
            norm_alias = _normalize_class_name(alias)
            prev = alias_to_sem_id.get(norm_alias)
            if prev is not None and prev != sem_id:
                raise ValueError(
                    f"Alias collision: '{norm_alias}' maps to both {prev} and {sem_id}"
                )
            alias_to_sem_id[norm_alias] = sem_id
    return alias_to_sem_id


_ALIAS_TO_SEMANTIC_ID: dict[str, int] = _build_alias_to_semantic_id()

# Minimal fallback: only for multi-token curated aliases (e.g. dining_table),
# matched against contiguous token windows.
_MULTI_TOKEN_ALIAS_RULES: list[tuple[tuple[str, ...], int]] = []
for alias, sem_id in _ALIAS_TO_SEMANTIC_ID.items():
    parts = tuple(p for p in alias.split("_") if p)
    if len(parts) >= 2:
        _MULTI_TOKEN_ALIAS_RULES.append((parts, sem_id))
_MULTI_TOKEN_ALIAS_RULES.sort(key=lambda x: (-len(x[0]), x[0], x[1]))


def _contains_contiguous_subsequence(tokens: tuple[str, ...], alias_tokens: tuple[str, ...]) -> bool:
    n = len(tokens)
    m = len(alias_tokens)
    if m == 0 or m > n:
        return False
    for i in range(0, n - m + 1):
        if tokens[i : i + m] == alias_tokens:
            return True
    return False


class SemanticTaxonomy:
    """Maps iGibson semantic information to L3MVN semantic IDs."""

    NUM_CHANNELS: int = NUM_SEMANTIC_CHANNELS

    @staticmethod
    def map_class_name_to_l3mvn_semantic_id(class_name: str | None) -> int:
        """Return L3MVN semantic id (0..15) for *class_name*."""
        norm_name = _normalize_class_name(class_name)
        if not norm_name:
            return BACKGROUND_SEMANTIC_ID

        # Explicit conservative unmapped list has priority.
        if norm_name in _EXPLICIT_UNMAPPED_ALIASES:
            return BACKGROUND_SEMANTIC_ID

        # 1) Exact curated alias match.
        sem_id = _ALIAS_TO_SEMANTIC_ID.get(norm_name)
        if sem_id is not None:
            return sem_id

        # 2) Minimal fallback: contiguous multi-token alias match.
        tokens = tuple(t for t in norm_name.split("_") if t)
        for alias_tokens, alias_sem_id in _MULTI_TOKEN_ALIAS_RULES:
            if _contains_contiguous_subsequence(tokens, alias_tokens):
                return alias_sem_id

        return BACKGROUND_SEMANTIC_ID

    @staticmethod
    def build_id_to_l3mvn_semantic_id(
        class_id_to_name: dict[int, str],
    ) -> dict[int, int]:
        """Build iGibson semantic-id -> L3MVN semantic-id lookup table."""
        return {
            cid: SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(name)
            for cid, name in class_id_to_name.items()
        }

    @staticmethod
    def remap_semantic_id_map(
        semantic_id_map: np.ndarray,
        class_id_to_name: dict[int, str],
    ) -> np.ndarray:
        """Convert iGibson semantic-id map to L3MVN semantic-id map."""
        if semantic_id_map.ndim != 2:
            raise ValueError(
                f"semantic_id_map must be 2-D (H, W), got shape {semantic_id_map.shape}"
            )

        id_to_sem_id = SemanticTaxonomy.build_id_to_l3mvn_semantic_id(class_id_to_name)
        remapped = np.zeros_like(semantic_id_map, dtype=np.int32)
        for class_id, sem_id in id_to_sem_id.items():
            remapped[semantic_id_map == class_id] = sem_id
        return remapped
