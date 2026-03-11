"""Semantic taxonomy: maps iGibson class names / semantic ids
to L3MVN semantic IDs.

Final responsibility split (L3MVN-compatible)
---------------------------------------------
- Adapter layer (EnvWrapper/ObsAdapter) should output Stage-1 semantic *id*
  single-channel.
- Stage-2 one-hot(16ch) generation belongs to
  ``Sem_Exp_Env_Agent._preprocess_obs()``, not this module.

This module therefore focuses on:
1) class-name -> L3MVN semantic id mapping
2) iGibson semantic-id map -> L3MVN semantic-id map remapping
"""

from __future__ import annotations

import numpy as np

# L3MVN semantic id space used before Stage-2 one-hot conversion:
# 0: unknown/background, 1..15: named categories, 16: optional extra.
NUM_SEMANTIC_CHANNELS: int = 16
BACKGROUND_SEMANTIC_ID: int = 0

# ---------------------------------------------------------------------------
# Keyword → L3MVN channel index rules
# ---------------------------------------------------------------------------
# Matching is substring-based on the lowercased class name.
# Longer keywords are checked before shorter ones so that more-specific strings
# win (e.g. "bathtub" before "tub", "dining_table" before "table").
# Within the same length, ties are resolved alphabetically for determinism.

_KEYWORD_RULES: list[tuple[str, int]] = [
    # 1 – chair
    ("highchair", 1),
    ("armchair", 1),
    ("chair", 1),
    ("stool", 1),
    ("bench", 1),
    # 2 – sofa
    ("loveseat", 2),
    ("couch", 2),
    ("sofa", 2),
    # 3 – plant
    ("potted", 3),
    ("flower", 3),
    ("plant", 3),
    # 4 – bed
    ("bed", 4),
    # 5 – toilet
    ("toilet", 5),
    # 6 – tv_monitor
    ("television", 6),
    ("monitor", 6),
    ("screen", 6),
    ("tv", 6),
    # 7 – bathtub
    ("bathtub", 7),
    ("tub", 7),
    # 8 – shower
    ("shower", 8),
    # 9 – fireplace
    ("fireplace", 9),
    # 10 – appliances
    ("refrigerator", 10),
    ("dishwasher", 10),
    ("microwave", 10),
    ("cooktop", 10),
    ("washer", 10),
    ("dryer", 10),
    ("fridge", 10),
    ("stove", 10),
    ("oven", 10),
    # 11 – towel
    ("towel", 11),
    # 12 – sink
    ("basin", 12),
    ("sink", 12),
    # 13 – chest_of_drawers
    ("chest_of_drawers", 13),
    ("dresser", 13),
    ("drawer", 13),
    # 14 – table
    ("coffee_table", 14),
    ("dining_table", 14),
    ("counter", 14),
    ("table", 14),
    ("desk", 14),
    # 15 – stairs
    ("stairs", 15),
    ("stair", 15),
]

# Pre-sorted: longer keywords first; alphabetical within same length.
_SORTED_RULES: list[tuple[str, int]] = sorted(
    _KEYWORD_RULES, key=lambda kv: (-len(kv[0]), kv[0])
)


class SemanticTaxonomy:
    """Maps iGibson semantic information to L3MVN semantic IDs.

    All public methods are static so the class can be used without
    instantiation and reused directly inside wrappers/adapters.
    """

    NUM_CHANNELS: int = NUM_SEMANTIC_CHANNELS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def map_class_name_to_l3mvn_semantic_id(class_name: str | None) -> int:
        """Return L3MVN semantic id (0..15) for *class_name*.

        Parameters
        ----------
        class_name : str or None
            iGibson object class name (e.g. ``"armchair.n.01"``).

        Returns
        -------
        int
            1–15 for a known L3MVN category, 0 for unknown/background.
        """
        if not class_name:
            return 0
        name_lower = class_name.lower()
        for keyword, idx in _SORTED_RULES:
            if keyword in name_lower:
                return idx
        return 0

    @staticmethod
    def build_id_to_l3mvn_semantic_id(
        class_id_to_name: dict[int, str],
    ) -> dict[int, int]:
        """Build iGibson semantic-id -> L3MVN semantic-id lookup table.

        Parameters
        ----------
        class_id_to_name : dict[int, str]
            iGibson GT id→name dictionary (e.g. from the scene's object list).

        Returns
        -------
        dict[int, int]
            id -> L3MVN semantic id (0 = unknown/background, 1..15 named).
        """
        return {
            cid: SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(name)
            for cid, name in class_id_to_name.items()
        }

    @staticmethod
    def remap_semantic_id_map(
        semantic_id_map: np.ndarray,
        class_id_to_name: dict[int, str],
    ) -> np.ndarray:
        """Convert iGibson semantic-id map to L3MVN semantic-id map.

        Parameters
        ----------
        semantic_id_map : np.ndarray, shape (H, W), integer dtype
            Per-pixel iGibson semantic class ids.
        class_id_to_name : dict[int, str]
            iGibson id -> class name dictionary.

        Returns
        -------
        np.ndarray, shape (H, W), dtype int32
            L3MVN semantic id map in {0..15}, where 0=unknown/background.
        """
        if semantic_id_map.ndim != 2:
            raise ValueError(
                f"semantic_id_map must be 2-D (H, W), got shape {semantic_id_map.shape}"
            )

        id_to_sem_id = SemanticTaxonomy.build_id_to_l3mvn_semantic_id(class_id_to_name)
        remapped = np.zeros_like(semantic_id_map, dtype=np.int32)
        for class_id, sem_id in id_to_sem_id.items():
            remapped[semantic_id_map == class_id] = sem_id
        return remapped
