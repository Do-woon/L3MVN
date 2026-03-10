"""Semantic taxonomy: maps iGibson class names / semantic-id maps
to L3MVN's 16-channel one-hot representation.

Channel layout
--------------
  0  : background / unmapped / unknown
  1  : chair
  2  : sofa
  3  : plant
  4  : bed
  5  : toilet
  6  : tv_monitor
  7  : bathtub
  8  : shower
  9  : fireplace
  10 : appliances
  11 : towel
  12 : sink
  13 : chest_of_drawers
  14 : table
  15 : stairs
"""

from __future__ import annotations

import numpy as np

NUM_CHANNELS: int = 16  # channel 0 = background + 15 L3MVN categories

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
    """Maps iGibson semantic information to L3MVN's 16-channel one-hot format.

    All public methods are static so the class can be used without
    instantiation and reused directly inside an ObsAdapter.
    """

    NUM_CHANNELS: int = NUM_CHANNELS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def map_class_name_to_l3mvn_index(class_name: str | None) -> int:
        """Return the L3MVN channel index (0–15) for *class_name*.

        Parameters
        ----------
        class_name : str or None
            iGibson object class name (e.g. ``"armchair.n.01"``).

        Returns
        -------
        int
            1–15 for a known L3MVN category, 0 for background / unknown.
        """
        if not class_name:
            return 0
        name_lower = class_name.lower()
        for keyword, idx in _SORTED_RULES:
            if keyword in name_lower:
                return idx
        return 0

    @staticmethod
    def build_id_to_l3mvn_index(class_id_to_name: dict[int, str]) -> dict[int, int]:
        """Build a mapping from iGibson semantic-id to L3MVN channel index.

        Parameters
        ----------
        class_id_to_name : dict[int, str]
            iGibson GT id→name dictionary (e.g. from the scene's object list).

        Returns
        -------
        dict[int, int]
            id → L3MVN channel index (0 = background).
        """
        return {
            cid: SemanticTaxonomy.map_class_name_to_l3mvn_index(name)
            for cid, name in class_id_to_name.items()
        }

    @staticmethod
    def semantic_id_map_to_one_hot(
        semantic_id_map: np.ndarray,
        class_id_to_name: dict[int, str],
    ) -> np.ndarray:
        """Convert a semantic id map to a 16-channel one-hot array.

        Parameters
        ----------
        semantic_id_map : np.ndarray, shape (H, W), integer dtype
            Per-pixel iGibson semantic class ids.
        class_id_to_name : dict[int, str]
            iGibson id → class name dictionary.

        Returns
        -------
        np.ndarray, shape (16, H, W), dtype uint8
            One-hot encoded semantic map.  Channel 0 is background/unknown.
            Every pixel has exactly one channel set to 1.
        """
        if semantic_id_map.ndim != 2:
            raise ValueError(
                f"semantic_id_map must be 2-D (H, W), got shape {semantic_id_map.shape}"
            )
        h, w = semantic_id_map.shape
        id_to_idx = SemanticTaxonomy.build_id_to_l3mvn_index(class_id_to_name)

        # Every pixel starts at channel 0 (background).
        # Known ids overwrite their pixels with the appropriate channel index.
        channel_map = np.zeros((h, w), dtype=np.int32)
        for cid, idx in id_to_idx.items():
            channel_map[semantic_id_map == cid] = idx

        one_hot = np.zeros((NUM_CHANNELS, h, w), dtype=np.uint8)
        for c in range(NUM_CHANNELS):
            one_hot[c] = channel_map == c

        return one_hot
