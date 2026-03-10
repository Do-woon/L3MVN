"""Unit tests for SemanticTaxonomy (envs/igibson/semantic_taxonomy.py)."""

import numpy as np
import pytest

from envs.igibson.semantic_taxonomy import SemanticTaxonomy


# ---------------------------------------------------------------------------
# Test 1: known class mapping
# ---------------------------------------------------------------------------

class TestKnownClassMapping:
    """Verify that each L3MVN category keyword maps to the expected index."""

    @pytest.mark.parametrize("name,expected", [
        # chair (1)
        ("chair",       1),
        ("armchair",    1),
        ("highchair",   1),
        ("stool",       1),
        ("bench",       1),
        # sofa (2)
        ("sofa",        2),
        ("couch",       2),
        ("loveseat",    2),
        # plant (3)
        ("plant",       3),
        ("potted plant", 3),
        ("flower",      3),
        # bed (4)
        ("bed",         4),
        # toilet (5)
        ("toilet",      5),
        # tv_monitor (6)
        ("tv",          6),
        ("television",  6),
        ("monitor",     6),
        ("screen",      6),
        # bathtub (7)
        ("bathtub",     7),
        ("tub",         7),
        # shower (8)
        ("shower",      8),
        # fireplace (9)
        ("fireplace",   9),
        # appliances (10)
        ("refrigerator", 10),
        ("fridge",      10),
        ("microwave",   10),
        ("oven",        10),
        ("dishwasher",  10),
        ("washer",      10),
        ("dryer",       10),
        ("stove",       10),
        ("cooktop",     10),
        # towel (11)
        ("towel",       11),
        # sink (12)
        ("sink",        12),
        ("basin",       12),
        # chest_of_drawers (13)
        ("dresser",     13),
        ("chest_of_drawers", 13),
        ("drawer",      13),
        # table (14)
        ("table",       14),
        ("desk",        14),
        ("counter",     14),
        ("coffee_table",  14),
        ("dining_table",  14),
        # stairs (15)
        ("stairs",      15),
        ("stair",       15),
    ])
    def test_keyword(self, name, expected):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index(name) == expected

    def test_case_insensitive_chair(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index("CHAIR") == 1

    def test_case_insensitive_sofa(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index("Sofa") == 2

    def test_igibson_style_armchair(self):
        # iGibson names often carry ".n.01" suffixes
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index("armchair.n.01") == 1

    def test_igibson_style_dining_table(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index("dining_table.n.01") == 14


# ---------------------------------------------------------------------------
# Test 2: unknown / background class mapping
# ---------------------------------------------------------------------------

class TestUnknownClassMapping:
    """Unmapped, empty, and None class names must all return channel 0."""

    @pytest.mark.parametrize("name", [
        "wall",
        "floor",
        "ceiling",
        "robot",
        "door",
        "window",
        "cabinet",   # not in any category keyword list
        "",
    ])
    def test_returns_zero(self, name):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index(name) == 0

    def test_none_returns_zero(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_index(None) == 0


# ---------------------------------------------------------------------------
# Test 3: mixed semantic id map
# ---------------------------------------------------------------------------

class TestMixedSemanticIdMap:
    """Verify shape, per-pixel channel assignment, and background channel."""

    # semantic_id_map layout (3, 4):
    #   id 1 → "chair"  → channel  1
    #   id 2 → "couch"  → channel  2
    #   id 3 → "wall"   → channel  0  (unknown)
    #   id 4 → "sink"   → channel 12

    SEM_MAP = np.array([
        [1, 2, 3, 4],
        [3, 1, 4, 2],
        [4, 3, 2, 1],
    ], dtype=np.int32)

    CLASS_ID_TO_NAME = {1: "chair", 2: "couch", 3: "wall", 4: "sink"}

    @pytest.fixture(autouse=True)
    def build_one_hot(self):
        self.one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(
            self.SEM_MAP, self.CLASS_ID_TO_NAME
        )

    def test_output_shape(self):
        assert self.one_hot.shape == (16, 3, 4)

    def test_chair_channel(self):
        # id=1 ("chair") at positions (0,0), (1,1), (2,3)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(self.one_hot[1], expected)

    def test_sofa_channel(self):
        # id=2 ("couch") at positions (0,1), (1,3), (2,2)
        expected = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(self.one_hot[2], expected)

    def test_unknown_to_channel0(self):
        # id=3 ("wall") at positions (0,2), (1,0), (2,1)
        expected = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(self.one_hot[0], expected)

    def test_sink_channel(self):
        # id=4 ("sink") at positions (0,3), (1,2), (2,0)
        expected = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(self.one_hot[12], expected)

    def test_unmapped_channels_are_zero(self):
        # Every channel except 0, 1, 2, 12 should be all-zeros
        used = {0, 1, 2, 12}
        for c in range(16):
            if c not in used:
                np.testing.assert_array_equal(
                    self.one_hot[c],
                    np.zeros((3, 4), dtype=np.uint8),
                    err_msg=f"channel {c} should be all zeros",
                )


# ---------------------------------------------------------------------------
# Test 4: one-hot integrity
# ---------------------------------------------------------------------------

class TestOneHotIntegrity:
    """Every pixel must have exactly one channel active (sum == 1)."""

    def test_channel_sum_is_one_sparse_map(self):
        # Only ids 1 and 2 are named; ids 3–9 are unmapped → go to channel 0.
        class_id_to_name = {1: "chair", 2: "sofa"}
        rng = np.random.default_rng(42)
        sem_map = rng.integers(0, 10, size=(8, 8), dtype=np.int32)
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(sem_map, class_id_to_name)
        channel_sums = one_hot.sum(axis=0)
        np.testing.assert_array_equal(channel_sums, np.ones((8, 8), dtype=np.uint8))

    def test_channel_sum_is_one_all_known(self):
        # Every id in the map is explicitly named.
        class_id_to_name = {0: "wall", 1: "bed", 2: "toilet", 3: "sink"}
        sem_map = np.array([[0, 1], [2, 3]], dtype=np.int32)
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(sem_map, class_id_to_name)
        channel_sums = one_hot.sum(axis=0)
        np.testing.assert_array_equal(channel_sums, np.ones((2, 2), dtype=np.uint8))

    def test_dtype_is_uint8(self):
        sem_map = np.zeros((4, 4), dtype=np.int32)
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(sem_map, {})
        assert one_hot.dtype == np.uint8

    def test_num_channels_constant(self):
        assert SemanticTaxonomy.NUM_CHANNELS == 16

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            SemanticTaxonomy.semantic_id_map_to_one_hot(
                np.zeros((4, 4, 4), dtype=np.int32), {}
            )
