"""Unit tests for SemanticTaxonomy (envs/igibson/semantic_taxonomy.py)."""

import numpy as np
import pytest

from envs.igibson.semantic_taxonomy import SemanticTaxonomy


class TestKnownClassMapping:
    """Verify curated alias mapping to L3MVN semantic ids."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("chair", 1),
            ("armchair", 1),
            ("office_chair", 1),
            ("folding_chair", 1),
            ("straight_chair", 1),
            ("swivel_chair", 1),
            ("rocking_chair", 1),
            ("highchair", 1),
            ("stool", 1),
            ("bench", 1),
            ("sofa", 2),
            ("chaise_longue", 2),
            ("couch", 2),
            ("loveseat", 2),
            ("sectional", 2),
            ("futon", 2),
            ("plant", 3),
            ("potted_plant", 3),
            ("pot_plant", 3),
            ("vase_plant", 3),
            ("flower", 3),
            ("bed", 4),
            ("bunk_bed", 4),
            ("crib", 4),
            ("toilet", 5),
            ("tv", 6),
            ("television", 6),
            ("monitor", 6),
            ("standing_tv", 6),
            ("wall_mounted_tv", 6),
            ("screen", 6),
            ("bathtub", 7),
            ("tub", 7),
            ("shower", 8),
            ("fireplace", 9),
            ("refrigerator", 10),
            ("fridge", 10),
            ("microwave", 10),
            ("oven", 10),
            ("dishwasher", 10),
            ("washer", 10),
            ("dryer", 10),
            ("stove", 10),
            ("griddle", 10),
            ("grill", 10),
            ("heater", 10),
            ("iron", 10),
            ("kettle", 10),
            ("range_hood", 10),
            ("toaster", 10),
            ("vacuum", 10),
            ("burner", 10),
            ("blender", 10),
            ("coffee_maker", 10),
            ("cooktop", 10),
            ("towel", 11),
            ("bath_towel", 11),
            ("hand_towel", 11),
            ("dishtowel", 11),
            ("rag", 11),
            ("sink", 12),
            ("basin", 12),
            ("bathroom_sink", 12),
            ("kitchen_sink", 12),
            ("dresser", 13),
            ("chest_of_drawers", 13),
            ("drawer", 13),
            ("bureau", 13),
            ("cabinet_dresser", 13),
            ("table", 14),
            ("desk", 14),
            ("counter", 14),
            ("countertop", 14),
            ("coffee_table", 14),
            ("dining_table", 14),
            ("breakfast_table", 14),
            ("side_table", 14),
            ("console_table", 14),
            ("gaming_table", 14),
            ("pedestal_table", 14),
            ("pool_table", 14),
            ("kitchen_counter", 14),
            ("stairs", 15),
            ("stair", 15),
        ],
    )
    def test_keyword(self, name, expected):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(name) == expected

    def test_case_insensitive(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("CHAIR") == 1
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("Sofa") == 2

    def test_igibson_style_suffix(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("armchair.n.01") == 1
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("dining_table.n.01") == 14

    def test_word_separator_normalization(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("kitchen-counter") == 14
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id("Bath Towel") == 11


class TestUnknownClassMapping:
    """Unmapped, empty, and None class names must return semantic id 0."""

    @pytest.mark.parametrize(
        "name",
        [
            "wall",
            "floor",
            "ceiling",
            "robot",
            "door",
            "window",
            "cabinet",
            "bookshelf",
            "bookcase",
            "shelf",
            "rack",
            "tv_stand",
            "ottoman",
            "",
        ],
    )
    def test_returns_zero(self, name):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(name) == 0

    def test_none_returns_zero(self):
        assert SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(None) == 0


class TestLookupAndRemap:
    """Verify id-table construction and per-pixel semantic-id remapping."""

    SEM_MAP = np.array(
        [
            [1, 2, 3, 4],
            [3, 1, 4, 2],
            [4, 3, 2, 1],
        ],
        dtype=np.int32,
    )
    CLASS_ID_TO_NAME = {1: "chair", 2: "couch", 3: "wall", 4: "sink"}

    def test_build_id_to_l3mvn_semantic_id(self):
        table = SemanticTaxonomy.build_id_to_l3mvn_semantic_id(self.CLASS_ID_TO_NAME)
        assert table[1] == 1
        assert table[2] == 2
        assert table[3] == 0
        assert table[4] == 12

    def test_remap_semantic_id_map_values(self):
        remapped = SemanticTaxonomy.remap_semantic_id_map(
            self.SEM_MAP, self.CLASS_ID_TO_NAME
        )
        expected = np.array(
            [
                [1, 2, 0, 12],
                [0, 1, 12, 2],
                [12, 0, 2, 1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(remapped, expected)

    def test_remap_dtype_and_shape(self):
        remapped = SemanticTaxonomy.remap_semantic_id_map(
            self.SEM_MAP, self.CLASS_ID_TO_NAME
        )
        assert remapped.dtype == np.int32
        assert remapped.shape == self.SEM_MAP.shape

    def test_remap_values_in_valid_range(self):
        remapped = SemanticTaxonomy.remap_semantic_id_map(
            self.SEM_MAP, self.CLASS_ID_TO_NAME
        )
        assert remapped.min() >= 0
        assert remapped.max() <= 15

    def test_unknown_id_defaults_to_zero(self):
        sem_map = np.array([[99, 1], [2, 3]], dtype=np.int32)
        remapped = SemanticTaxonomy.remap_semantic_id_map(
            sem_map, self.CLASS_ID_TO_NAME
        )
        assert remapped[0, 0] == 0

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            SemanticTaxonomy.remap_semantic_id_map(
                np.zeros((2, 2, 2), dtype=np.int32), self.CLASS_ID_TO_NAME
            )


class TestConstants:
    def test_num_channels_constant(self):
        assert SemanticTaxonomy.NUM_CHANNELS == 16
