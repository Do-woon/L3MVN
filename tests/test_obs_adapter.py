"""Unit tests for ObsAdapter (envs/igibson/obs_adapter.py).

Includes:
  - ObsAdapter unit tests (tests 1-4)
  - SemanticTaxonomy compatibility test (test 5)
"""

import numpy as np
import pytest

from envs.igibson.obs_adapter import ObsAdapter
from envs.igibson.semantic_taxonomy import SemanticTaxonomy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return ObsAdapter()


def make_inputs(h: int = 4, w: int = 5):
    """Return consistent (rgb, depth, semantic_id_map) test arrays."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    depth = rng.random(size=(h, w)).astype(np.float32)
    sem = rng.integers(0, 8, size=(h, w), dtype=np.int32)
    return rgb, depth, sem


# ---------------------------------------------------------------------------
# Test 1: basic happy path
# ---------------------------------------------------------------------------

class TestBasicHappyPath:
    def test_output_shape(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        assert obs.shape == (5, 4, 5)

    def test_output_dtype(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        assert obs.dtype == np.float32

    def test_arbitrary_spatial_size(self, adapter):
        rgb, depth, sem = make_inputs(120, 160)
        obs = adapter.adapt(rgb, depth, sem)
        assert obs.shape == (5, 120, 160)


# ---------------------------------------------------------------------------
# Test 2: depth with existing channel dimension
# ---------------------------------------------------------------------------

class TestDepthWithChannelDim:
    def test_depth_hw1_accepted(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        depth_hw1 = depth[:, :, np.newaxis]  # (4, 5, 1)
        obs = adapter.adapt(rgb, depth_hw1, sem)
        assert obs.shape == (5, 4, 5)
        assert obs.dtype == np.float32

    def test_depth_hw1_values_preserved(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        depth_hw1 = depth[:, :, np.newaxis]
        obs_2d = adapter.adapt(rgb, depth, sem)
        obs_3d = adapter.adapt(rgb, depth_hw1, sem)
        np.testing.assert_array_equal(obs_2d, obs_3d)


# ---------------------------------------------------------------------------
# Test 3: channel placement
# ---------------------------------------------------------------------------

class TestChannelPlacement:
    def test_rgb_channels_0_to_2(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        expected_rgb = rgb.astype(np.float32).transpose(2, 0, 1)  # (3, 4, 5)
        np.testing.assert_array_equal(obs[0:3], expected_rgb)

    def test_depth_channel_3(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        np.testing.assert_array_equal(obs[3], depth.astype(np.float32))

    def test_semantic_channel_4(self, adapter):
        rgb, depth, sem = make_inputs(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        np.testing.assert_array_equal(obs[4], sem.astype(np.float32))

    def test_rgb_values_unchanged(self, adapter):
        """RGB values must not be scaled/normalized."""
        rgb = np.full((4, 5, 3), 200, dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.zeros((4, 5), dtype=np.int32)
        obs = adapter.adapt(rgb, depth, sem)
        assert obs[0, 0, 0] == pytest.approx(200.0)

    def test_semantic_id_preserved_as_float(self, adapter):
        """Semantic IDs must be stored as-is (no remapping)."""
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.array([
            [0,  1,  2,  3,  4],
            [5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ], dtype=np.int32)
        obs = adapter.adapt(rgb, depth, sem)
        np.testing.assert_array_equal(obs[4], sem.astype(np.float32))


# ---------------------------------------------------------------------------
# Test 4: mismatched spatial shapes
# ---------------------------------------------------------------------------

class TestMismatchedSpatialShapes:
    def test_depth_wrong_spatial(self, adapter):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 6), dtype=np.float32)    # w differs
        sem = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)

    def test_semantic_wrong_spatial(self, adapter):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.zeros((3, 5), dtype=np.int32)         # h differs
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)

    def test_rgb_wrong_channels(self, adapter):
        rgb = np.zeros((4, 5, 4), dtype=np.uint8)     # 4 channels instead of 3
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)

    def test_rgb_wrong_ndim(self, adapter):
        rgb = np.zeros((4, 5), dtype=np.uint8)         # missing channel dim
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)

    def test_depth_wrong_ndim(self, adapter):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5, 2), dtype=np.float32)  # 2 depth channels
        sem = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)

    def test_semantic_wrong_ndim(self, adapter):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.zeros((4, 5, 1), dtype=np.int32)      # extra dim not allowed
        with pytest.raises(ValueError):
            adapter.adapt(rgb, depth, sem)


# ---------------------------------------------------------------------------
# Test 5: SemanticTaxonomy compatibility
# ---------------------------------------------------------------------------

class TestSemanticTaxonomyCompatibility:
    """Verify that ObsAdapter's channel-4 output feeds cleanly into
    SemanticTaxonomy.semantic_id_map_to_one_hot().

    ObsAdapter produces float32 ids; taxonomy expects integer ids.
    The cast from float32 → int32 is the caller's responsibility and
    is performed here explicitly to show the integration seam.
    """

    CLASS_ID_TO_NAME = {
        0: "wall",    # → channel 0 (unknown)
        1: "chair",   # → channel 1
        2: "sofa",    # → channel 2
        3: "sink",    # → channel 12
        4: "toilet",  # → channel 5
    }

    def _build_obs(self, adapter):
        """Build a (5, 4, 5) obs with deterministic semantic ids."""
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
            [4, 3, 1, 0, 2],
            [2, 4, 0, 1, 3],
        ], dtype=np.int32)
        return adapter.adapt(rgb, depth, sem), sem

    def test_one_hot_shape(self, adapter):
        obs, _ = self._build_obs(adapter)
        sem_channel = obs[4]                          # (4, 5) float32
        sem_int = sem_channel.astype(np.int32)        # cast: float32 → int32
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(
            sem_int, self.CLASS_ID_TO_NAME
        )
        assert one_hot.shape == (16, 4, 5)

    def test_one_hot_channel_sum_is_one(self, adapter):
        obs, _ = self._build_obs(adapter)
        sem_int = obs[4].astype(np.int32)
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(
            sem_int, self.CLASS_ID_TO_NAME
        )
        np.testing.assert_array_equal(
            one_hot.sum(axis=0),
            np.ones((4, 5), dtype=np.uint8),
        )

    def test_one_hot_chair_channel(self, adapter):
        """id=1 (chair) must activate channel 1 in the taxonomy output."""
        obs, sem_original = self._build_obs(adapter)
        sem_int = obs[4].astype(np.int32)
        one_hot = SemanticTaxonomy.semantic_id_map_to_one_hot(
            sem_int, self.CLASS_ID_TO_NAME
        )
        expected_chair = (sem_original == 1).astype(np.uint8)
        np.testing.assert_array_equal(one_hot[1], expected_chair)

    def test_round_trip_semantic_id_preserved(self, adapter):
        """Semantic ids must survive the ObsAdapter float32 round-trip."""
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        depth = np.zeros((4, 5), dtype=np.float32)
        sem = np.arange(20, dtype=np.int32).reshape(4, 5)
        obs = adapter.adapt(rgb, depth, sem)
        recovered = obs[4].astype(np.int32)
        np.testing.assert_array_equal(recovered, sem)
