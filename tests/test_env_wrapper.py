"""Unit tests for EnvWrapper (envs/igibson/env_wrapper.py).

All tests use mock objects — no iGibson runtime required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from envs.igibson.discrete_action_executor import (
    ACTION_FORWARD,
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_STOP,
    DiscreteActionExecutor,
)
from envs.igibson.env_wrapper import EnvWrapper
from envs.igibson.obs_adapter import ObsAdapter
from envs.igibson.semantic_taxonomy import SemanticTaxonomy

# ---- constants ----------------------------------------------------------
H, W = 4, 5
CLASS_ID_TO_NAME = {0: "wall", 1: "chair", 2: "sofa", 3: "sink"}


# ---- helpers ------------------------------------------------------------

def _make_rgb(h=H, w=W):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_depth(h=H, w=W):
    return np.random.default_rng(1).random((h, w)).astype(np.float32)


def _make_semantic(h=H, w=W):
    return np.random.default_rng(2).integers(0, 4, (h, w), dtype=np.int32)


def _fake_render_robot_cameras(modes):
    """Mimics iGibson renderer.render_robot_cameras return structure.

    VisionSensor calls this internally, passing its `raw_modalities` list.
    It expects a list of (H, W, 4) float32 arrays, one per mode.
    """
    frames = []
    for m in modes:
        if m == "rgb":
            # (H, W, 4) float32 [0,1]
            f = np.zeros((H, W, 4), dtype=np.float32)
            f[:, :, :3] = _make_rgb().astype(np.float32) / 255.0
            f[:, :, 3] = 1.0
            frames.append(f)
        elif m == "3d":
            # (H, W, 4) float32; channel 2 = -depth
            f = np.zeros((H, W, 4), dtype=np.float32)
            f[:, :, 2] = -_make_depth()
            frames.append(f)
        elif m == "seg":
            # (H, W, 4) float32; channel 0 = class_id / 512
            f = np.zeros((H, W, 4), dtype=np.float32)
            f[:, :, 0] = _make_semantic().astype(np.float32) / 512.0
            frames.append(f)
    return frames


class FakeExecutor:
    """Lightweight stand-in for DiscreteActionExecutor (pure-Python, no pybullet)."""

    EVE_ANGLE_MIN = -60
    EVE_ANGLE_MAX = 0

    def __init__(self):
        self._eve_angle = 0
        self._step = 30
        self.last_collision = False

    @property
    def eve_angle(self):
        return self._eve_angle

    def reset(self):
        self._eve_angle = 0

    def execute(self, action_id):
        if action_id == ACTION_LOOK_UP:
            self._eve_angle = min(self._eve_angle + self._step, self.EVE_ANGLE_MAX)
            return [0.0, 0.0, 0.0], False
        if action_id == ACTION_LOOK_DOWN:
            self._eve_angle = max(self._eve_angle - self._step, self.EVE_ANGLE_MIN)
            return [0.0, 0.0, 0.0], False
        if action_id == ACTION_STOP:
            return [0.0, 0.0, 0.0], False
        # FORWARD / TURN_*
        return [0.05, 0.0, 0.1], self.last_collision


def _build_wrapper(executor=None, max_steps=500, class_id_to_name=None) -> EnvWrapper:
    """Build an EnvWrapper backed by mocks.

    The mock Simulator exposes `.renderer.width`, `.renderer.height`
    and `.renderer.render_robot_cameras` so that VisionSensor works.
    """
    mock_env = MagicMock()
    mock_env.renderer.width = W
    mock_env.renderer.height = H
    mock_env.renderer.render_robot_cameras.side_effect = _fake_render_robot_cameras
    mock_env.step.return_value = None

    if executor is None:
        executor = FakeExecutor()
    if class_id_to_name is None:
        class_id_to_name = CLASS_ID_TO_NAME

    return EnvWrapper(
        igibson_env=mock_env,
        robot=MagicMock(),
        scene=MagicMock(),
        action_executor=executor,
        obs_adapter=ObsAdapter(),
        semantic_taxonomy=SemanticTaxonomy,
        goal_name="chair",
        goal_cat_id=1,
        class_id_to_name=class_id_to_name,
        max_steps=max_steps,
    )


# =========================================================================
# Test 1: reset happy path
# =========================================================================

class TestResetHappyPath:
    def test_obs_shape(self):
        w = _build_wrapper()
        obs, info = w.reset()
        assert obs.shape == (5, H, W)

    def test_obs_dtype(self):
        w = _build_wrapper()
        obs, _ = w.reset()
        assert obs.dtype == np.float32

    def test_sensor_pose_zero(self):
        w = _build_wrapper()
        _, info = w.reset()
        assert info["sensor_pose"] == [0.0, 0.0, 0.0]

    def test_eve_angle_zero(self):
        w = _build_wrapper()
        _, info = w.reset()
        assert info["eve_angle"] == 0

    def test_required_info_keys(self):
        w = _build_wrapper()
        _, info = w.reset()
        for key in ("sensor_pose", "eve_angle", "goal_cat_id", "goal_name", "clear_flag"):
            assert key in info, f"missing key: {key}"

    def test_goal_metadata(self):
        w = _build_wrapper()
        _, info = w.reset()
        assert info["goal_name"] == "chair"
        assert info["goal_cat_id"] == 1


# =========================================================================
# Test 2: forward step
# =========================================================================

class TestForwardStep:
    def test_obs_shape(self):
        w = _build_wrapper()
        w.reset()
        obs, fail_case, done, info = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert obs.shape == (5, H, W)

    def test_fail_case_keys(self):
        w = _build_wrapper()
        w.reset()
        _, fail_case, _, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        for key in ("collision", "success", "detection", "exploration"):
            assert key in fail_case

    def test_info_has_collision(self):
        w = _build_wrapper()
        w.reset()
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert "collision" in info

    def test_not_done(self):
        w = _build_wrapper()
        w.reset()
        _, _, done, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert done is False


# =========================================================================
# Test 3: look down step
# =========================================================================

class TestLookDownStep:
    def test_sensor_pose_zero(self):
        w = _build_wrapper()
        w.reset()
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        assert info["sensor_pose"] == [0.0, 0.0, 0.0]

    def test_eve_angle_decremented(self):
        w = _build_wrapper()
        w.reset()
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        assert info["eve_angle"] == -30


# =========================================================================
# Test 4: repeated look down clamp
# =========================================================================

class TestLookDownClamp:
    def test_clamp_at_minus60(self):
        w = _build_wrapper()
        w.reset()
        for _ in range(5):
            _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        assert info["eve_angle"] == -60


# =========================================================================
# Test 5: look up recovery
# =========================================================================

class TestLookUpRecovery:
    def test_recovery_to_zero(self):
        w = _build_wrapper()
        w.reset()
        # Go down twice → -60
        w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        # Come up once → -30
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_LOOK_UP})
        assert info["eve_angle"] == -30

    def test_recovery_fully(self):
        w = _build_wrapper()
        w.reset()
        w.plan_act_and_preprocess({"action": ACTION_LOOK_DOWN})
        w.plan_act_and_preprocess({"action": ACTION_LOOK_UP})
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_LOOK_UP})
        # Started at 0, down→-30, up→0, up→0 (clamped)
        assert info["eve_angle"] == 0


# =========================================================================
# Test 6: Stage 1 channel content
# =========================================================================

class TestSemanticBlock:
    def test_ch4_is_semantic_id_nonnegative(self):
        """ch4 must contain non-negative float32 semantic IDs."""
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_ch = obs[4]
        assert sem_ch.shape == (H, W)
        assert obs.dtype == np.float32
        assert np.all(sem_ch >= 0)

    def test_ch3_depth_nonnegative(self):
        """ch3 must be a non-negative float32 depth map."""
        w = _build_wrapper()
        obs, _ = w.reset()
        depth_ch = obs[3]
        assert depth_ch.shape == (H, W)
        assert depth_ch.dtype == np.float32
        assert np.all(depth_ch >= 0)

    def test_ch4_semantic_id_remapped_to_l3mvn_ids(self):
        """EnvWrapper must remap iGibson class ids to L3MVN semantic ids."""
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_ch = obs[4].astype(np.int32)
        # With CLASS_ID_TO_NAME {0:wall,1:chair,2:sofa,3:sink}
        # expected L3MVN ids are {0,1,2,12}.
        assert set(np.unique(sem_ch)).issubset({0, 1, 2, 12})

    def test_ch4_semantic_id_exact_mapping(self):
        """Per-pixel remap result must match SemanticTaxonomy mapping."""
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_raw = _make_semantic()
        expected = SemanticTaxonomy.remap_semantic_id_map(sem_raw, CLASS_ID_TO_NAME)
        np.testing.assert_array_equal(obs[4].astype(np.int32), expected)

    def test_unmapped_igibson_class_id_maps_to_zero(self):
        """Unknown class ids must fall back to background id 0."""
        class_id_to_name = {0: "wall", 1: "chair", 3: "sink"}  # id=2 omitted
        w = _build_wrapper(class_id_to_name=class_id_to_name)
        obs, _ = w.reset()
        sem_raw = _make_semantic()
        remapped = obs[4].astype(np.int32)
        np.testing.assert_array_equal(remapped[sem_raw == 2], 0)


# =========================================================================
# Test 7: done info keys
# =========================================================================

class TestDoneInfoKeys:
    def test_done_on_stop(self):
        w = _build_wrapper()
        w.reset()
        _, _, done, info = w.plan_act_and_preprocess({"action": ACTION_STOP})
        assert done is True

    def test_done_info_contains_metrics(self):
        w = _build_wrapper()
        w.reset()
        _, _, _, info = w.plan_act_and_preprocess({"action": ACTION_STOP})
        for key in ("spl", "success", "distance_to_goal"):
            assert key in info, f"missing key on done: {key}"


# =========================================================================
# Test 8: fail_case collision propagation
# =========================================================================

class TestCollisionPropagation:
    def test_collision_one(self):
        executor = FakeExecutor()
        executor.last_collision = True
        w = _build_wrapper(executor=executor)
        w.reset()
        _, fail_case, _, info = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert fail_case["collision"] == 1
        assert info["collision"] == 1

    def test_no_collision_zero(self):
        executor = FakeExecutor()
        executor.last_collision = False
        w = _build_wrapper(executor=executor)
        w.reset()
        _, fail_case, _, info = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert fail_case["collision"] == 0
        assert info["collision"] == 0


# =========================================================================
# Test 9: _check_done reacts to max_steps
# =========================================================================

class TestCheckDone:
    def test_done_on_max_steps(self):
        w = _build_wrapper(max_steps=3)
        w.reset()
        for _ in range(2):
            _, _, done, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
            assert done is False
        _, _, done, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert done is True

    def test_no_limit_when_none(self):
        w = _build_wrapper(max_steps=None)
        w.reset()
        for _ in range(20):
            _, _, done, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
            assert done is False

    def test_reset_clears_step_count(self):
        w = _build_wrapper(max_steps=2)
        w.reset()
        w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        # After reset, step count should be back to 0
        w.reset()
        _, _, done, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert done is False


# =========================================================================
# Test 10: Stage 1 depth and semantic ID channels
# =========================================================================

class TestPreprocessDepth:
    def test_ch3_depth_from_vision_sensor(self):
        """ch3 depth values come from VisionSensor (metres, [0, depth_high])."""
        w = _build_wrapper()
        obs, _ = w.reset()
        depth_ch = obs[3]
        assert depth_ch.shape == (H, W)
        assert depth_ch.dtype == np.float32
        # VisionSensor depth_high=5.0, so un-normalised depth <= 5.0 m
        assert depth_ch.max() <= 5.1  # small tolerance for float precision

    def test_ch4_semantic_id_is_integer_valued(self):
        """ch4 values are float32 representations of integer semantic IDs."""
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_ch = obs[4]
        # All values should be integer-valued (fractional part == 0)
        np.testing.assert_array_equal(sem_ch, np.round(sem_ch))

    def test_step_obs_also_stage1(self):
        """plan_act_and_preprocess also returns (5,H,W) Stage-1 obs."""
        w = _build_wrapper()
        w.reset()
        obs, _, _, _ = w.plan_act_and_preprocess({"action": ACTION_FORWARD})
        assert obs.shape == (5, H, W)
        assert obs.dtype == np.float32
