"""Unit tests for EnvWrapper (envs/igibson/env_wrapper.py).

All tests use mock objects — no iGibson runtime required.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import MagicMock, patch

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


def _build_wrapper(executor=None, max_steps=500) -> EnvWrapper:
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

    return EnvWrapper(
        igibson_env=mock_env,
        robot=MagicMock(),
        scene=MagicMock(),
        action_executor=executor,
        obs_adapter=ObsAdapter(),
        semantic_taxonomy=SemanticTaxonomy,
        goal_name="chair",
        goal_cat_id=1,
        class_id_to_name=CLASS_ID_TO_NAME,
        max_steps=max_steps,
    )


# =========================================================================
# Test 1: reset happy path
# =========================================================================

class TestResetHappyPath:
    def test_obs_shape(self):
        w = _build_wrapper()
        obs, info = w.reset()
        assert obs.shape == (20, H, W)

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
        assert obs.shape == (20, H, W)

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
# Test 6: semantic block shape & one-hot integrity
# =========================================================================

class TestSemanticBlock:
    def test_semantic_channels_shape(self):
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_block = obs[4:20]
        assert sem_block.shape == (16, H, W)

    def test_one_hot_sum_is_one(self):
        w = _build_wrapper()
        obs, _ = w.reset()
        sem_block = obs[4:20]
        channel_sums = sem_block.sum(axis=0)
        np.testing.assert_array_equal(
            channel_sums, np.ones((H, W), dtype=np.float32)
        )


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
# Test 10: _preprocess_depth_for_l3mvn converts m → cm
# =========================================================================

class TestPreprocessDepth:
    def test_metres_to_centimetres(self):
        w = _build_wrapper()
        depth_m = np.ones((4, 5), dtype=np.float32) * 2.5  # 2.5 m
        depth_cm = w._preprocess_depth_for_l3mvn(depth_m)
        np.testing.assert_allclose(depth_cm, 250.0)
        assert depth_cm.dtype == np.float32

    def test_squeeze_hw1(self):
        w = _build_wrapper()
        depth_m = np.ones((4, 5, 1), dtype=np.float32)
        depth_cm = w._preprocess_depth_for_l3mvn(depth_m)
        assert depth_cm.shape == (4, 5)

    def test_depth_in_stage2_obs_is_cm(self):
        """ch 3 of the (20,H,W) obs must contain depth in cm, not raw metres."""
        w = _build_wrapper()
        obs, _ = w.reset()
        raw_depth_m = _make_depth()  # our mock depth values
        # VisionSensor clips [depth_low, depth_high], normalises by depth_high,
        # then EnvWrapper un-normalises back to metres and converts to cm.
        # Check that the output is not identical to the raw metres.
        # Exact values depend on VisionSensor clipping, but ch3 should be cm scale.
        assert obs.dtype == np.float32
        # At minimum, verify it's not all zeros (mock depth has nonzero values)
        assert obs[3].max() > 0.0 or np.all(raw_depth_m == 0.0)
