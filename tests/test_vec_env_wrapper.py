"""Unit tests for SingleEnvVecWrapper (envs/igibson/vec_env_wrapper.py).

All tests use mock objects — no iGibson runtime required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from envs.igibson.vec_env_wrapper import SingleEnvVecWrapper

# ---- constants ----------------------------------------------------------
H, W = 4, 5

REQUIRED_INFO_KEYS = ("sensor_pose", "eve_angle", "goal_cat_id", "goal_name", "clear_flag")
REQUIRED_FAIL_CASE_KEYS = ("collision", "success", "detection", "exploration")


# ---- helpers ------------------------------------------------------------

def _make_obs(h=H, w=W) -> np.ndarray:
    return np.random.default_rng(42).random((5, h, w)).astype(np.float32)


def _make_info(extra: dict | None = None) -> dict:
    info = {
        "sensor_pose": [0.1, 0.2, 0.3],
        "eve_angle": -30,
        "goal_cat_id": 1,
        "goal_name": "chair",
        "clear_flag": 0,
    }
    if extra:
        info.update(extra)
    return info


def _make_fail_case(collision: int = 0) -> dict:
    return {"collision": collision, "success": 0, "detection": 0, "exploration": 0}


def _make_planner_inputs(m: int = H) -> dict:
    return {
        "map_pred": np.zeros((m, m), dtype=np.float32),
        "exp_pred": np.ones((m, m), dtype=np.float32),
        "pose_pred": np.array([1.0, 1.0, 0.0, 0, m, 0, m], dtype=np.float32),
        "goal": np.zeros((m, m), dtype=np.float32),
        "map_target": np.zeros((m, m), dtype=np.float32),
        "new_goal": False,
        "found_goal": 0,
        "wait": False,
    }


def _build_vec_wrapper(
    reset_obs=None,
    reset_info=None,
    step_obs=None,
    step_fail_case=None,
    step_done: bool = False,
    step_info=None,
) -> SingleEnvVecWrapper:
    """Build a SingleEnvVecWrapper backed by a mock EnvWrapper."""
    if reset_obs is None:
        reset_obs = _make_obs()
    if reset_info is None:
        reset_info = _make_info()
    if step_obs is None:
        step_obs = _make_obs()
    if step_fail_case is None:
        step_fail_case = _make_fail_case()
    if step_info is None:
        step_info = _make_info({"collision": step_fail_case["collision"]})

    mock_inner = MagicMock()
    mock_inner.reset.return_value = (reset_obs, reset_info)
    mock_inner.plan_act_and_preprocess.return_value = (
        step_obs, step_fail_case, step_done, step_info
    )

    return SingleEnvVecWrapper(mock_inner)


# =========================================================================
# Test 1: reset batch wrapping
# =========================================================================

class TestResetBatchWrapping:
    def test_obs_batch_shape(self):
        vec = _build_vec_wrapper()
        obs_batch, _ = vec.reset()
        assert obs_batch.shape == (1, 5, H, W)

    def test_obs_batch_dtype(self):
        vec = _build_vec_wrapper()
        obs_batch, _ = vec.reset()
        assert obs_batch.dtype == np.float32

    def test_infos_list_length(self):
        vec = _build_vec_wrapper()
        _, infos_list = vec.reset()
        assert isinstance(infos_list, list)
        assert len(infos_list) == 1

    def test_infos_list_element_is_dict(self):
        vec = _build_vec_wrapper()
        _, infos_list = vec.reset()
        assert isinstance(infos_list[0], dict)

    def test_obs_values_preserved(self):
        inner_obs = _make_obs()
        vec = _build_vec_wrapper(reset_obs=inner_obs)
        obs_batch, _ = vec.reset()
        np.testing.assert_array_equal(obs_batch[0], inner_obs)


# =========================================================================
# Test 2: plan_act_and_preprocess with single dict
# =========================================================================

class TestStepSingleDict:
    def _step(self):
        vec = _build_vec_wrapper()
        vec.reset()
        return vec.plan_act_and_preprocess(_make_planner_inputs())

    def test_obs_batch_shape(self):
        obs_batch, _, _, _ = self._step()
        assert obs_batch.shape == (1, 5, H, W)

    def test_obs_batch_dtype(self):
        obs_batch, _, _, _ = self._step()
        assert obs_batch.dtype == np.float32

    def test_fail_case_batch_length(self):
        _, fail_case_batch, _, _ = self._step()
        assert isinstance(fail_case_batch, list)
        assert len(fail_case_batch) == 1

    def test_done_batch_length(self):
        _, _, done_batch, _ = self._step()
        assert len(done_batch) == 1

    def test_infos_list_length(self):
        _, _, _, infos_list = self._step()
        assert isinstance(infos_list, list)
        assert len(infos_list) == 1

    def test_inner_wrapper_called_with_dict(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (_make_obs(), _make_info())
        mock_inner.plan_act_and_preprocess.return_value = (
            _make_obs(), _make_fail_case(), False, _make_info()
        )
        vec = SingleEnvVecWrapper(mock_inner)
        planner_inputs = _make_planner_inputs()
        vec.plan_act_and_preprocess(planner_inputs)
        assert mock_inner.plan_act_and_preprocess.call_count == 1
        assert mock_inner.plan_act_and_preprocess.call_args[0][0] is planner_inputs


# =========================================================================
# Test 3: plan_act_and_preprocess with list[dict]
# =========================================================================

class TestStepListOfDict:
    def test_list_input_unwrapped(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (_make_obs(), _make_info())
        mock_inner.plan_act_and_preprocess.return_value = (
            _make_obs(), _make_fail_case(), False, _make_info()
        )
        vec = SingleEnvVecWrapper(mock_inner)
        planner_inputs = _make_planner_inputs()
        vec.plan_act_and_preprocess([planner_inputs])
        # inner wrapper should receive the unwrapped single dict
        assert mock_inner.plan_act_and_preprocess.call_count == 1
        assert mock_inner.plan_act_and_preprocess.call_args[0][0] is planner_inputs

    def test_list_input_obs_shape(self):
        vec = _build_vec_wrapper()
        obs_batch, _, _, _ = vec.plan_act_and_preprocess([_make_planner_inputs()])
        assert obs_batch.shape == (1, 5, H, W)

    def test_tuple_input_unwrapped(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (_make_obs(), _make_info())
        mock_inner.plan_act_and_preprocess.return_value = (
            _make_obs(), _make_fail_case(), False, _make_info()
        )
        vec = SingleEnvVecWrapper(mock_inner)
        planner_inputs = _make_planner_inputs()
        vec.plan_act_and_preprocess((planner_inputs,))
        assert mock_inner.plan_act_and_preprocess.call_count == 1
        assert mock_inner.plan_act_and_preprocess.call_args[0][0] is planner_inputs

    def test_list_len_gt_one_raises(self):
        vec = _build_vec_wrapper()
        with pytest.raises(ValueError):
            vec.plan_act_and_preprocess([_make_planner_inputs(), _make_planner_inputs()])

    def test_tuple_len_gt_one_raises(self):
        vec = _build_vec_wrapper()
        with pytest.raises(ValueError):
            vec.plan_act_and_preprocess((_make_planner_inputs(), _make_planner_inputs()))


# =========================================================================
# Test 4: done wrapping
# =========================================================================

class TestDoneWrapping:
    def test_done_true_wrapped(self):
        vec = _build_vec_wrapper(step_done=True)
        vec.reset()
        _, _, done_batch, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        assert len(done_batch) == 1
        assert bool(done_batch[0]) is True

    def test_done_false_wrapped(self):
        vec = _build_vec_wrapper(step_done=False)
        vec.reset()
        _, _, done_batch, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        assert len(done_batch) == 1
        assert bool(done_batch[0]) is False

    def test_done_is_numpy_array(self):
        vec = _build_vec_wrapper(step_done=True)
        vec.reset()
        _, _, done_batch, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        assert isinstance(done_batch, np.ndarray)
        assert done_batch.dtype == bool


# =========================================================================
# Test 5: info passthrough
# =========================================================================

class TestInfoPassthrough:
    def test_reset_info_keys_preserved(self):
        info = _make_info()
        vec = _build_vec_wrapper(reset_info=info)
        _, infos_list = vec.reset()
        for key in REQUIRED_INFO_KEYS:
            assert key in infos_list[0], f"Missing key in reset info: {key}"

    def test_step_info_keys_preserved(self):
        info = _make_info({"collision": 0})
        vec = _build_vec_wrapper(step_info=info)
        vec.reset()
        _, _, _, infos_list = vec.plan_act_and_preprocess(_make_planner_inputs())
        for key in REQUIRED_INFO_KEYS:
            assert key in infos_list[0], f"Missing key in step info: {key}"

    def test_info_values_unchanged(self):
        info = _make_info()
        vec = _build_vec_wrapper(reset_info=info)
        _, infos_list = vec.reset()
        assert infos_list[0]["sensor_pose"] == [0.1, 0.2, 0.3]
        assert infos_list[0]["eve_angle"] == -30
        assert infos_list[0]["goal_name"] == "chair"


# =========================================================================
# Test 6: fail_case passthrough
# =========================================================================

class TestFailCasePassthrough:
    def test_required_keys_present(self):
        vec = _build_vec_wrapper(step_fail_case=_make_fail_case(collision=1))
        vec.reset()
        _, fail_case_batch, _, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        for key in REQUIRED_FAIL_CASE_KEYS:
            assert key in fail_case_batch[0], f"Missing key: {key}"

    def test_collision_value_preserved(self):
        vec = _build_vec_wrapper(step_fail_case=_make_fail_case(collision=1))
        vec.reset()
        _, fail_case_batch, _, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        assert fail_case_batch[0]["collision"] == 1

    def test_no_collision(self):
        vec = _build_vec_wrapper(step_fail_case=_make_fail_case(collision=0))
        vec.reset()
        _, fail_case_batch, _, _ = vec.plan_act_and_preprocess(_make_planner_inputs())
        assert fail_case_batch[0]["collision"] == 0


# =========================================================================
# Test 7: num_envs attribute
# =========================================================================

class TestNumEnvs:
    def test_num_envs_is_one(self):
        vec = _build_vec_wrapper()
        assert vec.num_envs == 1


# =========================================================================
# Test 8: async/step compatibility interface
# =========================================================================

class TestAsyncStepCompatibility:
    def test_step_async_then_wait(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (_make_obs(), _make_info())
        mock_inner.plan_act_and_preprocess.return_value = (
            _make_obs(), _make_fail_case(), False, _make_info()
        )
        vec = SingleEnvVecWrapper(mock_inner)

        planner_inputs = _make_planner_inputs()
        vec.step_async(planner_inputs)
        obs_batch, fail_case_batch, done_batch, infos_list = vec.step_wait()

        assert obs_batch.shape == (1, 5, H, W)
        assert len(fail_case_batch) == 1
        assert len(done_batch) == 1
        assert len(infos_list) == 1
        assert mock_inner.plan_act_and_preprocess.call_count == 1
        assert mock_inner.plan_act_and_preprocess.call_args[0][0] is planner_inputs

    def test_step_wait_without_async_raises(self):
        vec = _build_vec_wrapper()
        with pytest.raises(RuntimeError):
            vec.step_wait()

    def test_step_alias_calls_plan_act_and_preprocess(self):
        mock_inner = MagicMock()
        mock_inner.reset.return_value = (_make_obs(), _make_info())
        mock_inner.plan_act_and_preprocess.return_value = (
            _make_obs(), _make_fail_case(), False, _make_info()
        )
        vec = SingleEnvVecWrapper(mock_inner)

        planner_inputs = _make_planner_inputs()
        vec.step(planner_inputs)
        assert mock_inner.plan_act_and_preprocess.call_count == 1
        assert mock_inner.plan_act_and_preprocess.call_args[0][0] is planner_inputs
