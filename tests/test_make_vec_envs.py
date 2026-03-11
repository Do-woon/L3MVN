"""Unit tests for envs.make_vec_envs with iGibson integration.

These tests are split into:
1) Branching tests for make_vec_envs(args) using patch.object (no reload).
2) A focused _make_igibson_vec_envs(args) defaults test using fully mocked
   import chain via patch.dict(sys.modules).
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import envs as envs_mod


def _igibson_args(**overrides):
    defaults = dict(
        use_igibson=1,
        igibson_scene="Rs_int",
        goal_name="chair",
        goal_cat_id=1,
        igibson_assets_path="/fake/assets",
        igibson_dataset_path="/fake/ig_dataset",
        igibson_key_path="/fake/igibson.key",
        frame_width=160,
        frame_height=120,
        max_episode_length=500,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _habitat_args():
    return types.SimpleNamespace(
        use_igibson=0,
        task_config="tasks/objectnav_hm3d.yaml",
        device="cpu",
    )


class _FakeVecWrapper:
    num_envs = 1

    def reset(self):
        return None, [{}]

    def plan_act_and_preprocess(self, planner_inputs):
        return None, [{}], [], [{}]

    def close(self):
        pass


def _new_pkg(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    return mod


def _new_mod(name: str) -> types.ModuleType:
    return types.ModuleType(name)


def _build_mock_patches(vec_wrapper_return):
    """Return mocked module graph for _make_igibson_vec_envs import chain."""
    mods = {}

    # ---- igibson package tree ----
    igibson = _new_pkg("igibson")
    igibson.assets_path = "/fake/assets"
    igibson.ig_dataset_path = "/fake/ig_dataset"
    igibson.key_path = "/fake/igibson.key"
    mods["igibson"] = igibson

    mods["igibson.render"] = _new_pkg("igibson.render")
    mods["igibson.render.mesh_renderer"] = _new_pkg("igibson.render.mesh_renderer")
    m_mesh = _new_mod("igibson.render.mesh_renderer.mesh_renderer_settings")
    m_mesh.MeshRendererSettings = MagicMock(return_value=MagicMock())
    mods["igibson.render.mesh_renderer.mesh_renderer_settings"] = m_mesh

    mods["igibson.robots"] = _new_pkg("igibson.robots")
    m_robot = _new_mod("igibson.robots.turtlebot")
    robot_instance = MagicMock()
    m_robot.Turtlebot = MagicMock(return_value=robot_instance)
    mods["igibson.robots.turtlebot"] = m_robot

    mods["igibson.scenes"] = _new_pkg("igibson.scenes")
    m_scene = _new_mod("igibson.scenes.igibson_indoor_scene")
    scene_instance = MagicMock()
    scene_instance.get_random_point.return_value = (0, np.array([1.0, 2.0, 0.0]))
    m_scene.InteractiveIndoorScene = MagicMock(return_value=scene_instance)
    mods["igibson.scenes.igibson_indoor_scene"] = m_scene

    m_sim = _new_mod("igibson.simulator")
    sim_instance = MagicMock()
    m_sim.Simulator = MagicMock(return_value=sim_instance)
    mods["igibson.simulator"] = m_sim

    mods["igibson.utils"] = _new_pkg("igibson.utils")
    m_sem_utils = _new_mod("igibson.utils.semantics_utils")
    m_sem_utils.get_class_name_to_class_id = MagicMock(
        return_value={"chair": 1, "table": 2, "sofa": 3}
    )
    mods["igibson.utils.semantics_utils"] = m_sem_utils

    # ---- envs.igibson submodules imported inside _make_igibson_vec_envs ----
    m_exec = _new_mod("envs.igibson.discrete_action_executor")
    m_exec.DiscreteActionExecutor = MagicMock(return_value=MagicMock())
    mods["envs.igibson.discrete_action_executor"] = m_exec

    m_obs = _new_mod("envs.igibson.obs_adapter")
    m_obs.ObsAdapter = MagicMock(return_value=MagicMock())
    mods["envs.igibson.obs_adapter"] = m_obs

    m_tax = _new_mod("envs.igibson.semantic_taxonomy")
    m_tax.SemanticTaxonomy = type("SemanticTaxonomy", (), {})
    mods["envs.igibson.semantic_taxonomy"] = m_tax

    m_envw = _new_mod("envs.igibson.env_wrapper")
    m_envw.EnvWrapper = MagicMock(return_value=MagicMock())
    mods["envs.igibson.env_wrapper"] = m_envw

    m_vec = _new_mod("envs.igibson.vec_env_wrapper")
    m_vec.SingleEnvVecWrapper = MagicMock(return_value=vec_wrapper_return)
    mods["envs.igibson.vec_env_wrapper"] = m_vec

    return mods


class TestMakeVecEnvsBranching(unittest.TestCase):
    """Fast branch tests for make_vec_envs() without module reload."""

    def test_use_igibson_1_calls_igibson_builder(self):
        fw = _FakeVecWrapper()
        args = _igibson_args()
        with patch.object(envs_mod, "_make_igibson_vec_envs", return_value=fw) as p_ig:
            result = envs_mod.make_vec_envs(args)
            p_ig.assert_called_once_with(args)
            self.assertIs(result, fw)

    def test_returns_object_with_required_api(self):
        fw = _FakeVecWrapper()
        args = _igibson_args()
        with patch.object(envs_mod, "_make_igibson_vec_envs", return_value=fw):
            result = envs_mod.make_vec_envs(args)
        self.assertTrue(hasattr(result, "reset"))
        self.assertTrue(hasattr(result, "plan_act_and_preprocess"))
        self.assertTrue(hasattr(result, "close"))
        self.assertEqual(result.num_envs, 1)

    def test_use_igibson_0_uses_habitat_path(self):
        args = _habitat_args()
        fake_habitat_envs = MagicMock()
        fake_vecpytorch = MagicMock(return_value=MagicMock())
        with patch.object(envs_mod, "_make_igibson_vec_envs") as p_ig:
            with patch.object(envs_mod, "construct_envs21", return_value=fake_habitat_envs) as p_c21:
                with patch.object(envs_mod, "VecPyTorch", fake_vecpytorch):
                    _ = envs_mod.make_vec_envs(args)
        p_ig.assert_not_called()
        p_c21.assert_called_once()
        fake_vecpytorch.assert_called_once()

    def test_missing_use_igibson_defaults_to_habitat_path(self):
        args = types.SimpleNamespace(task_config="tasks/objectnav_hm3d.yaml", device="cpu")
        fake_habitat_envs = MagicMock()
        fake_vecpytorch = MagicMock(return_value=MagicMock())
        with patch.object(envs_mod, "_make_igibson_vec_envs") as p_ig:
            with patch.object(envs_mod, "construct_envs21", return_value=fake_habitat_envs) as p_c21:
                with patch.object(envs_mod, "VecPyTorch", fake_vecpytorch):
                    _ = envs_mod.make_vec_envs(args)
        p_ig.assert_not_called()
        p_c21.assert_called_once()


class TestMakeIGibsonVecEnvDefaults(unittest.TestCase):
    """Verify _make_igibson_vec_envs defaults without importing real iGibson."""

    def test_frame_defaults_used_when_missing(self):
        args = types.SimpleNamespace(
            use_igibson=1,
            igibson_scene="Rs_int",
            goal_name="sofa",
            goal_cat_id=3,
        )
        fw = _FakeVecWrapper()
        mock_patches = _build_mock_patches(vec_wrapper_return=fw)

        import sys

        with patch.dict(sys.modules, mock_patches):
            with patch.object(envs_mod, "_spawn_collision_free", return_value=1) as p_spawn:
                result = envs_mod._make_igibson_vec_envs(args)

        self.assertIs(result, fw)
        p_spawn.assert_called_once()
        sim_cls = mock_patches["igibson.simulator"].Simulator
        self.assertTrue(sim_cls.called)
        call_kwargs = sim_cls.call_args.kwargs
        self.assertEqual(call_kwargs.get("image_width"), 160)
        self.assertEqual(call_kwargs.get("image_height"), 120)


if __name__ == "__main__":
    unittest.main()
