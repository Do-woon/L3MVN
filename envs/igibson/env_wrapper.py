"""EnvWrapper: wraps an iGibson environment to satisfy the L3MVN single-env
contract (obs, info, fail_case, done).

Output obs is **Stage 1**: shape ``(5, H, W)``
  channels 0-2 : RGB (float32, raw values)
  channel  3   : Depth (float32, raw from VisionSensor, metres)
  channel  4   : Semantic ID (float32, raw GT integer id)

Dependencies
------------
- ``DiscreteActionExecutor`` — action execution + sensor_pose / collision
- ``ObsAdapter``             — Stage-1 raw obs ``(5, H, W)``
- ``VisionSensor``           — iGibson official sensor pipeline
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from igibson.sensors.vision_sensor import VisionSensor

from envs.igibson.discrete_action_executor import (
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_STOP,
    DiscreteActionExecutor,
)
from envs.igibson.obs_adapter import ObsAdapter
from envs.igibson.semantic_taxonomy import SemanticTaxonomy

class _SimAsEnv:
    """Lightweight shim so VisionSensor sees `env.config` and `env.simulator`."""

    def __init__(self, simulator, config: dict) -> None:
        self.simulator = simulator
        self.config = config


class EnvWrapper:
    """Single-env wrapper that produces L3MVN-compatible Stage-1 obs.

    Parameters
    ----------
    igibson_env : object
        A live iGibson ``Simulator`` instance (or duck-typed equivalent).
    robot : object
        A loaded iGibson robot (e.g. ``Turtlebot``).
    scene : object
        The loaded iGibson scene.
    action_executor : DiscreteActionExecutor
    obs_adapter : ObsAdapter
    semantic_taxonomy : SemanticTaxonomy (class, not instance)
    goal_name : str
        L3MVN goal object name (e.g. ``"chair"``).
    goal_cat_id : int
        L3MVN goal category index (1–15).
    class_id_to_name : dict[int, str]
        iGibson semantic id → category name mapping.
    max_steps : int or None
        Episode length limit.  ``None`` disables the limit.
    """

    def __init__(
        self,
        igibson_env,
        robot,
        scene,
        action_executor: DiscreteActionExecutor,
        obs_adapter: ObsAdapter,
        semantic_taxonomy: type[SemanticTaxonomy],
        goal_name: str,
        goal_cat_id: int,
        class_id_to_name: dict[int, str],
        max_steps: Optional[int] = 1000,
    ) -> None:
        self._env = igibson_env
        self._robot = robot
        self._scene = scene
        self._executor = action_executor
        self._obs_adapter = obs_adapter
        self._taxonomy = semantic_taxonomy
        self._goal_name = goal_name
        self._goal_cat_id = goal_cat_id
        self._class_id_to_name = class_id_to_name
        self._max_steps: Optional[int] = max_steps

        self._done: bool = False
        self._clear_flag: int = 0
        self._step_count: int = 0

        # Build VisionSensor via a lightweight env shim.
        self._vision_sensor = self._build_vision_sensor(igibson_env)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment and return (obs, info).

        Returns
        -------
        obs : np.ndarray, shape (5, H, W), dtype float32
        info : dict
        """
        self._executor.reset()
        self._done = False
        self._clear_flag = 0
        self._step_count = 0

        rgb, depth, semantic = self._get_sensors()
        obs = self._obs_adapter.adapt(rgb, depth, semantic)

        info = {
            "sensor_pose": [0.0, 0.0, 0.0],
            "eve_angle": self._executor.eve_angle,
            "goal_cat_id": self._goal_cat_id,
            "goal_name": self._goal_name,
            "clear_flag": self._clear_flag,
        }
        return obs, info

    def plan_act_and_preprocess(
        self, planner_inputs: dict
    ) -> tuple[np.ndarray, dict, bool, dict]:
        """Execute one action and return (obs, fail_case, done, info).

        Parameters
        ----------
        planner_inputs : dict
            Must contain ``"action"`` (int 0–5).

        Returns
        -------
        obs : np.ndarray, shape (5, H, W), dtype float32
        fail_case : dict
        done : bool
        info : dict
        """
        action = int(planner_inputs["action"])
        sensor_pose, collision = self._executor.execute(action)

        # Advance physics so the renderer reflects the new robot state.
        self._env.step()

        self._step_count += 1

        rgb, depth, semantic = self._get_sensors()
        obs = self._obs_adapter.adapt(rgb, depth, semantic)

        goal_reached = self._check_goal_reached(
            semantic, self._goal_cat_id, self._goal_name
        )
        self._done = self._check_done(action, goal_reached)

        collision_int = int(collision)
        fail_case = {
            "collision": collision_int,
            "success": 0,
            "detection": 0,
            "exploration": 0,
        }

        info: dict = {
            "sensor_pose": sensor_pose,
            "eve_angle": self._executor.eve_angle,
            "goal_cat_id": self._goal_cat_id,
            "goal_name": self._goal_name,
            "clear_flag": self._clear_flag,
            "collision": collision_int,
        }

        if self._done:
            info["spl"] = 0.0
            info["success"] = 0
            info["distance_to_goal"] = 0.0

        return obs, fail_case, self._done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vision_sensor(self, sim) -> VisionSensor:
        """Create VisionSensor backed by a lightweight env shim."""
        config = {
            "image_width": sim.renderer.width,
            "image_height": sim.renderer.height,
            "depth_noise_rate": 0.0,
            "depth_low": 0.5,
            "depth_high": 5.0,
        }
        env_shim = _SimAsEnv(sim, config)
        return VisionSensor(env_shim, modalities=["rgb", "depth", "seg"])

    def _get_sensors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Acquire rgb, depth, and semantic via VisionSensor.

        Returns
        -------
        rgb : (H, W, 3) uint8
        depth : (H, W) float32   (metres, normalised by depth_high then un-normalised here)
        semantic : (H, W) int32
        """
        env_shim = _SimAsEnv(self._env, self._vision_sensor.config)
        obs = self._vision_sensor.get_obs(env_shim)

        # rgb: (H, W, 3) float32 [0,1] → uint8 [0,255]
        rgb = (np.clip(obs["rgb"], 0.0, 1.0) * 255).astype(np.uint8)

        # depth: VisionSensor returns (H,W,1) float32 [0,1] normalised by
        # depth_high.  Convert back to metres for the raw depth value.
        depth_norm = obs["depth"]                  # (H, W, 1) float32
        depth = (depth_norm[:, :, 0] * self._vision_sensor.depth_high).astype(np.float32)

        # semantic: VisionSensor returns (H,W,1) int32 already un-normalised.
        semantic = obs["seg"][:, :, 0].astype(np.int32)

        return rgb, depth, semantic

    def _check_goal_reached(
        self,
        semantic_id_map: np.ndarray,
        goal_cat_id: int,
        goal_name: str,
    ) -> bool:
        """Placeholder: check whether the goal object is reached.

        TODO: implement real proximity / detection logic.
        """
        return False

    def _check_done(self, action: int, goal_reached: bool) -> bool:
        """Determine whether the episode is finished."""
        if action == ACTION_STOP:
            return True
        if goal_reached:
            return True
        if self._max_steps is not None and self._step_count >= self._max_steps:
            return True
        return False
