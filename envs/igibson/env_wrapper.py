"""EnvWrapper: wraps an iGibson environment to satisfy the L3MVN single-env
contract (obs, info, fail_case, done).

Output obs is **Stage 1**: shape ``(5, H, W)``
  channels 0-2 : RGB (float32, raw values)
  channel  3   : Depth (float32, raw from VisionSensor, metres)
  channel  4   : Semantic ID (float32, L3MVN semantic id single-channel)

Dependencies
------------
- ``DiscreteActionExecutor`` — action execution + sensor_pose / collision
- ``ObsAdapter``             — Stage-1 raw obs ``(5, H, W)``
- ``VisionSensor``           — iGibson official sensor pipeline
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import skimage.morphology

from igibson.sensors.vision_sensor import VisionSensor

from envs.igibson.discrete_action_executor import (
    ACTION_FORWARD,
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    DiscreteActionExecutor,
)
from envs.igibson.obs_adapter import ObsAdapter
from envs.igibson.semantic_taxonomy import SemanticTaxonomy
import envs.utils.pose as pu


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
    args : object, optional
        L3MVN args-like object. If provided, planner hyperparameters are read
        from it (map_resolution, map_size_cm, turn_angle, collision_threshold).
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
        args=None,
    ) -> None:
        self._env = igibson_env
        self._robot = robot
        self._scene = scene
        self._executor = action_executor
        self._obs_adapter = obs_adapter
        self._taxonomy = semantic_taxonomy
        self._goal_name = goal_name
        self._goal_cat_id = goal_cat_id
        self._class_id_to_name = class_id_to_name or {}
        self._max_steps: Optional[int] = max_steps

        self._map_resolution: int = int(getattr(args, "map_resolution", 5))
        self._map_size_cm: int = int(getattr(args, "map_size_cm", 400))
        self._turn_angle_deg: float = float(getattr(args, "turn_angle", 30.0))
        self._collision_threshold: float = float(
            getattr(args, "collision_threshold", 0.1)
        )

        self._id_to_l3mvn_sem_id = self._build_semantic_lookup(self._class_id_to_name)

        self._done: bool = False
        self._clear_flag: int = 0
        self._step_count: int = 0
        self._obs: Optional[np.ndarray] = None

        # Planner states (Habitat Sem_Exp parity).
        self._selem = skimage.morphology.disk(3)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._collision_map: Optional[np.ndarray] = None
        self._visited: Optional[np.ndarray] = None
        self._visited_vis: Optional[np.ndarray] = None
        self._col_width: int = 1
        self._curr_loc: Optional[list[float]] = None
        self._last_loc: Optional[list[float]] = None
        self._last_action: Optional[int] = None
        self._replan_count: int = 0
        self._collision_n: int = 0

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
        self._replan_count = 0
        self._collision_n = 0
        self._col_width = 1
        self._last_action = None
        self._curr_loc = None
        self._last_loc = None

        default_cells = max(1, self._map_size_cm // self._map_resolution)
        self._init_map_buffers(default_cells)

        rgb, depth, semantic = self._get_sensors()
        obs = self._obs_adapter.adapt(rgb, depth, semantic)
        self._obs = obs

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
        """Plan, execute one action, and return (obs, fail_case, done, info).

        Parameters
        ----------
        planner_inputs : dict
            Canonical L3MVN 8-key planner dict:
            ``map_pred, exp_pred, pose_pred, goal, map_target,
            new_goal, found_goal, wait``.

        Returns
        -------
        obs : np.ndarray, shape (5, H, W), dtype float32
        fail_case : dict
        done : bool
        info : dict
        """
        if bool(planner_inputs.get("wait", False)):
            self._last_action = None
            info = self._build_info(
                sensor_pose=[0.0, 0.0, 0.0],
                collision_int=0,
            )
            obs = (
                np.zeros_like(self._obs)
                if self._obs is not None
                else np.zeros(
                    (
                        5,
                        int(self._vision_sensor.config["image_height"]),
                        int(self._vision_sensor.config["image_width"]),
                    ),
                    dtype=np.float32,
                )
            )
            return obs, self._new_fail_case(0), False, info

        if bool(planner_inputs.get("new_goal", False)):
            self._clear_flag = 0

        action = self._plan(planner_inputs)
        sensor_pose, collision = self._executor.execute(action)

        # Advance physics so the renderer reflects the new robot state.
        self._env.step()

        self._step_count += 1

        rgb, depth, semantic = self._get_sensors()
        obs = self._obs_adapter.adapt(rgb, depth, semantic)
        self._obs = obs

        goal_reached = self._check_goal_reached(
            semantic, self._goal_cat_id, self._goal_name
        )
        self._done = self._check_done(action, goal_reached)

        collision_int = int(collision)
        fail_case = self._new_fail_case(collision_int)
        info = self._build_info(sensor_pose=sensor_pose, collision_int=collision_int)

        if self._done:
            info["spl"] = 0.0
            info["success"] = int(goal_reached)
            info["distance_to_goal"] = 0.0

        self._last_action = action
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
        semantic : (H, W) int32   (L3MVN semantic id space)
        """
        env_shim = _SimAsEnv(self._env, self._vision_sensor.config)
        obs = self._vision_sensor.get_obs(env_shim)

        # rgb: (H, W, 3) float32 [0,1] → uint8 [0,255]
        rgb = (np.clip(obs["rgb"], 0.0, 1.0) * 255).astype(np.uint8)

        # depth: VisionSensor returns (H,W,1) float32 [0,1] normalised by
        # depth_high.  Convert back to metres for the raw depth value.
        depth_norm = obs["depth"]                  # (H, W, 1) float32
        depth = (depth_norm[:, :, 0] * self._vision_sensor.depth_high).astype(np.float32)

        # semantic: VisionSensor returns iGibson class ids.
        # Convert to L3MVN semantic ids (Stage-1 single-channel id format).
        semantic_raw = obs["seg"][:, :, 0].astype(np.int32)
        semantic = np.zeros_like(semantic_raw, dtype=np.int32)
        for class_id, sem_id in self._id_to_l3mvn_sem_id.items():
            semantic[semantic_raw == class_id] = sem_id

        return rgb, depth, semantic

    def _build_semantic_lookup(self, class_id_to_name: dict[int, str]) -> dict[int, int]:
        """Build iGibson id -> L3MVN semantic id lookup once per wrapper."""
        return self._taxonomy.build_id_to_l3mvn_semantic_id(class_id_to_name)

    def _new_fail_case(self, collision_int: int) -> dict:
        """Return fail_case dict with required key set."""
        return {
            "collision": int(collision_int),
            "success": 0,
            "detection": 0,
            "exploration": 0,
        }

    def _build_info(self, sensor_pose, collision_int: int) -> dict:
        return {
            "sensor_pose": sensor_pose,
            "eve_angle": self._executor.eve_angle,
            "goal_cat_id": self._goal_cat_id,
            "goal_name": self._goal_name,
            "clear_flag": self._clear_flag,
            "collision": int(collision_int),
        }

    def close(self) -> None:
        """Disconnect / close the underlying iGibson environment or simulator.

        Calls ``disconnect()`` if available, falling back to ``close()``.
        Safe to call if neither method exists.
        """
        env = self._env
        if hasattr(env, "disconnect"):
            env.disconnect()
        elif hasattr(env, "close"):
            env.close()

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

    def _plan(self, planner_inputs: dict) -> int:
        """Habitat-compatible local planner: 8-key planner_inputs -> discrete action."""
        map_pred = np.rint(np.asarray(planner_inputs["map_pred"], dtype=np.float32))
        exp_pred = np.rint(np.asarray(planner_inputs["exp_pred"], dtype=np.float32))
        goal = np.asarray(planner_inputs["goal"], dtype=np.float32)
        pose_pred = np.asarray(planner_inputs["pose_pred"], dtype=np.float32)

        if pose_pred.shape[0] != 7:
            raise ValueError(
                f"pose_pred must have shape (7,), got {pose_pred.shape}"
            )

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred.tolist()
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        needed_size = max(gx2, gy2, gx1 + map_pred.shape[0], gy1 + map_pred.shape[1], 1)
        self._ensure_map_buffers(needed_size)
        assert self._visited is not None
        assert self._visited_vis is not None
        assert self._collision_map is not None

        if self._curr_loc is None:
            self._curr_loc = [float(start_x), float(start_y), float(start_o)]

        self._last_loc = self._curr_loc
        self._curr_loc = [float(start_x), float(start_y), float(start_o)]

        # Get current location in local map coordinates.
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / self._map_resolution - gx1),
            int(c * 100.0 / self._map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self._visited[gx1:gx2, gy1:gy2][start[0]:start[0] + 1, start[1]:start[1] + 1] = 1

        # Draw trajectory in local window for planning regularization.
        last_start_x, last_start_y = self._last_loc[0], self._last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [
            int(r * 100.0 / self._map_resolution - gx1),
            int(c * 100.0 / self._map_resolution - gy1),
        ]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self._visited_vis[gx1:gx2, gy1:gy2] = cv2.line(
            self._visited_vis[gx1:gx2, gy1:gy2].copy(),
            (int(last_start[1]), int(last_start[0])),
            (int(start[1]), int(start[0])),
            color=1,
            thickness=1,
        )

        # Collision map update from motion mismatch (same logic as Habitat path).
        if self._last_action == ACTION_FORWARD and not bool(planner_inputs.get("new_goal", False)):
            x1, y1, t1 = self._last_loc
            x2, y2, _ = self._curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self._col_width += 2
                if self._col_width == 7:
                    length = 4
                    buf = 3
                self._col_width = min(self._col_width, 5)
            else:
                self._col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < self._collision_threshold:
                self._collision_n += 1
                width = self._col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * (
                            (i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1))
                        )
                        wy = y1 + 0.05 * (
                            (i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1))
                        )
                        rr, cc = int(wy * 100 / self._map_resolution), int(
                            wx * 100 / self._map_resolution
                        )
                        rr, cc = pu.threshold_poses([rr, cc], self._collision_map.shape)
                        self._collision_map[rr, cc] = 1

        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)
        if replan:
            self._replan_count += 1
        else:
            self._replan_count = 0

        if self._collision_n > 20 or self._replan_count > 26:
            self._clear_flag = 1
            self._collision_n = 0

        if (stop and int(planner_inputs.get("found_goal", 0)) == 1) or self._replan_count > 26:
            return ACTION_STOP

        stg_x, stg_y = stg
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
        angle_agent = float(start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360
        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        eve_start_x = int(5 * math.sin(math.radians(angle_st_goal)) + start[0])
        eve_start_y = int(5 * math.cos(math.radians(angle_st_goal)) + start[1])
        eve_start_x = int(np.clip(eve_start_x, 0, map_pred.shape[0] - 1))
        eve_start_y = int(np.clip(eve_start_y, 0, map_pred.shape[1] - 1))

        if exp_pred[eve_start_x, eve_start_y] == 0 and self._executor.eve_angle > -60:
            return ACTION_LOOK_DOWN
        if exp_pred[eve_start_x, eve_start_y] == 1 and self._executor.eve_angle < 0:
            return ACTION_LOOK_UP
        if relative_angle > self._turn_angle_deg / 2.0:
            return ACTION_TURN_RIGHT
        if relative_angle < -self._turn_angle_deg / 2.0:
            return ACTION_TURN_LEFT
        return ACTION_FORWARD

    def _get_stg(
        self,
        grid: np.ndarray,
        start: list[int],
        goal: np.ndarray,
        planning_window: list[int],
    ) -> tuple[tuple[float, float], bool, bool]:
        """Compute short-term goal via FMM planner."""
        from envs.utils.fmm_planner import FMMPlanner

        gx1, gx2, gy1, gy2 = planning_window
        x1, y1 = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat: np.ndarray, value: float = 1.0) -> np.ndarray:
            h, w = mat.shape
            out = np.zeros((h + 2, w + 2), dtype=np.float32) + value
            out[1:h + 1, 1:w + 1] = mat
            return out

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2], self._selem
        ) != True
        assert self._collision_map is not None
        assert self._visited_vis is not None

        traversible[
            self._collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1
        ] = 0
        traversible[
            cv2.dilate(
                self._visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self._kernel
            )
            == 1
        ] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = add_boundary(traversible.astype(np.float32))
        goal = add_boundary(goal.astype(np.float32), value=0.0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(goal, selem) != True
        goal = 1 - goal * 1.0
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        return (stg_x, stg_y), replan, stop

    def _init_map_buffers(self, size: int) -> None:
        map_size = max(1, int(size))
        self._collision_map = np.zeros((map_size, map_size), dtype=np.float32)
        self._visited = np.zeros((map_size, map_size), dtype=np.float32)
        self._visited_vis = np.zeros((map_size, map_size), dtype=np.uint8)

    def _ensure_map_buffers(self, min_size: int) -> None:
        if self._collision_map is None or self._visited is None or self._visited_vis is None:
            self._init_map_buffers(min_size)
            return

        cur = self._collision_map.shape[0]
        if cur >= min_size:
            return

        new_size = max(min_size, int(cur * 1.5) + 1)
        new_collision = np.zeros((new_size, new_size), dtype=np.float32)
        new_visited = np.zeros((new_size, new_size), dtype=np.float32)
        new_visited_vis = np.zeros((new_size, new_size), dtype=np.uint8)

        new_collision[:cur, :cur] = self._collision_map
        new_visited[:cur, :cur] = self._visited
        new_visited_vis[:cur, :cur] = self._visited_vis

        self._collision_map = new_collision
        self._visited = new_visited
        self._visited_vis = new_visited_vis
