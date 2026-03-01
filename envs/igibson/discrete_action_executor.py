"""Discrete action executor for the iGibson adapter layer.

This module mirrors the 6-action contract used by L3MVN SemExp:
0 STOP, 1 FORWARD, 2 TURN_LEFT, 3 TURN_RIGHT, 4 LOOK_UP, 5 LOOK_DOWN.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from envs.utils.pose import get_rel_pose_change


class DiscreteActionExecutor:
    """Executes L3MVN-style discrete actions and reports adapter info."""

    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5

    MIN_EVE_ANGLE = -60
    MAX_EVE_ANGLE = 0

    def __init__(
        self,
        robot,
        forward_step_m: float = 0.25,
        turn_angle_deg: float = 30.0,
        eve_angle_step_deg: float = 30.0,
    ):
        self.robot = robot
        self.forward_step_m = float(forward_step_m)
        self.turn_angle_deg = float(turn_angle_deg)
        self.eve_angle_step_deg = float(eve_angle_step_deg)

        if self.forward_step_m <= 0:
            raise ValueError("forward_step_m must be > 0.")
        if self.turn_angle_deg <= 0:
            raise ValueError("turn_angle_deg must be > 0.")
        if self.eve_angle_step_deg <= 0:
            raise ValueError("eve_angle_step_deg must be > 0.")

        self.eve_angle = 0
        self._last_pose = None

    def reset(self) -> None:
        """Reset eve_angle to 0, record current robot pose as baseline."""
        self.eve_angle = 0
        self._set_camera_elevation(self.eve_angle)
        self._last_pose = self._get_robot_pose()

    def step(self, action_id: int) -> Dict[str, object]:
        """Execute one discrete action and return L3MVN-compatible info."""
        if action_id not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"Invalid action_id: {action_id}. Expected 0..5.")

        done = action_id == self.STOP
        sensor_pose: List[float] = [0.0, 0.0, 0.0]

        if action_id == self.FORWARD:
            sensor_pose = self._run_motion_and_measure(
                lambda: self._move_forward(self.forward_step_m)
            )
        elif action_id == self.TURN_LEFT:
            sensor_pose = self._run_motion_and_measure(
                lambda: self._turn_left(self.turn_angle_deg)
            )
        elif action_id == self.TURN_RIGHT:
            sensor_pose = self._run_motion_and_measure(
                lambda: self._turn_right(self.turn_angle_deg)
            )
        elif action_id == self.LOOK_UP:
            self.eve_angle = int(
                min(self.MAX_EVE_ANGLE, self.eve_angle + self.eve_angle_step_deg)
            )
            self._set_camera_elevation(self.eve_angle)
        elif action_id == self.LOOK_DOWN:
            self.eve_angle = int(
                max(self.MIN_EVE_ANGLE, self.eve_angle - self.eve_angle_step_deg)
            )
            self._set_camera_elevation(self.eve_angle)
        # action_id == STOP: no base motion, no camera tilt update.

        return {
            "sensor_pose": sensor_pose,
            "eve_angle": int(self.eve_angle),
            "done": bool(done),
        }

    def _run_motion_and_measure(self, motion_fn: Callable[[], None]) -> List[float]:
        if self._last_pose is None:
            self._last_pose = self._get_robot_pose()

        start_pose = self._last_pose
        motion_fn()
        end_pose = self._get_robot_pose()

        # Use the same relative-pose equation as L3MVN.
        dx, dy, do = get_rel_pose_change(end_pose, start_pose)
        self._last_pose = end_pose
        return [float(dx), float(dy), float(do)]

    def _get_robot_pose(self) -> Tuple[float, float, float]:
        # TODO(iGibson): wire to robot base pose query API and return
        # (x, y, yaw_rad) in world frame.
        getter = getattr(self.robot, "get_base_pose", None)
        if callable(getter):
            pose = getter()
            if len(pose) != 3:
                raise ValueError("robot.get_base_pose() must return 3 values.")
            return float(pose[0]), float(pose[1]), float(pose[2])
        raise NotImplementedError(
            "TODO: connect iGibson base pose query via robot.get_base_pose()."
        )

    def _move_forward(self, distance_m: float) -> None:
        # TODO(iGibson): wire to robot forward motion API (distance in meters).
        fn = getattr(self.robot, "move_forward", None)
        if callable(fn):
            fn(distance_m)
            return
        raise NotImplementedError(
            "TODO: connect iGibson forward motion via robot.move_forward(distance_m)."
        )

    def _turn_left(self, angle_deg: float) -> None:
        # TODO(iGibson): wire to robot left-turn API (angle in degrees).
        fn = getattr(self.robot, "turn_left", None)
        if callable(fn):
            fn(angle_deg)
            return
        raise NotImplementedError(
            "TODO: connect iGibson turn-left via robot.turn_left(angle_deg)."
        )

    def _turn_right(self, angle_deg: float) -> None:
        # TODO(iGibson): wire to robot right-turn API (angle in degrees).
        fn = getattr(self.robot, "turn_right", None)
        if callable(fn):
            fn(angle_deg)
            return
        raise NotImplementedError(
            "TODO: connect iGibson turn-right via robot.turn_right(angle_deg)."
        )

    def _set_camera_elevation(self, elevation_deg: int) -> None:
        # TODO(iGibson): wire to camera tilt API (elevation in degrees).
        fn = getattr(self.robot, "set_camera_elevation", None)
        if callable(fn):
            fn(elevation_deg)
            return
        raise NotImplementedError(
            "TODO: connect iGibson camera tilt via robot.set_camera_elevation(elevation_deg)."
        )
