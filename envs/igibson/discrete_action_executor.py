"""Discrete action executor for the iGibson adapter layer.

Action IDs (matching agents/sem_exp.py):
    0  STOP        – episode end trigger; no movement
    1  FORWARD     – translate forward by forward_step_m (teleport + collision check)
    2  TURN_LEFT   – rotate yaw +turn_angle_deg        (teleport + collision check)
    3  TURN_RIGHT  – rotate yaw -turn_angle_deg        (teleport + collision check)
    4  LOOK_UP     – increment eve_angle toward 0 deg  (internal state only, v1)
    5  LOOK_DOWN   – decrement eve_angle toward -60 deg (internal state only, v1)

Returns per execute():
    sensor_pose  : [dx, dy, do]  ego-frame delta (meters, radians)
                   compatible with infos['sensor_pose'] contract in L3MVN_ANALYSIS.md
    collision    : bool, True when the move was blocked and the robot was restored

TODO (post-v1):
    LOOK_UP/DOWN currently only manage the eve_angle state variable and do NOT
    tilt the physical camera.  Real camera elevation requires a robot that inherits
    ActiveCameraRobot (Fetch or Tiago in iGibson).  Switch the robot and send a
    velocity command to action[camera_control_idx_start] via its JointController.
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

import envs.utils.pose as pu

# ---------------------------------------------------------------------------
# Action ID constants (mirrors agents/sem_exp.py _plan logic)
# ---------------------------------------------------------------------------
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_LOOK_UP = 4
ACTION_LOOK_DOWN = 5

_ACTION_NAMES = {
    ACTION_STOP: "STOP",
    ACTION_FORWARD: "FORWARD",
    ACTION_TURN_LEFT: "TURN_LEFT",
    ACTION_TURN_RIGHT: "TURN_RIGHT",
    ACTION_LOOK_UP: "LOOK_UP",
    ACTION_LOOK_DOWN: "LOOK_DOWN",
}


class DiscreteActionExecutor:
    """Execute one discrete action per call using pybullet teleport.

    Parameters
    ----------
    robot : igibson.robots.robot_base.BaseRobot
        The loaded iGibson robot (e.g. Turtlebot).
    scene : igibson.scenes.scene_base.Scene
        The loaded scene; used to read ``scene.floor_body_ids`` so that floor
        contacts are excluded from the collision check.
    forward_step_m : float
        Distance (metres) to advance per FORWARD action.  Default 0.10 m.
    turn_angle_deg : float
        Rotation (degrees) per TURN_LEFT / TURN_RIGHT action.  Default 30°.
    eve_angle_step_deg : int
        Change in elevation angle (degrees) per LOOK_UP / LOOK_DOWN action.
        Default 30°.  Range is clamped to [−60, 0].
    """

    EVE_ANGLE_MIN: int = -60  # degrees
    EVE_ANGLE_MAX: int = 0    # degrees

    def __init__(
        self,
        robot,
        scene,
        forward_step_m: float = 0.10,
        turn_angle_deg: float = 30.0,
        eve_angle_step_deg: int = 30,
    ) -> None:
        self._robot = robot
        self._scene = scene
        self._forward_step_m = forward_step_m
        self._turn_angle_rad = math.radians(turn_angle_deg)
        self._eve_angle_step = eve_angle_step_deg
        self._eve_angle: int = 0
        # Approximate floor Z used by the collision-filter height guard.
        # scene.floor_heights is populated after import_scene() for both
        # StaticIndoorScene and InteractiveIndoorScene.
        try:
            self._floor_z = float(scene.floor_heights[0])
        except (AttributeError, IndexError, TypeError):
            self._floor_z = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def eve_angle(self) -> int:
        """Current camera elevation angle in degrees.  Range: [−60, 0]."""
        return self._eve_angle

    def reset(self) -> None:
        """Reset internal state (call at episode start)."""
        self._eve_angle = 0

    def execute(self, action_id: int):
        """Execute one discrete action.

        Parameters
        ----------
        action_id : int
            One of ACTION_STOP … ACTION_LOOK_DOWN (0–5).

        Returns
        -------
        sensor_pose : list[float]  length-3
            [dx, dy, do] ego-frame delta compatible with
            infos['sensor_pose'] as defined in L3MVN_ANALYSIS.md Section 3.
            Units: metres (dx, dy), radians (do).
        collision : bool
            True when the attempted move was blocked by an obstacle and the
            robot was teleported back to its previous pose.
        """
        if action_id == ACTION_STOP:
            return [0.0, 0.0, 0.0], False

        if action_id in (ACTION_LOOK_UP, ACTION_LOOK_DOWN):
            return self._execute_look(action_id)

        if action_id in (ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            return self._execute_move(action_id)

        raise ValueError("Unknown action_id: {}".format(action_id))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_look(self, action_id: int):
        """Handle LOOK_UP / LOOK_DOWN: only mutate eve_angle (v1)."""
        if action_id == ACTION_LOOK_UP:
            self._eve_angle = min(self._eve_angle + self._eve_angle_step, self.EVE_ANGLE_MAX)
        else:
            self._eve_angle = max(self._eve_angle - self._eve_angle_step, self.EVE_ANGLE_MIN)
        return [0.0, 0.0, 0.0], False

    def _execute_move(self, action_id: int):
        """Handle FORWARD / TURN_LEFT / TURN_RIGHT via teleport + collision check."""
        old_pos, old_orn = self._robot.get_position(), self._robot.get_orientation()
        before = self._pose_xyo(old_pos, old_orn)  # (x, y, yaw_rad)

        # Compute target pose
        yaw = before[2]
        if action_id == ACTION_FORWARD:
            new_pos = np.array([
                old_pos[0] + self._forward_step_m * math.cos(yaw),
                old_pos[1] + self._forward_step_m * math.sin(yaw),
                old_pos[2],
            ])
            new_orn = old_orn
        elif action_id == ACTION_TURN_LEFT:
            new_pos = old_pos
            new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw + self._turn_angle_rad])
        else:  # TURN_RIGHT
            new_pos = old_pos
            new_orn = p.getQuaternionFromEuler([0.0, 0.0, yaw - self._turn_angle_rad])

        collision = self._try_move(new_pos, new_orn, old_pos, old_orn)

        if collision:
            return [0.0, 0.0, 0.0], True

        after_pos, after_orn = self._robot.get_position(), self._robot.get_orientation()
        after = self._pose_xyo(after_pos, after_orn)
        dx, dy, do = pu.get_rel_pose_change(after, before)
        return [float(dx), float(dy), float(do)], False

    def _try_move(self, new_pos, new_orn, old_pos, old_orn) -> bool:
        """Teleport the robot to (new_pos, new_orn), check collisions, restore on hit.

        Returns True when a real obstacle was hit (floor/ceiling contacts are excluded).

        Collision filtering strategy (applied in order):
          1. ID-based: contacts with bodies in scene.floor_body_ids are excluded.
          2. Z-normal heuristic (fallback): contacts whose normal Z-component magnitude
             exceeds 0.7 are vertical surface contacts (floor or ceiling) and are
             excluded. This covers the case where scene.floor_body_ids is empty,
             as happens with InteractiveIndoorScene (Rs_int) in iGibson 2.x.
          3. Self-contact: contacts where bodyB == bodyA are excluded.
        """
        self._robot.set_position_orientation(new_pos, new_orn)
        p.performCollisionDetection()

        body_id = self._robot.get_body_ids()[0]
        contacts = p.getContactPoints(bodyA=body_id)

        hit = False
        if contacts:
            floor_ids = set(self._scene.floor_body_ids)
            for contact in contacts:
                body_b = contact[2]   # bodyB
                # 1. Self-contact
                if body_b == body_id:
                    continue
                # 2. ID-based floor exclusion
                if body_b in floor_ids:
                    continue
                # 3. Z-normal heuristic with height guard:
                #    |normalZ| > 0.7 normally means floor/ceiling, but a flat-top
                #    object (table, shelf, box) also produces a high-Z normal when
                #    the robot teleports onto it.  We add a height guard so that
                #    only contacts NEAR floor level are treated as floor contacts.
                #    Contacts whose contact point is more than 0.15 m above the
                #    scene floor are treated as real obstacles (올라타기 방지).
                contact_normal_z = contact[7][2]
                if abs(contact_normal_z) > 0.7:
                    contact_z = contact[5][2]  # world Z of contact point on robot
                    if contact_z < self._floor_z + 0.15:  # genuine floor contact
                        continue
                    # else: top surface of a raised object → real collision
                # This contact is a real lateral obstacle or raised-object surface.
                hit = True
                break

        if hit:
            self._robot.set_position_orientation(old_pos, old_orn)
        else:
            # Prevent physics drift: freeze joint velocities so that subsequent
            # s.step() calls do not move the robot away from the teleported pose.
            self._robot.keep_still()
        return hit

    @staticmethod
    def _pose_xyo(pos, orn) -> tuple:
        """Extract (x, y, yaw_radians) from position and quaternion."""
        yaw = p.getEulerFromQuaternion(orn)[2]
        return (float(pos[0]), float(pos[1]), float(yaw))
