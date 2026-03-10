"""Smoke test for DiscreteActionExecutor with iGibson GUI viewer.

Run from the workspace root:
    python tests/smoke_test_discrete_action.py

What this test does
-------------------
1. Opens the iGibson GUI (gui_interactive mode) with Rs_int scene and a Turtlebot.
2. Spawns the robot at a random traversable point on floor 0.
3. Executes a fixed action sequence and prints a debug table after each action.
4. Keeps the GUI open so you can visually inspect the robot's final position.
5. Prints the robot body ID and floor body IDs for API verification.

Scene / sensor settings aligned with turtlebot_static_nav.yaml
--------------------------------------------------------------
  scene:         Rs_int / InteractiveIndoorScene  (Rs with StaticIndoorScene would
                 be ideal, but only ig_dataset is present — no g_dataset/Rs)
  image_width:   640   (yaml: 640)
  image_height:  480   (yaml: 480)
  vertical_fov:  45    (yaml: 45)
  physics ts:    1/240 (iGibsonEnv default)
  render ts:     1/10  (iGibsonEnv action_timestep default)
  textures:      enabled (yaml: load_texture: true)
  trav_map_type: no_obj (yaml: trav_map_type: no_obj)
  spawn z-offset: 0.1  (yaml: initial_pos_z_offset: 0.1)

use_pb_gui note
---------------
Simulator(use_pb_gui=True) calls p.connect(p.GUI) for the PyBullet debug window.
This is a first-class Simulator constructor parameter and works alongside
mode='gui_interactive' (iGibson mesh renderer) — both windows open simultaneously.

Debug table columns
-------------------
step  action       dx(m)   dy(m)   do(deg)  collision  eve_angle  world_pos(x,y,z)
"""

import math
import signal
import sys
import time
import os

# ---------------------------------------------------------------------------
# Dataset path patch — must happen before any igibson sub-module is imported.
# The installed iGibson reads paths from its own global_config.yaml; we
# override those attributes directly after the top-level __init__ runs.
# ---------------------------------------------------------------------------
# sys.path.insert(0, "/mount/nas2/users/dukim/vla_ws/iGibson")

import igibson  # noqa: E402  (import after sys.path patch)

igibson.assets_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/assets"
igibson.ig_dataset_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset"
igibson.key_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key"

# ---------------------------------------------------------------------------
# Regular imports (after path patch)
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402

from igibson.simulator import Simulator  # noqa: E402
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene  # noqa: E402
from igibson.robots.turtlebot import Turtlebot  # noqa: E402
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings  # noqa: E402

# L3MVN workspace must be on PYTHONPATH (set via docker-compose or manually)
from envs.igibson.discrete_action_executor import (  # noqa: E402
    DiscreteActionExecutor,
    ACTION_STOP,
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_LOOK_UP,
    ACTION_LOOK_DOWN,
)

# ---------------------------------------------------------------------------
# Action sequence to execute
# ---------------------------------------------------------------------------
ACTION_SEQUENCE = [
    ACTION_FORWARD,
    ACTION_FORWARD,
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_FORWARD,
    ACTION_FORWARD,
    ACTION_FORWARD,
    ACTION_TURN_RIGHT,
    ACTION_LOOK_DOWN,
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_LOOK_UP,
    ACTION_TURN_RIGHT,
    ACTION_TURN_RIGHT,
    ACTION_TURN_RIGHT,
    ACTION_FORWARD,
]

ACTION_SEQUENCE.extend(
    [ACTION_FORWARD] * 100
)
ACTION_SEQUENCE.append(
    ACTION_STOP,
)

ACTION_NAMES = {
    ACTION_STOP: "STOP",
    ACTION_FORWARD: "FORWARD",
    ACTION_TURN_LEFT: "TURN_LEFT",
    ACTION_TURN_RIGHT: "TURN_RIGHT",
    ACTION_LOOK_UP: "LOOK_UP",
    ACTION_LOOK_DOWN: "LOOK_DOWN",
}

# Number of sim steps to run between actions so the GUI is visually responsive.
# With render_timestep=1/10 each s.step() is heavier (24 physics sub-steps);
# 2 steps = 0.2 simulated seconds is enough for the viewer to update.
SIM_STEPS_PER_ACTION = 10


def main():
    # ------------------------------------------------------------------
    # 1. Create simulator and scene
    # ------------------------------------------------------------------
    print("[smoke-test] Creating Simulator (gui_interactive) ...")
    # Renderer settings: match turtlebot_static_nav.yaml (load_texture=true,
    # no shadows, no PBR, optimized renderer).
    rendering_settings = MeshRendererSettings(
        enable_shadow=False,
        enable_pbr=False,
        msaa=False,
        optimized=True,
        load_textures=True,
    )
    # physics_timestep / render_timestep match iGibsonEnv defaults used by
    # turtlebot_static_nav.yaml (physics=1/240, action_timestep=1/10).
    s = Simulator(
        mode="gui_interactive",
        use_pb_gui=False, #True,           # also opens PyBullet debug window
        image_width=640,           # yaml: image_width: 640
        image_height=480,          # yaml: image_height: 480
        vertical_fov=45,           # yaml: vertical_fov: 45
        physics_timestep=1 / 240.0,
        render_timestep=1 / 10.0,
        rendering_settings=rendering_settings,
    )

    print("[smoke-test] Loading Rs_int scene ...")
    # turtlebot_static_nav.yaml uses scene: gibson / scene_id: Rs, but Gibson
    # dataset (g_dataset) is not present — only ig_dataset with *_int scenes.
    # Rs_int is the closest available scene.  trav_map_type="no_obj" and
    # trav_map_erosion=2 are taken directly from the yaml.
    scene = InteractiveIndoorScene(
        "Rs_int",
        trav_map_type="no_obj",   # yaml: trav_map_type: no_obj
        trav_map_resolution=0.1,  # yaml: trav_map_resolution: 0.1
        trav_map_erosion=2,       # yaml: trav_map_erosion: 2
        build_graph=True,         # yaml: build_graph: true
    )
    s.import_scene(scene)

    # ------------------------------------------------------------------
    # 2. Load robot
    # ------------------------------------------------------------------
    print("[smoke-test] Loading Turtlebot ...")
    robot = Turtlebot(action_type="continuous")
    s.import_object(robot)

    # ------------------------------------------------------------------
    # 3. Spawn at a collision-free traversable point
    # ------------------------------------------------------------------
    # trav_map_type="no_obj" computes walkability from the static geometry
    # only — interactive objects (chairs, sofas, tables …) are NOT marked as
    # obstacles.  This means get_random_point() can return a position that is
    # physically occupied by furniture.  We retry until the robot settles
    # without contacting any non-floor body.
    INITIAL_POS_Z_OFFSET = 0.1
    start_orn = np.array([0.0, 0.0, 0.0, 1.0])

    try:
        floor_z = float(scene.floor_heights[0])
    except (AttributeError, IndexError, TypeError):
        floor_z = 0.0
    floor_ids = set(scene.floor_body_ids)
    robot_body_id = robot.get_body_ids()[0]

    def _has_object_collision():
        """Return True if robot currently contacts any non-floor body."""
        p.performCollisionDetection()
        contacts = p.getContactPoints(bodyA=robot_body_id)
        if not contacts:
            return False
        for c in contacts:
            body_b = c[2]
            if body_b == robot_body_id or body_b in floor_ids:
                continue
            nz = c[7][2]   # contact normal Z on bodyB
            cz = c[5][2]   # contact point world Z on robot
            if abs(nz) > 0.7 and cz < floor_z + 0.15:
                continue   # genuine floor contact
            return True    # lateral or raised-object contact
        return False

    MAX_SPAWN_ATTEMPTS = 10
    start_pos = None
    for attempt in range(MAX_SPAWN_ATTEMPTS):
        _, pos = scene.get_random_point(floor=0)
        pos = pos.copy()
        pos[2] += INITIAL_POS_Z_OFFSET
        robot.set_position_orientation(pos, start_orn)
        robot.reset()
        robot.keep_still()
        # A few physics steps to let the robot settle and register contacts.
        for _ in range(5):
            s.step()
        if not _has_object_collision():
            start_pos = pos
            print(f"[smoke-test] Clean spawn after {attempt + 1} attempt(s).")
            break
        print(f"[smoke-test] Spawn attempt {attempt + 1}: object collision detected, retrying …")
    if start_pos is None:
        start_pos = pos
        print("[smoke-test] WARNING: no collision-free spawn found; using last position.")

    # Finish settling.
    for _ in range(15):
        s.step()

    # ------------------------------------------------------------------
    # 4. Print API info (body-ID verification)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("API verification")
    print("  robot.get_body_ids()  :", robot.get_body_ids())
    print("  scene.floor_body_ids  :", scene.floor_body_ids)
    print("  spawn position        :", start_pos)
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 5. Build executor
    # ------------------------------------------------------------------
    executor = DiscreteActionExecutor(
        robot=robot,
        scene=scene,
        forward_step_m=0.10,
        turn_angle_deg=30.0,
        eve_angle_step_deg=30,
    )

    # ------------------------------------------------------------------
    # 6. Print table header
    # ------------------------------------------------------------------
    header = (
        f"{'step':>4}  "
        f"{'action':<12}  "
        f"{'dx(m)':>8}  "
        f"{'dy(m)':>8}  "
        f"{'do(deg)':>8}  "
        f"{'collision':>9}  "
        f"{'eve_ang':>7}  "
        f"world_pos(x, y, z)"
    )
    print(header)
    print("-" * len(header))

    # ------------------------------------------------------------------
    # 7. Execute action sequence
    # ------------------------------------------------------------------
    for step, action_id in enumerate(ACTION_SEQUENCE):
        sensor_pose, collision = executor.execute(action_id)

        dx, dy, do = sensor_pose
        pos = robot.get_position()

        print(
            f"{step:>4}  "
            f"{ACTION_NAMES[action_id]:<12}  "
            f"{dx:>8.4f}  "
            f"{dy:>8.4f}  "
            f"{math.degrees(do):>8.3f}  "
            f"{'TRUE' if collision else 'false':>9}  "
            f"{executor.eve_angle:>7}  "
            f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
        )

        # Give the GUI time to render the updated state.
        for _ in range(SIM_STEPS_PER_ACTION):
            s.step()

    # ------------------------------------------------------------------
    # 8. Keep GUI open — Ctrl+C (or SIGTERM) terminates cleanly
    # ------------------------------------------------------------------
    print()
    print("[smoke-test] Sequence complete.  GUI window is active.")
    print("[smoke-test] Move the camera in the viewer window with mouse/keyboard.")
    print("[smoke-test] Press enter in this terminal (or Ctrl+C) to exit and clean up.")

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    s.disconnect()
    print("[smoke-test] Done.")


if __name__ == "__main__":
    main()
