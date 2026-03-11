"""Integration smoke test: EnvWrapper + real iGibson simulator.

Run from the workspace root:
    python tests/smoke_test_env_wrapper_igibson.py

This script wires together:
  - iGibson Simulator + InteractiveIndoorScene + Turtlebot
  - DiscreteActionExecutor
  - ObsAdapter
  - SemanticTaxonomy
  - EnvWrapper

and exercises reset() + a fixed action sequence through
plan_act_and_preprocess(), printing diagnostic info at each step
and asserting core L3MVN contracts.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# ---------------------------------------------------------------------------
# iGibson dataset path override (same as smoke_test_discrete_action.py)
# ---------------------------------------------------------------------------
import igibson

igibson.assets_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/assets"
igibson.ig_dataset_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset"
igibson.key_path = "/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key"

# ---------------------------------------------------------------------------
import pybullet as p
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.semantics_utils import get_class_name_to_class_id

from envs.igibson.discrete_action_executor import (
    ACTION_FORWARD,
    ACTION_LOOK_DOWN,
    ACTION_LOOK_UP,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    DiscreteActionExecutor,
)
from envs.igibson.env_wrapper import EnvWrapper
from envs.igibson.obs_adapter import ObsAdapter
from envs.igibson.semantic_taxonomy import SemanticTaxonomy

ACTION_NAMES = {
    ACTION_STOP: "STOP",
    ACTION_FORWARD: "FORWARD",
    ACTION_TURN_LEFT: "TURN_LEFT",
    ACTION_TURN_RIGHT: "TURN_RIGHT",
    ACTION_LOOK_UP: "LOOK_UP",
    ACTION_LOOK_DOWN: "LOOK_DOWN",
}

# Action sequence to exercise all 5 movement types
ACTION_SEQUENCE = [
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_LOOK_UP,
    ACTION_LOOK_DOWN,
] * 10

SIM_STEPS_PER_ACTION = 5


# ---------------------------------------------------------------------------
# Observation validation helpers
# ---------------------------------------------------------------------------

def summarize_observation(obs: np.ndarray) -> str:
    """Return a one-line diagnostic string for a (5, H, W) Stage-1 obs tensor."""
    rgb = obs[:3]
    depth = obs[3]
    sem_id = obs[4].astype(np.int32)
    uniq = np.unique(sem_id)
    return (
        f"rgb=[{rgb.min():.1f},{rgb.max():.1f},var={rgb.var():.1f}] "
        f"depth=[{depth.min():.2f},{depth.max():.2f},nuniq={len(np.unique(depth))}] "
        f"sem_id=[min={sem_id.min()},max={sem_id.max()},nuniq={len(uniq)}]"
    )


def _allowed_l3mvn_semantic_ids(class_id_to_name: dict[int, str]) -> set[int]:
    """Semantic IDs that may appear after taxonomy remap."""
    mapped = {
        SemanticTaxonomy.map_class_name_to_l3mvn_semantic_id(name)
        for name in class_id_to_name.values()
    }
    mapped.add(0)
    return mapped


def assert_stage1_observation_contract(
    obs: np.ndarray,
    class_id_to_name: dict[int, str],
) -> None:
    """Assert Stage-1 contract: non-empty + L3MVN semantic-id format."""
    rgb = obs[:3]
    depth = obs[3]
    sem_id = obs[4]

    # RGB: sum > 0 and variance > 0
    assert rgb.sum() > 0, "RGB block is all zeros"
    assert rgb.var() > 0, "RGB block has zero variance (solid colour)"

    # Depth: not all identical
    assert len(np.unique(depth)) > 1, "Depth block is a single uniform value"

    # Semantic ID: L3MVN id space (0..15), integer-valued.
    sem_id_i32 = sem_id.astype(np.int32)
    np.testing.assert_array_equal(sem_id, sem_id_i32.astype(np.float32))
    assert sem_id_i32.min() >= 0, f"Semantic ID has negative values: {sem_id_i32.min()}"
    assert sem_id_i32.max() <= 15, f"Semantic ID exceeds L3MVN range: {sem_id_i32.max()}"

    allowed = _allowed_l3mvn_semantic_ids(class_id_to_name)
    uniq = set(np.unique(sem_id_i32).tolist())
    assert uniq.issubset(allowed), (
        f"Unexpected semantic IDs found: {sorted(uniq - allowed)}; "
        f"allowed subset example: {sorted(list(allowed))[:16]}"
    )


def _build_class_id_to_name() -> dict[int, str]:
    """Invert iGibson's CLASS_NAME_TO_CLASS_ID → {id: name}."""
    name_to_id = get_class_name_to_class_id()
    return {v: k for k, v in name_to_id.items()}


def _spawn_collision_free(robot, scene, sim, *, max_attempts=10):
    """Place the robot at a collision-free traversable point."""
    start_orn = np.array([0.0, 0.0, 0.0, 1.0])
    z_offset = 0.1
    try:
        floor_z = float(scene.floor_heights[0])
    except (AttributeError, IndexError, TypeError):
        floor_z = 0.0
    floor_ids = set(scene.floor_body_ids)
    body_id = robot.get_body_ids()[0]

    def _has_collision():
        p.performCollisionDetection()
        for c in p.getContactPoints(bodyA=body_id):
            b = c[2]
            if b == body_id or b in floor_ids:
                continue
            if abs(c[7][2]) > 0.7 and c[5][2] < floor_z + 0.15:
                continue
            return True
        return False

    for attempt in range(max_attempts):
        _, pos = scene.get_random_point(floor=0)
        pos = pos.copy()
        pos[2] += z_offset
        robot.set_position_orientation(pos, start_orn)
        robot.reset()
        robot.keep_still()
        for _ in range(5):
            sim.step()
        if not _has_collision():
            print(f"  Clean spawn after {attempt + 1} attempt(s)")
            return
    print("  WARNING: using last spawn position (may have collision)")


def main():
    print("=" * 60)
    print("EnvWrapper integration smoke test")
    print("=" * 60)

    # ---- 1. Create simulator ----
    print("\n[1] Creating Simulator ...")
    settings = MeshRendererSettings(
        enable_shadow=False,
        enable_pbr=False,
        msaa=False,
        optimized=True,
        load_textures=True,
    )
    sim = Simulator(
        mode="gui_interactive",
        image_width=160,
        image_height=120,
        vertical_fov=45,
        physics_timestep=1 / 240.0,
        render_timestep=1 / 10.0,
        rendering_settings=settings,
    )

    print("[1] Loading scene Rs_int ...")
    scene = InteractiveIndoorScene(
        "Rs_int",
        trav_map_type="no_obj",
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        build_graph=True,
    )
    sim.import_scene(scene)

    print("[1] Loading Turtlebot ...")
    robot = Turtlebot(action_type="continuous")
    sim.import_object(robot)

    # ---- 2. Spawn robot ----
    print("[2] Spawning robot ...")
    _spawn_collision_free(robot, scene, sim)
    for _ in range(10):
        sim.step()

    # ---- 3. Build components ----
    print("[3] Building adapter components ...")
    executor = DiscreteActionExecutor(robot=robot, scene=scene)
    obs_adapter = ObsAdapter()
    class_id_to_name = _build_class_id_to_name()

    wrapper = EnvWrapper(
        igibson_env=sim,
        robot=robot,
        scene=scene,
        action_executor=executor,
        obs_adapter=obs_adapter,
        semantic_taxonomy=SemanticTaxonomy,
        goal_name="chair",
        goal_cat_id=1,
        class_id_to_name=class_id_to_name,
    )

    # ---- 4. Reset ----
    print("[4] Calling wrapper.reset() ...")
    obs, info = wrapper.reset()
    print(f"  obs shape : {obs.shape}")
    print(f"  obs dtype : {obs.dtype}")
    print(f"  sensor_pose: {info['sensor_pose']}")
    print(f"  eve_angle  : {info['eve_angle']}")
    print(f"  goal_name  : {info['goal_name']}")
    print(f"  obs detail : {summarize_observation(obs)}")

    assert obs.shape == (5, 120, 160), f"Expected (5,120,160), got {obs.shape}"
    assert obs.dtype == np.float32
    print("  [PASS] reset obs shape & dtype")

    assert_stage1_observation_contract(obs, class_id_to_name)
    print("  [PASS] reset observation satisfies Stage-1 contract (L3MVN semantic-id)")

    # ---- 5. Action sequence ----
    print(f"\n[5] Running action sequence: {[ACTION_NAMES[a] for a in ACTION_SEQUENCE]}")
    header = (
        f"{'step':>4}  {'action':<12}  "
        f"{'sensor_pose':>30}  {'eve_angle':>9}  "
        f"{'collision':>9}  {'done':>4}  obs_shape"
    )
    print(header)
    print("-" * len(header))

    for step_i, action_id in enumerate(ACTION_SEQUENCE):
        obs, fail_case, done, info = wrapper.plan_act_and_preprocess(
            {"action": action_id}
        )

        sp = info["sensor_pose"]
        sp_str = f"[{sp[0]:+.4f}, {sp[1]:+.4f}, {math.degrees(sp[2]):+.2f}°]"

        print(
            f"{step_i:>4}  {ACTION_NAMES[action_id]:<12}  "
            f"{sp_str:>30}  {info['eve_angle']:>9}  "
            f"{'TRUE' if info['collision'] else 'false':>9}  "
            f"{'TRUE' if done else 'false':>4}  {obs.shape}"
        )

        # --- per-step asserts ---
        assert obs.shape == (5, 120, 160)

        # fail_case required keys
        for k in ("collision", "success", "detection", "exploration"):
            assert k in fail_case, f"Missing fail_case key: {k}"

        # LOOK_UP / LOOK_DOWN → sensor_pose must be [0,0,0]
        if action_id in (ACTION_LOOK_UP, ACTION_LOOK_DOWN):
            assert info["sensor_pose"] == [0.0, 0.0, 0.0], (
                f"LOOK action should have zero sensor_pose, got {info['sensor_pose']}"
            )

        # Non-empty observation: check first and last steps
        if step_i == 0 or step_i == len(ACTION_SEQUENCE) - 1:
            assert_stage1_observation_contract(obs, class_id_to_name)
            print(f"       [PASS] step {step_i} observation satisfies Stage-1 contract")

        # Let physics settle
        for _ in range(SIM_STEPS_PER_ACTION):
            sim.step()

        if done:
            print(f"  Episode done at step {step_i}")
            break

    # ---- 6. Summary ----
    print("\n" + "=" * 60)
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)

    sim.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
