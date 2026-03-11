"""Integration smoke test: SingleEnvVecWrapper + real iGibson simulator.

Run from the workspace root:
    python tests/smoke_test_vec_env_wrapper_igibson.py [--save-obs [DIR]]

Flags
-----
--save-obs [DIR]
    Save RGB (PNG), depth (PNG, uint16) and raw obs (npy) for each step.
    DIR defaults to ``./smoke_obs_out`` when not specified.

This script wires together:
  - iGibson Simulator + InteractiveIndoorScene + Turtlebot
  - DiscreteActionExecutor
  - ObsAdapter
  - SemanticTaxonomy
  - EnvWrapper  (Stage 1, (5,H,W) output)
  - SingleEnvVecWrapper  (batch interface, (1,5,H,W) output)

and exercises reset() + a fixed action sequence through
plan_act_and_preprocess(), asserting the batch/list interface contract.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# iGibson dataset path override
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
from envs.igibson.vec_env_wrapper import SingleEnvVecWrapper

ACTION_NAMES = {
    ACTION_STOP: "STOP",
    ACTION_FORWARD: "FORWARD",
    ACTION_TURN_LEFT: "TURN_LEFT",
    ACTION_TURN_RIGHT: "TURN_RIGHT",
    ACTION_LOOK_UP: "LOOK_UP",
    ACTION_LOOK_DOWN: "LOOK_DOWN",
}

# Action sequence covering all 5 movement types (no STOP to avoid early done)
ACTION_SEQUENCE = [
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_LOOK_UP,
    ACTION_LOOK_DOWN,
] * 10

SIM_STEPS_PER_ACTION = 5

REQUIRED_INFO_KEYS = ("sensor_pose", "eve_angle", "goal_cat_id", "goal_name", "clear_flag")
REQUIRED_FAIL_CASE_KEYS = ("collision", "success", "detection", "exploration")


# ---------------------------------------------------------------------------
# Observation validation helpers (Stage 1)
# ---------------------------------------------------------------------------

def summarize_obs_batch(obs_batch: np.ndarray) -> str:
    """Return a one-line diagnostic string for a (1, 5, H, W) obs_batch."""
    obs = obs_batch[0]  # (5, H, W)
    rgb = obs[:3]
    depth = obs[3]
    sem_id = obs[4]
    return (
        f"rgb=[{rgb.min():.1f},{rgb.max():.1f},var={rgb.var():.2f}] "
        f"depth=[{depth.min():.3f},{depth.max():.3f}] "
        f"sem_id=[{sem_id.min():.0f},{sem_id.max():.0f}]"
    )


def assert_nonempty_stage1_observation(obs: np.ndarray) -> None:
    """Assert Stage-1 obs (5,H,W) has non-blank RGB, depth, and semantic ID."""
    rgb = obs[:3]
    depth = obs[3]
    sem_id = obs[4]

    assert rgb.sum() > 0, "RGB block is all zeros"
    assert rgb.var() > 0, "RGB block has zero variance (solid colour)"
    assert len(np.unique(depth)) > 1, "Depth block is a single uniform value"
    assert sem_id.max() >= 0, "Semantic ID block has no valid ids"


# ---------------------------------------------------------------------------
# Setup helpers (shared with existing smoke tests)
# ---------------------------------------------------------------------------

def _build_class_id_to_name() -> dict[int, str]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_obs(obs_batch: np.ndarray, tag: str, out_dir: str) -> None:
    """Save RGB (PNG), depth (PNG uint16), and raw array (npy) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    obs = obs_batch[0]  # (5, H, W)

    # RGB: channels 0-2 as (H, W, 3) uint8
    rgb_hwc = np.clip(obs[:3].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    Image.fromarray(rgb_hwc).save(os.path.join(out_dir, f"{tag}_rgb.png"))

    # Depth: channel 3, scaled to uint16 for PNG (preserves relative values)
    depth = obs[3]  # (H, W) float32
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth_u16 = ((depth - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    else:
        depth_u16 = np.zeros_like(depth, dtype=np.uint16)
    Image.fromarray(depth_u16).save(os.path.join(out_dir, f"{tag}_depth.png"))

    # Raw observation batch (npy)
    np.save(os.path.join(out_dir, f"{tag}_obs.npy"), obs_batch)

    print(f"  [SAVED] {out_dir}/{tag}_rgb.png, {tag}_depth.png, {tag}_obs.npy")


def main(save_obs: bool = False, out_dir: str = "./smoke_obs_out"):
    print("=" * 60)
    print("SingleEnvVecWrapper integration smoke test")
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
    print("[3] Building components ...")
    executor = DiscreteActionExecutor(robot=robot, scene=scene)
    obs_adapter = ObsAdapter()
    class_id_to_name = _build_class_id_to_name()

    inner_wrapper = EnvWrapper(
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
    vec_env = SingleEnvVecWrapper(inner_wrapper)
    print(f"  num_envs = {vec_env.num_envs}")

    # ---- 4. Reset ----
    print("[4] Calling vec_env.reset() ...")
    obs_batch, infos_list = vec_env.reset()

    print(f"  obs_batch.shape : {obs_batch.shape}")
    print(f"  obs_batch.dtype : {obs_batch.dtype}")
    print(f"  len(infos_list) : {len(infos_list)}")
    print(f"  infos_list[0]['sensor_pose'] : {infos_list[0]['sensor_pose']}")
    print(f"  infos_list[0]['eve_angle']   : {infos_list[0]['eve_angle']}")
    print(f"  infos_list[0]['goal_name']   : {infos_list[0]['goal_name']}")
    print(f"  obs detail      : {summarize_obs_batch(obs_batch)}")

    assert obs_batch.shape == (1, 5, 120, 160), (
        f"Expected (1,5,120,160), got {obs_batch.shape}"
    )
    assert obs_batch.dtype == np.float32
    assert len(infos_list) == 1
    assert isinstance(infos_list[0], dict)
    for key in REQUIRED_INFO_KEYS:
        assert key in infos_list[0], f"Missing key in reset infos_list[0]: {key}"
    print("  [PASS] reset batch shape, dtype, infos contract")

    assert_nonempty_stage1_observation(obs_batch[0])
    print("  [PASS] reset observation is non-empty (Stage 1)")

    if save_obs:
        _save_obs(obs_batch, "reset", out_dir)

    # ---- 5. Action sequence (as single dict) ----
    print(f"\n[5a] Running action sequence (single dict): {[ACTION_NAMES[a] for a in ACTION_SEQUENCE]}")
    header = (
        f"{'step':>4}  {'action':<12}  "
        f"{'sensor_pose':>30}  {'eve':>5}  "
        f"{'coll':>4}  {'done':>4}  obs_batch.shape"
    )
    print(header)
    print("-" * len(header))

    for step_i, action_id in enumerate(ACTION_SEQUENCE):
        obs_batch, fail_case_batch, done_batch, infos_list = (
            vec_env.plan_act_and_preprocess({"action": action_id})
        )

        sp = infos_list[0]["sensor_pose"]
        try:
            sp_str = f"[{sp[0]:+.4f}, {sp[1]:+.4f}, {math.degrees(sp[2]):+.2f}°]"
        except (TypeError, IndexError):
            sp_str = str(sp)

        print(
            f"{step_i:>4}  {ACTION_NAMES[action_id]:<12}  "
            f"{sp_str:>30}  {infos_list[0]['eve_angle']:>5}  "
            f"{fail_case_batch[0]['collision']:>4}  "
            f"{'T' if done_batch[0] else 'F':>4}  {obs_batch.shape}"
        )

        # --- batch contract asserts ---
        assert obs_batch.shape[0] == 1, f"obs_batch first dim must be 1, got {obs_batch.shape}"
        assert obs_batch.shape == (1, 5, 120, 160)
        assert len(fail_case_batch) == 1
        assert isinstance(fail_case_batch[0], dict)
        assert len(done_batch) == 1
        assert len(infos_list) == 1
        assert isinstance(infos_list[0], dict)

        for k in REQUIRED_FAIL_CASE_KEYS:
            assert k in fail_case_batch[0], f"Missing fail_case key: {k}"
        for k in REQUIRED_INFO_KEYS:
            assert k in infos_list[0], f"Missing info key: {k}"

        sem_id = obs_batch[0, 4]
        assert sem_id.min() >= 0, f"Semantic ID has negative values at step {step_i}"

        if step_i == 0 or step_i == len(ACTION_SEQUENCE) - 1:
            assert_nonempty_stage1_observation(obs_batch[0])
            print(f"       [PASS] step {step_i} obs non-empty")

        if save_obs:
            _save_obs(obs_batch, f"step_{step_i:03d}_{ACTION_NAMES[action_id]}", out_dir)

        for _ in range(SIM_STEPS_PER_ACTION):
            sim.step()

        if done_batch[0]:
            print(f"  Episode done at step {step_i}")
            break

    print("  [PASS] all per-step asserts passed (single dict input)")

    # ---- 6. Quick check: list[dict] input form ----
    print("\n[5b] Quick check: list[dict] planner_inputs form ...")
    vec_env.reset()
    obs_b2, fc_b2, done_b2, info_b2 = vec_env.plan_act_and_preprocess(
        [{"action": ACTION_FORWARD}]
    )
    assert obs_b2.shape == (1, 5, 120, 160)
    assert len(fc_b2) == 1
    assert len(done_b2) == 1
    assert len(info_b2) == 1
    print("  [PASS] list[dict] input form works")

    # ---- 7. Summary ----
    print("\n" + "=" * 60)
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)

    vec_env.close()
    return 0


if __name__ == "__main__":
    save = "--save-obs" in sys.argv
    # Determine output directory: next positional arg after --save-obs, if any.
    out_dir = "./smoke_obs_out"
    if save:
        idx = sys.argv.index("--save-obs")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            out_dir = sys.argv[idx + 1]
    sys.exit(main(save_obs=save, out_dir=out_dir))
