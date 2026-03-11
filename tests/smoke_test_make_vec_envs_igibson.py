"""Integration smoke test: make_vec_envs with iGibson backend (Stage 1).

Run from the workspace root:
    python tests/smoke_test_make_vec_envs_igibson.py

Validates that make_vec_envs(args) with use_igibson=1 returns a Stage-1
env object whose obs has shape (1, 5, H, W) as numpy ndarray.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pybullet as p

from envs.igibson.semantic_taxonomy import SemanticTaxonomy

# ---------------------------------------------------------------------------
# Minimal args namespace for the iGibson branch
# ---------------------------------------------------------------------------

def _make_args():
    args = types.SimpleNamespace(
        use_igibson=1,
        igibson_scene="Rs_int",
        goal_name="chair",
        goal_cat_id=1,
        # dataset paths
        igibson_assets_path="/mount/nas2/users/dukim/vla_ws/igibson/data/assets",
        igibson_dataset_path="/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset",
        igibson_key_path="/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key",
        # env dimensions
        frame_width=160,
        frame_height=120,
        max_episode_length=500,
    )
    return args


REQUIRED_INFO_KEYS = ("sensor_pose", "eve_angle", "goal_cat_id", "goal_name", "clear_flag")
REQUIRED_FAIL_CASE_KEYS = ("collision", "success", "detection", "exploration")
ALLOWED_NON_FLOOR_CONTACTS = 0

ACTION_SEQUENCE = [1, 2, 3, 4, 5] * 10  # FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN
ACTION_NAMES = {1: "FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT", 4: "LOOK_UP", 5: "LOOK_DOWN"}


# ---------------------------------------------------------------------------
# Observation validation (Stage 1)
# ---------------------------------------------------------------------------

def assert_nonempty_stage1_obs_batch(obs: np.ndarray) -> None:
    """Assert (1, 5, H, W) Stage-1 obs has non-blank RGB, depth, semantic ID."""
    o = obs[0]  # (5, H, W)
    rgb    = o[:3]
    depth  = o[3]
    sem_id = o[4]

    assert rgb.sum() > 0,               "RGB block is all zeros"
    assert rgb.var() > 0,               "RGB block has zero variance"
    assert len(np.unique(depth)) > 1,   "Depth block is a single uniform value"
    assert sem_id.max() >= 0,           "Semantic ID block has no valid ids"


def _unwrap_igibson_handles(vec_env):
    """Return (sim, scene, robot, class_id_to_name) from SingleEnvVecWrapper."""
    inner = getattr(vec_env, "_env", None)
    assert inner is not None, "SingleEnvVecWrapper must expose inner _env"
    sim = getattr(inner, "_env", None)
    scene = getattr(inner, "_scene", None)
    robot = getattr(inner, "_robot", None)
    class_id_to_name = getattr(inner, "_class_id_to_name", None)
    assert sim is not None and scene is not None and robot is not None
    assert isinstance(class_id_to_name, dict), "EnvWrapper must expose class_id_to_name dict"
    return sim, scene, robot, class_id_to_name


def _count_non_floor_contacts(robot, scene) -> int:
    """Count robot contacts excluding floor/self contacts."""
    body_id = robot.get_body_ids()[0]
    floor_ids = set(getattr(scene, "floor_body_ids", []) or [])
    try:
        floor_z = float(scene.floor_heights[0])
    except (AttributeError, IndexError, TypeError):
        floor_z = 0.0

    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=body_id)

    non_floor_contacts = 0
    for c in contacts:
        body_b = c[2]
        if body_b == body_id or body_b in floor_ids:
            continue
        if abs(c[7][2]) > 0.7 and c[5][2] < floor_z + 0.15:
            continue
        non_floor_contacts += 1
    return non_floor_contacts


def assert_stage1_semantic_contract(
    obs: np.ndarray,
    class_id_to_name: dict[int, str],
) -> None:
    """Validate Stage-1 semantic channel: integer-valued L3MVN semantic IDs."""
    sem_id = obs[0, 4]  # (H, W), float32 but integer-valued IDs
    sem_i32 = sem_id.astype(np.int32)
    np.testing.assert_array_equal(sem_id, sem_i32.astype(np.float32))
    assert sem_i32.min() >= 0, f"Semantic ID has negatives: {sem_i32.min()}"
    assert sem_i32.max() <= 15, f"Semantic ID exceeds L3MVN range: {sem_i32.max()}"

    allowed = set(
        SemanticTaxonomy.build_id_to_l3mvn_semantic_id(class_id_to_name).values()
    )
    allowed.add(0)
    uniq = set(np.unique(sem_i32).tolist())
    assert uniq.issubset(allowed), (
        f"Unexpected semantic IDs found: {sorted(uniq - allowed)} "
        f"(allowed subset example: {sorted(list(allowed))[:16]})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("make_vec_envs (iGibson branch, Stage 1) smoke test")
    print("=" * 60)

    args = _make_args()

    from envs import make_vec_envs

    envs = None
    try:
        # ---- 1. Build ----
        print("\n[1] make_vec_envs(args) ...")
        envs = make_vec_envs(args)
        print(f"  type(envs) = {type(envs).__name__}")
        print(f"  num_envs   = {envs.num_envs}")

        # Check required API surface
        assert hasattr(envs, "reset"),                   "missing reset()"
        assert hasattr(envs, "plan_act_and_preprocess"), "missing plan_act_and_preprocess()"
        assert hasattr(envs, "close"),                   "missing close()"
        print("  [PASS] API surface: reset / plan_act_and_preprocess / close")

        expected_shape = (1, 5, args.frame_height, args.frame_width)
        sim, scene, robot, class_id_to_name = _unwrap_igibson_handles(envs)

        # ---- 2. Reset ----
        print("\n[2] envs.reset() ...")
        obs, infos = envs.reset()

        print(f"  type(obs)  = {type(obs)}")
        print(f"  obs.shape  = {obs.shape}")
        print(f"  obs.dtype  = {obs.dtype}")
        print(f"  len(infos) = {len(infos)}")

        assert isinstance(obs, np.ndarray),            f"obs must be np.ndarray, got {type(obs)}"
        assert obs.shape == expected_shape,            f"Expected {expected_shape}, got {obs.shape}"
        assert obs.dtype == np.float32
        assert len(infos) == 1
        assert isinstance(infos[0], dict)
        for k in REQUIRED_INFO_KEYS:
            assert k in infos[0], f"Missing key in reset infos[0]: {k}"
        print("  [PASS] reset obs shape, dtype, infos contract")

        assert_nonempty_stage1_obs_batch(obs)
        print("  [PASS] reset obs is non-empty (RGB, depth, semantic ID)")

        assert_stage1_semantic_contract(obs, class_id_to_name)
        print("  [PASS] Stage-1 semantic channel uses integer L3MVN semantic IDs")

        non_floor_contacts = _count_non_floor_contacts(robot, scene)
        spawn_attempts = getattr(sim, "_spawn_collision_free_attempts", None)
        print(
            f"  spawn debug: attempts={spawn_attempts}, "
            f"non_floor_contacts_after_reset={non_floor_contacts}"
        )
        assert non_floor_contacts <= ALLOWED_NON_FLOOR_CONTACTS, (
            f"Spawn is in collision state: non-floor contacts={non_floor_contacts}"
        )
        print("  [PASS] reset spawn is collision-free")

        # ---- 3. Action sequence ----
        print(f"\n[3] Action sequence: {[ACTION_NAMES[a] for a in ACTION_SEQUENCE]}")
        header = (
            f"{'step':>4}  {'action':<12}  {'obs.shape':>14}  {'done':>4}  "
            f"{'sensor_pose':>28}  {'eve':>4}"
        )
        print(header)
        print("-" * len(header))

        for step_i, action_id in enumerate(ACTION_SEQUENCE):
            obs, fail_case_batch, done_batch, infos = envs.plan_act_and_preprocess(
                [{"action": action_id}]
            )

            sp = infos[0]["sensor_pose"]
            try:
                sp_str = f"[{sp[0]:+.3f},{sp[1]:+.3f},{math.degrees(sp[2]):+.1f}\u00b0]"
            except Exception:
                sp_str = str(sp)

            print(
                f"{step_i:>4}  {ACTION_NAMES[action_id]:<12}  "
                f"{str(obs.shape):>14}  "
                f"{'T' if done_batch[0] else 'F':>4}  "
                f"{sp_str:>28}  {infos[0]['eve_angle']:>4}"
            )

            # --- asserts ---
            assert isinstance(obs, np.ndarray)
            assert obs.shape == expected_shape, f"shape error at step {step_i}: {obs.shape}"
            assert obs.dtype == np.float32
            assert isinstance(fail_case_batch, list) and len(fail_case_batch) == 1
            for k in REQUIRED_FAIL_CASE_KEYS:
                assert k in fail_case_batch[0], f"Missing fail_case key: {k}"
            assert len(done_batch) == 1
            assert len(infos) == 1
            assert isinstance(infos[0], dict)
            assert "sensor_pose" in infos[0], "Missing infos[0]['sensor_pose']"
            assert "eve_angle" in infos[0], "Missing infos[0]['eve_angle']"

            if step_i == 0:
                assert_nonempty_stage1_obs_batch(obs)
                assert_stage1_semantic_contract(obs, class_id_to_name)
                print(f"       [PASS] step {step_i} obs non-empty")

            if done_batch[0]:
                print(f"  Episode done at step {step_i}")
                break

        print("  [PASS] all per-step asserts passed")

        # ---- 4. Summary ----
        print("\n" + "=" * 60)
        print("ALL ASSERTIONS PASSED")
        print("=" * 60)

    finally:
        if envs is not None:
            envs.close()
            print("[cleanup] envs.close() called")


if __name__ == "__main__":
    sys.exit(main())
