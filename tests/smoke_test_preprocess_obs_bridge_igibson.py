"""Integration smoke test: Stage1(make_vec_envs iGibson) -> _preprocess_obs() bridge.

Run from workspace root:
    python tests/smoke_test_preprocess_obs_bridge_igibson.py

This smoke test verifies:
1) make_vec_envs(args) iGibson branch returns Stage-1 batch obs (1,5,H,W)
2) Stage-1 single obs (5,H,W) can be fed into Sem_Exp_Env_Agent._preprocess_obs
3) Stage-2 output has shape (20,H,W) and semantic one-hot(16) structure
"""

from __future__ import annotations

import sys
import types

import numpy as np
from envs.igibson.semantic_taxonomy import L3MVN_CATEGORY_TO_ID, SemanticTaxonomy


def _make_args():
    return types.SimpleNamespace(
        # iGibson branch
        use_igibson=1,
        igibson_scene="Rs_int",
        goal_name="chair",
        goal_cat_id=1,
        igibson_assets_path="/mount/nas2/users/dukim/vla_ws/igibson/data/assets",
        igibson_dataset_path="/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset",
        igibson_key_path="/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key",
        frame_width=160,
        frame_height=120,
        max_episode_length=500,
        # preprocess-related args (Sem_Exp_Env_Agent._preprocess_obs contract)
        env_frame_width=160,   # ds == 1 path for this bridge smoke
        min_depth=0.5,
        max_depth=5.0,
        use_gtsem=1,           # avoid heavy semantic model path in this smoke
        num_sem_categories=16,
    )


REQUIRED_FAIL_CASE_KEYS = ("collision", "success", "detection", "exploration")
ALLOWED_NON_FLOOR_CONTACTS = 0
SEMANTIC_ID_TO_NAME = {v: k for k, v in L3MVN_CATEGORY_TO_ID.items()}
SEMANTIC_ID_TO_NAME[0] = "background_or_unmapped"
SEMANTIC_ID_TO_NAME[16] = "extra_channel_reserved"

def _build_preprocess_callable(args):
    """Build callable that directly invokes real Sem_Exp_Env_Agent._preprocess_obs."""
    try:
        from agents.sem_exp import Sem_Exp_Env_Agent

        agent_stub = object.__new__(Sem_Exp_Env_Agent)
        agent_stub.args = args
        agent_stub.rgb_vis = None

        def _real_preprocess(obs_stage1_single: np.ndarray) -> np.ndarray:
            return Sem_Exp_Env_Agent._preprocess_obs(agent_stub, obs_stage1_single)

        print("[bridge] Using real Sem_Exp_Env_Agent._preprocess_obs()")
        return _real_preprocess, "real"
    except Exception as exc:
        raise RuntimeError(
            "Failed to import/call real Sem_Exp_Env_Agent._preprocess_obs(). "
            "Fallback path is intentionally disabled in this smoke test."
        ) from exc


def _assert_stage1_batch_contract(obs_stage1_batch: np.ndarray, infos, args) -> None:
    assert isinstance(obs_stage1_batch, np.ndarray), "Stage1 batch obs must be np.ndarray"
    assert obs_stage1_batch.shape == (1, 5, args.frame_height, args.frame_width), (
        f"Expected Stage1 batch (1,5,{args.frame_height},{args.frame_width}), "
        f"got {obs_stage1_batch.shape}"
    )
    assert len(infos) == 1 and isinstance(infos[0], dict), "infos must be list[dict] len=1"


def _assert_stage1_single_contract(obs_stage1_single: np.ndarray, label: str) -> None:
    assert obs_stage1_single.ndim == 3 and obs_stage1_single.shape[0] == 5, (
        f"Stage1 single obs must be (5,H,W), got {obs_stage1_single.shape}"
    )
    rgb = obs_stage1_single[0:3]
    depth = obs_stage1_single[3]
    sem_id = obs_stage1_single[4]

    assert rgb.sum() > 0 and rgb.var() > 0, "Stage1 RGB is empty/constant"
    assert depth.shape == sem_id.shape, "Depth and semantic channels must align"
    assert np.unique(depth).size > 1, "Stage1 depth has no variation"

    sem_i32 = sem_id.astype(np.int32)
    np.testing.assert_array_equal(sem_id, sem_i32.astype(np.float32))
    assert sem_i32.min() >= 0, f"Semantic id contains negative values: {sem_i32.min()}"
    max_sem_id = int(SemanticTaxonomy.NUM_CHANNELS)
    assert sem_i32.max() <= max_sem_id, (
        f"Semantic id exceeds L3MVN range 0..{max_sem_id}: {sem_i32.max()}"
    )
    assert sem_i32.size > 0, "Semantic id channel is empty"
    uniq = np.unique(sem_i32)
    print(
        f"[{label}] Stage1 semantic stats: min={int(sem_i32.min())}, "
        f"max={int(sem_i32.max())}, n_unique={len(uniq)}"
    )


def _unwrap_class_id_to_name(vec_env) -> dict[int, str]:
    inner = getattr(vec_env, "_env", None)
    assert inner is not None, "SingleEnvVecWrapper must expose inner _env"
    class_id_to_name = getattr(inner, "_class_id_to_name", None)
    assert isinstance(class_id_to_name, dict), "EnvWrapper must expose class_id_to_name dict"
    return class_id_to_name


def _assert_spawn_collision_free(vec_env) -> None:
    import pybullet as p

    inner = getattr(vec_env, "_env", None)
    assert inner is not None, "SingleEnvVecWrapper must expose inner _env"
    sim = getattr(inner, "_env", None)
    scene = getattr(inner, "_scene", None)
    robot = getattr(inner, "_robot", None)
    assert sim is not None and scene is not None and robot is not None

    body_ids = robot.get_body_ids()
    assert body_ids, "Robot has no body ids; cannot evaluate collisions."
    body_id = body_ids[0]
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

    spawn_attempts = getattr(sim, "_spawn_collision_free_attempts", None)
    print(
        f"[reset] spawn debug: attempts={spawn_attempts}, "
        f"non_floor_contacts={non_floor_contacts}"
    )
    assert non_floor_contacts <= ALLOWED_NON_FLOOR_CONTACTS, (
        f"Spawn is in collision state: non-floor contacts={non_floor_contacts}"
    )
    print("[reset] spawn collision-free check: PASS")


def _print_stage1_semantic_distribution(obs_stage1_single: np.ndarray, label: str) -> None:
    sem_i32 = obs_stage1_single[4].astype(np.int32)
    uniq, counts = np.unique(sem_i32, return_counts=True)
    total = int(sem_i32.size)

    summary = []
    for sem_id, count in zip(uniq.tolist(), counts.tolist()):
        name = SEMANTIC_ID_TO_NAME.get(sem_id, "unknown_id")
        ratio = 100.0 * float(count) / float(total)
        summary.append(f"{sem_id}:{name} ({count}px, {ratio:.2f}%)")
    print(f"[{label}] Stage1 semantic IDs in frame: " + ", ".join(summary))


def _print_semantic_lookup_summary(class_id_to_name: dict[int, str]) -> None:
    id_to_sem = SemanticTaxonomy.build_id_to_l3mvn_semantic_id(class_id_to_name)
    sem_to_class_names: dict[int, list[str]] = {sid: [] for sid in range(0, 16)}
    for class_id, sem_id in id_to_sem.items():
        if 0 <= sem_id <= 15:
            sem_to_class_names[sem_id].append(class_id_to_name[class_id])

    print("[taxonomy] iGibson class-name mapping summary:")
    for sem_id in range(0, 16):
        names = sorted(sem_to_class_names[sem_id])
        sem_name = SEMANTIC_ID_TO_NAME.get(sem_id, "unknown_id")
        preview = ", ".join(names[:8]) if names else "-"
        more = "" if len(names) <= 8 else f", ... (+{len(names) - 8} more)"
        print(
            f"  sem_id={sem_id:2d} ({sem_name}): "
            f"mapped_class_names={len(names)}; sample=[{preview}{more}]"
        )


def _validate_stage2(
    label: str,
    obs_stage1_single: np.ndarray,
    obs_stage2_single: np.ndarray,
    args,
) -> None:
    h, w = obs_stage1_single.shape[1], obs_stage1_single.shape[2]
    expected_stage2_shape = (20, h, w)
    assert obs_stage2_single.shape == expected_stage2_shape, (
        f"{label}: expected Stage2 {expected_stage2_shape}, got {obs_stage2_single.shape}"
    )

    rgb_stage1 = obs_stage1_single[0:3]
    depth_stage1 = obs_stage1_single[3]
    rgb_stage2 = obs_stage2_single[0:3]
    depth_stage2 = obs_stage2_single[3]
    sem_block = obs_stage2_single[4:20]

    print(f"[{label}] Stage1 obs shape: {obs_stage1_single.shape}")
    print(f"[{label}] Stage2 obs shape: {obs_stage2_single.shape}")
    print(f"[{label}] Stage2 semantic block shape: {sem_block.shape}")

    # Channel placement checks
    np.testing.assert_allclose(rgb_stage2, rgb_stage1, atol=0.0, rtol=0.0)
    assert depth_stage2.shape == depth_stage1.shape
    assert not np.allclose(depth_stage2, depth_stage1), (
        f"{label}: depth appears unprocessed (Stage2 depth identical to Stage1 depth)."
    )

    # Semantic one-hot checks
    assert args.num_sem_categories == 16, "This bridge smoke assumes num_sem_categories == 16"
    assert sem_block.shape == (16, h, w), f"{label}: semantic block must be (16,H,W)"
    sem_sum = sem_block.sum(axis=0)
    semantic_id = obs_stage1_single[4].astype(np.int32)
    valid_mask = (semantic_id >= 1) & (semantic_id <= 16)
    bg_mask = ~valid_mask

    print(
        f"[{label}] semantic one-hot sum stats: min={float(sem_sum.min()):.3f}, "
        f"max={float(sem_sum.max()):.3f}"
    )
    if np.any(valid_mask):
        assert np.all(np.isclose(sem_sum[valid_mask], 1.0, atol=1e-6)), (
            f"{label}: valid semantic-id pixels must have semantic one-hot sum==1."
        )
    if np.any(bg_mask):
        assert np.all(np.isclose(sem_sum[bg_mask], 0.0, atol=1e-6)), (
            f"{label}: background/unmapped semantic-id pixels must have sum==0."
        )

    print(
        f"[{label}] depth before/after: "
        f"stage1[min={float(depth_stage1.min()):.4f}, max={float(depth_stage1.max()):.4f}] "
        f"-> stage2[min={float(depth_stage2.min()):.4f}, max={float(depth_stage2.max()):.4f}]"
    )


def main():
    print("=" * 72)
    print("Stage1(make_vec_envs iGibson) -> _preprocess_obs() bridge smoke test")
    print("=" * 72)

    args = _make_args()
    preprocess_obs_fn, preprocess_mode = _build_preprocess_callable(args)

    from envs import make_vec_envs

    envs = None
    try:
        print("\n[1] make_vec_envs(args) ...")
        envs = make_vec_envs(args)
        class_id_to_name = _unwrap_class_id_to_name(envs)
        _print_semantic_lookup_summary(class_id_to_name)

        print("\n[2] reset() Stage1 capture ...")
        obs_stage1_batch, infos = envs.reset()
        _assert_stage1_batch_contract(obs_stage1_batch, infos, args)
        print("[reset] collision: N/A (reset path has no fail_case)")
        _assert_spawn_collision_free(envs)
        obs0 = obs_stage1_batch[0]
        _assert_stage1_single_contract(obs0, "reset")
        _print_stage1_semantic_distribution(obs0, "reset")

        obs_stage2 = preprocess_obs_fn(obs0)
        _validate_stage2("reset", obs0, obs_stage2, args)

        print("\n[3] action=1 (FORWARD) Stage1 capture ...")
        obs_stage1_batch, fail_case_batch, done_batch, infos = envs.plan_act_and_preprocess(
            [{"action": 1}]
        )
        assert isinstance(fail_case_batch, list) and len(fail_case_batch) == 1
        for k in REQUIRED_FAIL_CASE_KEYS:
            assert k in fail_case_batch[0], f"Missing fail_case key: {k}"
        assert len(done_batch) == 1
        assert len(infos) == 1 and isinstance(infos[0], dict)
        assert "sensor_pose" in infos[0], "Missing infos[0]['sensor_pose']"
        assert "eve_angle" in infos[0], "Missing infos[0]['eve_angle']"

        _assert_stage1_batch_contract(obs_stage1_batch, infos, args)
        obs1 = obs_stage1_batch[0]
        _assert_stage1_single_contract(obs1, "step_action_1")
        _print_stage1_semantic_distribution(obs1, "step_action_1")

        obs_stage2_step = preprocess_obs_fn(obs1)
        _validate_stage2("step_action_1", obs1, obs_stage2_step, args)

        print("\n" + "=" * 72)
        print(f"ALL ASSERTIONS PASSED (preprocess mode: {preprocess_mode})")
        print("=" * 72)
    finally:
        if envs is not None:
            envs.close()
            print("[cleanup] envs.close() called")


if __name__ == "__main__":
    sys.exit(main())
