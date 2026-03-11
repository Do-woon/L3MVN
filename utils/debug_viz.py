from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import cv2
import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    arr = _to_numpy(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D image, got shape={arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _norm_to_u8(x: np.ndarray) -> np.ndarray:
    arr = _to_numpy(x).astype(np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    vmin = float(np.min(arr[finite]))
    vmax = float(np.max(arr[finite]))
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out * 255.0, 0, 255)
    return out.astype(np.uint8)


def _label_to_color(label_map: np.ndarray) -> np.ndarray:
    label = _to_numpy(label_map).astype(np.int32)
    label[label < 0] = 0
    vis = (label % 256).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def _binary_to_bgr(x: np.ndarray) -> np.ndarray:
    arr = (_to_numpy(x) > 0.5).astype(np.uint8) * 255
    return _ensure_bgr(arr)


def _make_titled_panel(
    img: np.ndarray,
    title: str,
    cell_hw: tuple[int, int] = (320, 320),
    header_h: int = 34,
) -> np.ndarray:
    h, w = cell_hw
    body = cv2.resize(_ensure_bgr(img), (w, h), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    panel[header_h:, :, :] = body
    panel[:header_h, :, :] = (32, 32, 32)
    cv2.putText(
        panel,
        str(title)[:48],
        (8, 23),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def _make_montage(panels: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if len(panels) == 0:
        return np.zeros((354, 320, 3), dtype=np.uint8)

    resized = [_ensure_bgr(p) for p in panels]
    h, w = resized[0].shape[:2]
    rows = int(math.ceil(len(resized) / float(cols)))
    blank = np.zeros_like(resized[0], dtype=np.uint8)
    while len(resized) < rows * cols:
        resized.append(blank.copy())
    row_imgs = []
    for r in range(rows):
        row_imgs.append(np.hstack(resized[r * cols:(r + 1) * cols]))
    return np.vstack(row_imgs)


class DebugVizDumper:
    """Lightweight PNG/json dumper for zeroshot loop internals."""

    def __init__(self, enabled: bool, out_dir: str, every: int = 1):
        self.enabled = bool(enabled)
        self.out_dir = out_dir
        self.every = max(1, int(every))

        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "obs"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "maps"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "maps", "local"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "maps", "planner"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "maps", "frontier"), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "meta"), exist_ok=True)

    @classmethod
    def from_args(cls, args):
        return cls(
            enabled=int(getattr(args, "debug_viz", 0)) == 1,
            out_dir=str(getattr(args, "debug_viz_dir", "./tmp/debug_viz")),
            every=int(getattr(args, "debug_viz_every", 1)),
        )

    def should_dump(self, step: int) -> bool:
        return self.enabled and (int(step) % self.every == 0)

    def dump_obs(
        self,
        step: int,
        env_idx: int,
        raw_obs_chw: np.ndarray,
        pre_obs_chw: np.ndarray,
        tag: str,
    ) -> None:
        if not self.should_dump(step):
            return

        raw = _to_numpy(raw_obs_chw).astype(np.float32)
        pre = _to_numpy(pre_obs_chw).astype(np.float32)
        panels = []

        if raw.ndim == 3 and raw.shape[0] >= 3:
            rgb = np.transpose(raw[:3], (1, 2, 0))
            panels.append(_make_titled_panel(_ensure_bgr(rgb[:, :, ::-1]), "raw_rgb"))
        if raw.ndim == 3 and raw.shape[0] >= 4:
            depth = _norm_to_u8(raw[3])
            panels.append(_make_titled_panel(cv2.applyColorMap(depth, cv2.COLORMAP_BONE), "raw_depth"))
        if raw.ndim == 3 and raw.shape[0] >= 5:
            sem = raw[4].astype(np.int32)
            panels.append(_make_titled_panel(_label_to_color(sem), "raw_sem_stage1"))

        if pre.ndim == 3 and pre.shape[0] >= 4:
            depth2 = _norm_to_u8(pre[3])
            panels.append(_make_titled_panel(cv2.applyColorMap(depth2, cv2.COLORMAP_BONE), "pre_depth"))
        if pre.ndim == 3 and pre.shape[0] > 4:
            sem_oh = pre[4:]
            sem_argmax = np.argmax(sem_oh, axis=0).astype(np.int32)
            panels.append(_make_titled_panel(_label_to_color(sem_argmax), "pre_sem_argmax"))

        montage = _make_montage(panels, cols=3)
        fn = os.path.join(
            self.out_dir,
            "obs",
            f"{tag}_step{int(step):06d}_env{int(env_idx):02d}.png",
        )
        cv2.imwrite(fn, montage)

    def dump_maps(
        self,
        step: int,
        env_idx: int,
        local_map_chw: np.ndarray,
        full_map_chw: np.ndarray,
        planner_input: dict,
        target_edge_map: Optional[np.ndarray] = None,
        target_point_map: Optional[np.ndarray] = None,
        local_goal_map: Optional[np.ndarray] = None,
        goal_cat_channel: Optional[int] = None,
    ) -> None:
        if not self.should_dump(step):
            return

        local_map = _to_numpy(local_map_chw).astype(np.float32)
        full_map = _to_numpy(full_map_chw).astype(np.float32)
        local_panels = []
        planner_panels = []
        frontier_panels = []

        local_panels.append(_make_titled_panel(_binary_to_bgr(local_map[0]), "local_obstacle"))
        local_panels.append(_make_titled_panel(_binary_to_bgr(local_map[1]), "local_explored"))
        local_panels.append(_make_titled_panel(_binary_to_bgr(local_map[2]), "local_current"))

        if goal_cat_channel is not None and 0 <= int(goal_cat_channel) < local_map.shape[0]:
            local_panels.append(
                _make_titled_panel(
                    _binary_to_bgr(local_map[int(goal_cat_channel)]),
                    f"local_goal_sem_ch{int(goal_cat_channel)}",
                )
            )

        local_panels.append(_make_titled_panel(_binary_to_bgr(full_map[0]), "full_obstacle"))
        local_panels.append(_make_titled_panel(_binary_to_bgr(full_map[1]), "full_explored"))

        map_pred = np.asarray(planner_input.get("map_pred", np.zeros_like(local_map[0])), dtype=np.float32)
        exp_pred = np.asarray(planner_input.get("exp_pred", np.zeros_like(local_map[0])), dtype=np.float32)
        goal = np.asarray(planner_input.get("goal", np.zeros_like(local_map[0])), dtype=np.float32)
        map_target = np.asarray(planner_input.get("map_target", np.zeros_like(local_map[0])), dtype=np.float32)

        planner_panels.append(_make_titled_panel(_binary_to_bgr(map_pred), "planner_map_pred"))
        planner_panels.append(_make_titled_panel(_binary_to_bgr(exp_pred), "planner_exp_pred"))
        planner_panels.append(_make_titled_panel(_binary_to_bgr(goal), "planner_goal"))
        planner_panels.append(_make_titled_panel(_label_to_color(map_target.astype(np.int32)), "planner_map_target"))

        if target_edge_map is not None:
            frontier_panels.append(_make_titled_panel(_binary_to_bgr(target_edge_map), "target_edge_map"))
        if target_point_map is not None:
            frontier_panels.append(
                _make_titled_panel(
                    _label_to_color(np.asarray(target_point_map, dtype=np.int32)),
                    "target_point_map",
                )
            )
        if local_goal_map is not None:
            frontier_panels.append(_make_titled_panel(_binary_to_bgr(local_goal_map), "selected_local_goal"))

        local_montage = _make_montage(local_panels, cols=3)
        local_fn = os.path.join(
            self.out_dir,
            "maps",
            "local",
            f"step{int(step):06d}_env{int(env_idx):02d}.png",
        )
        cv2.imwrite(local_fn, local_montage)

        planner_montage = _make_montage(planner_panels, cols=2)
        planner_fn = os.path.join(
            self.out_dir,
            "maps",
            "planner",
            f"step{int(step):06d}_env{int(env_idx):02d}.png",
        )
        cv2.imwrite(planner_fn, planner_montage)

        if len(frontier_panels) > 0:
            frontier_montage = _make_montage(frontier_panels, cols=3)
            frontier_fn = os.path.join(
                self.out_dir,
                "maps",
                "frontier",
                f"step{int(step):06d}_env{int(env_idx):02d}.png",
            )
            cv2.imwrite(frontier_fn, frontier_montage)

    def dump_meta(
        self,
        step: int,
        env_idx: int,
        planner_input: dict,
        done: bool,
        fail_case: dict,
        info: dict,
        selected_frontier_id: Optional[int] = None,
    ) -> None:
        if not self.should_dump(step):
            return

        pose_pred = np.asarray(planner_input.get("pose_pred", []), dtype=np.float32).tolist()
        payload = {
            "step": int(step),
            "env_idx": int(env_idx),
            "planner_keys": sorted(list(planner_input.keys())),
            "planner_shapes": {
                "map_pred": list(np.asarray(planner_input.get("map_pred", [])).shape),
                "exp_pred": list(np.asarray(planner_input.get("exp_pred", [])).shape),
                "goal": list(np.asarray(planner_input.get("goal", [])).shape),
                "map_target": list(np.asarray(planner_input.get("map_target", [])).shape),
                "pose_pred": list(np.asarray(planner_input.get("pose_pred", [])).shape),
            },
            "pose_pred": pose_pred,
            "found_goal": int(planner_input.get("found_goal", 0)),
            "new_goal": int(bool(planner_input.get("new_goal", False))),
            "wait": int(bool(planner_input.get("wait", False))),
            "selected_frontier_id": (
                None if selected_frontier_id is None else int(selected_frontier_id)
            ),
            "action": info.get("last_action", None),
            "done": bool(done),
            "fail_case": {
                "collision": int(fail_case.get("collision", 0)),
                "success": int(fail_case.get("success", 0)),
                "detection": int(fail_case.get("detection", 0)),
                "exploration": int(fail_case.get("exploration", 0)),
            },
            "info": {
                "goal_name": info.get("goal_name", None),
                "goal_cat_id": info.get("goal_cat_id", None),
                "eve_angle": info.get("eve_angle", None),
                "sensor_pose": info.get("sensor_pose", None),
                "clear_flag": info.get("clear_flag", None),
                "collision": info.get("collision", None),
            },
        }

        fn = os.path.join(
            self.out_dir,
            "meta",
            f"step{int(step):06d}_env{int(env_idx):02d}.json",
        )
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
