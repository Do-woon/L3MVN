"""ObsAdapter: converts iGibson sensor outputs to L3MVN Stage-1 raw observation.

Stage-1 contract (L3MVN env raw output)
----------------------------------------
  state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
  shape : (5, H, W)
  dtype : float32
  channels:
    0, 1, 2  – RGB
    3        – Depth (single channel)
    4        – Semantic ID (raw GT id, single channel)

SemanticTaxonomy integration (Stage-2 one-hot) is NOT this class's
responsibility.  ObsAdapter only satisfies the Stage-1 contract.
"""

from __future__ import annotations

import numpy as np


class ObsAdapter:
    """Adapts iGibson sensor outputs to the L3MVN Stage-1 observation tensor.

    All methods are instance methods for future extensibility, but the class
    holds no mutable state so it can safely be reused across steps.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        semantic_id_map: np.ndarray,
    ) -> np.ndarray:
        """Return the Stage-1 observation tensor.

        Parameters
        ----------
        rgb : np.ndarray, shape (H, W, 3)
            RGB image from iGibson, any numeric dtype.
        depth : np.ndarray, shape (H, W) or (H, W, 1)
            Depth image from iGibson, any numeric dtype.
        semantic_id_map : np.ndarray, shape (H, W)
            GT semantic id map from iGibson, any integer dtype.

        Returns
        -------
        np.ndarray, shape (5, H, W), dtype float32
            Channels 0-2: RGB, channel 3: depth, channel 4: semantic id.
        """
        self._validate_shapes(rgb, depth, semantic_id_map)

        rgb_hw3 = self._normalize_rgb(rgb)          # (H, W, 3) float32
        depth_hw1 = self._normalize_depth(depth)    # (H, W, 1) float32
        sem_hw1 = self._normalize_semantic(semantic_id_map)  # (H, W, 1) float32

        obs_hwc = np.concatenate([rgb_hw3, depth_hw1, sem_hw1], axis=2)  # (H, W, 5)
        obs = obs_hwc.transpose(2, 0, 1)  # (5, H, W)

        h, w = rgb.shape[:2]
        if obs.shape != (5, h, w):
            raise RuntimeError(
                f"Unexpected output shape {obs.shape}; expected (5, {h}, {w})"
            )
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_shapes(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        semantic_id_map: np.ndarray,
    ) -> None:
        """Raise ValueError if any input shape contract is violated."""
        # rgb must be (H, W, 3)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must have shape (H, W, 3), got {rgb.shape}"
            )
        h, w = rgb.shape[:2]

        # depth must be (H, W) or (H, W, 1)
        if depth.ndim == 2:
            if depth.shape != (h, w):
                raise ValueError(
                    f"depth spatial size {depth.shape} != rgb spatial size ({h}, {w})"
                )
        elif depth.ndim == 3:
            if depth.shape != (h, w, 1):
                raise ValueError(
                    f"depth must have shape ({h}, {w}, 1) or ({h}, {w}), got {depth.shape}"
                )
        else:
            raise ValueError(
                f"depth must be 2-D (H, W) or 3-D (H, W, 1), got ndim={depth.ndim}"
            )

        # semantic must be (H, W)
        if semantic_id_map.ndim != 2:
            raise ValueError(
                f"semantic_id_map must be 2-D (H, W), got ndim={semantic_id_map.ndim}"
            )
        if semantic_id_map.shape != (h, w):
            raise ValueError(
                f"semantic_id_map spatial size {semantic_id_map.shape} "
                f"!= rgb spatial size ({h}, {w})"
            )

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Cast rgb to float32 without changing values. Returns (H, W, 3)."""
        return rgb.astype(np.float32, copy=False)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Expand to (H, W, 1) and cast to float32."""
        if depth.ndim == 2:
            depth = depth[:, :, np.newaxis]
        return depth.astype(np.float32, copy=False)

    def _normalize_semantic(self, semantic_id_map: np.ndarray) -> np.ndarray:
        """Expand to (H, W, 1) and cast to float32."""
        return semantic_id_map[:, :, np.newaxis].astype(np.float32, copy=False)
