"""SingleEnvVecWrapper: thin batch/vector-like interface around a single EnvWrapper.

This wrapper makes one ``EnvWrapper`` instance look like the object returned by
L3MVN's ``make_vec_envs(args)``, without generalising to multiple environments.

Output contracts
----------------
- ``reset()``
    - ``obs_batch``  : ``np.ndarray``, shape ``(1, 5, H, W)``, dtype ``float32``
    - ``infos_list`` : ``list[dict]``, length 1

- ``plan_act_and_preprocess(planner_inputs)``
    - ``obs_batch``       : ``np.ndarray``, shape ``(1, 5, H, W)``
    - ``fail_case_batch`` : ``list[dict]``, length 1
    - ``done_batch``      : ``np.ndarray``, shape ``(1,)``, dtype ``bool``
    - ``infos_list``      : ``list[dict]``, length 1

``planner_inputs`` may be
    - a single ``dict``      (canonical 8-key planner_inputs)
    - a length-1 ``list[dict]``

The inner ``EnvWrapper`` still produces **Stage 1** obs ``(5, H, W)``.
"""

from __future__ import annotations

import numpy as np

from envs.igibson.env_wrapper import EnvWrapper


class SingleEnvVecWrapper:
    """Wraps a single ``EnvWrapper`` in a batch/vector-like interface.

    Parameters
    ----------
    env_wrapper : EnvWrapper
        A fully-constructed, ready-to-use ``EnvWrapper`` instance.
    """

    num_envs: int = 1

    def __init__(self, env_wrapper: EnvWrapper) -> None:
        self._env = env_wrapper
        # Keep Vec-like metadata when available.
        self.observation_space = getattr(env_wrapper, "observation_space", None)
        self.action_space = getattr(env_wrapper, "action_space", None)
        self._pending_actions = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        """Reset the inner env and return a batch-wrapped result.

        Returns
        -------
        obs_batch : np.ndarray, shape (1, 5, H, W), dtype float32
        infos_list : list[dict], length 1
        """
        obs, info = self._env.reset()
        return self._wrap_obs(obs), self._wrap_info(info)

    def plan_act_and_preprocess(
        self, planner_inputs
    ) -> tuple[np.ndarray, list[dict], np.ndarray, list[dict]]:
        """Execute one step and return batch-wrapped results.

        Parameters
        ----------
        planner_inputs : dict or list[dict]
            A single dict or a length-1 list of dicts.
            Canonical path uses Habitat-style 8-key planner_inputs.

        Returns
        -------
        obs_batch       : np.ndarray, shape (1, 5, H, W)
        fail_case_batch : list[dict], length 1
        done_batch      : np.ndarray, shape (1,), dtype bool
        infos_list      : list[dict], length 1
        """
        single_input = self._unwrap_planner_inputs(planner_inputs)
        obs, fail_case, done, info = self._env.plan_act_and_preprocess(single_input)
        return (
            self._wrap_obs(obs),
            self._wrap_fail_case(fail_case),
            self._wrap_done(done),
            self._wrap_info(info),
        )

    def close(self) -> None:
        """Close the inner env if it exposes a ``close()`` method."""
        if hasattr(self._env, "close"):
            self._env.close()

    def step(self, actions):
        """Compatibility helper: route to ``plan_act_and_preprocess``."""
        return self.plan_act_and_preprocess(actions)

    def step_async(self, actions) -> None:
        """Store pending actions for a later ``step_wait`` call."""
        self._pending_actions = actions

    def step_wait(self):
        """Execute actions previously passed to ``step_async``."""
        if self._pending_actions is None:
            raise RuntimeError("step_wait() called before step_async()")
        actions = self._pending_actions
        self._pending_actions = None
        return self.plan_act_and_preprocess(actions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_obs(obs: np.ndarray) -> np.ndarray:
        """Add batch dimension: (5, H, W) → (1, 5, H, W), guaranteed float32."""
        return obs.astype(np.float32, copy=False)[np.newaxis]  # (1, 5, H, W)

    @staticmethod
    def _wrap_info(info: dict) -> list[dict]:
        """Wrap scalar info dict in a length-1 list."""
        return [info]

    @staticmethod
    def _wrap_done(done: bool) -> np.ndarray:
        """Wrap scalar done flag in a length-1 boolean array."""
        return np.array([done], dtype=bool)

    @staticmethod
    def _wrap_fail_case(fail_case: dict) -> list[dict]:
        """Wrap scalar fail_case dict in a length-1 list."""
        return [fail_case]

    @staticmethod
    def _unwrap_planner_inputs(planner_inputs) -> dict:
        """Normalise planner_inputs to a single dict.

        Accepts:
        - ``dict``         — returned as-is
        - ``list[dict]``   — first element returned
        """
        if isinstance(planner_inputs, dict):
            return planner_inputs
        if isinstance(planner_inputs, (list, tuple)) and len(planner_inputs) == 1:
            first = planner_inputs[0]
            if not isinstance(first, dict):
                raise ValueError(
                    f"planner_inputs[0] must be a dict, got {type(first)}"
                )
            return first
        raise ValueError(
            f"planner_inputs must be a dict or a length-1 list/tuple of dicts, "
            f"got {type(planner_inputs)}"
        )
