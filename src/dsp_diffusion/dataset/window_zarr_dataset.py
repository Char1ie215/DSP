from __future__ import annotations

from typing import Dict

import numpy as np

from dsp_diffusion.dataset.zarr_dataset import ZarrDataset


class WindowZarrDataset(ZarrDataset):
    """Read one precomputed action window for each observation timestep.

    The regular OAT dataset expects an ``(N, D)`` action array and constructs a
    future window by slicing adjacent rows. Paired B-spline labels are already
    stored as ``(N, K, D)``; slicing adjacent rows would mix controls belonging
    to different policy queries. This dataset keeps the observation sampling
    unchanged and returns the single ``(K, D)`` window attached to the current
    observation.
    """

    def __init__(self, *args, n_action_steps: int, **kwargs) -> None:
        window_steps = int(n_action_steps)
        if window_steps < 1:
            raise ValueError("n_action_steps must be positive")
        super().__init__(*args, n_action_steps=1, **kwargs)
        action_array = self.replay_buffer[self.action_key]
        if len(action_array.shape) != 3:
            raise ValueError(
                "WindowZarrDataset requires an (N,K,D) action array, got "
                f"{action_array.shape}"
            )
        if int(action_array.shape[1]) != window_steps:
            raise ValueError(
                f"Configured action window {window_steps} does not match "
                f"stored shape {action_array.shape}"
            )
        self.window_action_steps = window_steps
        self.n_action_steps = window_steps

    def _sample_to_data(self, sample) -> Dict:
        obs_steps = self.n_obs_steps
        obs = {}
        for key in self.numeric_obs_keys:
            value = sample[key][:obs_steps]
            obs[key] = value.astype(np.float32) if value.dtype.kind == "f" else value
        for key in self.text_obs_keys:
            obs[key] = sample[key][0]

        current_index = max(obs_steps - 1, 0)
        action = sample[self.action_key][current_index].astype(np.float32)
        if action.shape != (
            self.window_action_steps,
            self.replay_buffer[self.action_key].shape[-1],
        ):
            raise RuntimeError(f"Unexpected window action shape {action.shape}")
        return {"obs": obs, "action": action}
