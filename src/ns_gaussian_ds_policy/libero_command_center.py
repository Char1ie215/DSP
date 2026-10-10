from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ns_gaussian_ds_policy.endpoint_field import (
    FIELD_ACTION_DIM,
    relative_center_chain_to_incremental_actions,
)
from ns_gaussian_ds_policy.libero_paired_curve import (
    COMMAND_DIM,
    GEOMETRY_DIM,
    LiberoPairedPoseActionFieldAdapter,
)


COMMAND_CENTER_DIM = FIELD_ACTION_DIM
DEFAULT_COMMAND_CENTER_HORIZON = 32


def _batched_rotvec_to_quaternion(rotation_vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(rotation_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("rotation_vectors must have shape (N,3)")
    angles = np.linalg.norm(vectors, axis=1)
    half_angles = 0.5 * angles
    scales = np.divide(
        np.sin(half_angles),
        angles,
        out=np.full_like(angles, 0.5),
        where=angles > 1.0e-12,
    )
    return np.concatenate(
        [vectors * scales[:, None], np.cos(half_angles)[:, None]],
        axis=1,
    )


def _batched_quaternion_multiply(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("quaternion batches must both have shape (N,4)")
    x1, y1, z1, w1 = np.moveaxis(left, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(right, -1, 0)
    product = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=1,
    )
    norms = np.linalg.norm(product, axis=1, keepdims=True)
    return product / np.maximum(norms, 1.0e-12)


def _batched_quaternion_to_rotvec(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (N,4)")
    quaternions = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True),
        1.0e-12,
    )
    quaternions = np.where(
        quaternions[:, 3:4] < 0.0,
        -quaternions,
        quaternions,
    )
    xyz_norm = np.linalg.norm(quaternions[:, :3], axis=1)
    angles = 2.0 * np.arctan2(
        xyz_norm,
        np.clip(quaternions[:, 3], -1.0, 1.0),
    )
    scales = np.divide(
        angles,
        xyz_norm,
        out=np.zeros_like(angles),
        where=xyz_norm > 1.0e-12,
    )
    return quaternions[:, :3] * scales[:, None]


def build_libero_command_center_windows(
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    horizon: int = DEFAULT_COMMAND_CENTER_HORIZON,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Integrate native LIBERO actions into observation-anchored centers.

    Row ``i`` owns a complete ``(horizon, 7)`` prediction target. Every pose
    in that window is relative to the state observed at row ``i``. Episode-end
    padding holds the terminal center instead of repeatedly integrating the
    final action.
    """

    actions = np.asarray(actions, dtype=np.float64)
    ends = np.asarray(episode_ends, dtype=np.int64)
    horizon = int(horizon)
    if actions.ndim != 2 or actions.shape[1] != COMMAND_CENTER_DIM:
        raise ValueError(
            f"LIBERO actions must have shape (N,{COMMAND_CENTER_DIM}), "
            f"got {actions.shape}"
        )
    if len(actions) == 0 or not np.all(np.isfinite(actions)):
        raise ValueError("actions must be a non-empty finite array")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if ends.ndim != 1 or len(ends) == 0 or int(ends[-1]) != len(actions):
        raise ValueError("episode_ends must terminate at the final action")
    if np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing")
    if float(position_action_scale) <= 0.0:
        raise ValueError("position_action_scale must be positive")
    if float(rotation_action_scale) <= 0.0:
        raise ValueError("rotation_action_scale must be positive")

    windows = np.empty(
        (len(actions), horizon, COMMAND_CENTER_DIM),
        dtype=np.float32,
    )
    padded_steps = 0
    start = 0
    offsets = np.arange(horizon, dtype=np.int64)[None]
    for raw_end in ends:
        end = int(raw_end)
        episode_length = end - start
        local_indices = np.minimum(
            np.arange(episode_length, dtype=np.int64)[:, None] + offsets,
            episode_length - 1,
        )
        future_actions = actions[start + local_indices]
        integrate = np.ones((episode_length, horizon), dtype=bool)
        if horizon > 1:
            integrate[:, 1:] = local_indices[:, 1:] != local_indices[:, :-1]
        padded_steps += int(np.size(integrate) - np.count_nonzero(integrate))

        physical_position_steps = (
            future_actions[..., :3]
            * float(position_action_scale)
            * integrate[..., None]
        )
        windows[start:end, :, :3] = (
            np.cumsum(physical_position_steps, axis=1)
            / float(position_action_scale)
        )

        orientation = np.zeros((episode_length, 4), dtype=np.float64)
        orientation[:, 3] = 1.0
        for step in range(horizon):
            delta = _batched_rotvec_to_quaternion(
                future_actions[:, step, 3:6]
                * float(rotation_action_scale)
            )
            integrated = _batched_quaternion_multiply(delta, orientation)
            orientation = np.where(
                integrate[:, step : step + 1],
                integrated,
                orientation,
            )
            windows[start:end, step, 3:6] = (
                _batched_quaternion_to_rotvec(orientation)
                / float(rotation_action_scale)
            )
        windows[start:end, :, 6] = future_actions[..., 6]
        start = end

    if not np.all(np.isfinite(windows)):
        raise RuntimeError("Generated command-center windows contain NaN or Inf")
    summary: dict[str, object] = {
        "steps": int(len(actions)),
        "episodes": int(len(ends)),
        "horizon": horizon,
        "action_dim": COMMAND_CENTER_DIM,
        "layout": "relative_position3 + relative_rotation_vector3 + gripper1",
        "origin": "current_observation",
        "source": "integrated_native_action",
        "position_action_scale": float(position_action_scale),
        "rotation_action_scale": float(rotation_action_scale),
        "episode_padding": "hold_terminal_center",
        "padded_fraction": float(padded_steps / windows.shape[0] / horizon),
        "max_abs": float(np.max(np.abs(windows))),
    }
    return np.ascontiguousarray(windows), summary


def command_centers_to_incremental_actions(
    centers: np.ndarray,
    *,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 2:
        return relative_center_chain_to_incremental_actions(
            centers,
            position_action_scale=float(position_action_scale),
            rotation_action_scale=float(rotation_action_scale),
        )
    if centers.ndim != 3 or centers.shape[-1] != COMMAND_CENTER_DIM:
        raise ValueError("centers must have shape (T,7) or (B,T,7)")
    return np.stack(
        [
            relative_center_chain_to_incremental_actions(
                sample,
                position_action_scale=float(position_action_scale),
                rotation_action_scale=float(rotation_action_scale),
            )
            for sample in centers
        ],
        axis=0,
    ).astype(np.float32)


class LiberoCommandCenterDirectAdapter(torch.nn.Module):
    """Difference predicted centers back into native LIBERO actions."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        position_action_scale: float = 0.05,
        rotation_action_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.position_action_scale = float(position_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)
        self.policy_batches = 0
        self.policy_env_queries = 0

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_command_center_direct"

    def get_execution_metrics(self) -> dict[str, object]:
        return {
            "executor": "command_center_direct",
            "representation": "action_integrated_command_center",
            "policy_batches": int(self.policy_batches),
            "policy_env_queries": int(self.policy_env_queries),
        }

    def predict_action(self, obs_dict: dict[str, Any], **kwargs):
        result = self.base_policy.predict_action(obs_dict, **kwargs)
        centers = result["action"]
        if centers.ndim != 3 or centers.shape[-1] != COMMAND_CENTER_DIM:
            raise ValueError(
                "Expected command centers with shape (B,T,7), got "
                f"{tuple(centers.shape)}"
            )
        actions = command_centers_to_incremental_actions(
            centers.detach().cpu().numpy(),
            position_action_scale=self.position_action_scale,
            rotation_action_scale=self.rotation_action_scale,
        )
        self.policy_batches += 1
        self.policy_env_queries += int(centers.shape[0])
        return {
            **result,
            "structured_action": centers,
            "action": torch.as_tensor(
                actions,
                device=centers.device,
                dtype=centers.dtype,
            ),
        }


class LiberoCommandCenterFieldAdapter(LiberoPairedPoseActionFieldAdapter):
    """Compile one action-integrated center chain into a cached field."""

    def __init__(self, base_policy: torch.nn.Module, **kwargs) -> None:
        super().__init__(base_policy, **kwargs)
        self.representation = "action_integrated_command_center"

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_command_center_field"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "command_center_field",
                "representation": self.representation,
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 3 or prediction.shape[-1] != COMMAND_CENTER_DIM:
            raise ValueError(
                "Expected command centers with shape (B,T,7), got "
                f"{tuple(prediction.shape)}"
            )
        centers = prediction.detach().cpu().numpy()
        actions = command_centers_to_incremental_actions(
            centers,
            position_action_scale=self.field_config.position_action_scale,
            rotation_action_scale=self.field_config.rotation_action_scale,
        )
        paired = np.concatenate(
            [
                actions[..., :GEOMETRY_DIM],
                actions[..., :COMMAND_DIM],
                centers[..., 6:7],
            ],
            axis=-1,
        ).astype(np.float32)
        return torch.as_tensor(
            paired,
            device=prediction.device,
            dtype=prediction.dtype,
        )
