from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GaussianParamTokenFieldAdapter(torch.nn.Module):
    """Wrap Gaussian-param policies and emit env actions.

    The first 16 dimensions are the common Gaussian command. Any dimensions
    after index 16 are copied into env action dimensions after index 7. This
    keeps LIBERO/RoboMimic at 16D -> 7D and supports RoboCasa at 21D -> 12D.
    """

    PARAM_DIM = 16

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        max_xyz_action: float = 1.0,
        max_rot_action: float = 1.0,
        step_scale: float = 1.0,
        center_steps: int = 16,
        lookahead: int = 3,
        temporal_decay: float = 0.7,
        replan_interval: int = 16,
        active_weight_floor: float = 0.35,
        min_sigma: float = 0.003,
        max_sigma: float = 0.1,
        min_gain: float = 1.0,
        max_gain: float = 220.0,
        input_xyz_scale: float = 1.0,
        policy_index_stride: int = 1,
        clip_gripper: bool = True,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.max_xyz_action = float(max_xyz_action)
        self.max_rot_action = float(max_rot_action)
        self.step_scale = float(step_scale)
        self.center_steps = int(max(center_steps, 1))
        self.lookahead = int(max(lookahead, 1))
        self.temporal_decay = float(np.clip(temporal_decay, 0.0, 1.0))
        self.replan_interval = int(max(replan_interval, 1))
        self.active_weight_floor = float(np.clip(active_weight_floor, 0.0, 1.0))
        self.min_sigma = float(max(min_sigma, 1.0e-6))
        self.max_sigma = float(max(max_sigma, self.min_sigma))
        self.min_gain = float(max(min_gain, 0.0))
        self.max_gain = float(max(max_gain, self.min_gain))
        self.input_xyz_scale = float(input_xyz_scale)
        self.policy_index_stride = int(max(policy_index_stride, 1))
        self.clip_gripper = bool(clip_gripper)
        self._centers: np.ndarray | None = None
        self._directions: np.ndarray | None = None
        self._sigma_parallel: np.ndarray | None = None
        self._sigma_perp: np.ndarray | None = None
        self._lambda_parallel: np.ndarray | None = None
        self._lambda_perp: np.ndarray | None = None
        self._weight_scale: np.ndarray | None = None
        self._gaussian_tails: np.ndarray | None = None
        self._raw_tails: np.ndarray | None = None
        self._age: np.ndarray | None = None
        self._num_centers = 0
        self._env_action_dim = 7

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        self._centers = None
        self._directions = None
        self._sigma_parallel = None
        self._sigma_perp = None
        self._lambda_parallel = None
        self._lambda_perp = None
        self._weight_scale = None
        self._gaussian_tails = None
        self._raw_tails = None
        self._age = None
        self._num_centers = 0
        self._env_action_dim = 7
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_gaussian_param_token_field"

    def _current_eef(self, obs_dict: dict[str, Any], batch_size: int) -> np.ndarray:
        if "robot0_eef_pos" not in obs_dict:
            raise KeyError("gaussian_param_token_field requires robot0_eef_pos in obs_dict")
        eef = obs_dict["robot0_eef_pos"]
        arr = eef.detach().cpu().numpy() if isinstance(eef, torch.Tensor) else np.asarray(eef)
        return arr.reshape(batch_size, -1, 3)[:, -1].astype(np.float32)

    @staticmethod
    def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1.0e-8:
            return np.asarray(fallback, dtype=np.float32).copy()
        return (np.asarray(vec, dtype=np.float32) / norm).astype(np.float32)

    def _policy_plan_indices(self, raw_steps: int) -> np.ndarray:
        max_by_chunk = 1 + (int(raw_steps) - 1) // int(self.policy_index_stride)
        steps = int(min(self.center_steps, max_by_chunk))
        return (np.arange(steps, dtype=np.int64) * int(self.policy_index_stride)).astype(np.int64)

    def _plan_from_policy(self, planned: np.ndarray, current_eef: np.ndarray) -> None:
        batch_size = int(planned.shape[0])
        indices = self._policy_plan_indices(int(planned.shape[1]))
        commands = planned[:, indices, :].astype(np.float32)
        raw_deltas = commands[:, :, 0:3] * float(self.input_xyz_scale)
        centers = current_eef[:, None, :] + np.cumsum(raw_deltas * float(self.step_scale), axis=1)

        directions = commands[:, :, 3:6].astype(np.float32)
        for b in range(batch_size):
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            for j in range(directions.shape[1]):
                if np.linalg.norm(raw_deltas[b, j]) > 1.0e-8:
                    fallback = self._normalize(raw_deltas[b, j], fallback)
                directions[b, j] = self._normalize(directions[b, j], fallback)
                fallback = directions[b, j]

        self._centers = centers.astype(np.float32)
        self._directions = directions.astype(np.float32)
        self._sigma_parallel = np.clip(commands[:, :, 7], self.min_sigma, self.max_sigma).astype(np.float32)
        self._sigma_perp = np.clip(commands[:, :, 8], self.min_sigma, self.max_sigma).astype(np.float32)
        self._lambda_parallel = np.clip(commands[:, :, 9], self.min_gain, self.max_gain).astype(np.float32)
        self._lambda_perp = np.clip(commands[:, :, 10], self.min_gain, self.max_gain).astype(np.float32)
        self._weight_scale = np.clip(commands[:, :, 11], 0.1, 4.0).astype(np.float32)
        self._gaussian_tails = commands[:, :, 12:16].astype(np.float32)
        self._raw_tails = commands[:, :, self.PARAM_DIM :].astype(np.float32)
        self._age = np.zeros((batch_size,), dtype=np.int64)
        self._num_centers = int(commands.shape[1])
        self._env_action_dim = 7 + int(self._raw_tails.shape[-1])

    def predict_action(self, obs_dict, **kwargs):
        result = self.base_policy.predict_action(obs_dict, **kwargs)
        planned_t = result["action"]
        if planned_t.shape[-1] < self.PARAM_DIM:
            raise ValueError(f"Expected at least 16D Gaussian param action, got {planned_t.shape}")

        planned = planned_t.detach().cpu().numpy().astype(np.float32)
        batch_size = int(planned.shape[0])
        current_eef = self._current_eef(obs_dict, batch_size)

        need_plan = (
            self._centers is None
            or self._directions is None
            or self._sigma_parallel is None
            or self._sigma_perp is None
            or self._lambda_parallel is None
            or self._lambda_perp is None
            or self._weight_scale is None
            or self._gaussian_tails is None
            or self._raw_tails is None
            or self._age is None
            or int(self._centers.shape[0]) != batch_size
            or np.any(self._age >= min(self.replan_interval, max(self._num_centers, 1)))
        )
        if need_plan:
            self._plan_from_policy(planned, current_eef)

        assert self._centers is not None
        assert self._directions is not None
        assert self._sigma_parallel is not None
        assert self._sigma_perp is not None
        assert self._lambda_parallel is not None
        assert self._lambda_perp is not None
        assert self._weight_scale is not None
        assert self._gaussian_tails is not None
        assert self._raw_tails is not None
        assert self._age is not None

        actions = np.zeros((batch_size, 1, self._env_action_dim), dtype=np.float32)
        for b in range(batch_size):
            active = int(min(self._age[b], self._num_centers - 1))
            end = int(min(self._num_centers, active + self.lookahead))
            vec = np.zeros((3,), dtype=np.float32)
            total = 0.0
            for j in range(active, end):
                residual = (self._centers[b, j] - current_eef[b]).astype(np.float32)
                direction = self._directions[b, j].astype(np.float32)
                parallel_mag = float(np.dot(residual, direction))
                parallel = direction * parallel_mag
                perp = residual - parallel
                sigma_parallel = float(max(self._sigma_parallel[b, j], 1.0e-6))
                sigma_perp = float(max(self._sigma_perp[b, j], 1.0e-6))
                spatial = float(
                    np.exp(
                        -0.5
                        * (
                            (parallel_mag / sigma_parallel) ** 2
                            + (float(np.linalg.norm(perp)) / sigma_perp) ** 2
                        )
                    )
                )
                if j == active:
                    spatial = max(spatial, self.active_weight_floor)
                weight = float(self.temporal_decay ** (j - active)) * spatial * float(self._weight_scale[b, j])
                local = (
                    parallel * float(self._lambda_parallel[b, j])
                    + perp * float(self._lambda_perp[b, j])
                ).astype(np.float32)
                vec += local * weight
                total += weight

            if total <= 1.0e-8:
                residual = (self._centers[b, active] - current_eef[b]).astype(np.float32)
                direction = self._directions[b, active].astype(np.float32)
                parallel_mag = float(np.dot(residual, direction))
                parallel = direction * parallel_mag
                perp = residual - parallel
                vec = (
                    parallel * float(self._lambda_parallel[b, active])
                    + perp * float(self._lambda_perp[b, active])
                ).astype(np.float32)
            else:
                vec = vec / float(total)

            norm = float(np.linalg.norm(vec))
            if norm > self.max_xyz_action:
                vec = vec / max(norm, 1.0e-8) * self.max_xyz_action
            vec = np.clip(vec, -self.max_xyz_action, self.max_xyz_action)
            gaussian_tail = self._gaussian_tails[b, int(min(active, self._gaussian_tails.shape[1] - 1))]
            rot = np.clip(gaussian_tail[:3], -self.max_rot_action, self.max_rot_action)
            gripper = float(gaussian_tail[3])
            if self.clip_gripper:
                gripper = float(np.clip(gripper, -1.0, 1.0))

            actions[b, 0, :3] = np.nan_to_num(vec, nan=0.0)
            actions[b, 0, 3:6] = np.nan_to_num(rot, nan=0.0)
            actions[b, 0, 6] = 0.0 if not np.isfinite(gripper) else gripper
            if self._raw_tails.shape[-1] > 0:
                raw_tail = self._raw_tails[b, int(min(active, self._raw_tails.shape[1] - 1))]
                actions[b, 0, 7:] = np.nan_to_num(raw_tail, nan=0.0)

        self._age += 1
        return {
            **result,
            "structured_action": planned_t,
            "action": torch.as_tensor(actions, device=planned_t.device, dtype=planned_t.dtype),
        }


def maybe_wrap_policy_for_rollout(policy: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    policy_cfg = getattr(getattr(cfg, "task", None), "policy", None)
    if bool(getattr(policy_cfg, "lazy_eval", False)):
        return policy
    adapter_cfg = getattr(policy_cfg, "rollout_adapter", None)
    if adapter_cfg is None:
        return policy
    adapter_type = str(getattr(adapter_cfg, "type", "")).strip()
    if adapter_type in {"", "none", "raw"}:
        return policy
    if adapter_type == "command_center_direct":
        from ns_gaussian_ds_policy.libero_command_center import (
            LiberoCommandCenterDirectAdapter,
        )

        return LiberoCommandCenterDirectAdapter(
            policy,
            position_action_scale=float(
                getattr(adapter_cfg, "position_action_scale", 0.05)
            ),
            rotation_action_scale=float(
                getattr(adapter_cfg, "rotation_action_scale", 0.5)
            ),
        )
    if adapter_type == "paired_pose_action_direct":
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoPairedPoseActionDirectAdapter,
        )

        return LiberoPairedPoseActionDirectAdapter(policy)
    if adapter_type == "paired_bspline_direct":
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoPairedBSplineDirectAdapter,
        )

        return LiberoPairedBSplineDirectAdapter(
            policy,
            source_horizon=int(
                getattr(adapter_cfg, "bspline_source_horizon", 32)
            ),
            degree=int(getattr(adapter_cfg, "bspline_degree", 3)),
            execution_steps=int(getattr(adapter_cfg, "plan_steps", 16)),
        )
    if adapter_type == "action_bspline_direct":
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoActionBSplineDirectAdapter,
        )

        return LiberoActionBSplineDirectAdapter(
            policy,
            source_horizon=int(
                getattr(adapter_cfg, "bspline_source_horizon", 32)
            ),
            degree=int(getattr(adapter_cfg, "bspline_degree", 3)),
            execution_steps=int(getattr(adapter_cfg, "plan_steps", 16)),
        )
    if adapter_type in {
        "command_center_field",
        "paired_pose_action_field",
        "paired_pose_action_hybrid_ds",
        "paired_bspline_field",
        "paired_bspline_hybrid_ds",
    }:
        from ns_gaussian_ds_policy.libero_command_center import (
            LiberoCommandCenterFieldAdapter,
        )
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoPairedBSplineFieldAdapter,
            LiberoPairedPoseActionFieldAdapter,
        )
        from ns_gaussian_ds_policy.libero_hybrid_ds import (
            LiberoPairedBSplineHybridDSAdapter,
            LiberoPairedPoseActionHybridDSAdapter,
        )

        adapter_classes = {
            "command_center_field": LiberoCommandCenterFieldAdapter,
            "paired_pose_action_field": LiberoPairedPoseActionFieldAdapter,
            "paired_pose_action_hybrid_ds": (
                LiberoPairedPoseActionHybridDSAdapter
            ),
            "paired_bspline_field": LiberoPairedBSplineFieldAdapter,
            "paired_bspline_hybrid_ds": LiberoPairedBSplineHybridDSAdapter,
        }
        adapter_class = adapter_classes[adapter_type]
        adapter_kwargs = dict(
            base_policy=policy,
            plan_steps=int(getattr(adapter_cfg, "plan_steps", 16)),
            position_action_scale=float(
                getattr(adapter_cfg, "position_action_scale", 0.05)
            ),
            rotation_action_scale=float(
                getattr(adapter_cfg, "rotation_action_scale", 0.5)
            ),
            position_tolerance=float(
                getattr(adapter_cfg, "position_tolerance", 0.002)
            ),
            rotation_tolerance=float(
                getattr(adapter_cfg, "rotation_tolerance", 0.05)
            ),
            max_position_action=float(
                getattr(adapter_cfg, "max_xyz_action", 1.0)
            ),
            max_rotation_action=float(
                getattr(adapter_cfg, "max_rot_action", 1.0)
            ),
            max_plan_steps=int(getattr(adapter_cfg, "max_plan_steps", 200)),
            gripper_only_position_threshold=float(
                getattr(
                    adapter_cfg,
                    "gripper_only_position_threshold",
                    0.002,
                )
            ),
            gripper_only_rotation_threshold=float(
                getattr(
                    adapter_cfg,
                    "gripper_only_rotation_threshold",
                    0.05,
                )
            ),
            gripper_close_command_threshold=float(
                getattr(adapter_cfg, "gripper_close_command_threshold", 0.5)
            ),
            gating_precision_scale=float(
                getattr(adapter_cfg, "gating_precision_scale", 25.0)
            ),
            position_recovery_gain=float(
                getattr(adapter_cfg, "position_recovery_gain", 1.0)
            ),
            rotation_recovery_gain=float(
                getattr(adapter_cfg, "rotation_recovery_gain", 1.0)
            ),
            support_interpolation_rcond=float(
                getattr(adapter_cfg, "support_interpolation_rcond", 1.0e-2)
            ),
            min_sigma=float(getattr(adapter_cfg, "min_sigma", 0.004)),
            terminal_stall_replan=bool(
                getattr(adapter_cfg, "terminal_stall_replan", True)
            ),
            tube_projection_enabled=bool(
                getattr(adapter_cfg, "tube_projection_enabled", True)
            ),
            tube_radius=float(getattr(adapter_cfg, "tube_radius", 0.01)),
            phase_conditioned_execution=bool(
                getattr(adapter_cfg, "phase_conditioned_execution", True)
            ),
            phase_position_tube_radius=float(
                getattr(adapter_cfg, "phase_position_tube_radius", 0.075)
            ),
            phase_rotation_tube_radius=float(
                getattr(adapter_cfg, "phase_rotation_tube_radius", 0.2)
            ),
            tube_contraction_rate=float(
                getattr(adapter_cfg, "tube_contraction_rate", 0.1)
            ),
            preserve_field_direction_on_saturation=bool(
                getattr(
                    adapter_cfg,
                    "direction_preserving_saturation",
                    True,
                )
            ),
        )
        if adapter_type in {
            "paired_bspline_field",
            "paired_bspline_hybrid_ds",
        }:
            adapter_kwargs.update(
                source_horizon=int(
                    getattr(adapter_cfg, "bspline_source_horizon", 32)
                ),
                degree=int(getattr(adapter_cfg, "bspline_degree", 3)),
                support_subdivisions=int(
                    getattr(adapter_cfg, "bspline_support_subdivisions", 1)
                ),
            )
        if adapter_type in {
            "paired_bspline_hybrid_ds",
            "paired_pose_action_hybrid_ds",
        }:
            projection_lookahead = getattr(
                adapter_cfg,
                "phase_projection_lookahead",
                None,
            )
            phase_advance_budget = getattr(
                adapter_cfg,
                "recovery_phase_advance_budget",
                None,
            )
            adapter_kwargs.update(
                phase_position_reentry_radius=float(
                    getattr(
                        adapter_cfg,
                        "phase_position_reentry_radius",
                        0.02,
                    )
                ),
                phase_rotation_reentry_radius=float(
                    getattr(
                        adapter_cfg,
                        "phase_rotation_reentry_radius",
                        0.1,
                    )
                ),
                phase_projection_lookahead=(
                    None
                    if projection_lookahead is None
                    else float(projection_lookahead)
                ),
                recovery_phase_advance_budget=(
                    None
                    if phase_advance_budget is None
                    else float(phase_advance_budget)
                ),
                phase_max_recovery_steps=int(
                    getattr(
                        adapter_cfg,
                        "phase_max_recovery_steps",
                        16,
                    )
                ),
                recovery_field_source=str(
                    getattr(
                        adapter_cfg,
                        "recovery_field_source",
                        "action",
                    )
                ),
                recovery_target_mode=str(
                    getattr(
                        adapter_cfg,
                        "recovery_target_mode",
                        "projected",
                    )
                ),
                freeze_recovery_phase=bool(
                    getattr(
                        adapter_cfg,
                        "freeze_recovery_phase",
                        False,
                    )
                ),
                autonomous_chunk_reentry=bool(
                    getattr(
                        adapter_cfg,
                        "autonomous_chunk_reentry",
                        False,
                    )
                ),
                replan_on_recovery_reentry=bool(
                    getattr(
                        adapter_cfg,
                        "replan_on_recovery_reentry",
                        False,
                    )
                ),
                rotation_clf_enabled=bool(
                    getattr(
                        adapter_cfg,
                        "rotation_clf_enabled",
                        False,
                    )
                ),
                rotation_clf_tube_radius=float(
                    getattr(
                        adapter_cfg,
                        "rotation_clf_tube_radius",
                        0.05,
                    )
                ),
                rotation_clf_contraction_rate=getattr(
                    adapter_cfg,
                    "rotation_clf_contraction_rate",
                    None,
                ),
                recovery_progress_patience=int(
                    getattr(
                        adapter_cfg,
                        "recovery_progress_patience",
                        0,
                    )
                ),
                recovery_progress_min_delta=float(
                    getattr(
                        adapter_cfg,
                        "recovery_progress_min_delta",
                        1.0e-3,
                    )
                ),
            )
        return adapter_class(**adapter_kwargs)
    if adapter_type != "gaussian_param_token_field":
        raise ValueError(f"Unsupported rollout adapter: {adapter_type}")
    return GaussianParamTokenFieldAdapter(
        policy,
        max_xyz_action=float(getattr(adapter_cfg, "max_xyz_action", 1.0)),
        max_rot_action=float(getattr(adapter_cfg, "max_rot_action", 1.0)),
        step_scale=float(getattr(adapter_cfg, "step_scale", 1.0)),
        center_steps=int(getattr(adapter_cfg, "center_steps", 16)),
        lookahead=int(getattr(adapter_cfg, "lookahead", 3)),
        temporal_decay=float(getattr(adapter_cfg, "temporal_decay", 0.7)),
        replan_interval=int(getattr(adapter_cfg, "replan_interval", 16)),
        active_weight_floor=float(getattr(adapter_cfg, "active_weight_floor", 0.35)),
        min_sigma=float(getattr(adapter_cfg, "min_sigma", 0.003)),
        max_sigma=float(getattr(adapter_cfg, "max_sigma", 0.1)),
        min_gain=float(getattr(adapter_cfg, "min_gain", 1.0)),
        max_gain=float(getattr(adapter_cfg, "max_gain", 220.0)),
        input_xyz_scale=float(getattr(adapter_cfg, "input_xyz_scale", 1.0)),
        policy_index_stride=int(getattr(adapter_cfg, "policy_index_stride", 1)),
        clip_gripper=bool(getattr(adapter_cfg, "clip_gripper", True)),
    )
