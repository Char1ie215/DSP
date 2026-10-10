from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ns_gaussian_ds_policy.libero_paired_curve import (
    LiberoPairedBSplineDirectAdapter,
)
from ns_gaussian_ds_policy.libero_paired_hybrid_profiles import (
    LAST_SAFE_V1_VALUES,
)


def last_safe_ensemble_kwargs() -> dict[str, Any]:
    """Map the immutable profile to its K=16 pose selector."""

    values = LAST_SAFE_V1_VALUES
    return {
        "samples": int(values["paired_pose_ensemble_samples"]),
        "source_horizon": int(values["paired_bspline_source_horizon"]),
        "selection_steps": int(values["paired_plan_steps"]),
        "degree": int(values["paired_bspline_degree"]),
        "support_subdivisions": int(
            values["paired_bspline_support_subdivisions"]
        ),
        "position_action_scale": float(values["position_action_scale"]),
        "rotation_action_scale": float(values["rotation_action_scale"]),
        "cluster_cutoff": float(
            values["paired_pose_ensemble_cluster_cutoff"]
        ),
        "minimum_cluster_fraction": float(
            values["paired_pose_ensemble_min_cluster_fraction"]
        ),
        "spread_quantile": 0.9,
        "dense_core_fraction": values[
            "paired_pose_ensemble_dense_core_fraction"
        ],
        "selection_mode": str(
            values["paired_pose_ensemble_selection_mode"]
        ),
        "distance_scale_mode": str(
            values["paired_pose_ensemble_distance_scale_mode"]
        ),
        "support_members_mode": str(
            values["paired_pose_ensemble_support_members_mode"]
        ),
    }


def last_safe_hybrid_kwargs(*, max_plan_steps: int = 1000) -> dict[str, Any]:
    """Map every executor setting from the canonical last-safe profile."""

    values = LAST_SAFE_V1_VALUES
    return {
        "source_horizon": int(values["paired_bspline_source_horizon"]),
        "degree": int(values["paired_bspline_degree"]),
        "support_subdivisions": int(
            values["paired_bspline_support_subdivisions"]
        ),
        "plan_steps": int(values["paired_plan_steps"]),
        "position_action_scale": float(values["position_action_scale"]),
        "rotation_action_scale": float(values["rotation_action_scale"]),
        "position_tolerance": float(values["position_tolerance"]),
        "rotation_tolerance": float(values["rotation_tolerance"]),
        "max_position_action": float(values["max_xyz_action"]),
        "max_rotation_action": float(values["max_rot_action"]),
        "max_plan_steps": int(max_plan_steps),
        "gripper_only_position_threshold": float(
            values["gripper_only_position_threshold"]
        ),
        "gripper_only_rotation_threshold": float(
            values["gripper_only_rotation_threshold"]
        ),
        "gripper_close_command_threshold": float(
            values["gripper_close_command_threshold"]
        ),
        "gating_precision_scale": float(values["gating_precision_scale"]),
        "position_recovery_gain": float(values["position_recovery_gain"]),
        "rotation_recovery_gain": float(values["rotation_recovery_gain"]),
        "support_interpolation_rcond": float(
            values["support_interpolation_rcond"]
        ),
        "min_sigma": float(values["center_field_min_sigma"]),
        "tube_projection_enabled": bool(values["tube_projection_enabled"]),
        "tube_radius": float(values["tube_radius"]),
        "tube_contraction_rate": float(values["tube_contraction_rate"]),
        "preserve_field_direction_on_saturation": bool(
            values["preserve_field_direction_on_saturation"]
        ),
        "phase_position_tube_radius": float(
            values["phase_position_tube_radius"]
        ),
        "phase_rotation_tube_radius": float(
            values["phase_rotation_tube_radius"]
        ),
        "phase_position_reentry_radius": float(
            values["phase_position_reentry_radius"]
        ),
        "phase_rotation_reentry_radius": float(
            values["phase_rotation_reentry_radius"]
        ),
        "phase_projection_lookahead": values["phase_projection_lookahead"],
        "recovery_phase_advance_budget": values[
            "recovery_phase_advance_budget"
        ],
        "phase_max_recovery_steps": int(values["phase_max_recovery_steps"]),
        "recovery_field_source": str(values["hybrid_recovery_field_source"]),
        "recovery_target_mode": str(values["hybrid_recovery_target_mode"]),
        "recovery_anchor_mode": str(values["hybrid_recovery_anchor_mode"]),
        "observed_anchor_reentry_radius": float(
            values["hybrid_observed_anchor_reentry_radius"]
        ),
        "observed_anchor_stable_steps": int(
            values["hybrid_observed_anchor_stable_steps"]
        ),
        "freeze_recovery_phase": bool(values["hybrid_freeze_recovery_phase"]),
        "autonomous_chunk_reentry": bool(
            values["hybrid_autonomous_chunk_reentry"]
        ),
        "replan_on_recovery_reentry": bool(
            values["hybrid_replan_on_recovery_reentry"]
        ),
        "rotation_clf_enabled": bool(values["hybrid_rotation_clf_enabled"]),
        "rotation_clf_tube_radius": float(
            values["hybrid_rotation_clf_radius"]
        ),
        "rotation_clf_contraction_rate": values[
            "hybrid_rotation_clf_contraction_rate"
        ],
        "recovery_progress_patience": int(
            values["hybrid_recovery_progress_patience"]
        ),
        "recovery_progress_min_delta": float(
            values["hybrid_recovery_progress_min_delta"]
        ),
        "field_speed_gain": float(values["hybrid_field_speed_gain"]),
        "support_confidence_outer_score": float(
            values["hybrid_support_confidence_outer_score"]
        ),
        "support_confidence_inner_score": float(
            values["hybrid_support_confidence_inner_score"]
        ),
        "recovery_trigger_mode": str(values["hybrid_recovery_trigger_mode"]),
        "disturbance_position_innovation_threshold": float(
            values["hybrid_disturbance_position_innovation_threshold"]
        ),
        "disturbance_rotation_innovation_threshold": float(
            values["hybrid_disturbance_rotation_innovation_threshold"]
        ),
        "disturbance_expected_motion_multiplier": float(
            values["hybrid_disturbance_expected_motion_multiplier"]
        ),
        "nominal_fusion_mode": str(values["hybrid_nominal_fusion_mode"]),
        "transverse_fusion_inner_ratio": float(
            values["hybrid_transverse_fusion_inner_ratio"]
        ),
        "transverse_fusion_outer_ratio": float(
            values["hybrid_transverse_fusion_outer_ratio"]
        ),
        "transverse_fusion_max_weight": float(
            values["hybrid_transverse_fusion_max_weight"]
        ),
        "adaptive_pose_support_enabled": bool(
            values["paired_pose_ensemble_adaptive_support"]
        ),
        "adaptive_pose_tube_enabled": bool(
            values["paired_pose_ensemble_adaptive_tube"]
        ),
        "pose_support_confidence": float(
            values["paired_pose_ensemble_support_confidence"]
        ),
        "pose_reentry_fraction": float(
            values["paired_pose_ensemble_reentry_fraction"]
        ),
        "pose_dense_core_confidence": float(
            values["paired_pose_ensemble_dense_core_confidence"]
        ),
    }


class CachedPairedBSplineDirectPolicy(torch.nn.Module):
    """Emit one cached native action per feedback controller step."""

    def __init__(self, direct_policy: LiberoPairedBSplineDirectAdapter) -> None:
        super().__init__()
        self.direct_policy = direct_policy
        self._actions: torch.Tensor | None = None
        self._structured_action: torch.Tensor | None = None
        self._cursor = 0
        self.action_steps = 0

    @property
    def device(self):
        return self.direct_policy.device

    @property
    def dtype(self):
        return self.direct_policy.dtype

    def reset(self):
        self._actions = None
        self._structured_action = None
        self._cursor = 0
        return self.direct_policy.reset()

    def get_policy_name(self):
        return self.direct_policy.get_policy_name() + "_cached"

    def get_observation_ports(self):
        return self.direct_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.direct_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.direct_policy.get_observation_encoder()

    def get_execution_metrics(self) -> dict[str, Any]:
        return {
            **self.direct_policy.get_execution_metrics(),
            "executor": "paired_bspline_k16_cached_direct",
            "cache_action_steps": int(self.action_steps),
            "query_cadence_steps": int(self.direct_policy.execution_steps),
        }

    def _query(self, obs_dict: dict[str, Any], **kwargs: Any) -> None:
        result = self.direct_policy.predict_action(obs_dict, **kwargs)
        actions = result["action"]
        if actions.ndim != 3 or int(actions.shape[-1]) != 7:
            raise ValueError(
                "Decoded paired B-spline actions must have shape (B,T,7), "
                f"got {tuple(actions.shape)}"
            )
        self._actions = actions
        self._structured_action = result.get("structured_action")
        self._cursor = 0

    @staticmethod
    def _batch_size(obs_dict: dict[str, Any]) -> int:
        for value in obs_dict.values():
            if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("Cannot infer observation batch size")

    def predict_action(self, obs_dict: dict[str, Any], **kwargs: Any):
        batch_size = self._batch_size(obs_dict)
        if (
            self._actions is None
            or int(self._actions.shape[0]) != batch_size
            or self._cursor >= int(self._actions.shape[1])
        ):
            self._query(obs_dict, **kwargs)
        assert self._actions is not None
        action = self._actions[:, self._cursor : self._cursor + 1]
        self._cursor += 1
        self.action_steps += int(action.shape[0])
        result = {"action": action}
        if self._structured_action is not None:
            result["structured_action"] = self._structured_action
        return result
