"""Canonical execution profiles for the paired LIBERO hybrid controller."""

from __future__ import annotations

from argparse import Namespace
from types import MappingProxyType
from typing import Any, Mapping


CUSTOM_PROFILE = "custom"
LAST_SAFE_V1 = "last_safe_v1"
LAST_SAFE_NO_FUSION_A = "last_safe_no_fusion_a"
LAST_SAFE_NO_FUSION_B = "last_safe_no_fusion_b"
CONCURRENT_WRENCH_20CM_V1 = "concurrent_wrench_20cm_v1"

METHOD_PROFILE_NAMES = (
    CUSTOM_PROFILE, LAST_SAFE_V1, LAST_SAFE_NO_FUSION_A, LAST_SAFE_NO_FUSION_B,
)
PERTURBATION_PROFILE_NAMES = (
    CUSTOM_PROFILE,
    CONCURRENT_WRENCH_20CM_V1,
)


LAST_SAFE_V1_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "max_xyz_action": 1.0,
        "max_rot_action": 0.5,
        "no_clip_gripper": False,
        "paired_plan_steps": 16,
        "paired_bspline_source_horizon": 32,
        "paired_bspline_degree": 3,
        "paired_bspline_support_subdivisions": 2,
        "paired_pose_ensemble_samples": 16,
        "paired_pose_ensemble_cluster_cutoff": 3.0,
        "paired_pose_ensemble_min_cluster_fraction": 0.5,
        "paired_pose_ensemble_selection_mode": "dominant_cluster",
        "paired_pose_ensemble_distance_scale_mode": "nearest_neighbor",
        "paired_pose_ensemble_support_members_mode": "selected_members",
        "paired_pose_ensemble_adaptive_support": True,
        "paired_pose_ensemble_adaptive_tube": True,
        "paired_pose_ensemble_support_confidence": 0.95,
        "paired_pose_ensemble_reentry_fraction": 0.8768,
        "paired_pose_ensemble_dense_core_fraction": None,
        "paired_pose_ensemble_dense_core_confidence": 0.9,
        "position_action_scale": 0.05,
        "rotation_action_scale": 0.5,
        "position_tolerance": 0.002,
        "rotation_tolerance": 0.05,
        "gripper_only_position_threshold": 0.002,
        "gripper_only_rotation_threshold": 0.05,
        "gripper_close_command_threshold": 0.5,
        "gating_precision_scale": 25.0,
        "position_recovery_gain": 1.0,
        "rotation_recovery_gain": 1.0,
        "support_interpolation_rcond": 0.01,
        "center_field_min_sigma": 0.003,
        "tube_projection_enabled": True,
        "tube_radius": 0.01,
        "phase_conditioned_execution": True,
        "phase_position_tube_radius": 0.04,
        "phase_rotation_tube_radius": 0.2,
        "phase_position_reentry_radius": 0.035072,
        "phase_rotation_reentry_radius": 0.1,
        "phase_projection_lookahead": None,
        "recovery_phase_advance_budget": None,
        "phase_max_recovery_steps": 0,
        "hybrid_recovery_field_source": "geometry",
        "hybrid_recovery_target_mode": "support_confidence_prefix",
        "hybrid_recovery_anchor_mode": "observed_perturbation_start",
        "hybrid_observed_anchor_reentry_radius": 0.005,
        "hybrid_observed_anchor_stable_steps": 2,
        "hybrid_freeze_recovery_phase": True,
        "hybrid_autonomous_chunk_reentry": True,
        "hybrid_replan_on_recovery_reentry": True,
        "hybrid_rotation_clf_enabled": True,
        "hybrid_rotation_clf_radius": 0.005,
        "hybrid_rotation_clf_contraction_rate": None,
        "hybrid_recovery_progress_patience": 0,
        "hybrid_recovery_progress_min_delta": 0.001,
        "hybrid_field_speed_gain": 2.0,
        "hybrid_support_confidence_outer_score": 0.05,
        "hybrid_support_confidence_inner_score": 0.1,
        "hybrid_recovery_trigger_mode": (
            "support_and_state_innovation"
        ),
        "hybrid_require_state_innovation": True,
        "hybrid_disturbance_position_innovation_threshold": 0.01,
        "hybrid_disturbance_rotation_innovation_threshold": 0.35,
        "hybrid_disturbance_expected_motion_multiplier": 1.5,
        "hybrid_nominal_fusion_mode": "transverse_outward_only",
        "hybrid_transverse_fusion_inner_ratio": 0.5,
        "hybrid_transverse_fusion_outer_ratio": 1.0,
        "hybrid_transverse_fusion_max_weight": 1.0,
        "tube_contraction_rate": 0.1,
        "preserve_field_direction_on_saturation": True,
    }
)


CONCURRENT_WRENCH_20CM_V1_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        "concurrent_perturbation_distance_cm": 20.0,
        "concurrent_perturbation_start_step": 3,
        "concurrent_perturbation_direction_x": 0.3882361,
        "concurrent_perturbation_direction_y": -0.9215599,
        "concurrent_perturbation_direction_z": 0.0,
        "concurrent_perturbation_ramp_steps": 2,
        "concurrent_perturbation_hold_steps": 2,
        "concurrent_perturbation_max_steps": 80,
        "concurrent_perturbation_stiffness": 1600.0,
        "concurrent_perturbation_integral_gain": 100.0,
        "concurrent_perturbation_integral_error_limit": 2.0,
        "concurrent_perturbation_damping": 60.0,
        "concurrent_perturbation_max_force": 300.0,
    }
)


def no_fusion_executor_overrides(method_profile: str) -> dict[str, Any]:
    """Executor-only delta shared by LIBERO, MimicGen and RoboCasa."""
    if method_profile == LAST_SAFE_V1:
        return {}
    if method_profile not in {LAST_SAFE_NO_FUSION_A, LAST_SAFE_NO_FUSION_B}:
        raise ValueError(f"Unknown paired executor profile: {method_profile}")
    return {
        "nominal_fusion_mode": "native_action",
        "require_state_innovation": method_profile == LAST_SAFE_NO_FUSION_A,
    }


def apply_evaluation_profiles(args: Namespace) -> Namespace:
    """Apply immutable named profiles to a parsed evaluator namespace."""
    method_profile = str(getattr(args, "method_profile", CUSTOM_PROFILE))
    if method_profile in {LAST_SAFE_V1, LAST_SAFE_NO_FUSION_A, LAST_SAFE_NO_FUSION_B}:
        if args.mode not in {
            "paired_bspline_direct",
            "paired_bspline_hybrid_ds",
            "paired_pose_action_direct",
            "paired_pose_action_hybrid_ds",
        }:
            raise ValueError(
                f"{LAST_SAFE_V1} does not support mode={args.mode!r}"
            )
        _assign(args, LAST_SAFE_V1_VALUES)
        if method_profile in {LAST_SAFE_NO_FUSION_A, LAST_SAFE_NO_FUSION_B}:
            args.hybrid_nominal_fusion_mode = "native_action"
            args.hybrid_require_state_innovation = method_profile == LAST_SAFE_NO_FUSION_A
    elif method_profile != CUSTOM_PROFILE:
        raise ValueError(f"unknown method profile: {method_profile}")

    perturbation_profile = str(
        getattr(args, "perturbation_profile", CUSTOM_PROFILE)
    )
    if perturbation_profile == CONCURRENT_WRENCH_20CM_V1:
        _assign(args, CONCURRENT_WRENCH_20CM_V1_VALUES)
    elif perturbation_profile != CUSTOM_PROFILE:
        raise ValueError(
            f"unknown perturbation profile: {perturbation_profile}"
        )
    return args


def _assign(args: Namespace, values: Mapping[str, Any]) -> None:
    missing = [name for name in values if not hasattr(args, name)]
    if missing:
        raise AttributeError(
            "evaluation namespace is missing profile fields: "
            + ", ".join(sorted(missing))
        )
    for name, value in values.items():
        setattr(args, name, value)
