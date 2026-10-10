from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.interpolate import BSpline

from ns_gaussian_ds_policy.autonomous_se3_field import (
    AutonomousSE3FitConfig,
    PhaseConditionedPoseActionFieldExecutor,
    PoseIndexedActionFieldExecutor,
)
from ns_gaussian_ds_policy.endpoint_field import (
    EndpointFieldConfig,
    build_command_endpoint_field_labels,
    compile_endpoint_field_plan,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_rotvec,
    relative_rotation_vector,
    rotvec_to_quaternion,
)


PAIRED_POSE_ACTION_DIM = 13
GEOMETRY_DIM = 6
COMMAND_DIM = 6
GRIPPER_INDEX = 12
LIBERO_ACTION_DIM = 7
LIBERO_ACTION_GRIPPER_INDEX = 6
PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS = 16
PAIRED_BSPLINE_DEFAULT_DEGREE = 3
PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON = 32


def open_uniform_bspline_knots(
    num_control_points: int,
    degree: int,
    source_horizon: int,
) -> np.ndarray:
    """Return a clamped, open-uniform knot vector in control-step units."""

    num_control_points = int(num_control_points)
    degree = int(degree)
    source_horizon = int(source_horizon)
    if degree < 1:
        raise ValueError("degree must be positive")
    if num_control_points <= degree:
        raise ValueError("num_control_points must exceed degree")
    if source_horizon < 2:
        raise ValueError("source_horizon must be at least two")
    terminal = float(source_horizon - 1)
    interior_count = num_control_points - degree - 1
    interior = (
        np.linspace(0.0, terminal, interior_count + 2, dtype=np.float64)[1:-1]
        if interior_count > 0
        else np.empty((0,), dtype=np.float64)
    )
    return np.concatenate(
        [
            np.zeros((degree + 1,), dtype=np.float64),
            interior,
            np.full((degree + 1,), terminal, dtype=np.float64),
        ]
    )


def paired_bspline_basis(
    phases: np.ndarray,
    *,
    num_control_points: int = PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
) -> np.ndarray:
    phases = np.asarray(phases, dtype=np.float64)
    knots = open_uniform_bspline_knots(
        num_control_points,
        degree,
        source_horizon,
    )
    coefficients = np.eye(int(num_control_points), dtype=np.float64)
    return np.asarray(
        BSpline(knots, coefficients, int(degree), extrapolate=False)(phases),
        dtype=np.float64,
    )


def paired_bspline_control_phases(
    *,
    num_control_points: int = PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
) -> np.ndarray:
    """Return Greville phases for controls on the shared spline basis."""

    knots = open_uniform_bspline_knots(
        int(num_control_points),
        int(degree),
        int(source_horizon),
    )
    return np.asarray(
        [
            np.mean(knots[index + 1 : index + int(degree) + 1])
            for index in range(int(num_control_points))
        ],
        dtype=np.float64,
    )


def _decode_discrete_gripper_controls(
    controls: np.ndarray,
    *,
    output_steps: int,
    degree: int,
    source_horizon: int,
) -> np.ndarray:
    return _decode_discrete_gripper_controls_at_phases(
        controls,
        output_phases=np.arange(int(output_steps), dtype=np.float64),
        degree=int(degree),
        source_horizon=int(source_horizon),
    )


def _decode_discrete_gripper_controls_at_phases(
    controls: np.ndarray,
    *,
    output_phases: np.ndarray,
    degree: int,
    source_horizon: int,
) -> np.ndarray:
    values = np.asarray(controls, dtype=np.float64)
    phases = paired_bspline_control_phases(
        num_control_points=int(values.shape[-1]),
        degree=int(degree),
        source_horizon=int(source_horizon),
    )
    output_phases = np.asarray(output_phases, dtype=np.float64).reshape(-1)
    flat = values.reshape(-1, values.shape[-1])
    decoded = np.stack(
        [
            np.interp(output_phases, phases, row)
            for row in flat
        ],
        axis=0,
    )
    discrete = np.where(decoded >= 0.0, 1.0, -1.0)
    return discrete.reshape(values.shape[:-1] + (len(output_phases),))


def _bspline_support_phases(
    output_steps: int,
    support_subdivisions: int,
) -> np.ndarray:
    output_steps = int(output_steps)
    support_subdivisions = int(support_subdivisions)
    if output_steps < 1:
        raise ValueError("output_steps must be positive")
    if support_subdivisions < 1:
        raise ValueError("support_subdivisions must be at least one")
    support_count = (output_steps - 1) * support_subdivisions + 1
    return (
        np.arange(support_count, dtype=np.float64)
        / float(support_subdivisions)
    )


def _paired_bspline_fit_operator(
    *,
    num_control_points: int,
    degree: int,
    source_horizon: int,
    velocity_weight: float,
    smoothness_weight: float,
    anchor_first_control: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if float(velocity_weight) < 0.0 or float(smoothness_weight) < 0.0:
        raise ValueError("fit weights must be non-negative")
    phases = np.arange(int(source_horizon), dtype=np.float64)
    basis = paired_bspline_basis(
        phases,
        num_control_points=num_control_points,
        degree=degree,
        source_horizon=source_horizon,
    )
    sample_difference = np.diff(np.eye(int(source_horizon)), axis=0)
    control_difference = np.diff(
        np.eye(int(num_control_points)),
        n=2,
        axis=0,
    )
    target_map = [np.eye(int(source_horizon), dtype=np.float64)]
    design = [basis]
    if float(velocity_weight) > 0.0:
        scale = np.sqrt(float(velocity_weight))
        design.append(scale * (sample_difference @ basis))
        target_map.append(scale * sample_difference)
    if float(smoothness_weight) > 0.0:
        scale = np.sqrt(float(smoothness_weight))
        design.append(scale * control_difference)
        target_map.append(
            np.zeros(
                (len(control_difference), int(source_horizon)),
                dtype=np.float64,
            )
        )
    design_matrix = np.concatenate(design, axis=0)
    target_matrix = np.concatenate(target_map, axis=0)
    if not anchor_first_control:
        return np.linalg.pinv(design_matrix) @ target_matrix, basis

    first_sample_map = np.zeros(
        (1, int(source_horizon)),
        dtype=np.float64,
    )
    first_sample_map[0, 0] = 1.0
    operator = np.zeros(
        (int(num_control_points), int(source_horizon)),
        dtype=np.float64,
    )
    operator[0] = first_sample_map
    residual_target_map = (
        target_matrix
        - design_matrix[:, :1] @ first_sample_map
    )
    operator[1:] = (
        np.linalg.pinv(design_matrix[:, 1:]) @ residual_target_map
    )
    return operator, basis


def _batched_relative_rotvec(
    current_quaternions: np.ndarray,
    target_quaternions: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_quaternions, dtype=np.float64)
    target = np.asarray(target_quaternions, dtype=np.float64)
    current = current / np.maximum(
        np.linalg.norm(current, axis=-1, keepdims=True),
        1.0e-12,
    )
    target = target / np.maximum(
        np.linalg.norm(target, axis=-1, keepdims=True),
        1.0e-12,
    )
    inverse = np.concatenate([-current[..., :3], current[..., 3:4]], axis=-1)
    x1, y1, z1, w1 = np.moveaxis(target, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(inverse, -1, 0)
    relative = np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )
    relative /= np.maximum(
        np.linalg.norm(relative, axis=-1, keepdims=True),
        1.0e-12,
    )
    relative = np.where(relative[..., 3:4] < 0.0, -relative, relative)
    xyz_norm = np.linalg.norm(relative[..., :3], axis=-1)
    angle = 2.0 * np.arctan2(
        xyz_norm,
        np.clip(relative[..., 3], -1.0, 1.0),
    )
    scale = np.divide(
        angle,
        xyz_norm,
        out=np.zeros_like(angle),
        where=xyz_norm > 1.0e-12,
    )
    return relative[..., :3] * scale[..., None]


def build_libero_paired_bspline_labels(
    positions: np.ndarray,
    quaternions: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    num_control_points: int = PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
    velocity_weight: float = 1.0,
    smoothness_weight: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit shared-phase relative-pose and native-action B-spline controls.

    Every observation owns one ``(K, 13)`` control matrix. The first six
    channels are a relative SE(3) pose curve and the final seven channels are
    the native LIBERO action curve, including the gripper command.
    """

    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (N,3), got {positions.shape}")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("quaternions must have shape (N,4)")
    if actions.shape != (len(positions), 7):
        raise ValueError("LIBERO actions must have shape (N,7)")
    if len(ends) == 0 or int(ends[-1]) != len(positions):
        raise ValueError("episode_ends must terminate at the final sample")
    if np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing")
    if float(position_action_scale) <= 0.0 or float(rotation_action_scale) <= 0.0:
        raise ValueError("action scales must be positive")

    geometry_operator, basis = _paired_bspline_fit_operator(
        num_control_points=int(num_control_points),
        degree=int(degree),
        source_horizon=int(source_horizon),
        velocity_weight=float(velocity_weight),
        smoothness_weight=float(smoothness_weight),
        anchor_first_control=True,
    )
    action_operator, _ = _paired_bspline_fit_operator(
        num_control_points=int(num_control_points),
        degree=int(degree),
        source_horizon=int(source_horizon),
        velocity_weight=float(velocity_weight),
        smoothness_weight=float(smoothness_weight),
        anchor_first_control=True,
    )
    controls = np.empty(
        (len(positions), int(num_control_points), PAIRED_POSE_ACTION_DIM),
        dtype=np.float32,
    )
    squared_error = 0.0
    squared_error_by_channel = np.zeros(
        (PAIRED_POSE_ACTION_DIM,),
        dtype=np.float64,
    )
    error_count = 0
    channel_error_count = 0
    max_abs_error = 0.0
    max_abs_error_by_channel = np.zeros(
        (PAIRED_POSE_ACTION_DIM,),
        dtype=np.float64,
    )
    gripper_mismatches = 0
    gripper_samples = 0
    start = 0
    offsets = np.arange(int(source_horizon), dtype=np.int64)[None]
    control_phases = paired_bspline_control_phases(
        num_control_points=int(num_control_points),
        degree=int(degree),
        source_horizon=int(source_horizon),
    )
    gripper_sample_indices = np.clip(
        np.rint(control_phases).astype(np.int64),
        0,
        int(source_horizon) - 1,
    )
    for raw_end in ends:
        end = int(raw_end)
        length = end - start
        local_indices = np.minimum(
            np.arange(length, dtype=np.int64)[:, None] + offsets,
            length - 1,
        )
        indices = local_indices + start
        relative_position = (
            positions[indices] - positions[start:end, None]
        ) / float(position_action_scale)
        current_quaternion = np.broadcast_to(
            quaternions[start:end, None],
            quaternions[indices].shape,
        )
        relative_rotation = _batched_relative_rotvec(
            current_quaternion,
            quaternions[indices],
        ) / float(rotation_action_scale)
        geometry = np.concatenate(
            [relative_position, relative_rotation],
            axis=-1,
        )
        future_actions = actions[indices]
        geometry_controls = np.einsum(
            "kh,lhd->lkd",
            geometry_operator,
            geometry,
        )
        action_controls = np.einsum(
            "kh,lhd->lkd",
            action_operator,
            future_actions[..., :COMMAND_DIM],
        )
        gripper_controls = future_actions[
            :, gripper_sample_indices, GRIPPER_INDEX - GEOMETRY_DIM
        ][..., None]
        episode_controls = np.concatenate(
            [geometry_controls, action_controls, gripper_controls],
            axis=-1,
        )
        controls[start:end] = episode_controls.astype(np.float32)

        targets = np.concatenate([geometry, future_actions], axis=-1)
        reconstructed = np.empty_like(targets)
        reconstructed[..., :GRIPPER_INDEX] = np.einsum(
            "hk,lkd->lhd",
            basis,
            episode_controls[..., :GRIPPER_INDEX],
        )
        reconstructed[..., GRIPPER_INDEX] = (
            _decode_discrete_gripper_controls(
                episode_controls[..., GRIPPER_INDEX],
                output_steps=int(source_horizon),
                degree=int(degree),
                source_horizon=int(source_horizon),
            )
        )
        error = reconstructed - targets
        squared_error += float(np.sum(np.square(error)))
        squared_error_by_channel += np.sum(np.square(error), axis=(0, 1))
        error_count += int(error.size)
        channel_error_count += int(np.prod(error.shape[:2]))
        max_abs_error = max(max_abs_error, float(np.max(np.abs(error))))
        max_abs_error_by_channel = np.maximum(
            max_abs_error_by_channel,
            np.max(np.abs(error), axis=(0, 1)),
        )
        gripper_mismatches += int(
            np.count_nonzero(
                reconstructed[..., GRIPPER_INDEX]
                != targets[..., GRIPPER_INDEX]
            )
        )
        gripper_samples += int(np.prod(targets.shape[:2]))
        start = end

    channel_rmse = np.sqrt(
        squared_error_by_channel / max(channel_error_count, 1)
    )
    summary: dict[str, object] = {
        "steps": int(len(controls)),
        "episodes": int(len(ends)),
        "control_points": int(num_control_points),
        "degree": int(degree),
        "source_horizon": int(source_horizon),
        "action_dim": PAIRED_POSE_ACTION_DIM,
        "position_action_scale": float(position_action_scale),
        "rotation_action_scale": float(rotation_action_scale),
        "velocity_weight": float(velocity_weight),
        "smoothness_weight": float(smoothness_weight),
        "fit_rmse": float(np.sqrt(squared_error / max(error_count, 1))),
        "fit_max_abs_error": float(max_abs_error),
        "fit_rmse_by_channel": channel_rmse.tolist(),
        "fit_max_abs_error_by_channel": max_abs_error_by_channel.tolist(),
        "continuous_fit_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(channel_rmse[:GRIPPER_INDEX])
                )
            )
        ),
        "geometry_fit_rmse": float(
            np.sqrt(np.mean(np.square(channel_rmse[:GEOMETRY_DIM])))
        ),
        "native_action_fit_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        channel_rmse[
                            GEOMETRY_DIM : GEOMETRY_DIM + COMMAND_DIM
                        ]
                    )
                )
            )
        ),
        "gripper_fit_rmse": float(channel_rmse[GRIPPER_INDEX]),
        "gripper_state_mismatch_fraction": float(
            gripper_mismatches / max(gripper_samples, 1)
        ),
        "layout": (
            "relative_pose_controls6 + native_arm_action_controls6 + "
            "discrete_gripper_controls1"
        ),
    }
    return np.ascontiguousarray(controls), summary


def build_libero_paired_pose_bspline_raw_action_labels(
    positions: np.ndarray,
    quaternions: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    num_control_points: int = PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
    velocity_weight: float = 1.0,
    smoothness_weight: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pair pose B-spline controls with exact native action steps.

    This is an action-representation ablation of the paired B-spline target.
    The first six channels remain the canonical pose B-spline controls. The
    final seven channels store the uncompressed future action at each native
    controller phase, so the output row count must equal the source horizon.
    """

    source_horizon = int(source_horizon)
    num_control_points = int(num_control_points)
    if num_control_points != source_horizon:
        raise ValueError(
            "Raw-action rows require num_control_points == source_horizon: "
            f"{num_control_points} != {source_horizon}"
        )
    actions = np.asarray(actions, dtype=np.float64)
    ends = np.asarray(episode_ends, dtype=np.int64)
    controls, paired_summary = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        ends,
        source_horizon=source_horizon,
        num_control_points=num_control_points,
        degree=int(degree),
        position_action_scale=float(position_action_scale),
        rotation_action_scale=float(rotation_action_scale),
        velocity_weight=float(velocity_weight),
        smoothness_weight=float(smoothness_weight),
    )

    offsets = np.arange(source_horizon, dtype=np.int64)[None]
    start = 0
    for raw_end in ends:
        end = int(raw_end)
        length = end - start
        local_indices = np.minimum(
            np.arange(length, dtype=np.int64)[:, None] + offsets,
            length - 1,
        )
        controls[start:end, :, GEOMETRY_DIM:] = actions[
            local_indices + start
        ].astype(np.float32)
        start = end

    channel_rmse = np.asarray(
        paired_summary["fit_rmse_by_channel"],
        dtype=np.float64,
    )
    channel_max = np.asarray(
        paired_summary["fit_max_abs_error_by_channel"],
        dtype=np.float64,
    )
    channel_rmse[GEOMETRY_DIM:] = 0.0
    channel_max[GEOMETRY_DIM:] = 0.0
    summary = {
        **paired_summary,
        "fit_rmse": float(np.sqrt(np.mean(np.square(channel_rmse)))),
        "fit_max_abs_error": float(np.max(channel_max)),
        "fit_rmse_by_channel": channel_rmse.tolist(),
        "fit_max_abs_error_by_channel": channel_max.tolist(),
        "continuous_fit_rmse": float(
            np.sqrt(np.mean(np.square(channel_rmse[:GRIPPER_INDEX])))
        ),
        "native_action_fit_rmse": 0.0,
        "gripper_fit_rmse": 0.0,
        "gripper_state_mismatch_fraction": 0.0,
        "action_representation": "exact_native_steps",
        "layout": (
            "relative_pose_controls6 + native_arm_action_steps6 + "
            "native_gripper_steps1"
        ),
    }
    return np.ascontiguousarray(controls), summary


def build_libero_action_bspline_labels(
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    num_control_points: int = PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    velocity_weight: float = 1.0,
    smoothness_weight: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit native LIBERO action windows with cubic B-spline controls."""

    actions = np.asarray(actions, dtype=np.float64)
    ends = np.asarray(episode_ends, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != LIBERO_ACTION_DIM:
        raise ValueError(
            "LIBERO actions must have shape (N,7), got "
            f"{actions.shape}"
        )
    if len(ends) == 0 or int(ends[-1]) != len(actions):
        raise ValueError("episode_ends must terminate at the final sample")
    if np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing")

    action_operator, basis = _paired_bspline_fit_operator(
        num_control_points=int(num_control_points),
        degree=int(degree),
        source_horizon=int(source_horizon),
        velocity_weight=float(velocity_weight),
        smoothness_weight=float(smoothness_weight),
        anchor_first_control=True,
    )
    controls = np.empty(
        (len(actions), int(num_control_points), LIBERO_ACTION_DIM),
        dtype=np.float32,
    )
    squared_error_by_channel = np.zeros(
        (LIBERO_ACTION_DIM,),
        dtype=np.float64,
    )
    max_abs_error_by_channel = np.zeros(
        (LIBERO_ACTION_DIM,),
        dtype=np.float64,
    )
    channel_error_count = 0
    gripper_mismatches = 0
    gripper_samples = 0
    offsets = np.arange(int(source_horizon), dtype=np.int64)[None]
    control_phases = paired_bspline_control_phases(
        num_control_points=int(num_control_points),
        degree=int(degree),
        source_horizon=int(source_horizon),
    )
    gripper_sample_indices = np.clip(
        np.rint(control_phases).astype(np.int64),
        0,
        int(source_horizon) - 1,
    )

    start = 0
    for raw_end in ends:
        end = int(raw_end)
        length = end - start
        local_indices = np.minimum(
            np.arange(length, dtype=np.int64)[:, None] + offsets,
            length - 1,
        )
        future_actions = actions[local_indices + start]
        arm_controls = np.einsum(
            "kh,lhd->lkd",
            action_operator,
            future_actions[..., :COMMAND_DIM],
        )
        gripper_controls = future_actions[
            :, gripper_sample_indices, LIBERO_ACTION_GRIPPER_INDEX
        ][..., None]
        episode_controls = np.concatenate(
            [arm_controls, gripper_controls],
            axis=-1,
        )
        controls[start:end] = episode_controls.astype(np.float32)

        reconstructed = np.empty_like(future_actions)
        reconstructed[..., :COMMAND_DIM] = np.einsum(
            "hk,lkd->lhd",
            basis,
            arm_controls,
        )
        reconstructed[..., LIBERO_ACTION_GRIPPER_INDEX] = (
            _decode_discrete_gripper_controls(
                episode_controls[..., LIBERO_ACTION_GRIPPER_INDEX],
                output_steps=int(source_horizon),
                degree=int(degree),
                source_horizon=int(source_horizon),
            )
        )
        error = reconstructed - future_actions
        squared_error_by_channel += np.sum(np.square(error), axis=(0, 1))
        max_abs_error_by_channel = np.maximum(
            max_abs_error_by_channel,
            np.max(np.abs(error), axis=(0, 1)),
        )
        channel_error_count += int(np.prod(error.shape[:2]))
        gripper_mismatches += int(
            np.count_nonzero(
                reconstructed[..., LIBERO_ACTION_GRIPPER_INDEX]
                != future_actions[..., LIBERO_ACTION_GRIPPER_INDEX]
            )
        )
        gripper_samples += int(np.prod(future_actions.shape[:2]))
        start = end

    channel_rmse = np.sqrt(
        squared_error_by_channel / max(channel_error_count, 1)
    )
    summary: dict[str, object] = {
        "steps": int(len(controls)),
        "episodes": int(len(ends)),
        "control_points": int(num_control_points),
        "degree": int(degree),
        "source_horizon": int(source_horizon),
        "action_dim": LIBERO_ACTION_DIM,
        "velocity_weight": float(velocity_weight),
        "smoothness_weight": float(smoothness_weight),
        "fit_rmse": float(np.sqrt(np.mean(np.square(channel_rmse)))),
        "fit_max_abs_error": float(np.max(max_abs_error_by_channel)),
        "fit_rmse_by_channel": channel_rmse.tolist(),
        "fit_max_abs_error_by_channel": max_abs_error_by_channel.tolist(),
        "native_action_fit_rmse": float(
            np.sqrt(np.mean(np.square(channel_rmse[:COMMAND_DIM])))
        ),
        "gripper_fit_rmse": float(
            channel_rmse[LIBERO_ACTION_GRIPPER_INDEX]
        ),
        "gripper_state_mismatch_fraction": float(
            gripper_mismatches / max(gripper_samples, 1)
        ),
        "layout": (
            "native_arm_action_controls6 + discrete_gripper_controls1"
        ),
    }
    return np.ascontiguousarray(controls), summary


def decode_libero_action_bspline_controls(
    controls: np.ndarray,
    *,
    output_steps: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
) -> np.ndarray:
    """Decode native-action B-spline controls at controller-step phases."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != LIBERO_ACTION_DIM:
        raise ValueError(
            "controls must end in (num_control_points, 7), got "
            f"{values.shape}"
        )
    num_control_points = int(values.shape[-2])
    output_steps = int(output_steps)
    if output_steps < 1 or output_steps > int(source_horizon):
        raise ValueError("output_steps must be in [1, source_horizon]")
    basis = paired_bspline_basis(
        np.arange(output_steps, dtype=np.float64),
        num_control_points=num_control_points,
        degree=int(degree),
        source_horizon=int(source_horizon),
    )
    decoded = np.empty(
        values.shape[:-2] + (output_steps, LIBERO_ACTION_DIM),
        dtype=np.float64,
    )
    decoded[..., :COMMAND_DIM] = np.einsum(
        "tk,...kd->...td",
        basis,
        values[..., :COMMAND_DIM],
    )
    decoded[..., LIBERO_ACTION_GRIPPER_INDEX] = (
        _decode_discrete_gripper_controls(
            values[..., LIBERO_ACTION_GRIPPER_INDEX],
            output_steps=output_steps,
            degree=int(degree),
            source_horizon=int(source_horizon),
        )
    )
    return np.clip(decoded, -1.0, 1.0).astype(np.float32)


def _decode_paired_pose_and_native_curves(
    pose_curve: np.ndarray,
    native_action_curve: np.ndarray,
    *,
    rotation_action_scale: float,
) -> np.ndarray:
    pose = np.asarray(pose_curve, dtype=np.float64)
    native_actions = np.asarray(native_action_curve, dtype=np.float64)
    if pose.shape[:-1] != native_actions.shape[:-1]:
        raise ValueError("pose and native-action curves must share phases")
    if pose.shape[-1] != GEOMETRY_DIM:
        raise ValueError("pose curve must have six channels")
    if native_actions.shape[-1] != LIBERO_ACTION_DIM:
        raise ValueError("native-action curve must have seven channels")

    support_steps = int(pose.shape[-2])
    flat_pose = pose.reshape(-1, support_steps, GEOMETRY_DIM)
    flat_actions = native_actions.reshape(
        -1,
        support_steps,
        LIBERO_ACTION_DIM,
    )
    decoded = np.empty(
        (len(flat_pose), support_steps, PAIRED_POSE_ACTION_DIM),
        dtype=np.float64,
    )
    for batch_index, sample in enumerate(flat_pose):
        relative_position = sample[:, :3] - sample[0:1, :3]
        raw_relative_rotation = sample[:, 3:6] * float(rotation_action_scale)
        relative_quaternions = np.stack(
            [
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                if np.linalg.norm(vector) <= 1.0e-12
                else np.concatenate(
                    [
                        vector
                        * (
                            np.sin(0.5 * np.linalg.norm(vector))
                            / np.linalg.norm(vector)
                        ),
                        [np.cos(0.5 * np.linalg.norm(vector))],
                    ]
                )
                for vector in raw_relative_rotation
            ],
            axis=0,
        )
        anchored_current = np.broadcast_to(
            relative_quaternions[0:1],
            relative_quaternions.shape,
        )
        anchored_rotvec = _batched_relative_rotvec(
            anchored_current,
            relative_quaternions,
        )
        position_delta = np.zeros_like(relative_position)
        rotation_delta = np.zeros_like(anchored_rotvec)
        if support_steps > 1:
            position_delta[:-1] = np.diff(relative_position, axis=0)
            rotation_delta[:-1] = _batched_relative_rotvec(
                relative_quaternions[:-1],
                relative_quaternions[1:],
            ) / float(rotation_action_scale)
        decoded[batch_index] = np.concatenate(
            [
                position_delta,
                rotation_delta,
                np.clip(flat_actions[batch_index], -1.0, 1.0),
            ],
            axis=-1,
        )
    return decoded.reshape(
        pose.shape[:-1] + (PAIRED_POSE_ACTION_DIM,)
    ).astype(np.float32)


def decode_libero_paired_bspline_controls(
    controls: np.ndarray,
    *,
    output_steps: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    rotation_action_scale: float = 0.5,
    support_subdivisions: int = 1,
) -> np.ndarray:
    """Decode paired controls on a shared, optionally densified phase grid."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != PAIRED_POSE_ACTION_DIM:
        raise ValueError(
            "controls must end in (num_control_points, 13), got "
            f"{values.shape}"
        )
    num_control_points = int(values.shape[-2])
    output_steps = int(output_steps)
    if output_steps < 1 or output_steps > int(source_horizon):
        raise ValueError("output_steps must be in [1, source_horizon]")
    support_phases = _bspline_support_phases(
        output_steps,
        int(support_subdivisions),
    )
    support_steps = len(support_phases)
    basis = paired_bspline_basis(
        support_phases,
        num_control_points=num_control_points,
        degree=int(degree),
        source_horizon=int(source_horizon),
    )
    pose_curve = np.einsum(
        "tk,...kd->...td",
        basis,
        values[..., :GEOMETRY_DIM],
    )
    native_action_curve = np.empty(
        values.shape[:-2] + (support_steps, LIBERO_ACTION_DIM),
        dtype=np.float64,
    )
    native_action_curve[..., :COMMAND_DIM] = np.einsum(
        "tk,...kd->...td",
        basis,
        values[..., GEOMETRY_DIM:GRIPPER_INDEX],
    )
    native_action_curve[..., LIBERO_ACTION_GRIPPER_INDEX] = (
        _decode_discrete_gripper_controls_at_phases(
            values[..., GRIPPER_INDEX],
            output_phases=support_phases,
            degree=int(degree),
            source_horizon=int(source_horizon),
        )
    )
    return _decode_paired_pose_and_native_curves(
        pose_curve,
        native_action_curve,
        rotation_action_scale=float(rotation_action_scale),
    )


def decode_libero_paired_pose_bspline_raw_action(
    controls: np.ndarray,
    *,
    output_steps: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
    rotation_action_scale: float = 0.5,
    support_subdivisions: int = 1,
) -> np.ndarray:
    """Decode pose controls while preserving raw native action phases."""

    values = np.asarray(controls, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != PAIRED_POSE_ACTION_DIM:
        raise ValueError(
            "controls must end in (source_horizon, 13), got "
            f"{values.shape}"
        )
    source_horizon = int(source_horizon)
    if int(values.shape[-2]) != source_horizon:
        raise ValueError(
            "Raw-action rows must equal source_horizon: "
            f"{values.shape[-2]} != {source_horizon}"
        )
    output_steps = int(output_steps)
    if output_steps < 1 or output_steps > source_horizon:
        raise ValueError("output_steps must be in [1, source_horizon]")
    support_phases = _bspline_support_phases(
        output_steps,
        int(support_subdivisions),
    )
    basis = paired_bspline_basis(
        support_phases,
        num_control_points=source_horizon,
        degree=int(degree),
        source_horizon=source_horizon,
    )
    pose_curve = np.einsum(
        "tk,...kd->...td",
        basis,
        values[..., :GEOMETRY_DIM],
    )

    native_phases = np.arange(source_horizon, dtype=np.float64)
    raw_actions = values[..., GEOMETRY_DIM:]
    flat_actions = raw_actions.reshape(-1, source_horizon, LIBERO_ACTION_DIM)
    interpolated = np.empty(
        (len(flat_actions), len(support_phases), LIBERO_ACTION_DIM),
        dtype=np.float64,
    )
    for batch_index, sample in enumerate(flat_actions):
        for channel in range(COMMAND_DIM):
            interpolated[batch_index, :, channel] = np.interp(
                support_phases,
                native_phases,
                sample[:, channel],
            )
        interpolated[batch_index, :, LIBERO_ACTION_GRIPPER_INDEX] = np.where(
            np.interp(
                support_phases,
                native_phases,
                sample[:, LIBERO_ACTION_GRIPPER_INDEX],
            )
            >= 0.0,
            1.0,
            -1.0,
        )
    native_action_curve = interpolated.reshape(
        values.shape[:-2] + (len(support_phases), LIBERO_ACTION_DIM)
    )
    return _decode_paired_pose_and_native_curves(
        pose_curve,
        native_action_curve,
        rotation_action_scale=float(rotation_action_scale),
    )


def build_libero_paired_pose_action_labels(
    positions: np.ndarray,
    quaternions: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    *,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build aligned ``geometry6 + command6 + gripper1`` LIBERO labels."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"LIBERO actions must have shape (N,7), got {actions.shape}")
    labels, target_indices = build_command_endpoint_field_labels(
        positions,
        quaternions,
        actions,
        episode_ends,
        stride=1,
        position_action_scale=float(position_action_scale),
        rotation_action_scale=float(rotation_action_scale),
    )
    if labels.shape != (len(actions), PAIRED_POSE_ACTION_DIM):
        raise RuntimeError(
            "Unexpected paired LIBERO label shape: "
            f"{labels.shape} != {(len(actions), PAIRED_POSE_ACTION_DIM)}"
        )
    max_abs_before_clip = float(np.max(np.abs(labels)))
    clipped_fraction = float(np.mean(np.abs(labels) > 1.0))
    if clip:
        labels = np.clip(labels, -1.0, 1.0)
    elif max_abs_before_clip > 1.0 + 1.0e-5:
        raise ValueError(
            "Paired LIBERO labels exceed [-1,1] with clipping disabled: "
            f"max_abs={max_abs_before_clip:.6f}"
        )
    summary: dict[str, float | int] = {
        "steps": int(len(labels)),
        "episodes": int(len(np.asarray(episode_ends))),
        "action_dim": PAIRED_POSE_ACTION_DIM,
        "position_action_scale": float(position_action_scale),
        "rotation_action_scale": float(rotation_action_scale),
        "max_abs_before_clip": max_abs_before_clip,
        "clipped_fraction": clipped_fraction,
    }
    return (
        np.ascontiguousarray(labels, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int64),
        summary,
    )


@dataclass(frozen=True)
class _GripperBoundary:
    action_index: int
    command: float


@dataclass(frozen=True)
class PairedPoseEnsembleSupport:
    """Pose-only uncertainty attached to one selected paired prediction."""

    selected_sample_index: int
    cluster_members: tuple[int, ...]
    cluster_confident: bool
    position_covariances: np.ndarray
    position_spread_radii: np.ndarray
    rotation_spread_radii: np.ndarray
    position_distance_scale: float
    rotation_distance_scale: float
    position_dense_core_covariances: np.ndarray | None = None
    position_dense_core_spread_radii: np.ndarray | None = None
    dense_core_fraction: float | None = None
    sample_weights: np.ndarray | None = None
    selected_density: float | None = None
    effective_sample_size: float | None = None
    kernel_mass_fraction: float | None = None
    spread_quantile: float = 0.9


def find_gripper_only_boundary(
    paired_actions: np.ndarray,
    *,
    position_action_scale: float,
    rotation_action_scale: float,
    position_threshold: float,
    rotation_threshold: float,
    close_threshold: float = 0.5,
) -> _GripperBoundary | None:
    """Find the first discrete gripper event with negligible SE(3) motion."""

    actions = np.asarray(paired_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != PAIRED_POSE_ACTION_DIM:
        raise ValueError(
            "paired_actions must have shape "
            f"(T,{PAIRED_POSE_ACTION_DIM}), got {actions.shape}"
        )
    for index in range(1, len(actions)):
        previous_closed = bool(actions[index - 1, GRIPPER_INDEX] > close_threshold)
        current_closed = bool(actions[index, GRIPPER_INDEX] > close_threshold)
        if previous_closed == current_closed:
            continue
        position_delta = float(
            np.linalg.norm(actions[index, :3]) * float(position_action_scale)
        )
        rotation_delta = float(
            np.linalg.norm(actions[index, 3:6]) * float(rotation_action_scale)
        )
        if (
            position_delta <= float(position_threshold)
            and rotation_delta <= float(rotation_threshold)
        ):
            return _GripperBoundary(
                action_index=index,
                command=float(actions[index, GRIPPER_INDEX]),
            )
    return None


class LiberoPairedBSplinePoseMedoidPolicy(torch.nn.Module):
    """Select one paired plan by pose medoid and retain pose uncertainty.

    The selected sample is returned without modifying any of its pose,
    action, or gripper controls. Other sampled actions are ignored. Their
    pose curves are used only to identify a dominant pose mode and estimate
    transverse support around the selected pose curve.
    """

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        samples: int = 8,
        source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
        selection_steps: int | None = None,
        degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
        support_subdivisions: int = 2,
        position_action_scale: float = 0.05,
        rotation_action_scale: float = 0.5,
        cluster_cutoff: float = 3.0,
        minimum_cluster_fraction: float = 0.5,
        spread_quantile: float = 0.9,
        dense_core_fraction: float | None = None,
        selection_mode: str = "dominant_cluster",
        distance_scale_mode: str = "nearest_neighbor",
        support_members_mode: str = "selected_members",
        kde_position_bandwidth: float = 0.01,
        kde_rotation_bandwidth: float = 0.25,
        kde_gripper_bandwidth: float = 1.0,
        kde_support_mass: float = 0.9,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.samples = int(samples)
        self.source_horizon = int(source_horizon)
        self.selection_steps = (
            self.source_horizon
            if selection_steps is None
            else int(selection_steps)
        )
        self.degree = int(degree)
        self.support_subdivisions = int(support_subdivisions)
        self.position_action_scale = float(position_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)
        self.cluster_cutoff = float(cluster_cutoff)
        self.minimum_cluster_fraction = float(minimum_cluster_fraction)
        self.spread_quantile = float(spread_quantile)
        self.selection_mode = str(selection_mode)
        self.distance_scale_mode = str(distance_scale_mode)
        self.support_members_mode = str(support_members_mode)
        self.kde_position_bandwidth = float(kde_position_bandwidth)
        self.kde_rotation_bandwidth = float(kde_rotation_bandwidth)
        self.kde_gripper_bandwidth = float(kde_gripper_bandwidth)
        self.kde_support_mass = float(kde_support_mass)
        self.dense_core_fraction = (
            None
            if dense_core_fraction is None
            else float(dense_core_fraction)
        )
        if self.samples < 2:
            raise ValueError("samples must be at least two")
        if self.source_horizon < 2:
            raise ValueError("source_horizon must be at least two")
        if not 1 <= self.selection_steps <= self.source_horizon:
            raise ValueError(
                "selection_steps must be in [1, source_horizon]"
            )
        if self.support_subdivisions < 1:
            raise ValueError("support_subdivisions must be at least one")
        if self.position_action_scale <= 0.0:
            raise ValueError("position_action_scale must be positive")
        if self.rotation_action_scale <= 0.0:
            raise ValueError("rotation_action_scale must be positive")
        if self.cluster_cutoff <= 0.0:
            raise ValueError("cluster_cutoff must be positive")
        if not 0.0 < self.minimum_cluster_fraction <= 1.0:
            raise ValueError("minimum_cluster_fraction must be in (0,1]")
        if not 0.0 < self.spread_quantile <= 1.0:
            raise ValueError("spread_quantile must be in (0,1]")
        if self.selection_mode not in {
            "dominant_cluster",
            "all_samples",
            "manifold_kde",
        }:
            raise ValueError(
                "selection_mode must be dominant_cluster, all_samples, "
                "or manifold_kde"
            )
        if self.distance_scale_mode not in {
            "nearest_neighbor",
            "pairwise_median",
        }:
            raise ValueError(
                "distance_scale_mode must be nearest_neighbor or "
                "pairwise_median"
            )
        if self.support_members_mode not in {
            "selected_members",
            "all_samples",
            "kde_responsibility",
        }:
            raise ValueError(
                "support_members_mode must be selected_members or "
                "all_samples, or kde_responsibility"
            )
        if (
            self.support_members_mode == "kde_responsibility"
            and self.selection_mode != "manifold_kde"
        ):
            raise ValueError(
                "kde_responsibility support requires manifold_kde selection"
            )
        for name, value in (
            ("kde_position_bandwidth", self.kde_position_bandwidth),
            ("kde_rotation_bandwidth", self.kde_rotation_bandwidth),
            ("kde_gripper_bandwidth", self.kde_gripper_bandwidth),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.kde_support_mass <= 1.0:
            raise ValueError("kde_support_mass must be in (0,1]")
        if (
            self.dense_core_fraction is not None
            and not 0.0 < self.dense_core_fraction <= 1.0
        ):
            raise ValueError("dense_core_fraction must be in (0,1]")
        self.selection_decoded_steps = min(
            self.selection_steps * self.support_subdivisions,
            (self.source_horizon - 1) * self.support_subdivisions + 1,
        )

        self.query_batches = 0
        self.env_queries = 0
        self.base_samples = 0
        self._pending_supports: list[PairedPoseEnsembleSupport] = []
        self._cluster_sizes: list[float] = []
        self._cluster_fractions: list[float] = []
        self._position_spreads: list[float] = []
        self._rotation_spreads: list[float] = []
        self._dense_core_sizes: list[float] = []
        self._dense_core_spreads: list[float] = []
        self._cluster_confidences: list[float] = []
        self._selected_densities: list[float] = []
        self._effective_sample_sizes: list[float] = []
        self._kernel_mass_fractions: list[float] = []

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        self._pending_supports = []
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_pose_medoid"

    def _decode_controls(self, controls: np.ndarray) -> np.ndarray:
        return decode_libero_paired_bspline_controls(
            controls,
            output_steps=self.source_horizon,
            source_horizon=self.source_horizon,
            degree=self.degree,
            rotation_action_scale=self.rotation_action_scale,
            support_subdivisions=self.support_subdivisions,
        )

    @staticmethod
    def _batch_size(obs_dict: dict[str, Any]) -> int:
        for value in obs_dict.values():
            if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        raise ValueError("Cannot infer observation batch size")

    def _repeat_batch_value(self, value: Any, batch_size: int) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                return torch.repeat_interleave(value, self.samples, dim=0)
            return value
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                return np.repeat(value, self.samples, axis=0)
            return value
        if isinstance(value, list) and len(value) == batch_size:
            return [item for item in value for _ in range(self.samples)]
        if isinstance(value, tuple) and len(value) == batch_size:
            return tuple(item for item in value for _ in range(self.samples))
        return value

    def _relative_pose_centers(
        self,
        decoded: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(decoded, dtype=np.float64)
        sample_count, support_count = values.shape[:2]
        increments = values[..., :3] * self.position_action_scale
        positions = np.concatenate(
            [
                np.zeros((sample_count, 1, 3), dtype=np.float64),
                np.cumsum(increments, axis=1),
            ],
            axis=1,
        )
        orientations = np.zeros(
            (sample_count, support_count + 1, 4),
            dtype=np.float64,
        )
        orientations[:, 0, 3] = 1.0
        for sample_index in range(sample_count):
            orientation = orientations[sample_index, 0].copy()
            for support_index in range(support_count):
                delta = rotvec_to_quaternion(
                    values[sample_index, support_index, 3:6]
                    * self.rotation_action_scale
                )
                orientation = quaternion_multiply(delta, orientation)
                orientations[
                    sample_index,
                    support_index + 1,
                ] = orientation
        return positions, orientations

    @staticmethod
    def _pairwise_position_distance(positions: np.ndarray) -> np.ndarray:
        delta = positions[:, None] - positions[None, :]
        return np.sqrt(
            np.mean(np.sum(np.square(delta), axis=-1), axis=-1)
        )

    @staticmethod
    def _pairwise_rotation_distance(orientations: np.ndarray) -> np.ndarray:
        sample_count = int(len(orientations))
        distance = np.zeros((sample_count, sample_count), dtype=np.float64)
        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                errors = np.stack(
                    [
                        relative_rotation_vector(current, target)
                        for current, target in zip(
                            orientations[left],
                            orientations[right],
                            strict=True,
                        )
                    ],
                    axis=0,
                )
                value = float(
                    np.sqrt(np.mean(np.sum(np.square(errors), axis=-1)))
                )
                distance[left, right] = value
                distance[right, left] = value
        return distance

    @staticmethod
    def _pairwise_gripper_distance(grippers: np.ndarray) -> np.ndarray:
        closed = np.asarray(grippers, dtype=np.float64) > 0.0
        delta = closed[:, None] != closed[None, :]
        return np.sqrt(np.mean(delta.astype(np.float64), axis=-1))

    def _manifold_kde_selection(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        grippers: np.ndarray,
    ) -> tuple[list[int], int, bool, np.ndarray, float, float, float]:
        """Select an observed paired sample at the densest pose mode."""

        position = self._pairwise_position_distance(positions)
        rotation = self._pairwise_rotation_distance(orientations)
        gripper = self._pairwise_gripper_distance(grippers)
        squared_distance = (
            np.square(position / self.kde_position_bandwidth)
            + np.square(rotation / self.kde_rotation_bandwidth)
            + np.square(gripper / self.kde_gripper_bandwidth)
        )
        kernel = np.exp(-0.5 * squared_distance)
        densities = np.mean(kernel, axis=1)
        selected = int(np.argmax(densities))
        weights = np.asarray(kernel[selected], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        weight_square_sum = float(np.sum(np.square(weights)))
        effective_sample_size = (
            weight_sum * weight_sum / max(weight_square_sum, 1.0e-12)
        )
        kernel_mass_fraction = weight_sum / float(len(weights))

        order = np.argsort(-weights, kind="stable")
        cumulative = np.cumsum(weights[order])
        member_count = int(
            np.searchsorted(
                cumulative,
                self.kde_support_mass * weight_sum,
                side="left",
            )
            + 1
        )
        members = sorted(int(index) for index in order[:member_count])
        required = float(
            np.ceil(self.minimum_cluster_fraction * len(weights))
        )
        confident = effective_sample_size + 1.0e-12 >= required
        return (
            members,
            selected,
            confident,
            weights,
            float(densities[selected]),
            float(effective_sample_size),
            float(kernel_mass_fraction),
        )

    @staticmethod
    def _nearest_neighbor_scale(distance: np.ndarray) -> float:
        values = np.asarray(distance, dtype=np.float64)
        masked = values.copy()
        np.fill_diagonal(masked, np.inf)
        nearest = np.min(masked, axis=1)
        finite = nearest[np.isfinite(nearest) & (nearest > 1.0e-12)]
        if len(finite) == 0:
            return 1.0
        return float(max(np.median(finite), 1.0e-12))

    @staticmethod
    def _pairwise_median_scale(distance: np.ndarray) -> float:
        values = np.asarray(distance, dtype=np.float64)
        upper = values[np.triu_indices(len(values), k=1)]
        finite = upper[np.isfinite(upper) & (upper > 1.0e-12)]
        if len(finite) == 0:
            return 1.0
        return float(max(np.median(finite), 1.0e-12))

    def _distance_scale(self, distance: np.ndarray) -> float:
        if self.distance_scale_mode == "pairwise_median":
            return self._pairwise_median_scale(distance)
        return self._nearest_neighbor_scale(distance)

    def _normalized_pose_distance(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        position = self._pairwise_position_distance(positions)
        rotation = self._pairwise_rotation_distance(orientations)
        position_scale = self._distance_scale(position)
        rotation_scale = self._distance_scale(rotation)
        components = []
        if float(np.max(position)) > 1.0e-12:
            components.append(position / position_scale)
        if float(np.max(rotation)) > 1.0e-12:
            components.append(rotation / rotation_scale)
        if not components:
            combined = np.zeros_like(position)
        else:
            stacked = np.stack(components, axis=0)
            combined = np.sqrt(np.mean(np.square(stacked), axis=0))
        return combined, position_scale, rotation_scale

    @staticmethod
    def _cluster_medoid(distance: np.ndarray, members: list[int]) -> int:
        if len(members) == 1:
            return int(members[0])
        submatrix = distance[np.ix_(members, members)]
        return int(members[int(np.argmin(np.mean(submatrix, axis=1)))])

    def _dominant_cluster(
        self,
        distance: np.ndarray,
    ) -> tuple[list[int], bool]:
        clusters = [[index] for index in range(len(distance))]
        while len(clusters) > 1:
            best: tuple[float, int, int] | None = None
            for left in range(len(clusters)):
                for right in range(left + 1, len(clusters)):
                    linkage = float(
                        np.max(
                            distance[np.ix_(clusters[left], clusters[right])]
                        )
                    )
                    candidate = (linkage, left, right)
                    if best is None or candidate < best:
                        best = candidate
            if best is None or best[0] > self.cluster_cutoff:
                break
            _, left, right = best
            clusters[left] = sorted(clusters[left] + clusters[right])
            del clusters[right]

        def rank(members: list[int]) -> tuple[float, float, int]:
            compactness = float(
                np.mean(distance[np.ix_(members, members)])
            )
            return (-float(len(members)), compactness, min(members))

        dominant = min(clusters, key=rank)
        required = int(
            np.ceil(self.minimum_cluster_fraction * len(distance))
        )
        if len(dominant) < required:
            return list(range(len(distance))), False
        return dominant, len(clusters) == 1 or len(dominant) >= required

    @staticmethod
    def _path_tangents(reference: np.ndarray) -> np.ndarray:
        points = np.asarray(reference, dtype=np.float64)
        tangent = np.zeros_like(points)
        if len(points) == 1:
            tangent[0, 0] = 1.0
            return tangent
        tangent[0] = points[1] - points[0]
        tangent[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangent[1:-1] = points[2:] - points[:-2]
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        for index in range(len(tangent)):
            norm = float(np.linalg.norm(tangent[index]))
            if norm > 1.0e-12:
                fallback = tangent[index] / norm
            tangent[index] = fallback
        return tangent

    def _pose_support(
        self,
        positions: np.ndarray,
        orientations: np.ndarray,
        members: list[int],
        medoid: int,
        *,
        cluster_confident: bool,
        position_scale: float,
        rotation_scale: float,
        member_weights: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        selected_density: float | None = None,
        effective_sample_size: float | None = None,
        kernel_mass_fraction: float | None = None,
    ) -> PairedPoseEnsembleSupport:
        reference = positions[medoid]
        tangent = self._path_tangents(reference)
        projector = (
            np.eye(3, dtype=np.float64)[None]
            - np.einsum("ni,nj->nij", tangent, tangent)
        )
        offsets = positions[members] - reference[None]
        transverse = np.einsum("nij,mnj->mni", projector, offsets)
        position_distances = np.linalg.norm(transverse, axis=-1)
        weights = None
        if member_weights is not None:
            weights = np.asarray(member_weights, dtype=np.float64)
            if weights.shape != (len(members),):
                raise ValueError("member_weights must align with members")
            if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
                raise ValueError("member_weights must have positive mass")
            normalized_weights = weights / float(np.sum(weights))
            covariances = np.einsum(
                "m,mni,mnj->nij",
                normalized_weights,
                transverse,
                transverse,
            )
            position_radii = self._weighted_quantile_per_phase(
                position_distances,
                weights,
                self.spread_quantile,
            )
        else:
            covariances = np.einsum(
                "mni,mnj->nij",
                transverse,
                transverse,
            ) / float(max(len(members), 1))
            position_radii = np.quantile(
                position_distances,
                self.spread_quantile,
                axis=0,
            )

        dense_core_covariances = None
        dense_core_radii = None
        if self.dense_core_fraction is not None:
            core_size = max(
                1,
                int(np.ceil(self.dense_core_fraction * len(members))),
            )
            dense_core_covariances = np.zeros_like(covariances)
            dense_core_radii = np.zeros(
                (len(reference),),
                dtype=np.float64,
            )
            for phase in range(len(reference)):
                order = np.argsort(
                    position_distances[:, phase],
                    kind="stable",
                )
                selected = order[:core_size]
                core_offsets = transverse[selected, phase]
                if weights is None:
                    dense_core_covariances[phase] = (
                        core_offsets.T @ core_offsets
                    ) / float(core_size)
                    dense_core_radii[phase] = float(
                        np.quantile(
                            position_distances[selected, phase],
                            self.spread_quantile,
                        )
                    )
                else:
                    core_weights = weights[selected]
                    core_weights = core_weights / float(
                        np.sum(core_weights)
                    )
                    dense_core_covariances[phase] = np.einsum(
                        "m,mi,mj->ij",
                        core_weights,
                        core_offsets,
                        core_offsets,
                    )
                    dense_core_radii[phase] = float(
                        self._weighted_quantile(
                            position_distances[selected, phase],
                            core_weights,
                            self.spread_quantile,
                        )
                    )

        rotation_distances = np.zeros(
            (len(members), len(reference)),
            dtype=np.float64,
        )
        for row, member in enumerate(members):
            rotation_distances[row] = [
                np.linalg.norm(relative_rotation_vector(current, target))
                for current, target in zip(
                    orientations[medoid],
                    orientations[member],
                    strict=True,
                )
            ]
        if weights is None:
            rotation_radii = np.quantile(
                rotation_distances,
                self.spread_quantile,
                axis=0,
            )
        else:
            rotation_radii = self._weighted_quantile_per_phase(
                rotation_distances,
                weights,
                self.spread_quantile,
            )
        return PairedPoseEnsembleSupport(
            selected_sample_index=int(medoid),
            cluster_members=tuple(int(index) for index in members),
            cluster_confident=bool(cluster_confident),
            position_covariances=np.ascontiguousarray(covariances),
            position_spread_radii=np.asarray(
                position_radii,
                dtype=np.float64,
            ),
            rotation_spread_radii=np.asarray(
                rotation_radii,
                dtype=np.float64,
            ),
            position_distance_scale=float(position_scale),
            rotation_distance_scale=float(rotation_scale),
            position_dense_core_covariances=(
                None
                if dense_core_covariances is None
                else np.ascontiguousarray(dense_core_covariances)
            ),
            position_dense_core_spread_radii=(
                None
                if dense_core_radii is None
                else np.ascontiguousarray(dense_core_radii)
            ),
            dense_core_fraction=self.dense_core_fraction,
            sample_weights=(
                None
                if sample_weights is None
                else np.asarray(sample_weights, dtype=np.float64)
            ),
            selected_density=selected_density,
            effective_sample_size=effective_sample_size,
            kernel_mass_fraction=kernel_mass_fraction,
            spread_quantile=float(self.spread_quantile),
        )

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or weights.shape != values.shape:
            raise ValueError("weighted quantile inputs must be aligned vectors")
        order = np.argsort(values, kind="stable")
        sorted_weights = weights[order]
        threshold = float(quantile) * float(np.sum(sorted_weights))
        index = int(
            np.searchsorted(np.cumsum(sorted_weights), threshold, side="left")
        )
        return float(values[order[min(index, len(order) - 1)]])

    @classmethod
    def _weighted_quantile_per_phase(
        cls,
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.asarray(
            [
                cls._weighted_quantile(values[:, phase], weights, quantile)
                for phase in range(values.shape[1])
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quantile_metrics(
        values: list[float],
        prefix: str,
    ) -> dict[str, float]:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_mean": float(np.mean(array)),
            f"{prefix}_p90": float(np.quantile(array, 0.9)),
            f"{prefix}_max": float(np.max(array)),
        }

    def consume_pose_ensemble_support(
        self,
    ) -> list[PairedPoseEnsembleSupport]:
        supports = self._pending_supports
        self._pending_supports = []
        return supports

    def get_ensemble_metrics(self) -> dict[str, object]:
        selection_names = {
            "dominant_cluster": "pose_only_dominant_cluster_medoid",
            "all_samples": "pose_only_all_sample_medoid",
            "manifold_kde": "pose_gripper_manifold_kde_mode",
        }
        metrics: dict[str, object] = {
            "pose_ensemble_samples_per_query": self.samples,
            "pose_ensemble_query_batches": int(self.query_batches),
            "pose_ensemble_env_queries": int(self.env_queries),
            "pose_ensemble_base_samples": int(self.base_samples),
            "pose_ensemble_selection": selection_names[self.selection_mode],
            "pose_ensemble_action_aggregation": "none_selected_pair_exact",
            "pose_ensemble_selection_native_steps": self.selection_steps,
            "pose_ensemble_selection_decoded_steps": (
                self.selection_decoded_steps
            ),
            "pose_ensemble_cluster_cutoff": self.cluster_cutoff,
            "pose_ensemble_minimum_cluster_fraction": (
                self.minimum_cluster_fraction
            ),
            "pose_ensemble_selection_mode": self.selection_mode,
            "pose_ensemble_distance_scale_mode": self.distance_scale_mode,
            "pose_ensemble_support_members_mode": self.support_members_mode,
            "pose_ensemble_spread_quantile": self.spread_quantile,
            "pose_ensemble_dense_core_fraction": self.dense_core_fraction,
            "pose_ensemble_kde_position_bandwidth": (
                self.kde_position_bandwidth
            ),
            "pose_ensemble_kde_rotation_bandwidth": (
                self.kde_rotation_bandwidth
            ),
            "pose_ensemble_kde_gripper_bandwidth": (
                self.kde_gripper_bandwidth
            ),
            "pose_ensemble_kde_support_mass": self.kde_support_mass,
        }
        metrics.update(
            self._quantile_metrics(
                self._cluster_sizes,
                "pose_ensemble_cluster_size",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._cluster_fractions,
                "pose_ensemble_cluster_fraction",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._position_spreads,
                "pose_ensemble_position_spread",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._rotation_spreads,
                "pose_ensemble_rotation_spread",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._dense_core_sizes,
                "pose_ensemble_dense_core_size",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._dense_core_spreads,
                "pose_ensemble_dense_core_spread",
            )
        )
        if self._cluster_confidences:
            metrics["pose_ensemble_cluster_confident_fraction"] = float(
                np.mean(self._cluster_confidences)
            )
        metrics.update(
            self._quantile_metrics(
                self._selected_densities,
                "pose_ensemble_kde_selected_density",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._effective_sample_sizes,
                "pose_ensemble_kde_effective_sample_size",
            )
        )
        metrics.update(
            self._quantile_metrics(
                self._kernel_mass_fractions,
                "pose_ensemble_kde_kernel_mass_fraction",
            )
        )
        return metrics

    def predict_action(self, obs_dict, **kwargs):
        batch_size = self._batch_size(obs_dict)
        repeated_obs = {
            key: self._repeat_batch_value(value, batch_size)
            for key, value in obs_dict.items()
        }
        result = self.base_policy.predict_action(repeated_obs, **kwargs)
        controls = result["action"]
        expected_batch = batch_size * self.samples
        if int(controls.shape[0]) != expected_batch:
            raise RuntimeError(
                f"Policy returned batch {controls.shape[0]} for "
                f"{batch_size} envs x {self.samples} samples"
            )
        if controls.shape[-1] != PAIRED_POSE_ACTION_DIM:
            raise ValueError(
                f"Expected {PAIRED_POSE_ACTION_DIM}D paired controls, "
                f"got {controls.shape}"
            )

        grouped = controls.reshape(
            batch_size,
            self.samples,
            *controls.shape[1:],
        )
        decoded = self._decode_controls(grouped.detach().cpu().numpy())
        selected = []
        supports = []
        for batch_index in range(batch_size):
            decoded_group = decoded[
                batch_index,
                :,
                : self.selection_decoded_steps,
            ]
            positions, orientations = self._relative_pose_centers(
                decoded_group
            )
            sample_weights = None
            selected_density = None
            effective_sample_size = None
            kernel_mass_fraction = None
            if self.selection_mode == "manifold_kde":
                (
                    members,
                    medoid,
                    confident,
                    sample_weights,
                    selected_density,
                    effective_sample_size,
                    kernel_mass_fraction,
                ) = self._manifold_kde_selection(
                    positions,
                    orientations,
                    decoded_group[..., GRIPPER_INDEX],
                )
                position_scale = self.kde_position_bandwidth
                rotation_scale = self.kde_rotation_bandwidth
            else:
                distance, position_scale, rotation_scale = (
                    self._normalized_pose_distance(positions, orientations)
                )
            if self.selection_mode == "all_samples":
                members = list(range(self.samples))
                confident = True
                medoid = self._cluster_medoid(distance, members)
            elif self.selection_mode == "dominant_cluster":
                members, confident = self._dominant_cluster(distance)
                medoid = self._cluster_medoid(distance, members)
            if self.support_members_mode == "all_samples":
                support_members = list(range(self.samples))
            else:
                support_members = members
            member_weights = None
            if self.support_members_mode == "kde_responsibility":
                if sample_weights is None:
                    raise RuntimeError(
                        "KDE support requested without KDE sample weights"
                    )
                member_weights = sample_weights[support_members]
            support = self._pose_support(
                positions,
                orientations,
                support_members,
                medoid,
                cluster_confident=(
                    confident
                    or self.support_members_mode == "all_samples"
                ),
                position_scale=position_scale,
                rotation_scale=rotation_scale,
                member_weights=member_weights,
                sample_weights=sample_weights,
                selected_density=selected_density,
                effective_sample_size=effective_sample_size,
                kernel_mass_fraction=kernel_mass_fraction,
            )
            selected.append(grouped[batch_index, medoid])
            supports.append(support)
            self._cluster_sizes.append(float(len(members)))
            self._cluster_fractions.append(
                float(len(members)) / float(self.samples)
            )
            self._position_spreads.append(
                float(np.mean(support.position_spread_radii))
            )
            self._rotation_spreads.append(
                float(np.mean(support.rotation_spread_radii))
            )
            if support.dense_core_fraction is not None:
                self._dense_core_sizes.append(
                    float(
                        max(
                            1,
                            int(
                                np.ceil(
                                    support.dense_core_fraction
                                    * len(support.cluster_members)
                                )
                            ),
                        )
                    )
                )
                if support.position_dense_core_spread_radii is not None:
                    self._dense_core_spreads.append(
                        float(
                            np.mean(
                                support.position_dense_core_spread_radii
                            )
                        )
                    )
            self._cluster_confidences.append(
                float(support.cluster_confident)
            )
            if selected_density is not None:
                self._selected_densities.append(selected_density)
            if effective_sample_size is not None:
                self._effective_sample_sizes.append(effective_sample_size)
            if kernel_mass_fraction is not None:
                self._kernel_mass_fractions.append(kernel_mass_fraction)

        selected_controls = torch.stack(selected, dim=0)
        self._pending_supports = supports
        self.query_batches += 1
        self.env_queries += batch_size
        self.base_samples += expected_batch
        return {
            **result,
            "action": selected_controls,
            "pose_ensemble_support": supports,
        }


class LiberoPairedPoseBSplineRawActionMedoidPolicy(
    LiberoPairedBSplinePoseMedoidPolicy
):
    """Select pose medoids for pose-control plus raw-action predictions."""

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_pose_bspline_raw_action_medoid"
        )

    def _decode_controls(self, controls: np.ndarray) -> np.ndarray:
        return decode_libero_paired_pose_bspline_raw_action(
            controls,
            output_steps=self.source_horizon,
            source_horizon=self.source_horizon,
            degree=self.degree,
            rotation_action_scale=self.rotation_action_scale,
            support_subdivisions=self.support_subdivisions,
        )


class LiberoPairedPoseActionMedoidPolicy(
    LiberoPairedBSplinePoseMedoidPolicy
):
    """Select one exact raw pose/action sequence by its pose medoid."""

    def __init__(self, base_policy: torch.nn.Module, **kwargs) -> None:
        subdivisions = int(kwargs.pop("support_subdivisions", 1))
        if subdivisions != 1:
            raise ValueError(
                "Raw pose/action sequences require support_subdivisions=1"
            )
        super().__init__(
            base_policy,
            support_subdivisions=1,
            **kwargs,
        )

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_raw_pose_action_medoid"

    def _decode_controls(self, controls: np.ndarray) -> np.ndarray:
        return np.asarray(controls)


class LiberoPairedPoseActionDirectAdapter(torch.nn.Module):
    """Execute the native command half of a paired prediction directly."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        execution_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.representation = "paired_sampled_curve"
        self.execution_steps = (
            None if execution_steps is None else int(execution_steps)
        )
        if self.execution_steps is not None and self.execution_steps < 1:
            raise ValueError("execution_steps must be positive")
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
        return self.base_policy.get_policy_name() + "_paired_pose_action_direct"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {
            "executor": "paired_pose_action_direct",
            "representation": self.representation,
            "policy_batches": int(self.policy_batches),
            "policy_env_queries": int(self.policy_env_queries),
            "execution_steps": self.execution_steps,
        }
        get_ensemble_metrics = getattr(
            self.base_policy,
            "get_ensemble_metrics",
            None,
        )
        if callable(get_ensemble_metrics):
            metrics.update(get_ensemble_metrics())
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

    def _arm_commands(self, paired: torch.Tensor) -> torch.Tensor:
        return paired[..., GEOMETRY_DIM : GEOMETRY_DIM + COMMAND_DIM]

    def predict_action(self, obs_dict, **kwargs):
        result = self.base_policy.predict_action(obs_dict, **kwargs)
        structured = result["action"]
        paired = self._decode_prediction(structured)
        if self.execution_steps is not None:
            paired = paired[:, : self.execution_steps]
        if paired.shape[-1] != PAIRED_POSE_ACTION_DIM:
            raise ValueError(
                f"Expected {PAIRED_POSE_ACTION_DIM}D paired action, got {paired.shape}"
            )
        self.policy_batches += 1
        self.policy_env_queries += int(paired.shape[0])
        action = torch.cat(
            [self._arm_commands(paired), paired[..., -1:]],
            dim=-1,
        )
        return {**result, "structured_action": structured, "action": action}


class LiberoPairedBSplineDirectAdapter(LiberoPairedPoseActionDirectAdapter):
    """Decode shared-phase paired B-spline controls to native actions."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
        degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
        execution_steps: int = 16,
    ) -> None:
        super().__init__(base_policy)
        self.representation = "paired_bspline"
        self.source_horizon = int(source_horizon)
        self.degree = int(degree)
        self.execution_steps = int(execution_steps)
        if not 1 <= self.execution_steps <= self.source_horizon:
            raise ValueError(
                "execution_steps must be in [1, source_horizon]"
            )

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_paired_bspline_direct"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_bspline_direct",
                "source_horizon": self.source_horizon,
                "degree": self.degree,
                "execution_steps": self.execution_steps,
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        decoded = decode_libero_paired_bspline_controls(
            prediction.detach().cpu().numpy(),
            output_steps=self.execution_steps,
            source_horizon=self.source_horizon,
            degree=self.degree,
        )
        return torch.as_tensor(
            decoded,
            device=prediction.device,
            dtype=prediction.dtype,
        )


class LiberoPairedPoseBSplineRawActionDirectAdapter(
    LiberoPairedBSplineDirectAdapter
):
    """Execute exact raw actions paired with the predicted pose B-spline."""

    def __init__(self, base_policy: torch.nn.Module, **kwargs) -> None:
        super().__init__(base_policy, **kwargs)
        self.representation = "paired_pose_bspline_raw_action"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_pose_bspline_raw_action_direct"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_pose_bspline_raw_action_direct",
                "representation": self.representation,
                "action_representation": "exact_native_steps",
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        decoded = decode_libero_paired_pose_bspline_raw_action(
            prediction.detach().cpu().numpy(),
            output_steps=self.execution_steps,
            source_horizon=self.source_horizon,
            degree=self.degree,
        )
        return torch.as_tensor(
            decoded,
            device=prediction.device,
            dtype=prediction.dtype,
        )


class LiberoActionBSplineDirectAdapter(torch.nn.Module):
    """Decode action-only B-spline controls and execute native commands."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
        degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
        execution_steps: int = 16,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.representation = "action_bspline"
        self.source_horizon = int(source_horizon)
        self.degree = int(degree)
        self.execution_steps = int(execution_steps)
        if not 1 <= self.execution_steps <= self.source_horizon:
            raise ValueError(
                "execution_steps must be in [1, source_horizon]"
            )
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
        return self.base_policy.get_policy_name() + "_action_bspline_direct"

    def get_execution_metrics(self) -> dict[str, object]:
        return {
            "executor": "action_bspline_direct",
            "representation": self.representation,
            "policy_batches": int(self.policy_batches),
            "policy_env_queries": int(self.policy_env_queries),
            "source_horizon": self.source_horizon,
            "degree": self.degree,
            "execution_steps": self.execution_steps,
        }

    def predict_action(self, obs_dict, **kwargs):
        result = self.base_policy.predict_action(obs_dict, **kwargs)
        controls = result["action"]
        decoded = decode_libero_action_bspline_controls(
            controls.detach().cpu().numpy(),
            output_steps=self.execution_steps,
            source_horizon=self.source_horizon,
            degree=self.degree,
        )
        action = torch.as_tensor(
            decoded,
            device=controls.device,
            dtype=controls.dtype,
        )
        self.policy_batches += 1
        self.policy_env_queries += int(action.shape[0])
        return {
            **result,
            "structured_action": controls,
            "action": action,
        }


class LiberoPairedPoseActionFieldAdapter(torch.nn.Module):
    """Execute paired LIBERO predictions as cached pose-indexed Gaussian fields."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        plan_steps: int = 16,
        position_action_scale: float = 0.05,
        rotation_action_scale: float = 0.5,
        position_tolerance: float = 0.002,
        rotation_tolerance: float = 0.05,
        max_position_action: float = 1.0,
        max_rotation_action: float = 1.0,
        max_plan_steps: int = 200,
        gripper_only_position_threshold: float = 0.002,
        gripper_only_rotation_threshold: float = 0.05,
        gripper_close_command_threshold: float = 0.5,
        truncate_at_gripper_only_transition: bool = True,
        gating_precision_scale: float = 25.0,
        position_recovery_gain: float = 1.0,
        rotation_recovery_gain: float = 1.0,
        support_interpolation_rcond: float = 1.0e-2,
        min_sigma: float = 0.004,
        terminal_stall_replan: bool = True,
        tube_projection_enabled: bool = True,
        tube_radius: float = 0.01,
        phase_conditioned_execution: bool = True,
        phase_position_tube_radius: float = 0.075,
        phase_rotation_tube_radius: float = 0.2,
        phase_stride: int = 1,
        tube_contraction_rate: float = 0.1,
        preserve_field_direction_on_saturation: bool = True,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.plan_steps = int(plan_steps)
        if self.plan_steps < 1:
            raise ValueError("plan_steps must be positive")
        self.phase_stride = int(phase_stride)
        if self.phase_stride < 1:
            raise ValueError("phase_stride must be at least one")
        self.compiled_plan_steps = self.plan_steps * self.phase_stride
        self.gripper_only_position_threshold = float(
            gripper_only_position_threshold
        )
        self.gripper_only_rotation_threshold = float(
            gripper_only_rotation_threshold
        )
        self.gripper_close_command_threshold = float(
            gripper_close_command_threshold
        )
        self.truncate_at_gripper_only_transition = bool(
            truncate_at_gripper_only_transition
        )
        self.gating_precision_scale = float(gating_precision_scale)
        self.position_recovery_gain = float(position_recovery_gain)
        self.rotation_recovery_gain = float(rotation_recovery_gain)
        self.field_fit_config = AutonomousSE3FitConfig(
            support_interpolation_rcond=float(support_interpolation_rcond)
        )
        self.phase_conditioned_execution = bool(phase_conditioned_execution)
        self.phase_position_tube_radius = float(
            phase_position_tube_radius
        )
        self.phase_rotation_tube_radius = float(phase_rotation_tube_radius)
        if (
            self.phase_position_tube_radius <= 0.0
            or self.phase_rotation_tube_radius <= 0.0
        ):
            raise ValueError("phase tube radii must be positive")
        self.representation = "paired_sampled_curve"
        self.field_config = EndpointFieldConfig(
            position_action_scale=float(position_action_scale),
            rotation_action_scale=float(rotation_action_scale),
            field_speed_scale=1.0,
            position_tolerance=float(position_tolerance),
            rotation_tolerance=float(rotation_tolerance),
            max_position_action=float(max_position_action),
            max_rotation_action=float(max_rotation_action),
            replan_after_targets=self.compiled_plan_steps,
            max_plan_steps=int(max_plan_steps),
            gripper_contact_progress=False,
            gripper_close_transition_replan=False,
            gripper_command_mode="spatial_progress",
            gripper_close_command_threshold=float(
                gripper_close_command_threshold
            ),
            terminal_stall_replan=bool(terminal_stall_replan),
            min_sigma=float(min_sigma),
            clf_projection_enabled=bool(tube_projection_enabled),
            clf_position_tube_radius=float(tube_radius),
            clf_position_contraction_rate=float(tube_contraction_rate),
            clf_projection_mode="smooth_tube",
            preserve_field_direction_on_saturation=bool(
                preserve_field_direction_on_saturation
            ),
        )
        self._executors: list[PoseIndexedActionFieldExecutor | None] = []
        self._last_structured_action: torch.Tensor | None = None
        self.policy_batches = 0
        self.policy_env_queries = 0
        self.completed_plans = 0
        self.gripper_event_replans = 0
        self.replan_requests = 0
        self.plan_completion_reasons: Counter[str] = Counter()
        self.replan_reasons: Counter[str] = Counter()
        self.native_action_steps = 0
        self.field_recovery_steps = 0

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        self._executors = []
        self._last_structured_action = None
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_paired_pose_action_field"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = {
            "executor": "paired_pose_action_field",
            "representation": self.representation,
            "policy_batches": int(self.policy_batches),
            "policy_env_queries": int(self.policy_env_queries),
            "completed_plans": int(self.completed_plans),
            "gripper_event_replans": int(self.gripper_event_replans),
            "replan_requests": int(self.replan_requests),
            "plan_completion_reasons": dict(
                sorted(self.plan_completion_reasons.items())
            ),
            "replan_reasons": dict(sorted(self.replan_reasons.items())),
            "native_action_steps": int(self.native_action_steps),
            "field_recovery_steps": int(self.field_recovery_steps),
            "truncate_at_gripper_only_transition": (
                self.truncate_at_gripper_only_transition
            ),
            "phase_conditioned_execution": self.phase_conditioned_execution,
            "nominal_execution": (
                "exact_native_policy_command"
                if self.phase_conditioned_execution
                else "cached_gaussian_pose_action_field"
            ),
            "recovery_execution": "cached_gaussian_pose_action_field",
            "support_interpolation_rcond": float(
                self.field_fit_config.support_interpolation_rcond
            ),
            "phase_position_tube_radius": self.phase_position_tube_radius,
            "phase_rotation_tube_radius": self.phase_rotation_tube_radius,
            "native_plan_steps": self.plan_steps,
            "field_support_steps": self.compiled_plan_steps,
            "phase_stride": self.phase_stride,
            "tube_projection_enabled": bool(
                self.field_config.clf_projection_enabled
            ),
            "tube_radius": float(self.field_config.clf_position_tube_radius),
            "direction_preserving_saturation": bool(
                self.field_config.preserve_field_direction_on_saturation
            ),
        }
        get_ensemble_metrics = getattr(
            self.base_policy,
            "get_ensemble_metrics",
            None,
        )
        if callable(get_ensemble_metrics):
            metrics.update(get_ensemble_metrics())
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

    def _phase_executor_class(self):
        return PhaseConditionedPoseActionFieldExecutor

    def _phase_executor_kwargs(self) -> dict[str, object]:
        return {
            "position_tube_radius": self.phase_position_tube_radius,
            "rotation_tube_radius": self.phase_rotation_tube_radius,
            "nominal_execution": "native_action",
            "phase_stride": self.phase_stride,
        }

    @staticmethod
    def _latest_numpy(
        obs_dict: dict[str, Any],
        key: str,
        width: int,
    ) -> np.ndarray:
        if key not in obs_dict:
            raise KeyError(f"paired_pose_action_field requires {key} in obs_dict")
        value = obs_dict[key]
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        return np.asarray(array, dtype=np.float64).reshape(len(array), -1, width)[:, -1]

    def _compile_executor(
        self,
        paired_prediction: np.ndarray,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> PoseIndexedActionFieldExecutor:
        paired = np.asarray(paired_prediction, dtype=np.float32)
        if paired.ndim != 2 or paired.shape[1] != PAIRED_POSE_ACTION_DIM:
            raise ValueError(
                "paired prediction must have shape "
                f"(T,{PAIRED_POSE_ACTION_DIM}), got {paired.shape}"
            )
        paired = np.ascontiguousarray(paired[: self.compiled_plan_steps])
        if len(paired) < self.compiled_plan_steps:
            raise ValueError(
                "paired prediction has fewer field supports than requested: "
                f"{len(paired)} < {self.compiled_plan_steps}"
            )
        boundary_view = self._native_phase_boundary_view(paired)
        boundary = (
            find_gripper_only_boundary(
                boundary_view,
                position_action_scale=(
                    self.field_config.position_action_scale
                ),
                rotation_action_scale=(
                    self.field_config.rotation_action_scale
                ),
                position_threshold=self.gripper_only_position_threshold,
                rotation_threshold=self.gripper_only_rotation_threshold,
                close_threshold=self.gripper_close_command_threshold,
            )
            if self.truncate_at_gripper_only_transition
            else None
        )
        if boundary is not None and self.phase_stride > 1:
            boundary = _GripperBoundary(
                action_index=int(boundary.action_index) * self.phase_stride,
                command=float(boundary.command),
            )
        end = len(paired) if boundary is None else int(boundary.action_index)
        if end < 1:
            raise RuntimeError("A paired field needs at least one arm command")
        geometry = np.concatenate(
            [paired[:end, :GEOMETRY_DIM], paired[:end, GRIPPER_INDEX : GRIPPER_INDEX + 1]],
            axis=1,
        ).astype(np.float32)
        commands = paired[:end, GEOMETRY_DIM : GEOMETRY_DIM + COMMAND_DIM]
        plan = compile_endpoint_field_plan(
            geometry,
            start_position=np.asarray(position, dtype=np.float64),
            start_quaternion=normalize_quaternion(quaternion),
            config=self.field_config,
        )
        executor_class = (
            self._phase_executor_class()
            if self.phase_conditioned_execution
            else PoseIndexedActionFieldExecutor
        )
        executor_kwargs = {}
        if self.phase_conditioned_execution:
            executor_kwargs.update(self._phase_executor_kwargs())
        return executor_class(
            plan,
            controller_actions=np.asarray(commands, dtype=np.float64),
            config=self.field_config,
            fit_config=self.field_fit_config,
            gating_precision_scale=self.gating_precision_scale,
            position_recovery_gain=self.position_recovery_gain,
            rotation_recovery_gain=self.rotation_recovery_gain,
            terminal_gripper_command=(
                None if boundary is None else float(boundary.command)
            ),
            **executor_kwargs,
        )

    def _native_phase_boundary_view(
        self,
        paired: np.ndarray,
    ) -> np.ndarray:
        if self.phase_stride == 1:
            return paired
        native = np.ascontiguousarray(paired[:: self.phase_stride]).copy()
        rotation_scale = float(self.field_config.rotation_action_scale)
        for native_index in range(len(native)):
            start = native_index * self.phase_stride
            stop = min(start + self.phase_stride, len(paired))
            native[native_index, :3] = np.sum(
                paired[start:stop, :3],
                axis=0,
            )
            orientation = np.array(
                [0.0, 0.0, 0.0, 1.0],
                dtype=np.float64,
            )
            for delta in paired[start:stop, 3:6]:
                orientation = quaternion_multiply(
                    rotvec_to_quaternion(
                        np.asarray(delta, dtype=np.float64) * rotation_scale
                    ),
                    orientation,
                )
            native[native_index, 3:6] = (
                quaternion_to_rotvec(orientation) / rotation_scale
            )
        return native

    def _query_and_compile(
        self,
        obs_dict: dict[str, Any],
        positions: np.ndarray,
        quaternions: np.ndarray,
        env_indices: list[int],
        **kwargs,
    ) -> None:
        query_obs = {}
        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                query_obs[key] = value[env_indices]
            elif isinstance(value, np.ndarray) and len(value) == len(positions):
                query_obs[key] = value[env_indices]
            else:
                query_obs[key] = value
        result = self.base_policy.predict_action(query_obs, **kwargs)
        structured_tensor = result["action"]
        paired_tensor = self._decode_prediction(structured_tensor)
        if paired_tensor.shape[-1] != PAIRED_POSE_ACTION_DIM:
            raise ValueError(
                f"Expected {PAIRED_POSE_ACTION_DIM}D paired action, "
                f"got {paired_tensor.shape}"
            )
        paired = paired_tensor.detach().cpu().numpy().astype(np.float32)
        self._last_structured_action = structured_tensor
        self.policy_batches += 1
        if len(paired) != len(env_indices):
            raise RuntimeError(
                f"Policy returned batch {len(paired)} for {len(env_indices)} envs"
            )
        self.policy_env_queries += len(paired)
        for row, env_index in enumerate(env_indices):
            self._executors[env_index] = self._compile_executor(
                paired[row],
                positions[env_index],
                quaternions[env_index],
            )

    def predict_action(self, obs_dict, **kwargs):
        positions = self._latest_numpy(obs_dict, "robot0_eef_pos", 3)
        quaternions = self._latest_numpy(obs_dict, "robot0_eef_quat", 4)
        gripper_qpos = self._latest_numpy(obs_dict, "robot0_gripper_qpos", 2)
        batch_size = int(len(positions))
        if len(self._executors) != batch_size:
            self._executors = [None] * batch_size

        missing = [
            index
            for index, executor in enumerate(self._executors)
            if executor is None
        ]
        if missing:
            self._query_and_compile(
                obs_dict,
                positions,
                quaternions,
                missing,
                **kwargs,
            )

        steps = []
        replan = []
        for index, executor in enumerate(self._executors):
            assert executor is not None
            phase_action_active = bool(
                isinstance(executor, PhaseConditionedPoseActionFieldExecutor)
                and executor.action_index < executor.target_count
            )
            step = executor.step(
                positions[index],
                quaternions[index],
                gripper_qpos[index],
            )
            action_will_be_replaced = bool(
                step.replan_required
                and step.replan_reason == "recovery_reentry"
            )
            if phase_action_active and not action_will_be_replaced:
                if executor.last_phase_frozen:
                    self.field_recovery_steps += 1
                else:
                    self.native_action_steps += 1
            if step.plan_complete or step.replan_required:
                self.completed_plans += int(step.plan_complete)
                self.replan_requests += int(step.replan_required)
                if step.plan_complete:
                    completion_reason = (
                        step.replan_reason or "target_reached"
                    )
                    self.plan_completion_reasons[completion_reason] += 1
                if step.replan_required:
                    replan_reason = step.replan_reason or "unspecified"
                    self.replan_reasons[replan_reason] += 1
                self.gripper_event_replans += int(
                    step.replan_reason == "gripper_only_transition"
                )
                replan.append(index)
            steps.append(step)

        if replan:
            self._query_and_compile(
                obs_dict,
                positions,
                quaternions,
                replan,
                **kwargs,
            )
            for index in replan:
                executor = self._executors[index]
                assert executor is not None
                phase_action_active = bool(
                    isinstance(
                        executor,
                        PhaseConditionedPoseActionFieldExecutor,
                    )
                    and executor.action_index < executor.target_count
                )
                steps[index] = executor.step(
                    positions[index],
                    quaternions[index],
                    gripper_qpos[index],
                )
                if phase_action_active:
                    if executor.last_phase_frozen:
                        self.field_recovery_steps += 1
                    else:
                        self.native_action_steps += 1

        actions = np.stack([step.action for step in steps], axis=0)[:, None]
        action_tensor = torch.as_tensor(
            actions,
            device=self.device,
            dtype=self.dtype,
        )
        result = {"action": action_tensor}
        if self._last_structured_action is not None:
            result["structured_action"] = self._last_structured_action
        return result


class LiberoPairedBSplineFieldAdapter(LiberoPairedPoseActionFieldAdapter):
    """Compile decoded paired B-spline curves into the shared Gaussian field."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        source_horizon: int = PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
        degree: int = PAIRED_BSPLINE_DEFAULT_DEGREE,
        support_subdivisions: int = 1,
        **kwargs,
    ) -> None:
        self.support_subdivisions = int(support_subdivisions)
        if self.support_subdivisions < 1:
            raise ValueError("support_subdivisions must be at least one")
        super().__init__(
            base_policy,
            phase_stride=self.support_subdivisions,
            **kwargs,
        )
        self.representation = "paired_bspline"
        self.source_horizon = int(source_horizon)
        self.degree = int(degree)
        if self.plan_steps > self.source_horizon:
            raise ValueError("plan_steps cannot exceed source_horizon")
        decoded_support_steps = (
            (self.source_horizon - 1) * self.support_subdivisions + 1
        )
        if self.compiled_plan_steps > decoded_support_steps:
            raise ValueError(
                "Densified plan needs support beyond the B-spline horizon"
            )

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_paired_bspline_field"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_bspline_field",
                "source_horizon": self.source_horizon,
                "degree": self.degree,
                "support_subdivisions": self.support_subdivisions,
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        decoded = decode_libero_paired_bspline_controls(
            prediction.detach().cpu().numpy(),
            output_steps=self.source_horizon,
            source_horizon=self.source_horizon,
            degree=self.degree,
            rotation_action_scale=self.field_config.rotation_action_scale,
            support_subdivisions=self.support_subdivisions,
        )
        return torch.as_tensor(
            decoded,
            device=prediction.device,
            dtype=prediction.dtype,
        )


class LiberoPairedPoseBSplineRawActionFieldAdapter(
    LiberoPairedBSplineFieldAdapter
):
    """Compile pose controls while retaining native action-step commands."""

    def __init__(self, base_policy: torch.nn.Module, **kwargs) -> None:
        super().__init__(base_policy, **kwargs)
        self.representation = "paired_pose_bspline_raw_action"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_pose_bspline_raw_action_field"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_pose_bspline_raw_action_field",
                "representation": self.representation,
                "action_representation": "exact_native_steps",
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        decoded = decode_libero_paired_pose_bspline_raw_action(
            prediction.detach().cpu().numpy(),
            output_steps=self.source_horizon,
            source_horizon=self.source_horizon,
            degree=self.degree,
            rotation_action_scale=self.field_config.rotation_action_scale,
            support_subdivisions=self.support_subdivisions,
        )
        return torch.as_tensor(
            decoded,
            device=prediction.device,
            dtype=prediction.dtype,
        )


class LiberoPairedBSplineGeometryDirectAdapter(
    LiberoPairedBSplineDirectAdapter
):
    """Execute decoded pose-curve tangents as native delta-pose commands."""

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        geometry_command_gain: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(base_policy, **kwargs)
        if float(geometry_command_gain) <= 0.0:
            raise ValueError("geometry_command_gain must be positive")
        self.geometry_command_gain = float(geometry_command_gain)
        self.representation = "paired_bspline_geometry_direct"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_paired_bspline_geometry_direct"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_bspline_geometry_direct",
                "representation": self.representation,
                "arm_source": "decoded_pose_center_difference",
                "action_bspline_arm_channels_used": False,
                "geometry_command_gain": self.geometry_command_gain,
            }
        )
        return metrics

    def _arm_commands(self, paired: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.geometry_command_gain * paired[..., :GEOMETRY_DIM],
            min=-1.0,
            max=1.0,
        )


class LiberoPairedBSplinePoseFieldAdapter(
    LiberoPairedBSplineFieldAdapter
):
    """Execute only the paired checkpoint's pose spline as a cached field.

    The pose curve supplies both field centers and center-to-center nominal
    velocity. The paired arm-action curve is discarded; only its shared
    gripper schedule remains.
    """

    def __init__(self, base_policy: torch.nn.Module, *args, **kwargs) -> None:
        kwargs["phase_conditioned_execution"] = False
        super().__init__(base_policy, *args, **kwargs)
        self.representation = "paired_bspline_pose_only_field"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_paired_bspline_pose_only_field"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_bspline_pose_only_field",
                "representation": self.representation,
                "geometry_source": "paired_pose_bspline",
                "nominal_velocity_source": "pose_center_difference",
                "arm_action_bspline_used": False,
                "shared_gripper_schedule_used": True,
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        paired = super()._decode_prediction(prediction)
        pose_only = paired.clone()
        pose_only[..., GEOMETRY_DIM : GEOMETRY_DIM + COMMAND_DIM] = (
            paired[..., :GEOMETRY_DIM] * float(self.support_subdivisions)
        )
        return pose_only


class LiberoPairedBSplineIntegratedFieldAdapter(
    LiberoPairedBSplineFieldAdapter
):
    """Use the paired checkpoint's native actions for both motion and geometry.

    The action half is unchanged. The compiler integrates those same native
    arm commands into its center chain, which isolates center construction from
    every other paired-field execution choice.
    """

    def __init__(self, base_policy: torch.nn.Module, *args, **kwargs) -> None:
        super().__init__(base_policy, *args, **kwargs)
        self.representation = "paired_bspline_action_integrated"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_paired_bspline_integrated_field"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_bspline_integrated_field",
                "representation": self.representation,
                "geometry_source": "integrated_native_action",
            }
        )
        return metrics

    def _decode_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        paired = super()._decode_prediction(prediction)
        integrated = paired.clone()
        integrated[..., :GEOMETRY_DIM] = paired[
            ..., GEOMETRY_DIM : GEOMETRY_DIM + COMMAND_DIM
        ] / float(self.support_subdivisions)
        return integrated
