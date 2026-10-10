from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ns_gaussian_ds_policy.endpoint_field import (
    FIELD_ACTION_DIM,
    EndpointFieldConfig,
    EndpointFieldPlan,
    EndpointFieldStep,
    normalize_quaternion,
    quaternion_multiply,
    relative_rotation_vector,
    rotvec_to_quaternion,
)
from ns_gaussian_ds_policy.state_chain_partition import partition_state_chain


@dataclass(frozen=True)
class AutonomousSE3FitConfig:
    phase_sample_step: float = 0.25
    position_tube_radius: float = 0.004
    rotation_tube_radius: float = 0.05
    position_cross_track_gain: float = 1.0
    rotation_cross_track_gain: float = 1.0
    contraction_floor: float = 0.1
    ridge: float = 1.0e-2
    support_interpolation_rcond: float = 1.0e-3
    projection_iterations: int = 120
    rotation_support_scale: float = 1.5
    intersection_position_scale: float = 0.01
    intersection_rotation_scale: float = 0.2
    intersection_margin: float = 0.25
    tangent_alignment_cosine: float = 0.95

    def __post_init__(self) -> None:
        positive = {
            "phase_sample_step": self.phase_sample_step,
            "position_tube_radius": self.position_tube_radius,
            "rotation_tube_radius": self.rotation_tube_radius,
            "position_cross_track_gain": self.position_cross_track_gain,
            "rotation_cross_track_gain": self.rotation_cross_track_gain,
            "contraction_floor": self.contraction_floor,
            "rotation_support_scale": self.rotation_support_scale,
            "intersection_position_scale": self.intersection_position_scale,
            "intersection_rotation_scale": self.intersection_rotation_scale,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.ridge) < 0.0:
            raise ValueError("ridge must be non-negative")
        if not 0.0 <= float(self.support_interpolation_rcond) < 1.0:
            raise ValueError("support_interpolation_rcond must be in [0,1)")
        if int(self.projection_iterations) < 1:
            raise ValueError("projection_iterations must be positive")
        if float(self.intersection_margin) < 0.0:
            raise ValueError("intersection_margin must be non-negative")
        if not -1.0 <= float(self.tangent_alignment_cosine) <= 1.0:
            raise ValueError("tangent_alignment_cosine must be in [-1,1]")


@dataclass(frozen=True)
class _AutonomousSE3Chunk:
    center_start: int
    center_end: int
    support_positions: np.ndarray
    support_orientations: np.ndarray
    position_precisions: np.ndarray
    rotation_precisions: np.ndarray
    position_velocities: np.ndarray
    position_matrices: np.ndarray
    rotation_velocities: np.ndarray
    rotation_matrices: np.ndarray
    metadata: dict[str, float | str]

    @property
    def num_supports(self) -> int:
        return int(len(self.support_positions))


def _pose_at_phase(
    plan: EndpointFieldPlan,
    phase: float,
    *,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    phase = float(np.clip(phase, 0.0, float(target_count)))
    if phase >= float(target_count) - 1.0e-12:
        return (
            plan.position_centers[target_count].copy(),
            plan.orientation_centers[target_count].copy(),
        )
    segment_index = min(int(np.floor(phase)), target_count - 1)
    alpha = float(phase - segment_index)
    positions = plan.position_centers
    position = (
        (1.0 - alpha) * positions[segment_index]
        + alpha * positions[segment_index + 1]
    )
    orientation = quaternion_multiply(
        rotvec_to_quaternion(
            alpha * plan.segment_rotation_vectors[segment_index]
        ),
        plan.orientation_centers[segment_index],
    )
    return position, orientation


def _lookahead_velocity(
    plan: EndpointFieldPlan,
    phase: float,
    *,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    position, orientation = _pose_at_phase(
        plan,
        phase,
        target_count=target_count,
    )
    next_position, next_orientation = _pose_at_phase(
        plan,
        min(float(phase) + 1.0, float(target_count)),
        target_count=target_count,
    )
    return (
        next_position - position,
        relative_rotation_vector(orientation, next_orientation),
    )


def _command_at_phase(
    nominal_twists: np.ndarray,
    phase: float,
    *,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if float(phase) >= float(target_count) - 1.0e-12:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    index = min(int(np.floor(float(phase))), target_count - 1)
    alpha = float(np.clip(float(phase) - index, 0.0, 1.0))
    next_index = min(index + 1, target_count - 1)
    command = (
        (1.0 - alpha) * nominal_twists[index]
        + alpha * nominal_twists[next_index]
    )
    return command[:3].copy(), command[3:6].copy()


def _project_command_onto_geometry(
    geometry_velocity: np.ndarray,
    command_velocity: np.ndarray,
) -> np.ndarray:
    """Use command speed along the center-chain tangent without changing path."""

    geometry_velocity = np.asarray(geometry_velocity, dtype=np.float64).reshape(3)
    command_velocity = np.asarray(command_velocity, dtype=np.float64).reshape(3)
    geometry_speed = float(np.linalg.norm(geometry_velocity))
    if geometry_speed <= 1.0e-12:
        return command_velocity.copy()
    tangent = geometry_velocity / geometry_speed
    commanded_progress = float(np.dot(command_velocity, tangent))
    if commanded_progress <= 1.0e-12:
        return geometry_velocity.copy()
    return commanded_progress * tangent


def _nominal_velocity(
    plan: EndpointFieldPlan,
    phase: float,
    *,
    target_count: int,
    nominal_twists: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    geometry_position, geometry_rotation = _lookahead_velocity(
        plan,
        phase,
        target_count=target_count,
    )
    if nominal_twists is None:
        return geometry_position, geometry_rotation
    command_position, command_rotation = _command_at_phase(
        nominal_twists,
        phase,
        target_count=target_count,
    )
    return (
        _project_command_onto_geometry(geometry_position, command_position),
        _project_command_onto_geometry(geometry_rotation, command_rotation),
    )


def _orthogonal_basis(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return (
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
    direction = vector / norm
    axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, axis)
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second = np.cross(direction, first)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    return first, second


def _state_embedding(
    plan: EndpointFieldPlan,
    *,
    target_count: int,
    fit_config: AutonomousSE3FitConfig,
) -> np.ndarray:
    positions = (
        plan.position_centers[: target_count + 1]
        / float(fit_config.intersection_position_scale)
    )
    start = plan.orientation_centers[0]
    rotations = np.stack(
        [
            relative_rotation_vector(start, quaternion)
            for quaternion in plan.orientation_centers[: target_count + 1]
        ],
        axis=0,
    ) / float(fit_config.intersection_rotation_scale)
    return np.concatenate([positions, rotations], axis=1)


def partition_se3_center_chain(
    plan: EndpointFieldPlan,
    *,
    target_count: int,
    fit_config: AutonomousSE3FitConfig | None = None,
) -> tuple[tuple[int, int], ...]:
    fit_config = fit_config or AutonomousSE3FitConfig()
    embedding = _state_embedding(
        plan,
        target_count=int(target_count),
        fit_config=fit_config,
    )
    return partition_state_chain(
        embedding,
        intersection_margin=float(fit_config.intersection_margin),
        tangent_alignment_cosine=float(
            fit_config.tangent_alignment_cosine
        ),
    )


def _chunk_support_state(
    plan: EndpointFieldPlan,
    center_start: int,
    center_end: int,
    fit_config: AutonomousSE3FitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_positions = plan.position_centers[center_start : center_end + 1]
    support_orientations = plan.orientation_centers[
        center_start : center_end + 1
    ]
    precision_indices = np.minimum(
        np.arange(center_start, center_end + 1),
        plan.num_targets - 1,
    )
    position_precisions = plan.segment_precisions[precision_indices]
    rotation_angles = np.linalg.norm(
        plan.segment_rotation_vectors[precision_indices],
        axis=1,
    )
    rotation_sigmas = np.maximum(
        float(fit_config.rotation_tube_radius),
        float(fit_config.rotation_support_scale) * rotation_angles,
    )
    rotation_precisions = 1.0 / np.square(rotation_sigmas)
    return (
        support_positions,
        support_orientations,
        position_precisions,
        rotation_precisions,
    )


def _normalize_quaternion_rows(
    quaternions: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"{name} must contain nonzero quaternions")
    return values / norms


def _pairwise_relative_rotation_vectors(
    support_orientations: np.ndarray,
    target_orientations: np.ndarray,
) -> np.ndarray:
    """Vectorized world-frame rotations from each support to each target."""

    supports = _normalize_quaternion_rows(
        support_orientations,
        name="support_orientations",
    )
    targets = _normalize_quaternion_rows(
        target_orientations,
        name="target_orientations",
    )
    target_xyz = targets[:, None, :3]
    target_w = targets[:, None, 3:4]
    inverse_support_xyz = -supports[None, :, :3]
    inverse_support_w = supports[None, :, 3:4]
    relative_xyz = (
        target_w * inverse_support_xyz
        + inverse_support_w * target_xyz
        + np.cross(target_xyz, inverse_support_xyz, axis=-1)
    )
    relative_w = (
        target_w * inverse_support_w
        - np.sum(target_xyz * inverse_support_xyz, axis=-1, keepdims=True)
    )
    relative = np.concatenate([relative_xyz, relative_w], axis=-1)
    relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
    relative = np.where(relative[..., 3:4] < 0.0, -relative, relative)

    xyz = relative[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1)
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
    return xyz * scale[..., None]


def _gating_and_offsets(
    positions: np.ndarray,
    orientations: np.ndarray,
    support_positions: np.ndarray,
    support_orientations: np.ndarray,
    position_precisions: np.ndarray,
    rotation_precisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_offsets = positions[:, None, :] - support_positions[None, :, :]
    position_mahalanobis = np.einsum(
        "nki,kij,nkj->nk",
        position_offsets,
        position_precisions,
        position_offsets,
    )
    rotation_offsets = _pairwise_relative_rotation_vectors(
        support_orientations,
        orientations,
    )
    rotation_mahalanobis = (
        np.sum(np.square(rotation_offsets), axis=2)
        * rotation_precisions[None, :]
    )
    log_weights = -0.5 * (position_mahalanobis + rotation_mahalanobis)
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(np.maximum(log_weights, -80.0))
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights, position_offsets, rotation_offsets


def _chunk_training_samples(
    plan: EndpointFieldPlan,
    *,
    center_start: int,
    center_end: int,
    target_count: int,
    fit_config: AutonomousSE3FitConfig,
    nominal_twists: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phases = np.arange(
        float(center_start),
        float(center_end),
        float(fit_config.phase_sample_step),
        dtype=np.float64,
    )
    phases = np.unique(
        np.concatenate(
            [
                phases,
                np.arange(center_start, center_end + 1, dtype=np.float64),
                np.array([float(center_end)], dtype=np.float64),
            ]
        )
    )
    positions: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    target_rotations: list[np.ndarray] = []
    sample_weights: list[float] = []

    for phase in phases:
        position, orientation = _pose_at_phase(
            plan,
            float(phase),
            target_count=target_count,
        )
        nominal_position, nominal_rotation = _nominal_velocity(
            plan,
            float(phase),
            target_count=target_count,
            nominal_twists=nominal_twists,
        )
        positions.append(position)
        orientations.append(orientation)
        target_positions.append(nominal_position)
        target_rotations.append(nominal_rotation)
        sample_weights.append(4.0)

        position_first, position_second = _orthogonal_basis(
            nominal_position
        )
        for normal in (position_first, position_second):
            for sign in (-1.0, 1.0):
                offset = (
                    sign
                    * float(fit_config.position_tube_radius)
                    * normal
                )
                positions.append(position + offset)
                orientations.append(orientation)
                target_positions.append(
                    nominal_position
                    - float(fit_config.position_cross_track_gain) * offset
                )
                target_rotations.append(nominal_rotation)
                sample_weights.append(1.0)

        rotation_first, rotation_second = _orthogonal_basis(
            nominal_rotation
        )
        for normal in (rotation_first, rotation_second):
            for sign in (-1.0, 1.0):
                offset = (
                    sign
                    * float(fit_config.rotation_tube_radius)
                    * normal
                )
                positions.append(position)
                orientations.append(
                    quaternion_multiply(
                        rotvec_to_quaternion(offset),
                        orientation,
                    )
                )
                target_positions.append(nominal_position)
                target_rotations.append(
                    nominal_rotation
                    - float(fit_config.rotation_cross_track_gain) * offset
                )
                sample_weights.append(1.0)

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(orientations, dtype=np.float64),
        np.asarray(target_positions, dtype=np.float64),
        np.asarray(target_rotations, dtype=np.float64),
        np.asarray(sample_weights, dtype=np.float64),
    )


def _contractive_prior_matrix(
    velocity: np.ndarray,
    *,
    contraction_floor: float,
    cross_track_gain: float,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1.0e-12:
        return -max(contraction_floor, cross_track_gain) * np.eye(3)
    tangent = velocity / speed
    tangent_projection = np.outer(tangent, tangent)
    return (
        -float(contraction_floor) * tangent_projection
        -max(float(contraction_floor), float(cross_track_gain))
        * (np.eye(3) - tangent_projection)
    )


def _fit_contracting_affine_mixture(
    gating: np.ndarray,
    offsets: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    prior_velocities: np.ndarray,
    prior_matrices: np.ndarray,
    *,
    contraction_floor: float,
    ridge: float,
    projection_iterations: int,
    speed_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count, support_count, dimension = offsets.shape
    if dimension != 3:
        raise ValueError("The SE(3) affine fitter expects 3-D tangent vectors")
    feature_block_size = 1 + dimension
    feature_count = support_count * feature_block_size
    feature_tensor = np.concatenate(
        [
            gating[:, :, None],
            gating[:, :, None] * offsets,
        ],
        axis=2,
    )
    sqrt_weights = np.sqrt(sample_weights)
    design = feature_tensor.reshape(sample_count, feature_count)
    weighted_design = design * sqrt_weights[:, None]
    weighted_targets = targets * sqrt_weights[:, None]

    def pack(
        velocities: np.ndarray,
        matrices: np.ndarray,
    ) -> np.ndarray:
        blocks = np.empty(
            (support_count, feature_block_size, dimension),
            dtype=np.float64,
        )
        blocks[:, 0] = velocities
        blocks[:, 1:] = np.swapaxes(matrices, 1, 2)
        return blocks.reshape(feature_count, dimension)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blocks = parameters.reshape(
            support_count,
            feature_block_size,
            dimension,
        )
        return blocks[:, 0], np.swapaxes(blocks[:, 1:], 1, 2)

    prior = pack(
        prior_velocities,
        prior_matrices,
    )
    gram = weighted_design.T @ weighted_design
    linear_term = weighted_design.T @ weighted_targets
    if ridge > 0.0:
        gram += float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
        linear_term += float(ridge) * prior

    def project(parameters: np.ndarray) -> np.ndarray:
        velocities, matrices = unpack(parameters)
        velocities = velocities.copy()
        speeds = np.linalg.norm(velocities, axis=1)
        over_limit = speeds > float(speed_limit)
        if np.any(over_limit):
            velocities[over_limit] *= (
                float(speed_limit) / speeds[over_limit]
            )[:, None]
        symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
        skew = 0.5 * (matrices - np.swapaxes(matrices, 1, 2))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.minimum(eigenvalues, -float(contraction_floor))
        matrices = (
            (eigenvectors * eigenvalues[:, None, :])
            @ np.swapaxes(eigenvectors, 1, 2)
            + skew
        )
        return pack(velocities, matrices)

    try:
        parameters = np.linalg.solve(gram, linear_term)
    except np.linalg.LinAlgError:
        parameters = np.linalg.lstsq(gram, linear_term, rcond=None)[0]
    parameters = project(parameters)
    accelerated = parameters.copy()
    acceleration = 1.0
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1.0e-12)
    for _ in range(int(projection_iterations)):
        previous = parameters
        parameters = project(
            accelerated
            - (gram @ accelerated - linear_term) / lipschitz
        )
        next_acceleration = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * acceleration * acceleration)
        )
        accelerated = parameters + (
            (acceleration - 1.0) / next_acceleration
        ) * (parameters - previous)
        acceleration = next_acceleration

    velocities, matrices = unpack(parameters)
    return velocities.copy(), matrices.copy()


def _mixture_output(
    gating: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
    matrices: np.ndarray,
) -> np.ndarray:
    local = velocities[None] + np.einsum(
        "kij,nkj->nki",
        matrices,
        offsets,
    )
    return np.sum(gating[:, :, None] * local, axis=1)


def _enforce_terminal_output(
    gating: np.ndarray,
    offsets: np.ndarray,
    velocities: np.ndarray,
    matrices: np.ndarray,
    desired: np.ndarray,
) -> np.ndarray:
    current = _mixture_output(
        gating,
        offsets,
        velocities,
        matrices,
    )[0]
    residual = current - np.asarray(desired, dtype=np.float64).reshape(3)
    weights = gating[0]
    denominator = float(np.dot(weights, weights))
    if denominator <= 1.0e-12:
        return velocities
    return velocities - (weights / denominator)[:, None] * residual[None]


def _calibrate_affine_intercepts(
    gating: np.ndarray,
    offsets: np.ndarray,
    matrices: np.ndarray,
    targets: np.ndarray,
    *,
    rcond: float,
) -> tuple[np.ndarray, float, float, int]:
    """Fit bounded local intercepts to the blended support labels."""

    gating = np.asarray(gating, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    matrices = np.asarray(matrices, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    support_count = int(gating.shape[0])
    if gating.shape != (support_count, support_count):
        raise ValueError(
            "Support interpolation requires a square gating matrix, got "
            f"{gating.shape}"
        )
    local_linear = np.einsum("kij,nkj->nki", matrices, offsets)
    linear_mixture = np.sum(gating[:, :, None] * local_linear, axis=1)
    right_hand_side = targets - linear_mixture
    condition = float(np.linalg.cond(gating))
    intercepts, _, rank, _ = np.linalg.lstsq(
        gating,
        right_hand_side,
        rcond=float(rcond),
    )
    reconstructed = _mixture_output(
        gating,
        offsets,
        intercepts,
        matrices,
    )
    max_error = float(np.max(np.linalg.norm(reconstructed - targets, axis=1)))
    return intercepts, max_error, condition, int(rank)


def _compile_pose_indexed_action_chunk(
    plan: EndpointFieldPlan,
    *,
    center_start: int,
    center_end: int,
    controller_twists: np.ndarray,
    fit_config: AutonomousSE3FitConfig,
    gating_precision_scale: float,
    position_recovery_gain: float,
    rotation_recovery_gain: float,
) -> _AutonomousSE3Chunk:
    """Bind each pre-action pose to its native controller command."""

    if center_end <= center_start:
        raise ValueError("A pose-indexed action chunk needs at least one action")
    support_positions = plan.position_centers[center_start:center_end]
    support_orientations = plan.orientation_centers[center_start:center_end]
    position_precisions = (
        plan.segment_precisions[center_start:center_end]
        * float(gating_precision_scale)
    )
    rotation_angles = np.linalg.norm(
        plan.segment_rotation_vectors[center_start:center_end],
        axis=1,
    )
    rotation_sigmas = np.maximum(
        float(fit_config.rotation_tube_radius),
        float(fit_config.rotation_support_scale) * rotation_angles,
    ) / np.sqrt(float(gating_precision_scale))
    rotation_precisions = 1.0 / np.square(rotation_sigmas)
    gating, position_offsets, rotation_offsets = _gating_and_offsets(
        support_positions,
        support_orientations,
        support_positions,
        support_orientations,
        position_precisions,
        rotation_precisions,
    )
    support_count = int(center_end - center_start)
    position_matrices = np.repeat(
        (-float(position_recovery_gain) * np.eye(3, dtype=np.float64))[None],
        support_count,
        axis=0,
    )
    rotation_matrices = np.repeat(
        (-float(rotation_recovery_gain) * np.eye(3, dtype=np.float64))[None],
        support_count,
        axis=0,
    )
    targets = np.asarray(
        controller_twists[center_start:center_end],
        dtype=np.float64,
    )
    position_velocities, position_error, position_condition, position_rank = (
        _calibrate_affine_intercepts(
            gating,
            position_offsets,
            position_matrices,
            targets[:, :3],
            rcond=fit_config.support_interpolation_rcond,
        )
    )
    rotation_velocities, rotation_error, rotation_condition, rotation_rank = (
        _calibrate_affine_intercepts(
            gating,
            rotation_offsets,
            rotation_matrices,
            targets[:, 3:6],
            rcond=fit_config.support_interpolation_rcond,
        )
    )
    return _AutonomousSE3Chunk(
        center_start=int(center_start),
        center_end=int(center_end),
        support_positions=support_positions.copy(),
        support_orientations=support_orientations.copy(),
        position_precisions=position_precisions.copy(),
        rotation_precisions=rotation_precisions.copy(),
        position_velocities=position_velocities,
        position_matrices=position_matrices,
        rotation_velocities=rotation_velocities,
        rotation_matrices=rotation_matrices,
        metadata={
            "progression_source": "autonomous_pose_gating",
            "action_binding": "pre_action_pose",
            "gating_precision_scale": float(gating_precision_scale),
            "support_interpolation_rcond": float(
                fit_config.support_interpolation_rcond
            ),
            "position_support_reconstruction_max_error": position_error,
            "rotation_support_reconstruction_max_error": rotation_error,
            "position_gating_condition": position_condition,
            "rotation_gating_condition": rotation_condition,
            "position_gating_rank": float(position_rank),
            "rotation_gating_rank": float(rotation_rank),
        },
    )


def _compile_chunk(
    plan: EndpointFieldPlan,
    *,
    center_start: int,
    center_end: int,
    target_count: int,
    field_config: EndpointFieldConfig,
    fit_config: AutonomousSE3FitConfig,
    nominal_twists: np.ndarray | None = None,
) -> _AutonomousSE3Chunk:
    (
        support_positions,
        support_orientations,
        position_precisions,
        rotation_precisions,
    ) = _chunk_support_state(
        plan,
        center_start,
        center_end,
        fit_config,
    )
    (
        sample_positions,
        sample_orientations,
        target_positions,
        target_rotations,
        sample_weights,
    ) = _chunk_training_samples(
        plan,
        center_start=center_start,
        center_end=center_end,
        target_count=target_count,
        fit_config=fit_config,
        nominal_twists=nominal_twists,
    )
    gating, position_offsets, rotation_offsets = _gating_and_offsets(
        sample_positions,
        sample_orientations,
        support_positions,
        support_orientations,
        position_precisions,
        rotation_precisions,
    )
    support_phases = np.arange(
        center_start,
        center_end + 1,
        dtype=np.float64,
    )
    prior_position_velocities = []
    prior_rotation_velocities = []
    prior_position_matrices = []
    prior_rotation_matrices = []
    for phase in support_phases:
        position_velocity, rotation_velocity = _nominal_velocity(
            plan,
            float(phase),
            target_count=target_count,
            nominal_twists=nominal_twists,
        )
        prior_position_velocities.append(position_velocity)
        prior_rotation_velocities.append(rotation_velocity)
        prior_position_matrices.append(
            _contractive_prior_matrix(
                position_velocity,
                contraction_floor=float(fit_config.contraction_floor),
                cross_track_gain=float(
                    fit_config.position_cross_track_gain
                ),
            )
        )
        prior_rotation_matrices.append(
            _contractive_prior_matrix(
                rotation_velocity,
                contraction_floor=float(fit_config.contraction_floor),
                cross_track_gain=float(
                    fit_config.rotation_cross_track_gain
                ),
            )
        )
    prior_position_velocities = np.asarray(
        prior_position_velocities,
        dtype=np.float64,
    )
    prior_rotation_velocities = np.asarray(
        prior_rotation_velocities,
        dtype=np.float64,
    )
    prior_position_matrices = np.asarray(
        prior_position_matrices,
        dtype=np.float64,
    )
    prior_rotation_matrices = np.asarray(
        prior_rotation_matrices,
        dtype=np.float64,
    )
    position_speed_limit = max(
        4.0 * float(np.max(np.linalg.norm(target_positions, axis=1))),
        float(field_config.position_action_scale),
    )
    rotation_speed_limit = max(
        4.0 * float(np.max(np.linalg.norm(target_rotations, axis=1))),
        float(field_config.rotation_action_scale),
    )
    position_velocities, position_matrices = (
        _fit_contracting_affine_mixture(
            gating,
            position_offsets,
            target_positions,
            sample_weights,
            prior_position_velocities,
            prior_position_matrices,
            contraction_floor=float(fit_config.contraction_floor),
            ridge=float(fit_config.ridge),
            projection_iterations=int(fit_config.projection_iterations),
            speed_limit=position_speed_limit,
        )
    )
    rotation_velocities, rotation_matrices = (
        _fit_contracting_affine_mixture(
            gating,
            rotation_offsets,
            target_rotations,
            sample_weights,
            prior_rotation_velocities,
            prior_rotation_matrices,
            contraction_floor=float(fit_config.contraction_floor),
            ridge=float(fit_config.ridge),
            projection_iterations=int(fit_config.projection_iterations),
            speed_limit=rotation_speed_limit,
        )
    )

    terminal_position, terminal_orientation = _pose_at_phase(
        plan,
        float(center_end),
        target_count=target_count,
    )
    terminal_position_velocity, terminal_rotation_velocity = (
        _nominal_velocity(
            plan,
            float(center_end),
            target_count=target_count,
            nominal_twists=nominal_twists,
        )
    )
    terminal_gating, terminal_position_offsets, terminal_rotation_offsets = (
        _gating_and_offsets(
            terminal_position[None],
            terminal_orientation[None],
            support_positions,
            support_orientations,
            position_precisions,
            rotation_precisions,
        )
    )
    position_velocities = _enforce_terminal_output(
        terminal_gating,
        terminal_position_offsets,
        position_velocities,
        position_matrices,
        terminal_position_velocity,
    )
    rotation_velocities = _enforce_terminal_output(
        terminal_gating,
        terminal_rotation_offsets,
        rotation_velocities,
        rotation_matrices,
        terminal_rotation_velocity,
    )

    fitted_positions = _mixture_output(
        gating,
        position_offsets,
        position_velocities,
        position_matrices,
    )
    fitted_rotations = _mixture_output(
        gating,
        rotation_offsets,
        rotation_velocities,
        rotation_matrices,
    )
    position_eigenvalues = np.linalg.eigvalsh(
        0.5
        * (
            position_matrices
            + np.swapaxes(position_matrices, 1, 2)
        )
    )
    rotation_eigenvalues = np.linalg.eigvalsh(
        0.5
        * (
            rotation_matrices
            + np.swapaxes(rotation_matrices, 1, 2)
        )
    )
    return _AutonomousSE3Chunk(
        center_start=int(center_start),
        center_end=int(center_end),
        support_positions=support_positions.copy(),
        support_orientations=support_orientations.copy(),
        position_precisions=position_precisions.copy(),
        rotation_precisions=rotation_precisions.copy(),
        position_velocities=position_velocities,
        position_matrices=position_matrices,
        rotation_velocities=rotation_velocities,
        rotation_matrices=rotation_matrices,
        metadata={
            "progression_source": (
                "geometry" if nominal_twists is None else "command_projected"
            ),
            "position_fit_rmse": float(
                np.sqrt(np.mean(np.square(fitted_positions - target_positions)))
            ),
            "rotation_fit_rmse": float(
                np.sqrt(np.mean(np.square(fitted_rotations - target_rotations)))
            ),
            "position_symmetric_eigenvalue_max": float(
                np.max(position_eigenvalues)
            ),
            "rotation_symmetric_eigenvalue_max": float(
                np.max(rotation_eigenvalues)
            ),
        },
    )


class AutonomousSE3FieldExecutor:
    """Execute fixed state-only SE(3) fields with event-based replanning."""

    _GRIPPER_TRACKING_OPEN = "tracking_open"
    _GRIPPER_TRACKING_CLOSED = "tracking_closed"
    _GRIPPER_CLOSING = "closing"

    def __init__(
        self,
        plan: EndpointFieldPlan,
        config: EndpointFieldConfig | None = None,
        fit_config: AutonomousSE3FitConfig | None = None,
        nominal_commands: np.ndarray | None = None,
        *,
        _compile_chunks: bool = True,
    ) -> None:
        self.plan = plan
        self.config = config or EndpointFieldConfig()
        self.fit_config = fit_config or AutonomousSE3FitConfig()
        self.target_count = min(
            int(self.config.replan_after_targets),
            int(self.plan.num_targets),
        )
        self.nominal_twists: np.ndarray | None = None
        if nominal_commands is not None:
            commands = np.asarray(nominal_commands, dtype=np.float64)
            if commands.shape != (self.plan.num_targets, 6):
                raise ValueError(
                    "nominal_commands must have shape "
                    f"({self.plan.num_targets},6), got {commands.shape}"
                )
            if not np.all(np.isfinite(commands)):
                raise ValueError("nominal_commands must be finite")
            self.nominal_twists = commands.copy()
            self.nominal_twists[:, :3] *= float(
                self.config.position_action_scale
            )
            self.nominal_twists[:, 3:6] *= float(
                self.config.rotation_action_scale
            )
        self.chunk_ranges = partition_se3_center_chain(
            plan,
            target_count=self.target_count,
            fit_config=self.fit_config,
        )
        self.chunks = (
            tuple(
                _compile_chunk(
                    plan,
                    center_start=start,
                    center_end=end,
                    target_count=self.target_count,
                    field_config=self.config,
                    fit_config=self.fit_config,
                    nominal_twists=self.nominal_twists,
                )
                for start, end in self.chunk_ranges
            )
            if _compile_chunks
            else ()
        )
        self.current_chunk = 0
        self.target_index = 0
        self.steps_on_target = 0
        self.total_steps = 0
        self.false_attractor_steps = 0
        self.gripper_contact_stall_steps = 0
        self.terminal_stall_steps = 0
        self.previous_position: np.ndarray | None = None
        self.previous_quaternion: np.ndarray | None = None
        self.last_actual_position_step: float | None = None
        self.last_actual_rotation_step: float | None = None
        self.last_commanded_position_step = 0.0
        self.last_commanded_rotation_step = 0.0
        self.gripper_execution_state = self._GRIPPER_TRACKING_OPEN
        self.gripper_state_initialized = False
        self.gripper_desired_closed = self._gripper_target_is_closed(
            self.plan.gripper_targets[0]
        )
        self.previous_gripper_aperture: float | None = None
        self.gripper_close_start_aperture: float | None = None
        self.last_gripper_aperture_step: float | None = None
        self.last_gripper_closure_delta = 0.0
        self.last_gripper_command = float(
            np.clip(self.plan.gripper_targets[0], -1.0, 1.0)
        )
        self.gripper_phase = 0.0
        self.continuous_phase = 0.0
        self.last_center_transition_alpha = 0.0
        self.last_geometric_phase = 0.0
        self.last_geometric_on_path = False
        self.last_geometric_position_error = 0.0
        self.last_geometric_rotation_error = 0.0
        self.last_gripper_progress_capture_radius = 0.0
        self.last_clf_applied = False
        self.last_clf_distance = 0.0
        self.last_clf_correction_norm = 0.0

    @property
    def replan_target_index(self) -> int:
        return self.target_count - 1

    @property
    def current_field(self) -> _AutonomousSE3Chunk:
        return self.chunks[self.current_chunk]

    @property
    def compile_metadata(self) -> dict[str, object]:
        return {
            "executor": "autonomous_se3",
            "progression_source": (
                "geometry"
                if self.nominal_twists is None
                else "command_projected"
            ),
            "chunk_ranges": [list(value) for value in self.chunk_ranges],
            "chunks": [dict(chunk.metadata) for chunk in self.chunks],
        }

    def _chunk_gating_and_offsets(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        chunk: _AutonomousSE3Chunk | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk = self.current_field if chunk is None else chunk
        return _gating_and_offsets(
            position[None],
            quaternion[None],
            chunk.support_positions,
            chunk.support_orientations,
            chunk.position_precisions,
            chunk.rotation_precisions,
        )

    def mixture_weights(
        self,
        position,
        quaternion=None,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if quaternion is None:
            quaternion = self.plan.orientation_centers[
                min(self.target_index, self.target_count)
            ]
        quaternion = normalize_quaternion(quaternion)
        weights, _, _ = self._chunk_gating_and_offsets(
            position,
            quaternion,
        )
        return weights[0]

    def _pose_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk = self.current_field
        gating, position_offsets, rotation_offsets = (
            self._chunk_gating_and_offsets(position, quaternion, chunk)
        )
        position_delta = _mixture_output(
            gating,
            position_offsets,
            chunk.position_velocities,
            chunk.position_matrices,
        )[0]
        rotation_delta = _mixture_output(
            gating,
            rotation_offsets,
            chunk.rotation_velocities,
            chunk.rotation_matrices,
        )[0]
        position_delta = self._clf_project_position_delta(
            position,
            position_delta,
        )
        return position_delta, rotation_delta, gating[0]

    def _closest_chunk_position(self, position: np.ndarray) -> np.ndarray:
        start, end = self.chunk_ranges[self.current_chunk]
        closest = self.plan.position_centers[start]
        closest_distance = float(np.linalg.norm(position - closest))
        for segment_index in range(start, end):
            segment_start = self.plan.position_centers[segment_index]
            segment_end = self.plan.position_centers[segment_index + 1]
            segment = segment_end - segment_start
            denominator = float(np.dot(segment, segment))
            alpha = (
                0.0
                if denominator <= 1.0e-12
                else float(
                    np.clip(
                        np.dot(position - segment_start, segment) / denominator,
                        0.0,
                        1.0,
                    )
                )
            )
            candidate = segment_start + alpha * segment
            distance = float(np.linalg.norm(position - candidate))
            if distance < closest_distance:
                closest = candidate
                closest_distance = distance
        return np.asarray(closest, dtype=np.float64)

    def _clf_project_position_delta(
        self,
        position: np.ndarray,
        position_delta: np.ndarray,
    ) -> np.ndarray:
        self.last_clf_applied = False
        self.last_clf_correction_norm = 0.0
        if not bool(self.config.clf_projection_enabled):
            self.last_clf_distance = 0.0
            return position_delta

        closest = self._closest_chunk_position(position)
        offset = position - closest
        squared_distance = float(np.dot(offset, offset))
        self.last_clf_distance = float(np.sqrt(squared_distance))
        if self.last_clf_distance <= float(
            self.config.clf_position_tube_radius
        ) or squared_distance <= 1.0e-12:
            return position_delta

        contraction = float(self.config.clf_position_contraction_rate)
        if self.config.clf_projection_mode == "smooth_tube":
            radial_target = contraction * (
                self.last_clf_distance
                - float(self.config.clf_position_tube_radius)
            )
            required_inner_product = radial_target * self.last_clf_distance
        else:
            required_inner_product = contraction * squared_distance
        violation = float(
            np.dot(offset, position_delta) + required_inner_product
        )
        if violation <= 0.0:
            return position_delta

        correction = (violation / squared_distance) * offset
        self.last_clf_applied = True
        self.last_clf_correction_norm = float(np.linalg.norm(correction))
        return position_delta - correction

    def _segment_projection(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        segment_index: int,
    ) -> tuple[float, float, float]:
        start_position = self.plan.position_centers[segment_index]
        end_position = self.plan.position_centers[segment_index + 1]
        segment = end_position - start_position
        denominator = float(np.dot(segment, segment))
        position_progress = (
            0.0
            if denominator <= 1.0e-12
            else float(np.dot(position - start_position, segment) / denominator)
        )
        rotation = self.plan.segment_rotation_vectors[segment_index]
        rotation_denominator = float(np.dot(rotation, rotation))
        rotation_progress = (
            position_progress
            if rotation_denominator <= 1.0e-12
            else float(
                np.dot(
                    relative_rotation_vector(
                        self.plan.orientation_centers[segment_index],
                        quaternion,
                    ),
                    rotation,
                )
                / rotation_denominator
            )
        )
        alpha = float(
            np.clip(
                position_progress if denominator > 1.0e-12 else rotation_progress,
                0.0,
                1.0,
            )
        )
        closest_position, closest_orientation = _pose_at_phase(
            self.plan,
            float(segment_index) + alpha,
            target_count=self.target_count,
        )
        position_error = float(np.linalg.norm(position - closest_position))
        rotation_error = float(
            np.linalg.norm(
                relative_rotation_vector(quaternion, closest_orientation)
            )
        )
        return alpha, position_error, rotation_error

    def _geometric_phase(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        update: bool,
    ) -> tuple[float, int, bool]:
        start, end = self.chunk_ranges[self.current_chunk]
        candidates = []
        for segment_index in range(start, end):
            alpha, position_error, rotation_error = self._segment_projection(
                position,
                quaternion,
                segment_index,
            )
            candidates.append(
                (
                    float(segment_index) + alpha,
                    segment_index,
                    position_error,
                    rotation_error,
                )
            )
        minimum_position_error = min(value[2] for value in candidates)
        position_tie_tolerance = max(
            1.0e-6,
            0.25 * float(self.config.position_tolerance),
        )
        position_candidates = [
            value
            for value in candidates
            if value[2] <= minimum_position_error + position_tie_tolerance
        ]
        phase, segment_index, position_error, rotation_error = min(
            position_candidates,
            key=lambda value: (value[3], value[2], value[0]),
        )
        capture = max(
            float(self.config.position_tolerance),
            float(self.config.intermediate_capture_fraction)
            * max(float(self.plan.segment_lengths[segment_index]), 1.0e-12),
        )
        on_path = bool(position_error <= capture)
        spatial_phase = max(self.gripper_phase, phase) if on_path else self.gripper_phase
        spatial_phase = min(spatial_phase, float(self.target_count - 1))
        if update:
            self.gripper_phase = spatial_phase
            self.continuous_phase = spatial_phase
            next_target = min(int(np.floor(spatial_phase)), self.replan_target_index)
            if next_target != self.target_index:
                self.target_index = next_target
                self.steps_on_target = 0
            self.last_center_transition_alpha = float(
                spatial_phase - np.floor(spatial_phase)
            )
            self.last_geometric_phase = float(phase)
            self.last_geometric_on_path = bool(on_path)
            self.last_geometric_position_error = float(position_error)
            self.last_geometric_rotation_error = float(rotation_error)
            self.last_gripper_progress_capture_radius = float(capture)
        return spatial_phase, segment_index, on_path

    def segment_progress(
        self,
        position,
        quaternion=None,
        target_index: int | None = None,
    ) -> float:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if quaternion is None:
            quaternion = self.plan.orientation_centers[
                min(self.target_index, self.target_count)
            ]
        quaternion = normalize_quaternion(quaternion)
        index = self.target_index if target_index is None else int(target_index)
        index = min(max(index, 0), self.target_count - 1)
        alpha, _, _ = self._segment_projection(
            position,
            quaternion,
            index,
        )
        return alpha

    def _chunk_terminal_reached(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> bool:
        _, end = self.chunk_ranges[self.current_chunk]
        position_error = float(
            np.linalg.norm(position - self.plan.position_centers[end])
        )
        rotation_error = float(
            np.linalg.norm(
                relative_rotation_vector(
                    quaternion,
                    self.plan.orientation_centers[end],
                )
            )
        )
        position_tolerance = float(self.config.position_tolerance)
        if self.current_chunk < len(self.chunks) - 1:
            incoming = max(end - 1, 0)
            position_tolerance = max(
                position_tolerance,
                float(self.config.intermediate_capture_fraction)
                * max(float(self.plan.segment_lengths[incoming]), 1.0e-12),
            )
        return bool(
            position_error <= position_tolerance
            and rotation_error <= float(self.config.rotation_tolerance)
        )

    def _advance_chunks(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        while self.current_chunk < len(self.chunks) - 1:
            if not self._chunk_terminal_reached(position, quaternion):
                break
            self.current_chunk += 1

    def _final_errors(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self.target_count
        return (
            self.plan.position_centers[index] - position,
            relative_rotation_vector(
                quaternion,
                self.plan.orientation_centers[index],
            ),
        )

    def _terminal_stall_detected(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        position_delta: np.ndarray,
        rotation_delta: np.ndarray,
        position_error: np.ndarray,
    ) -> bool:
        self.last_commanded_position_step = float(
            np.linalg.norm(position_delta)
        )
        self.last_commanded_rotation_step = float(
            np.linalg.norm(rotation_delta)
        )

        has_previous_state = self.previous_position is not None
        if has_previous_state:
            assert self.previous_quaternion is not None
            self.last_actual_position_step = float(
                np.linalg.norm(position - self.previous_position)
            )
            self.last_actual_rotation_step = float(
                np.linalg.norm(
                    relative_rotation_vector(
                        self.previous_quaternion,
                        quaternion,
                    )
                )
            )
        else:
            self.last_actual_position_step = None
            self.last_actual_rotation_step = None

        self.previous_position = position.copy()
        self.previous_quaternion = quaternion.copy()

        near_terminal = bool(
            np.linalg.norm(position_error)
            <= float(self.config.terminal_stall_radius)
        )
        actual_motion_small = bool(
            has_previous_state
            and self.last_actual_position_step is not None
            and self.last_actual_rotation_step is not None
            and self.last_actual_position_step
            <= float(self.config.terminal_stall_position_step_threshold)
            and self.last_actual_rotation_step
            <= float(self.config.terminal_stall_rotation_step_threshold)
        )
        command_is_active = bool(
            self.last_commanded_position_step
            >= float(self.config.terminal_stall_position_command_threshold)
            or self.last_commanded_rotation_step
            >= float(self.config.terminal_stall_rotation_command_threshold)
        )
        stalled = bool(
            self.config.terminal_stall_replan
            and near_terminal
            and actual_motion_small
            and command_is_active
        )
        self.terminal_stall_steps = (
            self.terminal_stall_steps + 1 if stalled else 0
        )
        return self.terminal_stall_steps >= int(
            self.config.terminal_stall_patience
        )

    def _begin_gripper_closure(self, aperture: float) -> None:
        self.gripper_execution_state = self._GRIPPER_CLOSING
        self.gripper_close_start_aperture = float(aperture)
        self.gripper_contact_stall_steps = 0
        self.last_gripper_closure_delta = 0.0

    def _observe_gripper_closure(
        self,
        gripper_qpos: np.ndarray | None,
    ) -> str | None:
        measured_aperture = self._gripper_aperture(gripper_qpos)
        if measured_aperture is None:
            self.last_gripper_aperture_step = None
            return None

        previous_aperture = self.previous_gripper_aperture
        self.last_gripper_aperture_step = (
            None
            if previous_aperture is None
            else abs(measured_aperture - previous_aperture)
        )
        self.previous_gripper_aperture = measured_aperture

        if not self.gripper_state_initialized:
            self.gripper_state_initialized = True
            if self.gripper_desired_closed:
                if measured_aperture > float(
                    self.config.gripper_closed_qpos_threshold
                ):
                    self._begin_gripper_closure(measured_aperture)
                else:
                    self.gripper_execution_state = (
                        self._GRIPPER_TRACKING_CLOSED
                    )
            else:
                self.gripper_execution_state = self._GRIPPER_TRACKING_OPEN
            return None

        if self.gripper_execution_state != self._GRIPPER_CLOSING:
            self.gripper_contact_stall_steps = 0
            return None

        assert self.gripper_close_start_aperture is not None
        self.last_gripper_closure_delta = max(
            0.0,
            self.gripper_close_start_aperture - measured_aperture,
        )
        if measured_aperture <= float(
            self.config.gripper_closed_qpos_threshold
        ):
            self.gripper_execution_state = self._GRIPPER_TRACKING_CLOSED
            self.gripper_contact_stall_steps = 0
            if bool(self.config.gripper_close_transition_replan):
                return "gripper_close_transition"
            return None

        fingers_stalled = bool(
            self.last_gripper_aperture_step is not None
            and self.last_gripper_aperture_step
            <= float(self.config.gripper_contact_aperture_step_threshold)
            + 1.0e-12
        )
        stalled_contact = bool(
            self.config.gripper_contact_progress
            and self.last_gripper_closure_delta
            >= float(self.config.gripper_contact_min_aperture_delta)
            and fingers_stalled
        )
        self.gripper_contact_stall_steps = (
            self.gripper_contact_stall_steps + 1 if stalled_contact else 0
        )
        if self.gripper_contact_stall_steps < int(
            self.config.gripper_contact_stall_patience
        ):
            return None

        self.gripper_execution_state = self._GRIPPER_TRACKING_CLOSED
        return "gripper_contact_progress"

    def _set_gripper_command_state(
        self,
        command: float,
        gripper_qpos: np.ndarray | None,
    ) -> float:
        command = float(np.clip(command, -1.0, 1.0))
        requested_closed = self._gripper_target_is_closed(command)
        measured_aperture = self._gripper_aperture(gripper_qpos)

        if self.gripper_execution_state == self._GRIPPER_CLOSING:
            self.gripper_desired_closed = requested_closed
            if not requested_closed:
                self.gripper_execution_state = self._GRIPPER_TRACKING_OPEN
                self.gripper_close_start_aperture = None
                self.gripper_contact_stall_steps = 0
                self.last_gripper_closure_delta = 0.0
            return command

        transition_to_closed = bool(
            requested_closed and not self.gripper_desired_closed
        )
        self.gripper_desired_closed = requested_closed
        if transition_to_closed and self.gripper_state_initialized:
            if (
                measured_aperture is not None
                and measured_aperture
                > float(self.config.gripper_closed_qpos_threshold)
            ):
                self._begin_gripper_closure(measured_aperture)
                return command
            self.gripper_execution_state = self._GRIPPER_TRACKING_CLOSED
        elif not requested_closed:
            self.gripper_execution_state = self._GRIPPER_TRACKING_OPEN
            self.gripper_close_start_aperture = None
            self.gripper_contact_stall_steps = 0
            self.last_gripper_closure_delta = 0.0
        return command

    def _gripper_aperture(
        self,
        gripper_qpos: np.ndarray | None,
    ) -> float | None:
        if gripper_qpos is None:
            return None
        qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if len(qpos) == 0 or not np.all(np.isfinite(qpos)):
            return None
        return float(np.max(np.abs(qpos)))

    def _gripper_target_is_closed(self, target: float) -> bool:
        return bool(
            float(target) > float(self.config.gripper_close_command_threshold)
        )

    def _gripper_target_reached(
        self,
        target: float,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        if gripper_qpos is None:
            return True
        aperture = self._gripper_aperture(gripper_qpos)
        if aperture is None:
            return False
        measured_closed = bool(
            aperture <= float(self.config.gripper_closed_qpos_threshold)
        )
        return measured_closed == self._gripper_target_is_closed(target)

    def _gripper_command(self, phase: float, target_index: int) -> float:
        targets = np.clip(
            np.asarray(self.plan.gripper_targets, dtype=np.float64),
            -1.0,
            1.0,
        )
        if self.config.gripper_command_mode == "target" or len(targets) == 1:
            return float(targets[min(max(target_index, 0), len(targets) - 1)])
        sample_phases = np.arange(len(targets), dtype=np.float64)
        return float(
            np.interp(
                np.clip(phase, 0.0, sample_phases[-1]),
                sample_phases,
                targets,
            )
        )

    def _field_action(
        self,
        position_delta: np.ndarray,
        rotation_delta: np.ndarray,
        gripper_command: float,
    ) -> np.ndarray:
        speed_scale = float(self.config.field_speed_scale)
        position_action = (
            speed_scale
            * position_delta
            / float(self.config.position_action_scale)
        )
        rotation_action = (
            speed_scale
            * rotation_delta
            / float(self.config.rotation_action_scale)
        )
        if bool(self.config.preserve_field_direction_on_saturation):
            position_peak = float(np.max(np.abs(position_action)))
            position_limit = float(self.config.max_position_action)
            if position_peak > position_limit:
                position_action *= position_limit / position_peak
            rotation_peak = float(np.max(np.abs(rotation_action)))
            rotation_limit = float(self.config.max_rotation_action)
            if rotation_peak > rotation_limit:
                rotation_action *= rotation_limit / rotation_peak
        else:
            position_action = np.clip(
                position_action,
                -float(self.config.max_position_action),
                float(self.config.max_position_action),
            )
            rotation_action = np.clip(
                rotation_action,
                -float(self.config.max_rotation_action),
                float(self.config.max_rotation_action),
            )
        return np.concatenate(
            [position_action, rotation_action, [gripper_command]]
        ).astype(np.float32)

    def _make_step(
        self,
        *,
        action: np.ndarray,
        position_error: np.ndarray,
        rotation_error: np.ndarray,
        coverage: float,
        plan_complete: bool,
        replan_required: bool,
        replan_reason: str | None = None,
    ) -> EndpointFieldStep:
        return EndpointFieldStep(
            action=np.asarray(action, dtype=np.float32),
            target_index=int(self.target_index),
            position_error=float(np.linalg.norm(position_error)),
            rotation_error=float(np.linalg.norm(rotation_error)),
            coverage=float(coverage),
            plan_complete=bool(plan_complete),
            replan_required=bool(replan_required),
            replan_reason=replan_reason,
            center_transition_alpha=float(self.last_center_transition_alpha),
        )

    def preview_action(self, position, quaternion) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        position_delta, rotation_delta, weights = self._pose_field(
            position,
            quaternion,
        )
        phase, segment_index, _ = self._geometric_phase(
            position,
            quaternion,
            update=False,
        )
        gripper = self._gripper_command(phase, segment_index)
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        return self._make_step(
            action=self._field_action(
                position_delta,
                rotation_delta,
                gripper,
            ),
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=float(np.max(weights)),
            plan_complete=False,
            replan_required=False,
        )

    def step(
        self,
        position,
        quaternion,
        gripper_qpos=None,
    ) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        gripper_completion = self._observe_gripper_closure(gripper_qpos)
        if gripper_completion is not None:
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
                replan_reason=gripper_completion,
            )

        self._advance_chunks(position, quaternion)
        final_position_error = float(np.linalg.norm(position_error))
        strict_position_reached = bool(
            final_position_error <= float(self.config.position_tolerance)
        )
        final_pose_reached = bool(
            self.total_steps > 0
            and np.linalg.norm(rotation_error)
            <= float(self.config.rotation_tolerance)
            and strict_position_reached
        )
        final_gripper_target = float(
            np.clip(
                self.plan.gripper_targets[self.replan_target_index],
                -1.0,
                1.0,
            )
        )
        final_gripper_reached = self._gripper_target_reached(
            final_gripper_target,
            gripper_qpos,
        )
        if final_pose_reached and final_gripper_reached:
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
            )
        if self.total_steps >= int(self.config.max_plan_steps):
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=0.0,
                plan_complete=False,
                replan_required=True,
                replan_reason="hard_timeout",
            )

        if final_pose_reached:
            action = np.zeros((FIELD_ACTION_DIM,), dtype=np.float32)
            gripper = self._set_gripper_command_state(
                final_gripper_target,
                gripper_qpos,
            )
            action[-1] = gripper
            self.last_gripper_command = gripper
            self.last_commanded_position_step = 0.0
            self.last_commanded_rotation_step = 0.0
            self.previous_position = position.copy()
            self.previous_quaternion = quaternion.copy()
            self.steps_on_target += 1
            self.total_steps += 1
            return self._make_step(
                action=action,
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=False,
                replan_required=False,
            )

        position_delta, rotation_delta, weights = self._pose_field(
            position,
            quaternion,
        )
        low_speed = bool(
            np.linalg.norm(position_delta)
            <= float(self.config.false_attractor_position_speed_threshold)
            and np.linalg.norm(rotation_delta)
            <= float(self.config.false_attractor_rotation_speed_threshold)
        )
        far_from_target = bool(
            np.linalg.norm(position_error)
            > float(self.config.false_attractor_error_multiplier)
            * float(self.config.position_tolerance)
            or np.linalg.norm(rotation_error)
            > float(self.config.false_attractor_error_multiplier)
            * float(self.config.rotation_tolerance)
        )
        self.false_attractor_steps = (
            self.false_attractor_steps + 1
            if low_speed and far_from_target
            else 0
        )
        if self.false_attractor_steps >= int(
            self.config.false_attractor_patience
        ):
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=float(np.max(weights)),
                plan_complete=False,
                replan_required=True,
                replan_reason="false_attractor",
            )

        if self._terminal_stall_detected(
            position,
            quaternion,
            position_delta,
            rotation_delta,
            position_error,
        ):
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=float(np.max(weights)),
                plan_complete=False,
                replan_required=True,
                replan_reason="terminal_stall",
            )

        phase, segment_index, _ = self._geometric_phase(
            position,
            quaternion,
            update=True,
        )
        gripper = self._gripper_command(phase, segment_index)
        gripper = self._set_gripper_command_state(gripper, gripper_qpos)
        action = self._field_action(position_delta, rotation_delta, gripper)
        self.last_gripper_command = gripper
        self.steps_on_target += 1
        self.total_steps += 1
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=float(np.max(weights)),
            plan_complete=False,
            replan_required=False,
        )


class ControllerAwareSE3FieldExecutor(AutonomousSE3FieldExecutor):
    """Preserve policy commands on-path and add pose recovery off-path.

    The cached controller actions remain in controller-action coordinates.
    They are not fitted as measured state velocities. The center chain is used
    only as a state-dependent reference for progress and recovery.
    """

    def __init__(
        self,
        plan: EndpointFieldPlan,
        controller_actions: np.ndarray,
        config: EndpointFieldConfig | None = None,
        fit_config: AutonomousSE3FitConfig | None = None,
        *,
        feedforward_position_radius: float = 0.01,
        feedforward_rotation_radius: float = 0.2,
        position_recovery_gain: float = 1.0,
        rotation_recovery_gain: float = 1.0,
    ) -> None:
        super().__init__(
            plan,
            config=config,
            fit_config=fit_config,
            _compile_chunks=False,
        )
        actions = np.asarray(controller_actions, dtype=np.float64)
        if actions.shape != (self.plan.num_targets, 6):
            raise ValueError(
                "controller_actions must have shape "
                f"({self.plan.num_targets},6), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("controller_actions must be finite")
        positive = {
            "feedforward_position_radius": feedforward_position_radius,
            "feedforward_rotation_radius": feedforward_rotation_radius,
            "position_recovery_gain": position_recovery_gain,
            "rotation_recovery_gain": rotation_recovery_gain,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.controller_actions = actions.copy()
        # Action-space references use monotonic spatial progress, so a short
        # command must not be treated as a completed autonomous-field chunk
        # before that command is issued. Keep the complete action horizon in
        # one ordered reference even if the geometric path nearly intersects.
        self.chunk_ranges = ((0, self.target_count),)
        self.chunks = (
            _compile_chunk(
                self.plan,
                center_start=0,
                center_end=self.target_count,
                target_count=self.target_count,
                field_config=self.config,
                fit_config=self.fit_config,
                nominal_twists=None,
            ),
        )
        self.current_chunk = 0
        self.feedforward_position_radius = float(
            feedforward_position_radius
        )
        self.feedforward_rotation_radius = float(
            feedforward_rotation_radius
        )
        self.position_recovery_gain = float(position_recovery_gain)
        self.rotation_recovery_gain = float(rotation_recovery_gain)
        self.last_feedforward_weight = 1.0
        self.last_reference_position_error = 0.0
        self.last_reference_rotation_error = 0.0

    @property
    def compile_metadata(self) -> dict[str, object]:
        metadata = dict(super().compile_metadata)
        metadata.update(
            {
                "executor": "controller_aware_se3",
                "progression_source": "spatial_reference",
                "on_path_action": "raw_policy_command",
                "feedforward_position_radius": (
                    self.feedforward_position_radius
                ),
                "feedforward_rotation_radius": (
                    self.feedforward_rotation_radius
                ),
                "position_recovery_gain": self.position_recovery_gain,
                "rotation_recovery_gain": self.rotation_recovery_gain,
            }
        )
        return metadata

    def _controller_action_at_phase(self, phase: float) -> np.ndarray:
        phase = float(
            np.clip(phase, 0.0, float(max(self.target_count - 1, 0)))
        )
        index = min(int(np.floor(phase)), self.target_count - 1)
        alpha = float(np.clip(phase - index, 0.0, 1.0))
        next_index = min(index + 1, self.target_count - 1)
        return (
            (1.0 - alpha) * self.controller_actions[index]
            + alpha * self.controller_actions[next_index]
        )

    def _pose_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase, _, _ = self._geometric_phase(
            position,
            quaternion,
            update=False,
        )
        reference_position, reference_orientation = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        position_error = reference_position - position
        rotation_error = relative_rotation_vector(
            quaternion,
            reference_orientation,
        )
        command = self._controller_action_at_phase(phase)
        command_position = (
            command[:3] * float(self.config.position_action_scale)
        )
        command_rotation = (
            command[3:6] * float(self.config.rotation_action_scale)
        )

        position_ratio = (
            float(np.linalg.norm(position_error))
            / self.feedforward_position_radius
        )
        rotation_ratio = (
            float(np.linalg.norm(rotation_error))
            / self.feedforward_rotation_radius
        )
        feedforward_weight = float(
            np.exp(-0.5 * (position_ratio**2 + rotation_ratio**2))
        )
        position_delta = (
            feedforward_weight * command_position
            + self.position_recovery_gain * position_error
        )
        rotation_delta = (
            feedforward_weight * command_rotation
            + self.rotation_recovery_gain * rotation_error
        )
        position_delta = self._clf_project_position_delta(
            position,
            position_delta,
        )
        gating, _, _ = self._chunk_gating_and_offsets(
            position,
            quaternion,
        )
        self.last_feedforward_weight = feedforward_weight
        self.last_reference_position_error = float(
            np.linalg.norm(position_error)
        )
        self.last_reference_rotation_error = float(
            np.linalg.norm(rotation_error)
        )
        return position_delta, rotation_delta, gating[0]


class PoseIndexedActionFieldExecutor(AutonomousSE3FieldExecutor):
    """Autonomous Gaussian field from measured pose to controller action.

    Native pose actions are attached to the poses immediately before those
    actions are issued. Gaussian-affine experts are calibrated jointly so the
    blended field reproduces every attached action at its support pose. The
    affine terms add pose recovery away from the commanded center chain.
    """

    def __init__(
        self,
        plan: EndpointFieldPlan,
        controller_actions: np.ndarray,
        config: EndpointFieldConfig | None = None,
        fit_config: AutonomousSE3FitConfig | None = None,
        *,
        gating_precision_scale: float = 25.0,
        position_recovery_gain: float = 1.0,
        rotation_recovery_gain: float = 1.0,
        terminal_gripper_command: float | None = None,
    ) -> None:
        super().__init__(
            plan,
            config=config,
            fit_config=fit_config,
            _compile_chunks=False,
        )
        actions = np.asarray(controller_actions, dtype=np.float64)
        if actions.shape != (self.plan.num_targets, 6):
            raise ValueError(
                "controller_actions must have shape "
                f"({self.plan.num_targets},6), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("controller_actions must be finite")
        for name, value in {
            "gating_precision_scale": gating_precision_scale,
            "position_recovery_gain": position_recovery_gain,
            "rotation_recovery_gain": rotation_recovery_gain,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not np.isclose(float(self.config.field_speed_scale), 1.0):
            raise ValueError(
                "pose-indexed action fields require field_speed_scale=1 to "
                "preserve native controller actions"
            )
        self.controller_actions = actions.copy()
        self.controller_twists = actions.copy()
        self.controller_twists[:, :3] *= float(
            self.config.position_action_scale
        )
        self.controller_twists[:, 3:6] *= float(
            self.config.rotation_action_scale
        )
        self.position_recovery_gain = float(position_recovery_gain)
        self.rotation_recovery_gain = float(rotation_recovery_gain)
        self.gating_precision_scale = float(gating_precision_scale)
        if terminal_gripper_command is None:
            self.terminal_gripper_command = None
        else:
            command = float(terminal_gripper_command)
            if not np.isfinite(command):
                raise ValueError(
                    "terminal_gripper_command must be finite"
                )
            self.terminal_gripper_command = float(
                np.clip(command, -1.0, 1.0)
            )
        self.terminal_gripper_issued = False
        self.chunks = tuple(
            _compile_pose_indexed_action_chunk(
                self.plan,
                center_start=start,
                center_end=end,
                controller_twists=self.controller_twists,
                fit_config=self.fit_config,
                gating_precision_scale=self.gating_precision_scale,
                position_recovery_gain=self.position_recovery_gain,
                rotation_recovery_gain=self.rotation_recovery_gain,
            )
            for start, end in self.chunk_ranges
        )
        self.current_chunk = 0

    @property
    def compile_metadata(self) -> dict[str, object]:
        metadata = dict(super().compile_metadata)
        metadata.update(
            {
                "executor": "pose_indexed_action_field",
                "progression_source": "autonomous_pose_gating",
                "action_binding": "pre_action_pose",
                "on_support_action": "exact_native_policy_command",
                "gating_precision_scale": self.gating_precision_scale,
                "position_recovery_gain": self.position_recovery_gain,
                "rotation_recovery_gain": self.rotation_recovery_gain,
                "terminal_gripper_command": self.terminal_gripper_command,
            }
        )
        return metadata

    def _terminal_gripper_action(
        self,
        gripper_qpos: np.ndarray | None,
        *,
        mutate_gripper: bool,
    ) -> np.ndarray:
        if self.terminal_gripper_command is None:
            raise RuntimeError("No terminal gripper command is configured")
        command = self.terminal_gripper_command
        if mutate_gripper:
            command = self._set_gripper_command_state(command, gripper_qpos)
        action = np.zeros((FIELD_ACTION_DIM,), dtype=np.float32)
        action[-1] = command
        return action

    def _terminal_gripper_complete_step(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> EndpointFieldStep:
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        return self._make_step(
            action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=1.0,
            plan_complete=True,
            replan_required=False,
            replan_reason="gripper_only_transition",
        )

    def preview_action(self, position, quaternion) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        if self.terminal_gripper_command is not None:
            if self.terminal_gripper_issued:
                return self._terminal_gripper_complete_step(
                    position,
                    quaternion,
                )
            position_error, rotation_error = self._final_errors(
                position,
                quaternion,
            )
            if (
                np.linalg.norm(position_error)
                <= float(self.config.position_tolerance)
                and np.linalg.norm(rotation_error)
                <= float(self.config.rotation_tolerance)
            ):
                return self._make_step(
                    action=self._terminal_gripper_action(
                        None,
                        mutate_gripper=False,
                    ),
                    position_error=position_error,
                    rotation_error=rotation_error,
                    coverage=1.0,
                    plan_complete=False,
                    replan_required=False,
                )
        return super().preview_action(position, quaternion)

    def step(
        self,
        position,
        quaternion,
        gripper_qpos=None,
    ) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        if (
            self.terminal_gripper_command is not None
            and self.terminal_gripper_issued
        ):
            return self._terminal_gripper_complete_step(
                position,
                quaternion,
            )

        field_step = super().step(
            position,
            quaternion,
            gripper_qpos,
        )
        if not (
            self.terminal_gripper_command is not None
            and not self.terminal_gripper_issued
            and field_step.plan_complete
            and not field_step.replan_required
            and field_step.replan_reason is None
        ):
            return field_step

        action = self._terminal_gripper_action(
            gripper_qpos,
            mutate_gripper=True,
        )
        self.last_gripper_command = float(action[-1])
        self.last_commanded_position_step = 0.0
        self.last_commanded_rotation_step = 0.0
        self.previous_position = position.copy()
        self.previous_quaternion = quaternion.copy()
        self.terminal_gripper_issued = True
        self.total_steps += 1
        self.steps_on_target += 1
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=1.0,
            plan_complete=False,
            replan_required=False,
        )


class PhaseConditionedPoseActionFieldExecutor(
    PoseIndexedActionFieldExecutor
):
    """Advance an ordered phase while retaining pose-field feedback.

    The command index advances once per nominal control step, matching action
    chunking. A large deviation from the cached pose tube freezes that index;
    the Gaussian pose-action field then drives the measured state back toward
    the tube. ``nominal_execution`` selects whether motion inside the tube uses
    the native command or the same cached field used for recovery. In both
    cases, the ordered phase binds the gripper command and plan horizon.
    """

    def __init__(
        self,
        plan: EndpointFieldPlan,
        controller_actions: np.ndarray,
        config: EndpointFieldConfig | None = None,
        fit_config: AutonomousSE3FitConfig | None = None,
        *,
        position_tube_radius: float = 0.05,
        rotation_tube_radius: float = 0.5,
        gating_precision_scale: float = 25.0,
        position_recovery_gain: float = 1.0,
        rotation_recovery_gain: float = 1.0,
        terminal_gripper_command: float | None = None,
        nominal_execution: str = "native_action",
        phase_stride: int = 1,
    ) -> None:
        super().__init__(
            plan,
            controller_actions=controller_actions,
            config=config,
            fit_config=fit_config,
            gating_precision_scale=gating_precision_scale,
            position_recovery_gain=position_recovery_gain,
            rotation_recovery_gain=rotation_recovery_gain,
            terminal_gripper_command=terminal_gripper_command,
        )
        for name, value in {
            "position_tube_radius": position_tube_radius,
            "rotation_tube_radius": rotation_tube_radius,
        }.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.position_tube_radius = float(position_tube_radius)
        self.rotation_tube_radius = float(rotation_tube_radius)
        if nominal_execution not in {"native_action", "gaussian_field"}:
            raise ValueError(
                "nominal_execution must be 'native_action' or "
                "'gaussian_field'"
            )
        self.nominal_execution = str(nominal_execution)
        self.phase_stride = int(phase_stride)
        if self.phase_stride < 1:
            raise ValueError("phase_stride must be at least one")
        self.action_index = 0
        self.last_phase_frozen = False
        self.last_tube_phase = 0.0
        self.last_tube_position_error = 0.0
        self.last_tube_rotation_error = 0.0

    @property
    def compile_metadata(self) -> dict[str, object]:
        metadata = dict(super().compile_metadata)
        metadata.update(
            {
                "executor": (
                    "phase_conditioned_pose_action_field"
                    if self.nominal_execution == "native_action"
                    else "phase_gated_pose_action_field"
                ),
                "progression_source": "ordered_command_index",
                "nominal_execution": (
                    "exact_native_policy_command"
                    if self.nominal_execution == "native_action"
                    else "cached_gaussian_pose_action_field"
                ),
                "recovery_execution": "cached_gaussian_pose_action_field",
                "phase_freeze_condition": "outside_pose_tube",
                "position_tube_radius": self.position_tube_radius,
                "rotation_tube_radius": self.rotation_tube_radius,
                "gripper_binding": "ordered_command_index",
                "phase_stride": self.phase_stride,
                "terminal_gripper_command": self.terminal_gripper_command,
            }
        )
        return metadata

    def _select_chunk_for_action(self, action_index: int) -> None:
        index = min(max(int(action_index), 0), self.target_count - 1)
        for chunk_index, (start, end) in enumerate(self.chunk_ranges):
            if start <= index < end:
                self.current_chunk = chunk_index
                return
        self.current_chunk = len(self.chunk_ranges) - 1

    def _nearest_pose_on_chain(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[float, float, float]:
        candidates = []
        for segment_index in range(self.target_count):
            alpha, position_error, rotation_error = self._segment_projection(
                position,
                quaternion,
                segment_index,
            )
            candidates.append(
                (
                    float(segment_index) + alpha,
                    position_error,
                    rotation_error,
                )
            )
        minimum_position_error = min(value[1] for value in candidates)
        tie_tolerance = max(
            1.0e-6,
            0.25 * float(self.config.position_tolerance),
        )
        near_candidates = [
            value
            for value in candidates
            if value[1] <= minimum_position_error + tie_tolerance
        ]
        phase, position_error, rotation_error = min(
            near_candidates,
            key=lambda value: (value[2], value[1], value[0]),
        )
        self.last_tube_phase = float(phase)
        self.last_tube_position_error = float(position_error)
        self.last_tube_rotation_error = float(rotation_error)
        self.last_geometric_phase = float(phase)
        return phase, position_error, rotation_error

    def _record_action_state(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        action: np.ndarray,
    ) -> None:
        if self.previous_position is None:
            self.last_actual_position_step = None
            self.last_actual_rotation_step = None
        else:
            assert self.previous_quaternion is not None
            self.last_actual_position_step = float(
                np.linalg.norm(position - self.previous_position)
            )
            self.last_actual_rotation_step = float(
                np.linalg.norm(
                    relative_rotation_vector(
                        self.previous_quaternion,
                        quaternion,
                    )
                )
            )
        self.previous_position = position.copy()
        self.previous_quaternion = quaternion.copy()
        self.last_commanded_position_step = float(
            np.linalg.norm(action[:3])
            * float(self.config.position_action_scale)
        )
        self.last_commanded_rotation_step = float(
            np.linalg.norm(action[3:6])
            * float(self.config.rotation_action_scale)
        )

    def _ordered_action(
        self,
        action_index: int,
        gripper_qpos: np.ndarray | None,
    ) -> np.ndarray:
        index = min(max(int(action_index), 0), self.target_count - 1)
        gripper = self._set_gripper_command_state(
            float(self.plan.gripper_targets[index]),
            gripper_qpos,
        )
        return np.concatenate(
            [self.controller_actions[index], [gripper]]
        ).astype(np.float32)

    def _action_for_state(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
        *,
        mutate_gripper: bool,
    ) -> tuple[np.ndarray, float, bool]:
        _, position_error, rotation_error = self._nearest_pose_on_chain(
            position,
            quaternion,
        )
        frozen = bool(
            position_error > self.position_tube_radius
            or rotation_error > self.rotation_tube_radius
        )
        self.last_phase_frozen = frozen
        self.last_geometric_on_path = not frozen
        self._select_chunk_for_action(self.action_index)
        weights = self.mixture_weights(position, quaternion)
        coverage = float(np.max(weights))
        if not frozen and self.nominal_execution == "native_action":
            if mutate_gripper:
                action = self._ordered_action(
                    self.action_index,
                    gripper_qpos,
                )
            else:
                index = min(self.action_index, self.target_count - 1)
                action = np.concatenate(
                    [
                        self.controller_actions[index],
                        [self.plan.gripper_targets[index]],
                    ]
                ).astype(np.float32)
            return action, coverage, False

        position_delta, rotation_delta, weights = self._pose_field(
            position,
            quaternion,
        )
        index = min(self.action_index, self.target_count - 1)
        gripper = float(self.plan.gripper_targets[index])
        if mutate_gripper:
            gripper = self._set_gripper_command_state(
                gripper,
                gripper_qpos,
            )
        action = self._field_action(
            position_delta,
            rotation_delta,
            gripper,
        )
        return action, float(np.max(weights)), frozen

    def preview_action(self, position, quaternion) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        if self.action_index >= self.target_count:
            if (
                self.terminal_gripper_command is not None
                and not self.terminal_gripper_issued
            ):
                self.target_index = self.replan_target_index
                return self._make_step(
                    action=self._terminal_gripper_action(
                        None,
                        mutate_gripper=False,
                    ),
                    position_error=position_error,
                    rotation_error=rotation_error,
                    coverage=1.0,
                    plan_complete=False,
                    replan_required=False,
                )
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
                replan_reason=(
                    "gripper_only_transition"
                    if self.terminal_gripper_command is not None
                    else "ordered_action_horizon"
                ),
            )
        self.target_index = int(self.action_index)
        action, coverage, _ = self._action_for_state(
            position,
            quaternion,
            None,
            mutate_gripper=False,
        )
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=coverage,
            plan_complete=False,
            replan_required=False,
        )

    def step(
        self,
        position,
        quaternion,
        gripper_qpos=None,
    ) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        gripper_completion = self._observe_gripper_closure(gripper_qpos)
        if (
            gripper_completion is not None
            and self.terminal_gripper_command is None
        ):
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
                replan_reason=gripper_completion,
            )
        if self.action_index >= self.target_count:
            if (
                self.terminal_gripper_command is not None
                and not self.terminal_gripper_issued
            ):
                self.target_index = self.replan_target_index
                action = self._terminal_gripper_action(
                    gripper_qpos,
                    mutate_gripper=True,
                )
                self.last_gripper_command = float(action[-1])
                self._record_action_state(position, quaternion, action)
                self.terminal_gripper_issued = True
                self.total_steps += 1
                self.steps_on_target += 1
                return self._make_step(
                    action=action,
                    position_error=position_error,
                    rotation_error=rotation_error,
                    coverage=1.0,
                    plan_complete=False,
                    replan_required=False,
                )
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
                replan_reason=(
                    "gripper_only_transition"
                    if self.terminal_gripper_command is not None
                    else "ordered_action_horizon"
                ),
            )
        if self.total_steps >= int(self.config.max_plan_steps):
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=0.0,
                plan_complete=False,
                replan_required=True,
                replan_reason="hard_timeout",
            )

        issued_index = int(self.action_index)
        self.target_index = issued_index
        action, coverage, frozen = self._action_for_state(
            position,
            quaternion,
            gripper_qpos,
            mutate_gripper=True,
        )
        self.last_gripper_command = float(action[-1])
        self._record_action_state(position, quaternion, action)
        self.total_steps += 1
        self.steps_on_target += 1
        if not frozen:
            self.action_index += self.phase_stride
            self.continuous_phase = float(self.action_index)
            self.gripper_phase = float(issued_index)
            self.last_center_transition_alpha = 0.0
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=coverage,
            plan_complete=False,
            replan_required=False,
        )
