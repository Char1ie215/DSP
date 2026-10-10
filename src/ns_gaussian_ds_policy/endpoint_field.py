from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FIELD_GEOMETRY_DIM = 6
FIELD_ACTION_DIM = 7
COMMAND_FIELD_EXTRA_DIM = 6
COMMAND_FIELD_CORE_DIM = FIELD_ACTION_DIM + COMMAND_FIELD_EXTRA_DIM


def _as_float_array(value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Field inputs must be finite")
    return array


def normalize_quaternion(quaternion) -> np.ndarray:
    quat = _as_float_array(quaternion, (4,))
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion norm must be positive")
    return quat / norm


def quaternion_inverse(quaternion) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def quaternion_multiply(left, right) -> np.ndarray:
    """Hamilton product for xyzw quaternions: R(out) = R(left) R(right)."""

    x1, y1, z1, w1 = normalize_quaternion(left)
    x2, y2, z2, w2 = normalize_quaternion(right)
    result = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion(result)


def rotvec_to_quaternion(rotation_vector) -> np.ndarray:
    rotvec = _as_float_array(rotation_vector, (3,))
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    half = 0.5 * angle
    xyz = rotvec * (np.sin(half) / angle)
    return normalize_quaternion(np.concatenate([xyz, [np.cos(half)]]))


def quaternion_to_rotvec(quaternion) -> np.ndarray:
    quat = normalize_quaternion(quaternion)
    if quat[3] < 0.0:
        quat = -quat
    xyz_norm = float(np.linalg.norm(quat[:3]))
    if xyz_norm <= 1.0e-12:
        return np.zeros((3,), dtype=np.float64)
    angle = 2.0 * np.arctan2(xyz_norm, float(np.clip(quat[3], -1.0, 1.0)))
    return quat[:3] * (angle / xyz_norm)


def relative_rotation_vector(current_quaternion, target_quaternion) -> np.ndarray:
    """World-frame rotation taking current orientation to target orientation."""

    relative = quaternion_multiply(
        target_quaternion,
        quaternion_inverse(current_quaternion),
    )
    return quaternion_to_rotvec(relative)


def endpoint_target_indices(episode_ends, stride: int = 1) -> np.ndarray:
    """Return future indices without crossing demonstration boundaries."""

    if int(stride) < 1:
        raise ValueError("stride must be at least 1")
    ends = np.asarray(episode_ends, dtype=np.int64)
    if (
        ends.ndim != 1
        or len(ends) == 0
        or ends[0] <= 0
        or np.any(np.diff(ends) <= 0)
    ):
        raise ValueError("episode_ends must be a non-empty increasing vector")

    targets = np.arange(int(ends[-1]), dtype=np.int64)
    start = 0
    for raw_end in ends:
        end = int(raw_end)
        targets[start:end] = np.minimum(
            np.arange(start, end, dtype=np.int64) + int(stride),
            end - 1,
        )
        start = end
    return targets


def build_endpoint_field_labels(
    positions,
    quaternions,
    actions,
    episode_ends,
    *,
    stride: int = 1,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode measured future poses as normalized local endpoint commands."""

    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {positions.shape}")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("quaternions must be (N,4)")
    if actions.ndim != 2 or actions.shape[0] != len(positions) or actions.shape[1] < 7:
        raise ValueError("actions must be (N,>=7)")
    if float(position_action_scale) <= 0.0 or float(rotation_action_scale) <= 0.0:
        raise ValueError("action scales must be positive")

    targets = endpoint_target_indices(episode_ends, stride=int(stride))
    if len(targets) != len(positions):
        raise ValueError(
            f"episode_ends describe {len(targets)} steps, but poses have {len(positions)}"
        )
    position_delta = (
        positions[targets] - positions
    ) / float(position_action_scale)
    rotation_delta = np.stack(
        [
            relative_rotation_vector(quaternions[index], quaternions[target])
            for index, target in enumerate(targets)
        ],
        axis=0,
    ) / float(rotation_action_scale)
    labels = np.concatenate(
        [position_delta, rotation_delta, actions[:, 6:7]],
        axis=1,
    ).astype(np.float32)
    if labels.shape != (len(positions), FIELD_ACTION_DIM):
        raise RuntimeError(f"Unexpected label shape {labels.shape}")
    if not np.all(np.isfinite(labels)):
        raise RuntimeError("Generated labels contain NaN or Inf")
    return labels, targets


def build_command_endpoint_field_labels(
    positions,
    quaternions,
    actions,
    episode_ends,
    *,
    stride: int = 1,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode center geometry and retain the demonstrated OSC command.

    The first six channels describe the measured future SE(3) center. The
    next six retain the source delta action that produced the demonstration.
    Gripper and any environment-specific auxiliary channels follow unchanged.
    """

    actions = np.asarray(actions, dtype=np.float64)
    field_labels, targets = build_endpoint_field_labels(
        positions,
        quaternions,
        actions,
        episode_ends,
        stride=stride,
        position_action_scale=position_action_scale,
        rotation_action_scale=rotation_action_scale,
    )
    labels = np.concatenate(
        [
            field_labels[:, :FIELD_GEOMETRY_DIM],
            actions[:, :FIELD_GEOMETRY_DIM],
            field_labels[:, FIELD_GEOMETRY_DIM:FIELD_ACTION_DIM],
            actions[:, FIELD_ACTION_DIM:],
        ],
        axis=1,
    ).astype(np.float32)
    expected_shape = (len(actions), actions.shape[1] + COMMAND_FIELD_EXTRA_DIM)
    if labels.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected command-field label shape {labels.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(labels)):
        raise RuntimeError("Generated command-field labels contain NaN or Inf")
    return labels, targets


def build_action_integrated_relative_center_chain(
    actions,
    *,
    n_obs_steps: int,
    source_indices=None,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> np.ndarray:
    """Integrate a sampled action window into future centers.

    Output row ``n_obs_steps - 1`` is the first commanded center after the
    current observation. Later rows are cumulative SE(3) centers relative to
    that same observation. Earlier rows represent the current center and are
    never selected by Diffusion Policy at inference.

    ``source_indices`` identifies repeated rows introduced by episode-end
    padding. A padded terminal action is integrated once and then held rather
    than being applied repeatedly.
    """

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < FIELD_ACTION_DIM:
        raise ValueError("actions must have shape (T,>=7)")
    if len(actions) == 0 or not np.all(np.isfinite(actions)):
        raise ValueError("actions must be a non-empty finite array")
    n_obs_steps = int(n_obs_steps)
    if n_obs_steps < 1 or n_obs_steps > len(actions):
        raise ValueError("n_obs_steps must be in [1, len(actions)]")
    if float(position_action_scale) <= 0.0:
        raise ValueError("position_action_scale must be positive")
    if float(rotation_action_scale) <= 0.0:
        raise ValueError("rotation_action_scale must be positive")

    if source_indices is None:
        source_indices = np.arange(len(actions), dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
        if source_indices.shape != (len(actions),):
            raise ValueError("source_indices must have shape (T,)")
        if np.any(np.diff(source_indices) < 0):
            raise ValueError("source_indices must be nondecreasing")

    anchor = n_obs_steps - 1
    labels = np.zeros_like(actions, dtype=np.float64)
    if anchor > 0:
        labels[:anchor, FIELD_GEOMETRY_DIM:] = actions[
            :anchor,
            FIELD_GEOMETRY_DIM:,
        ]

    cumulative_position = np.zeros((3,), dtype=np.float64)
    cumulative_orientation = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    previous_source_index: int | None = None
    for output_index in range(anchor, len(actions)):
        source_index = int(source_indices[output_index])
        should_integrate = (
            output_index == anchor
            or source_index != previous_source_index
        )
        if should_integrate:
            command = actions[output_index]
            cumulative_position += (
                command[:3] * float(position_action_scale)
            )
            cumulative_orientation = quaternion_multiply(
                rotvec_to_quaternion(
                    command[3:6] * float(rotation_action_scale)
                ),
                cumulative_orientation,
            )
        labels[output_index, :3] = (
            cumulative_position / float(position_action_scale)
        )
        labels[output_index, 3:6] = (
            quaternion_to_rotvec(cumulative_orientation)
            / float(rotation_action_scale)
        )
        labels[output_index, FIELD_GEOMETRY_DIM:] = actions[
            output_index,
            FIELD_GEOMETRY_DIM:,
        ]
        previous_source_index = source_index

    labels = labels.astype(np.float32)
    if not np.all(np.isfinite(labels)):
        raise RuntimeError("Generated relative center chain contains NaN or Inf")
    return labels


def relative_center_chain_to_incremental_actions(
    relative_centers,
    *,
    position_action_scale: float = 0.05,
    rotation_action_scale: float = 0.5,
) -> np.ndarray:
    """Difference cumulative relative centers into adjacent SE(3) motions."""

    centers = np.asarray(relative_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] < FIELD_ACTION_DIM:
        raise ValueError("relative_centers must have shape (K,>=7)")
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        raise ValueError("relative_centers must be a non-empty finite array")
    if float(position_action_scale) <= 0.0:
        raise ValueError("position_action_scale must be positive")
    if float(rotation_action_scale) <= 0.0:
        raise ValueError("rotation_action_scale must be positive")

    actions = np.zeros_like(centers, dtype=np.float64)
    previous_position = np.zeros((3,), dtype=np.float64)
    previous_orientation = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    for index, center in enumerate(centers):
        target_position = (
            center[:3] * float(position_action_scale)
        )
        target_orientation = rotvec_to_quaternion(
            center[3:6] * float(rotation_action_scale)
        )
        actions[index, :3] = (
            target_position - previous_position
        ) / float(position_action_scale)
        actions[index, 3:6] = relative_rotation_vector(
            previous_orientation,
            target_orientation,
        ) / float(rotation_action_scale)
        actions[index, FIELD_GEOMETRY_DIM:] = center[
            FIELD_GEOMETRY_DIM:
        ]
        previous_position = target_position
        previous_orientation = target_orientation

    actions = actions.astype(np.float32)
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("Differenced center chain contains NaN or Inf")
    return actions


def split_command_endpoint_field_actions(
    predicted_actions,
    *,
    environment_action_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a command-field prediction into field, command, and aux parts."""

    actions = np.asarray(predicted_actions)
    if actions.ndim < 1:
        raise ValueError("predicted_actions must have at least one dimension")
    if environment_action_dim is None:
        environment_action_dim = int(actions.shape[-1]) - COMMAND_FIELD_EXTRA_DIM
    environment_action_dim = int(environment_action_dim)
    if environment_action_dim < FIELD_ACTION_DIM:
        raise ValueError("environment_action_dim must be at least 7")
    expected_dim = environment_action_dim + COMMAND_FIELD_EXTRA_DIM
    if int(actions.shape[-1]) != expected_dim:
        raise ValueError(
            f"Command-field predictions must end in {expected_dim} channels, "
            f"got {actions.shape}"
        )

    field_actions = np.concatenate(
        [
            actions[..., :FIELD_GEOMETRY_DIM],
            actions[..., 2 * FIELD_GEOMETRY_DIM : COMMAND_FIELD_CORE_DIM],
        ],
        axis=-1,
    )
    nominal_commands = actions[
        ..., FIELD_GEOMETRY_DIM : 2 * FIELD_GEOMETRY_DIM
    ]
    auxiliary_actions = actions[..., COMMAND_FIELD_CORE_DIM:]
    return field_actions, nominal_commands, auxiliary_actions


@dataclass(frozen=True)
class EndpointFieldPlan:
    start_position: np.ndarray
    start_orientation: np.ndarray
    position_targets: np.ndarray
    orientation_targets: np.ndarray
    gripper_targets: np.ndarray
    segment_lengths: np.ndarray
    segment_directions: np.ndarray
    segment_rotation_vectors: np.ndarray
    segment_precisions: np.ndarray

    def __post_init__(self) -> None:
        start_position = np.asarray(self.start_position)
        start_orientation = np.asarray(self.start_orientation)
        positions = np.asarray(self.position_targets)
        orientations = np.asarray(self.orientation_targets)
        grippers = np.asarray(self.gripper_targets)
        lengths = np.asarray(self.segment_lengths)
        directions = np.asarray(self.segment_directions)
        rotation_vectors = np.asarray(self.segment_rotation_vectors)
        precisions = np.asarray(self.segment_precisions)
        if start_position.shape != (3,):
            raise ValueError("start_position must be (3,)")
        if start_orientation.shape != (4,):
            raise ValueError("start_orientation must be (4,)")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"position_targets must be (K,3), got {positions.shape}")
        if orientations.shape != (len(positions), 4):
            raise ValueError("orientation_targets must be (K,4)")
        if grippers.shape != (len(positions),):
            raise ValueError("gripper_targets must be (K,)")
        if lengths.shape != (len(positions),):
            raise ValueError("segment_lengths must be (K,)")
        if directions.shape != (len(positions), 3):
            raise ValueError("segment_directions must be (K,3)")
        if rotation_vectors.shape != (len(positions), 3):
            raise ValueError("segment_rotation_vectors must be (K,3)")
        if precisions.shape != (len(positions), 3, 3):
            raise ValueError("segment_precisions must be (K,3,3)")
        if len(positions) == 0:
            raise ValueError("A field plan needs at least one target")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                start_position,
                start_orientation,
                positions,
                orientations,
                grippers,
                lengths,
                directions,
                rotation_vectors,
                precisions,
            )
        ):
            raise ValueError("Field plan values must be finite")

    @property
    def num_targets(self) -> int:
        return int(len(self.position_targets))

    @property
    def position_centers(self) -> np.ndarray:
        return np.concatenate(
            [self.start_position[None], self.position_targets],
            axis=0,
        )

    @property
    def orientation_centers(self) -> np.ndarray:
        return np.concatenate(
            [self.start_orientation[None], self.orientation_targets],
            axis=0,
        )


@dataclass(frozen=True)
class EndpointFieldConfig:
    position_action_scale: float = 0.05
    rotation_action_scale: float = 0.5
    field_speed_scale: float = 1.0
    position_tolerance: float = 0.002
    rotation_tolerance: float = 0.05
    max_position_action: float = 1.0
    max_rotation_action: float = 1.0
    replan_after_targets: int = 8
    max_plan_steps: int = 200
    gripper_contact_progress: bool = True
    gripper_close_transition_replan: bool = True
    gripper_command_mode: str = "target"
    gripper_close_command_threshold: float = 0.5
    gripper_closed_qpos_threshold: float = 0.03
    gripper_contact_position_tolerance: float = 0.005
    gripper_contact_stall_patience: int = 3
    gripper_contact_min_aperture_delta: float = 0.002
    gripper_contact_aperture_step_threshold: float = 1.0e-4
    false_attractor_patience: int = 10
    false_attractor_position_speed_threshold: float = 1.0e-4
    false_attractor_rotation_speed_threshold: float = 1.0e-3
    false_attractor_error_multiplier: float = 2.0
    terminal_stall_replan: bool = True
    terminal_stall_radius: float = 0.02
    terminal_stall_patience: int = 10
    terminal_stall_position_step_threshold: float = 5.0e-5
    terminal_stall_rotation_step_threshold: float = 1.0e-3
    terminal_stall_position_command_threshold: float = 1.0e-3
    terminal_stall_rotation_command_threshold: float = 1.0e-2
    sigma_parallel_scale: float = 1.5
    sigma_perpendicular_scale: float = 0.75
    min_sigma: float = 0.004
    segment_cross_track_gain: float = 1.0
    future_weight_decay: float = 0.1
    past_weight_decay: float = 0.05
    intermediate_progress_threshold: float = 0.8
    intermediate_capture_fraction: float = 0.25
    clf_projection_enabled: bool = False
    clf_position_tube_radius: float = 0.01
    clf_position_contraction_rate: float = 0.1
    clf_projection_mode: str = "quadratic"
    preserve_field_direction_on_saturation: bool = False

    def __post_init__(self) -> None:
        positive = {
            "position_action_scale": self.position_action_scale,
            "rotation_action_scale": self.rotation_action_scale,
            "field_speed_scale": self.field_speed_scale,
            "position_tolerance": self.position_tolerance,
            "rotation_tolerance": self.rotation_tolerance,
            "max_position_action": self.max_position_action,
            "max_rotation_action": self.max_rotation_action,
            "gripper_closed_qpos_threshold": self.gripper_closed_qpos_threshold,
            "gripper_contact_position_tolerance": (
                self.gripper_contact_position_tolerance
            ),
            "gripper_contact_min_aperture_delta": (
                self.gripper_contact_min_aperture_delta
            ),
            "gripper_contact_aperture_step_threshold": (
                self.gripper_contact_aperture_step_threshold
            ),
            "false_attractor_position_speed_threshold": (
                self.false_attractor_position_speed_threshold
            ),
            "false_attractor_rotation_speed_threshold": (
                self.false_attractor_rotation_speed_threshold
            ),
            "false_attractor_error_multiplier": self.false_attractor_error_multiplier,
            "terminal_stall_radius": self.terminal_stall_radius,
            "terminal_stall_position_step_threshold": (
                self.terminal_stall_position_step_threshold
            ),
            "terminal_stall_rotation_step_threshold": (
                self.terminal_stall_rotation_step_threshold
            ),
            "terminal_stall_position_command_threshold": (
                self.terminal_stall_position_command_threshold
            ),
            "terminal_stall_rotation_command_threshold": (
                self.terminal_stall_rotation_command_threshold
            ),
            "sigma_parallel_scale": self.sigma_parallel_scale,
            "sigma_perpendicular_scale": self.sigma_perpendicular_scale,
            "min_sigma": self.min_sigma,
            "clf_position_tube_radius": self.clf_position_tube_radius,
            "clf_position_contraction_rate": (
                self.clf_position_contraction_rate
            ),
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.segment_cross_track_gain) < 0.0:
            raise ValueError("segment_cross_track_gain must be non-negative")
        if int(self.replan_after_targets) < 1:
            raise ValueError("replan_after_targets must be at least 1")
        if int(self.max_plan_steps) < 1:
            raise ValueError("max_plan_steps must be at least 1")
        if int(self.false_attractor_patience) < 1:
            raise ValueError("false_attractor_patience must be at least 1")
        if int(self.gripper_contact_stall_patience) < 1:
            raise ValueError("gripper_contact_stall_patience must be at least 1")
        if int(self.terminal_stall_patience) < 1:
            raise ValueError("terminal_stall_patience must be at least 1")
        if not -1.0 <= float(self.gripper_close_command_threshold) <= 1.0:
            raise ValueError("gripper_close_command_threshold must be in [-1,1]")
        if self.gripper_command_mode not in {"target", "spatial_progress"}:
            raise ValueError(
                "gripper_command_mode must be 'target' or 'spatial_progress'"
            )
        if self.clf_projection_mode not in {"quadratic", "smooth_tube"}:
            raise ValueError(
                "clf_projection_mode must be 'quadratic' or 'smooth_tube'"
            )
        if (
            float(self.gripper_contact_position_tolerance)
            < float(self.position_tolerance)
        ):
            raise ValueError(
                "gripper_contact_position_tolerance must be at least "
                "position_tolerance"
            )
        if not 0.0 < float(self.future_weight_decay) <= 1.0:
            raise ValueError("future_weight_decay must be in (0,1]")
        if not 0.0 < float(self.past_weight_decay) <= 1.0:
            raise ValueError("past_weight_decay must be in (0,1]")
        if not 0.0 < float(self.intermediate_progress_threshold) <= 1.0:
            raise ValueError("intermediate_progress_threshold must be in (0,1]")
        if not 0.0 < float(self.intermediate_capture_fraction) <= 1.0:
            raise ValueError("intermediate_capture_fraction must be in (0,1]")


@dataclass(frozen=True)
class EndpointFieldStep:
    action: np.ndarray
    target_index: int
    position_error: float
    rotation_error: float
    coverage: float
    plan_complete: bool
    replan_required: bool
    replan_reason: str | None = None
    center_transition_alpha: float = 1.0


def compile_endpoint_field_plan(
    predicted_actions,
    start_position,
    start_quaternion,
    config: EndpointFieldConfig | None = None,
) -> EndpointFieldPlan:
    """Compile predicted local center increments into full-pose field targets."""

    config = config or EndpointFieldConfig()
    actions = np.asarray(predicted_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != FIELD_ACTION_DIM:
        raise ValueError(f"predicted_actions must be (K,{FIELD_ACTION_DIM}), got {actions.shape}")
    if len(actions) == 0 or not np.all(np.isfinite(actions)):
        raise ValueError("predicted_actions must be a non-empty finite array")

    position = _as_float_array(start_position, (3,)).copy()
    orientation = normalize_quaternion(start_quaternion)
    positions = []
    orientations = []

    for command in actions:
        displacement = command[:3] * float(config.position_action_scale)
        position = position + displacement
        delta_quaternion = rotvec_to_quaternion(
            command[3:6] * float(config.rotation_action_scale)
        )
        orientation = quaternion_multiply(delta_quaternion, orientation)
        positions.append(position.copy())
        orientations.append(orientation.copy())

    position_targets = np.asarray(positions, dtype=np.float64)
    position_centers = np.concatenate(
        [_as_float_array(start_position, (3,))[None], position_targets],
        axis=0,
    )
    orientation_targets = np.asarray(orientations, dtype=np.float64)
    orientation_centers = np.concatenate(
        [normalize_quaternion(start_quaternion)[None], orientation_targets],
        axis=0,
    )
    segments = np.diff(position_centers, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    segment_directions = _segment_directions(segments, segment_lengths)
    segment_rotation_vectors = np.stack(
        [
            relative_rotation_vector(current, target)
            for current, target in zip(
                orientation_centers[:-1],
                orientation_centers[1:],
                strict=True,
            )
        ],
        axis=0,
    )
    local_scales = segment_lengths.copy()
    if len(local_scales) > 1:
        local_scales[:-1] = np.maximum(local_scales[:-1], local_scales[1:])
    local_scales = np.maximum(local_scales, float(config.min_sigma))
    segment_precisions = np.stack(
        [
            _anisotropic_precision(
                direction,
                sigma_parallel=max(
                    float(config.min_sigma),
                    float(config.sigma_parallel_scale) * scale,
                ),
                sigma_perpendicular=max(
                    float(config.min_sigma),
                    float(config.sigma_perpendicular_scale) * scale,
                ),
            )
            for direction, scale in zip(
                segment_directions,
                local_scales,
                strict=True,
            )
        ],
        axis=0,
    )

    return EndpointFieldPlan(
        start_position=_as_float_array(start_position, (3,)).copy(),
        start_orientation=normalize_quaternion(start_quaternion),
        position_targets=position_targets,
        orientation_targets=orientation_targets,
        gripper_targets=actions[:, 6].astype(np.float64),
        segment_lengths=segment_lengths,
        segment_directions=segment_directions,
        segment_rotation_vectors=segment_rotation_vectors,
        segment_precisions=segment_precisions,
    )


def _segment_directions(segments: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    directions = np.zeros_like(segments)
    valid = lengths > 1.0e-12
    directions[valid] = segments[valid] / lengths[valid, None]
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for index in range(len(directions)):
        if valid[index]:
            fallback = directions[index]
        else:
            future = np.flatnonzero(valid[index + 1 :])
            directions[index] = (
                directions[index + 1 + int(future[0])]
                if len(future)
                else fallback
            )
    return directions


def _anisotropic_precision(
    tangent: np.ndarray,
    *,
    sigma_parallel: float,
    sigma_perpendicular: float,
) -> np.ndarray:
    direction = np.asarray(tangent, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    parallel = np.outer(direction, direction)
    perpendicular = np.eye(3, dtype=np.float64) - parallel
    return (
        parallel / float(sigma_parallel) ** 2
        + perpendicular / float(sigma_perpendicular) ** 2
    )


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm <= 1.0e-12:
        return vector
    return vector * (float(max_norm) / norm)


class EndpointGatedFieldExecutor:
    """Execute a center chain as a state-progressed segment-expert mixture.

    Every adjacent center pair defines one local SE(3) expert. The position
    component advances along the segment and removes cross-track error; the
    rotation component does the analogous operation on the quaternion geodesic.
    Ordered spatial weights mix the experts into one field. Measured pose, not
    wall-clock time, advances progress.
    """

    def __init__(
        self,
        plan: EndpointFieldPlan,
        config: EndpointFieldConfig | None = None,
    ) -> None:
        self.plan = plan
        self.config = config or EndpointFieldConfig()
        self.target_index = 0
        self.steps_on_target = 0
        self.total_steps = 0
        self.false_attractor_steps = 0
        self.gripper_open_observed = False
        self.continuous_phase = 0.0
        self.last_center_transition_alpha = 1.0

    @property
    def replan_target_index(self) -> int:
        return min(
            int(self.config.replan_after_targets),
            int(self.plan.num_targets),
        ) - 1

    def _errors(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        target_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self.target_index if target_index is None else int(target_index)
        position_error = self.plan.position_targets[index] - position
        rotation_error = relative_rotation_vector(
            quaternion,
            self.plan.orientation_targets[index],
        )
        return position_error, rotation_error

    def _replan_target_reached(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        return bool(
            self.target_index == self.replan_target_index
            and self._target_progress_reached(
                position,
                quaternion,
                gripper_qpos,
            )
        )

    def _target_progress_reached(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        if self.steps_on_target == 0:
            return False

        centers = self.plan.position_centers
        start = centers[self.target_index]
        target = centers[self.target_index + 1]
        segment = target - start
        length = float(self.plan.segment_lengths[self.target_index])
        target_error = float(np.linalg.norm(target - position))
        rotation_error = relative_rotation_vector(
            quaternion,
            self.plan.orientation_targets[self.target_index],
        )
        orientation_reached = bool(
            np.linalg.norm(rotation_error) <= float(self.config.rotation_tolerance)
        )
        if self.target_index == self.replan_target_index:
            position_reached = bool(
                target_error <= float(self.config.position_tolerance)
            )
            return bool(
                orientation_reached
                and (
                    position_reached
                    or self._gripper_contact_progress_reached(
                        gripper_qpos,
                        target_error,
                    )
                )
            )
        capture = max(
            float(self.config.position_tolerance),
            float(self.config.intermediate_capture_fraction) * length,
        )
        if length <= 1.0e-12:
            position_reached = bool(
                target_error <= float(self.config.position_tolerance)
            )
            return bool(
                orientation_reached
                and (
                    position_reached
                    or self._gripper_contact_progress_reached(
                        gripper_qpos,
                        target_error,
                    )
                )
            )

        progress = float(np.dot(position - start, segment) / (length * length))
        projection = start + progress * segment
        lateral_error = float(np.linalg.norm(position - projection))
        position_reached = bool(
            target_error <= capture
            or (
                progress >= float(self.config.intermediate_progress_threshold)
                and lateral_error <= capture
            )
        )
        return bool(
            orientation_reached
            and (
                position_reached
                or self._gripper_contact_progress_reached(
                    gripper_qpos,
                    target_error,
                )
            )
        )

    def _gripper_contact_progress_reached(
        self,
        gripper_qpos: np.ndarray | None,
        target_error: float,
    ) -> bool:
        if not bool(self.config.gripper_contact_progress):
            return False
        if gripper_qpos is None:
            return False
        if (
            float(self.plan.gripper_targets[self.target_index])
            <= float(self.config.gripper_close_command_threshold)
        ):
            return False
        qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if len(qpos) == 0 or not np.all(np.isfinite(qpos)):
            return False
        gripper_closed = bool(
            np.max(np.abs(qpos))
            <= float(self.config.gripper_closed_qpos_threshold)
        )
        return bool(
            gripper_closed
            and target_error
            <= float(self.config.gripper_contact_position_tolerance)
        )

    def _gripper_close_transition_reached(
        self,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        if not bool(self.config.gripper_close_transition_replan):
            return False
        if gripper_qpos is None:
            return False
        qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if len(qpos) == 0 or not np.all(np.isfinite(qpos)):
            return False

        aperture = float(np.max(np.abs(qpos)))
        if aperture > float(self.config.gripper_closed_qpos_threshold):
            self.gripper_open_observed = True
            return False
        if not self.gripper_open_observed:
            return False
        if (
            float(self.plan.gripper_targets[self.target_index])
            <= float(self.config.gripper_close_command_threshold)
        ):
            return False

        self.gripper_open_observed = False
        return True

    def _intermediate_target_reached(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        return bool(
            self.target_index < self.replan_target_index
            and self._target_progress_reached(
                position,
                quaternion,
                gripper_qpos,
            )
        )

    def _advance_progress(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
    ) -> None:
        if self._intermediate_target_reached(
            position,
            quaternion,
            gripper_qpos,
        ):
            self.target_index += 1
            self.steps_on_target = 0

    def _position_segment_progress(
        self,
        position: np.ndarray,
        segment_index: int,
    ) -> tuple[float, bool]:
        centers = self.plan.position_centers
        start = centers[segment_index]
        target = centers[segment_index + 1]
        segment = target - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1.0e-12:
            return 1.0, False
        raw_progress = float(np.dot(position - start, segment) / denominator)
        projection = start + raw_progress * segment
        lateral_error = float(np.linalg.norm(position - projection))
        capture = max(
            float(self.config.position_tolerance),
            float(self.config.intermediate_capture_fraction)
            * float(self.plan.segment_lengths[segment_index]),
        )
        progress = float(np.clip(raw_progress, 0.0, 1.0))
        return (progress if lateral_error <= capture else 0.0), True

    def _rotation_segment_progress(
        self,
        quaternion: np.ndarray,
        segment_index: int,
    ) -> tuple[float, bool]:
        segment_rotation = self.plan.segment_rotation_vectors[segment_index]
        angle_squared = float(np.dot(segment_rotation, segment_rotation))
        if angle_squared <= 1.0e-12:
            return 1.0, False

        start = self.plan.orientation_centers[segment_index]
        from_start = relative_rotation_vector(start, quaternion)
        raw_progress = float(
            np.dot(from_start, segment_rotation) / angle_squared
        )
        progress = float(np.clip(raw_progress, 0.0, 1.0))
        closest = quaternion_multiply(
            rotvec_to_quaternion(progress * segment_rotation),
            start,
        )
        cross_track_error = float(
            np.linalg.norm(relative_rotation_vector(quaternion, closest))
        )
        return (
            progress
            if cross_track_error <= float(self.config.rotation_tolerance)
            else 0.0
        ), True

    def segment_progress(
        self,
        position,
        quaternion=None,
        target_index: int | None = None,
    ) -> float:
        """Return shared geometric progress along one SE(3) segment."""

        index = self.target_index if target_index is None else int(target_index)
        index = min(max(index, 0), self.plan.num_targets - 1)
        measured_position = _as_float_array(position, (3,))
        progress_components: list[float] = []

        position_progress, position_moves = self._position_segment_progress(
            measured_position,
            index,
        )
        if position_moves:
            progress_components.append(position_progress)

        if quaternion is not None:
            rotation_progress, rotation_moves = self._rotation_segment_progress(
                normalize_quaternion(quaternion),
                index,
            )
            if rotation_moves:
                progress_components.append(rotation_progress)

        if not progress_components:
            return 1.0
        return float(min(progress_components))

    def _ordered_gaussian_weights(
        self,
        position: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        measured_position = _as_float_array(position, (3,))
        segment_starts = self.plan.position_centers[:-1]
        offsets = measured_position[None] - segment_starts
        mahalanobis = np.einsum(
            "ki,kij,kj->k",
            offsets,
            self.plan.segment_precisions,
            offsets,
        )
        indices = np.arange(self.plan.num_targets, dtype=np.int64)
        order_distance = indices.astype(np.float64) - float(phase)
        log_weights = -0.5 * mahalanobis
        log_weights += np.maximum(order_distance, 0.0) * np.log(
            float(self.config.future_weight_decay)
        )
        log_weights += np.maximum(-order_distance, 0.0) * np.log(
            float(self.config.past_weight_decay)
        )
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            weights = np.zeros((self.plan.num_targets,), dtype=np.float64)
            weights[self.target_index] = 1.0
            return weights
        maximum = float(np.max(log_weights[finite]))
        weights = np.zeros((self.plan.num_targets,), dtype=np.float64)
        weights[finite] = np.exp(np.maximum(log_weights[finite] - maximum, -80.0))
        return weights / float(np.sum(weights))

    def _mixture_weights_and_phase(
        self,
        position,
        quaternion=None,
        *,
        update_phase: bool,
    ) -> tuple[np.ndarray, float]:
        measured_position = _as_float_array(position, (3,))
        progress = self.segment_progress(
            measured_position,
            quaternion,
            target_index=self.target_index,
        )
        phase = max(
            self.continuous_phase,
            float(self.target_index) + progress,
        )
        phase = min(
            phase,
            float(self.replan_target_index),
        )
        if update_phase:
            self.continuous_phase = phase
            self.last_center_transition_alpha = float(
                phase - np.floor(phase)
            )
        weights = self._ordered_gaussian_weights(
            measured_position,
            phase,
        )
        return weights, phase

    def mixture_weights(self, position, quaternion=None) -> np.ndarray:
        """Return state-progressed Gaussian weights over all segments."""

        weights, _ = self._mixture_weights_and_phase(
            position,
            quaternion,
            update_phase=True,
        )
        return weights

    def _position_segment_field(
        self,
        position: np.ndarray,
        segment_index: int,
    ) -> np.ndarray:
        centers = self.plan.position_centers
        start = centers[segment_index]
        target = centers[segment_index + 1]
        direction = self.plan.segment_directions[segment_index]
        length = float(self.plan.segment_lengths[segment_index])
        if length <= 1.0e-12:
            local = target - position
        else:
            offset = position - start
            parallel_offset = float(np.dot(offset, direction))
            progress = float(np.clip(parallel_offset / length, 0.0, 1.0))
            closest = start + progress * (target - start)
            cross_track_error = closest - position
            remaining = float(np.dot(target - position, direction))
            forward = float(np.clip(remaining, -length, length)) * direction
            local = (
                forward
                + float(self.config.segment_cross_track_gain) * cross_track_error
            )
        return _clip_vector_norm(
            local,
            max(length, float(self.config.min_sigma)),
        )

    def _position_field(self, position: np.ndarray, weights: np.ndarray) -> np.ndarray:
        local_fields = np.stack(
            [
                self._position_segment_field(position, index)
                for index in range(self.plan.num_targets)
            ],
            axis=0,
        )
        return np.sum(weights[:, None] * local_fields, axis=0)

    def _rotation_segment_field(
        self,
        quaternion: np.ndarray,
        segment_index: int,
    ) -> np.ndarray:
        orientations = self.plan.orientation_centers
        start = orientations[segment_index]
        target = orientations[segment_index + 1]
        segment_rotation = self.plan.segment_rotation_vectors[segment_index]
        angle = float(np.linalg.norm(segment_rotation))
        if angle <= 1.0e-12:
            local = relative_rotation_vector(quaternion, target)
        else:
            direction = segment_rotation / angle
            from_start = relative_rotation_vector(start, quaternion)
            progress = float(
                np.clip(np.dot(from_start, segment_rotation) / (angle * angle), 0.0, 1.0)
            )
            closest = quaternion_multiply(
                rotvec_to_quaternion(progress * segment_rotation),
                start,
            )
            cross_track_error = relative_rotation_vector(quaternion, closest)
            target_error = relative_rotation_vector(quaternion, target)
            remaining = float(np.dot(target_error, direction))
            forward = float(np.clip(remaining, -angle, angle)) * direction
            local = (
                forward
                + float(self.config.segment_cross_track_gain) * cross_track_error
            )
        max_rotation = max(angle, float(self.config.rotation_tolerance))
        return _clip_vector_norm(local, max_rotation)

    def _rotation_field(self, quaternion: np.ndarray, weights: np.ndarray) -> np.ndarray:
        local_fields = np.stack(
            [
                self._rotation_segment_field(quaternion, index)
                for index in range(self.plan.num_targets)
            ],
            axis=0,
        )
        return np.sum(weights[:, None] * local_fields, axis=0)

    def _false_attractor_detected(
        self,
        position_delta: np.ndarray,
        rotation_delta: np.ndarray,
        position_error: np.ndarray,
        rotation_error: np.ndarray,
    ) -> bool:
        low_field_speed = bool(
            np.linalg.norm(position_delta)
            <= float(self.config.false_attractor_position_speed_threshold)
            and np.linalg.norm(rotation_delta)
            <= float(self.config.false_attractor_rotation_speed_threshold)
        )
        error_multiplier = float(self.config.false_attractor_error_multiplier)
        far_from_target = bool(
            np.linalg.norm(position_error)
            > error_multiplier * float(self.config.position_tolerance)
            or np.linalg.norm(rotation_error)
            > error_multiplier * float(self.config.rotation_tolerance)
        )
        if low_field_speed and far_from_target:
            self.false_attractor_steps += 1
        else:
            self.false_attractor_steps = 0
        return self.false_attractor_steps >= int(
            self.config.false_attractor_patience
        )

    def _action_from_fields(
        self,
        position_delta: np.ndarray,
        rotation_delta: np.ndarray,
        target_index: int,
        phase: float,
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
        gripper_command = self._gripper_command(
            target_index=target_index,
            phase=phase,
        )
        return np.concatenate(
            [
                position_action,
                rotation_action,
                [gripper_command],
            ]
        ).astype(np.float32)

    def _gripper_command(self, *, target_index: int, phase: float) -> float:
        targets = np.clip(
            np.asarray(self.plan.gripper_targets, dtype=np.float64),
            -1.0,
            1.0,
        )
        if self.config.gripper_command_mode == "target" or len(targets) == 1:
            index = min(max(int(target_index), 0), len(targets) - 1)
            return float(targets[index])

        sample_phases = np.arange(len(targets), dtype=np.float64)
        spatial_phase = float(np.clip(phase, 0.0, sample_phases[-1]))
        return float(np.interp(spatial_phase, sample_phases, targets))

    def preview_action(
        self,
        position,
        quaternion,
    ) -> EndpointFieldStep:
        """Evaluate the current field without advancing executor state."""

        measured_position = _as_float_array(position, (3,))
        measured_quaternion = normalize_quaternion(quaternion)
        weights, phase = self._mixture_weights_and_phase(
            measured_position,
            measured_quaternion,
            update_phase=False,
        )
        position_delta = self._position_field(measured_position, weights)
        rotation_delta = self._rotation_field(measured_quaternion, weights)
        position_error, rotation_error = self._errors(
            measured_position,
            measured_quaternion,
        )
        index = int(self.target_index)
        return EndpointFieldStep(
            action=self._action_from_fields(
                position_delta,
                rotation_delta,
                index,
                phase,
            ),
            target_index=index,
            position_error=float(np.linalg.norm(position_error)),
            rotation_error=float(np.linalg.norm(rotation_error)),
            coverage=float(weights[index]),
            plan_complete=False,
            replan_required=False,
            center_transition_alpha=float(phase - np.floor(phase)),
        )

    def step(
        self,
        position,
        quaternion,
        gripper_qpos=None,
    ) -> EndpointFieldStep:
        measured_position = _as_float_array(position, (3,))
        measured_quaternion = normalize_quaternion(quaternion)
        measured_gripper_qpos = (
            None
            if gripper_qpos is None
            else np.asarray(gripper_qpos, dtype=np.float64)
        )
        if self._gripper_close_transition_reached(measured_gripper_qpos):
            position_error, rotation_error = self._errors(
                measured_position,
                measured_quaternion,
            )
            weights = self.mixture_weights(
                measured_position,
                measured_quaternion,
            )
            return EndpointFieldStep(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                target_index=int(self.target_index),
                position_error=float(np.linalg.norm(position_error)),
                rotation_error=float(np.linalg.norm(rotation_error)),
                coverage=float(weights[self.target_index]),
                plan_complete=True,
                replan_required=False,
                replan_reason="gripper_close_transition",
            )
        if self._replan_target_reached(
            measured_position,
            measured_quaternion,
            measured_gripper_qpos,
        ):
            position_error, rotation_error = self._errors(
                measured_position,
                measured_quaternion,
            )
            weights = self.mixture_weights(
                measured_position,
                measured_quaternion,
            )
            return EndpointFieldStep(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                target_index=self.replan_target_index + 1,
                position_error=float(np.linalg.norm(position_error)),
                rotation_error=float(np.linalg.norm(rotation_error)),
                coverage=float(weights[self.target_index]),
                plan_complete=True,
                replan_required=False,
            )

        if self.total_steps >= int(self.config.max_plan_steps):
            position_error, rotation_error = self._errors(
                measured_position,
                measured_quaternion,
            )
            weights = self.mixture_weights(
                measured_position,
                measured_quaternion,
            )
            return EndpointFieldStep(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                target_index=int(self.target_index),
                position_error=float(np.linalg.norm(position_error)),
                rotation_error=float(np.linalg.norm(rotation_error)),
                coverage=float(weights[self.target_index]),
                plan_complete=False,
                replan_required=True,
                replan_reason="hard_timeout",
            )

        self._advance_progress(
            measured_position,
            measured_quaternion,
            measured_gripper_qpos,
        )
        weights = self.mixture_weights(
            measured_position,
            measured_quaternion,
        )
        position_delta = self._position_field(measured_position, weights)
        rotation_delta = self._rotation_field(measured_quaternion, weights)
        position_error, rotation_error = self._errors(
            measured_position,
            measured_quaternion,
        )
        if self._false_attractor_detected(
            position_delta,
            rotation_delta,
            position_error,
            rotation_error,
        ):
            return EndpointFieldStep(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                target_index=int(self.target_index),
                position_error=float(np.linalg.norm(position_error)),
                rotation_error=float(np.linalg.norm(rotation_error)),
                coverage=float(weights[self.target_index]),
                plan_complete=False,
                replan_required=True,
                replan_reason="false_attractor",
            )

        index = int(self.target_index)
        action = self._action_from_fields(
            position_delta,
            rotation_delta,
            index,
            self.continuous_phase,
        )
        self.steps_on_target += 1
        self.total_steps += 1
        return EndpointFieldStep(
            action=action,
            target_index=index,
            position_error=float(np.linalg.norm(position_error)),
            rotation_error=float(np.linalg.norm(rotation_error)),
            coverage=float(weights[index]),
            plan_complete=False,
            replan_required=False,
            center_transition_alpha=float(self.last_center_transition_alpha),
        )
