from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def geometry_center_indices(
    points: Any,
    *,
    max_error: float,
    max_gap: int | None = None,
) -> np.ndarray:
    """Select on-trajectory centers with a deterministic geometric error bound."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 1:
        raise ValueError(f"Expected a non-empty [time, dim] trajectory, got {points.shape}.")
    max_error = float(max_error)
    if max_error < 0.0:
        raise ValueError("max_error must be non-negative.")
    if max_gap is not None and int(max_gap) < 1:
        raise ValueError("max_gap must be positive when provided.")
    if len(points) == 1:
        return np.array([0], dtype=np.int64)

    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        denominator = float(np.dot(segment, segment))
        candidates = points[start + 1 : end]
        if denominator <= 1.0e-12:
            distances = np.linalg.norm(candidates - points[start], axis=1)
        else:
            ratios = np.clip(
                ((candidates - points[start]) @ segment) / denominator,
                0.0,
                1.0,
            )
            projections = points[start] + ratios[:, None] * segment
            distances = np.linalg.norm(candidates - projections, axis=1)
        offset = int(np.argmax(distances))
        if float(distances[offset]) > max_error:
            split = start + 1 + offset
            keep[split] = True
            stack.extend(((start, split), (split, end)))

    indices = np.flatnonzero(keep)
    if max_gap is not None:
        gap = int(max_gap)
        additions: list[int] = []
        for start, end in zip(indices[:-1], indices[1:]):
            additions.extend(range(int(start) + gap, int(end), gap))
        if additions:
            indices = np.unique(np.concatenate((indices, additions))).astype(np.int64)
    return indices


def make_gaussian_center_label(
    trajectory: Any,
    center_indices: Any,
    *,
    timestep: int,
    num_centers: int,
) -> np.ndarray:
    """Encode future on-trajectory centers as rows of [delta_time, position]."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    center_indices = np.asarray(center_indices, dtype=np.int64).reshape(-1)
    timestep = int(timestep)
    num_centers = int(num_centers)
    if trajectory.ndim != 2 or len(trajectory) < 1:
        raise ValueError(f"Expected trajectory [time, dim], got {trajectory.shape}.")
    if not 0 <= timestep < len(trajectory):
        raise IndexError(f"timestep {timestep} is outside trajectory length {len(trajectory)}.")
    if num_centers < 1:
        raise ValueError("num_centers must be positive.")
    if len(center_indices) < 1 or center_indices[0] != 0:
        raise ValueError("center_indices must include the trajectory start.")

    future = center_indices[center_indices > timestep]
    selected = np.concatenate((np.array([timestep], dtype=np.int64), future))
    selected = selected[:num_centers]
    if len(selected) < num_centers:
        selected = np.pad(selected, (0, num_centers - len(selected)), mode="edge")

    previous = np.concatenate((np.array([timestep - 1], dtype=np.int64), selected[:-1]))
    delta_times = np.maximum(selected - previous, 1).astype(np.float64)
    label = np.empty((num_centers, trajectory.shape[1] + 1), dtype=np.float32)
    label[:, 0] = delta_times
    label[:, 1:] = trajectory[selected].astype(np.float32)
    return label


@dataclass(frozen=True)
class GaussianCenterPlan:
    """A predicted center/timing command compiled into phase-conditioned fields."""

    centers: np.ndarray
    center_times: np.ndarray
    segment_durations: np.ndarray
    segment_lengths: np.ndarray
    segment_directions: np.ndarray
    segment_velocities: np.ndarray
    segment_speeds: np.ndarray
    segment_sigma_parallel: np.ndarray
    segment_sigma_perp: np.ndarray

    @property
    def num_segments(self) -> int:
        return int(len(self.segment_lengths))

    @property
    def end_time(self) -> float:
        return float(self.center_times[-1])


def compile_gaussian_center_plan(
    command: Any,
    *,
    start_position: Any,
    min_duration: float = 0.25,
    max_duration: float = 40.0,
    support_scale: float = 1.25,
    cross_track_support_scale: float = 0.5,
    support_floor: float = 1.0,
    center_min: float | None = None,
    center_max: float | None = None,
) -> GaussianCenterPlan:
    """Compile direct [delta_time, center] predictions without a spline."""

    command = np.asarray(command, dtype=np.float64)
    start_position = np.asarray(start_position, dtype=np.float64).reshape(-1)
    if command.ndim != 2 or command.shape[1] != len(start_position) + 1:
        raise ValueError(
            "Expected command [num_centers, 1 + position_dim], got "
            f"{command.shape} for start position {start_position.shape}."
        )
    if len(command) < 1:
        raise ValueError("At least one predicted center is required.")

    durations = np.nan_to_num(
        command[:, 0],
        nan=min_duration,
        posinf=max_duration,
        neginf=min_duration,
    )
    durations = np.clip(durations, float(min_duration), float(max_duration))
    predicted_centers = np.nan_to_num(command[:, 1:], nan=0.0)
    if center_min is not None or center_max is not None:
        lower = -np.inf if center_min is None else float(center_min)
        upper = np.inf if center_max is None else float(center_max)
        predicted_centers = np.clip(predicted_centers, lower, upper)

    centers = np.concatenate((start_position[None], predicted_centers), axis=0)
    center_times = np.concatenate((np.array([0.0]), np.cumsum(durations)))
    vectors = np.diff(centers, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    segment_velocities = vectors / durations[:, None]
    directions = np.zeros_like(vectors)
    fallback = np.zeros_like(start_position)
    fallback[0] = 1.0
    for index, vector in enumerate(vectors):
        norm = float(lengths[index])
        if norm > 1.0e-10:
            fallback = vector / norm
        directions[index] = fallback

    length_scales = np.maximum(lengths, float(support_floor))

    return GaussianCenterPlan(
        centers=centers,
        center_times=center_times,
        segment_durations=durations,
        segment_lengths=lengths,
        segment_directions=directions,
        segment_velocities=segment_velocities,
        segment_speeds=lengths / durations,
        segment_sigma_parallel=np.maximum(
            float(support_scale) * length_scales,
            support_floor,
        ),
        segment_sigma_perp=np.maximum(
            float(cross_track_support_scale) * length_scales,
            support_floor,
        ),
    )


@dataclass(frozen=True)
class GaussianCenterFieldStep:
    action: np.ndarray
    velocity: np.ndarray
    phase: float
    active_segment: int
    done: bool


class OrderedGaussianCenterFieldExecutor:
    """Mix ordered next-center attractors into a phase-conditioned field."""

    def __init__(
        self,
        plan: GaussianCenterPlan,
        *,
        phase_support_scale: float = 0.1,
        spatial_weight_floor: float = 0.1,
        cross_track_gain: float = 0.2,
        reference_tracking_gain: float = 1.0,
        terminal_gain: float = 0.35,
        phase_tracking_scale: float = 1.0,
        min_phase_rate: float = 1.0,
        capture_fraction: float = 0.12,
        capture_floor: float = 0.5,
        max_speed: float | None = None,
        lookahead: int | None = None,
        active_weight_floor: float | None = None,
    ) -> None:
        if plan.num_segments < 1:
            raise ValueError("A field plan requires at least one segment.")
        if phase_support_scale <= 0.0:
            raise ValueError("phase_support_scale must be positive.")
        if phase_tracking_scale <= 0.0:
            raise ValueError("phase_tracking_scale must be positive.")
        self.plan = plan
        # Legacy configs remain loadable, but lookahead no longer truncates the
        # mixture: every outgoing center segment participates in every query.
        del lookahead
        if active_weight_floor is not None:
            spatial_weight_floor = float(active_weight_floor)
        self.phase_support_scale = float(phase_support_scale)
        self.spatial_weight_floor = float(np.clip(spatial_weight_floor, 0.0, 1.0))
        self.cross_track_gain = float(max(cross_track_gain, 0.0))
        self.reference_tracking_gain = float(max(reference_tracking_gain, 0.0))
        self.terminal_gain = float(max(terminal_gain, 0.0))
        self.phase_tracking_scale = float(phase_tracking_scale)
        self.min_phase_rate = float(np.clip(min_phase_rate, 0.0, 1.0))
        self.capture_fraction = float(np.clip(capture_fraction, 0.0, 0.49))
        self.capture_floor = float(max(capture_floor, 0.0))
        self.max_speed = None if max_speed is None else float(max(max_speed, 0.0))
        self.reset()

    def reset(self) -> None:
        self.phase = 0.0

    def _phase_weights(self, phase: float) -> np.ndarray:
        # A command issued at integer phase s belongs to the segment whose
        # discrete interval contains s. Transitions are centered half a control
        # step between adjacent segments, so nominal integer-step execution does
        # not average an incoming and outgoing corner velocity prematurely.
        starts = self.plan.center_times[:-1] - 0.5
        ends = self.plan.center_times[1:] - 0.5
        query = float(phase)
        distance = np.maximum(np.maximum(starts - query, query - ends), 0.0)
        normalized = distance / max(self.phase_support_scale, 1.0e-6)
        return np.exp(np.maximum(-0.5 * np.square(normalized), -80.0))

    def _segment_vector(
        self,
        position: np.ndarray,
        index: int,
    ) -> tuple[np.ndarray, float, float]:
        start = self.plan.centers[index]
        target = self.plan.centers[index + 1]
        direction = self.plan.segment_directions[index]
        length = float(self.plan.segment_lengths[index])
        offset = position - start
        parallel_offset = float(np.dot(offset, direction))
        perpendicular = offset - parallel_offset * direction

        if length <= 1.0e-10:
            local = self.terminal_gain * (target - position)
            cross_error = target - position
        else:
            progress = float(np.clip(parallel_offset / length, 0.0, 1.0))
            closest = start + progress * (target - start)
            cross_error = closest - position
            remaining = float(np.dot(target - position, direction))
            nominal_speed = float(self.plan.segment_speeds[index])
            forward_speed = float(
                np.clip(remaining, -nominal_speed, nominal_speed)
            )
            local = forward_speed * direction
            local = local + self.cross_track_gain * cross_error

        exponent = -0.5 * (
            (parallel_offset / self.plan.segment_sigma_parallel[index]) ** 2
            + (
                np.linalg.norm(perpendicular)
                / self.plan.segment_sigma_perp[index]
            )
            ** 2
        )
        return (
            local,
            float(np.exp(max(float(exponent), -80.0))),
            float(np.linalg.norm(cross_error)),
        )

    def mixture_weights(self, position: Any, phase: float | None = None) -> np.ndarray:
        """Return normalized all-segment expert weights for diagnostics."""

        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.shape != self.plan.centers.shape[1:]:
            raise ValueError(
                f"Expected position shape {self.plan.centers.shape[1:]}, got {position.shape}."
            )
        query_phase = self.phase if phase is None else float(phase)
        phase_weights = self._phase_weights(query_phase)
        spatial_weights = np.empty(self.plan.num_segments, dtype=np.float64)
        for index in range(self.plan.num_segments):
            _, spatial, _ = self._segment_vector(position, index)
            spatial_weights[index] = spatial
        spatial_weights = self.spatial_weight_floor + (
            1.0 - self.spatial_weight_floor
        ) * spatial_weights
        weights = phase_weights * spatial_weights
        total = float(np.sum(weights))
        if total <= 1.0e-12:
            return phase_weights / max(float(np.sum(phase_weights)), 1.0e-12)
        return weights / total

    def vector_field(self, position: Any, phase: float | None = None) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.shape != self.plan.centers.shape[1:]:
            raise ValueError(
                f"Expected position shape {self.plan.centers.shape[1:]}, got {position.shape}."
            )
        query_phase = self.phase if phase is None else float(phase)
        weights = self.mixture_weights(position, phase=query_phase)
        vector = np.zeros_like(position)
        for index, weight in enumerate(weights):
            local, _, _ = self._segment_vector(position, index)
            vector += float(weight) * local
        # Preserve the policy's predicted timing in closed loop. The local
        # experts provide the nominal motion direction and spatial recovery;
        # this phase-reference term only becomes nonzero when the measured
        # agent lags or is displaced from the nominal center chain.
        reference = interpolate_center_plan(self.plan, query_phase)
        vector += self.reference_tracking_gain * (reference - position)
        if self.max_speed is not None:
            speed = float(np.linalg.norm(vector))
            if speed > self.max_speed:
                vector *= self.max_speed / max(speed, 1.0e-12)
        return vector

    def _phase_rate(self, position: np.ndarray, phase: float) -> float:
        reference = interpolate_center_plan(self.plan, phase)
        tracking_error = float(np.linalg.norm(reference - position))
        index = self._segment_at_phase(phase)
        support = (
            self.phase_tracking_scale
            * float(self.plan.segment_sigma_perp[index])
        )
        recovery_gate = np.exp(
            -0.5 * (tracking_error / max(support, 1.0e-6)) ** 2
        )
        return self.min_phase_rate + (1.0 - self.min_phase_rate) * float(recovery_gate)

    def _segment_at_phase(self, phase: float) -> int:
        index = int(np.searchsorted(self.plan.center_times, phase, side="right") - 1)
        return int(np.clip(index, 0, self.plan.num_segments - 1))

    def step(self, position: Any, dt: float = 1.0) -> GaussianCenterFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        phase_rate = self._phase_rate(position, self.phase)
        velocity = self.vector_field(position, phase=self.phase)
        action = position + dt * velocity
        self.phase = min(self.phase + dt * phase_rate, self.plan.end_time)
        index = self._segment_at_phase(self.phase)
        goal_error = float(np.linalg.norm(action - self.plan.centers[-1]))
        final_length = float(self.plan.segment_lengths[-1])
        tolerance = max(
            self.capture_floor,
            self.capture_fraction * max(final_length, self.capture_floor),
        )
        done = self.phase >= self.plan.end_time - 1.0e-8 and goal_error <= tolerance
        return GaussianCenterFieldStep(
            action=action,
            velocity=velocity,
            phase=float(self.phase),
            active_segment=index,
            done=bool(done),
        )


def interpolate_center_plan(plan: GaussianCenterPlan, phase: float) -> np.ndarray:
    """Open-loop piecewise-linear baseline for the same centers and timing."""

    phase = float(np.clip(phase, 0.0, plan.end_time))
    index = int(np.searchsorted(plan.center_times, phase, side="right") - 1)
    index = int(np.clip(index, 0, plan.num_segments - 1))
    start_time = float(plan.center_times[index])
    ratio = (phase - start_time) / float(plan.segment_durations[index])
    return plan.centers[index] + np.clip(ratio, 0.0, 1.0) * (
        plan.centers[index + 1] - plan.centers[index]
    )
