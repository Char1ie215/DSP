from __future__ import annotations

import numpy as np

from ns_gaussian_ds_policy.gaussian_center_field import (
    OrderedGaussianCenterFieldExecutor,
    compile_gaussian_center_plan,
    geometry_center_indices,
    interpolate_center_plan,
    make_gaussian_center_label,
)


def test_geometry_centers_stay_on_trajectory_and_keep_corner() -> None:
    trajectory = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]]
    )
    indices = geometry_center_indices(trajectory, max_error=0.1)
    np.testing.assert_array_equal(indices, [0, 2, 4])
    np.testing.assert_allclose(trajectory[indices], [[0, 0], [2, 0], [2, 2]])


def test_geometry_centers_enforce_maximum_temporal_gap() -> None:
    trajectory = np.stack((np.arange(10), np.zeros(10)), axis=1)
    indices = geometry_center_indices(trajectory, max_error=0.1, max_gap=3)
    np.testing.assert_array_equal(indices, [0, 3, 6, 9])


def test_label_contains_direct_centers_and_delta_times() -> None:
    trajectory = np.stack((np.arange(8), np.arange(8) * 2), axis=1)
    label = make_gaussian_center_label(
        trajectory,
        np.array([0, 2, 5, 7]),
        timestep=1,
        num_centers=4,
    )
    np.testing.assert_allclose(label[:, 0], [1, 1, 3, 2])
    np.testing.assert_allclose(
        label[:, 1:],
        trajectory[[1, 2, 5, 7]],
    )


def test_center_plan_uses_predicted_timing_without_spline() -> None:
    command = np.array([[2.0, 2.0, 0.0], [3.0, 2.0, 3.0]])
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    np.testing.assert_allclose(plan.centers, [[0, 0], [2, 0], [2, 3]])
    np.testing.assert_allclose(plan.center_times, [0, 2, 5])
    np.testing.assert_allclose(plan.segment_speeds, [1, 1])
    np.testing.assert_allclose(
        plan.segment_velocities,
        [[1.0, 0.0], [0.0, 1.0]],
    )
    np.testing.assert_allclose(interpolate_center_plan(plan, 3.5), [2, 1.5])


def test_field_corrects_cross_track_error() -> None:
    command = np.array([[2.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_support_scale=0.1,
        cross_track_gain=1.0,
    )
    velocity = executor.vector_field([0.5, 1.0])
    assert velocity[0] > 0.0
    assert velocity[1] < 0.0


def test_field_mixes_every_segment_with_normalized_weights() -> None:
    command = np.array(
        [[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]
    )
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(plan, phase_support_scale=0.5)
    weights = executor.mixture_weights([0.5, 0.1], phase=0.75)
    assert weights.shape == (plan.num_segments,)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0)


def test_field_velocity_is_continuous_across_center_time() -> None:
    command = np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]
    )
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(plan, phase_support_scale=0.6)
    before = executor.vector_field([1.0, 0.0], phase=1.0 - 1.0e-6)
    after = executor.vector_field([1.0, 0.0], phase=1.0 + 1.0e-6)
    assert np.linalg.norm(after - before) < 1.0e-5


def test_phase_disambiguates_self_intersection_direction() -> None:
    command = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, -1.0],
        ]
    )
    plan = compile_gaussian_center_plan(command, start_position=[-1.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_support_scale=0.25,
        spatial_weight_floor=0.1,
        cross_track_gain=0.0,
    )
    early = executor.vector_field([0.0, 0.0], phase=1.0)
    late = executor.vector_field([0.0, 0.0], phase=3.0)
    assert early[1] > 0.0
    assert late[1] < 0.0


def test_tracking_error_smoothly_slows_phase_progress() -> None:
    command = np.array([[2.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    on_path = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_tracking_scale=0.5,
        min_phase_rate=0.1,
    )
    off_path = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_tracking_scale=0.5,
        min_phase_rate=0.1,
    )
    on_path.phase = 1.0
    off_path.phase = 1.0
    on_path_step = on_path.step([1.0, 0.0])
    off_path_step = off_path.step([0.0, 0.0])
    assert on_path_step.phase > off_path_step.phase
    assert 1.1 <= off_path_step.phase < 2.0


def test_segment_field_points_to_next_center_at_timed_speed() -> None:
    command = np.array([[4.0, 8.0, 0.0], [2.0, 8.0, 4.0]])
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_support_scale=0.05,
        cross_track_gain=0.2,
    )
    velocity = executor.vector_field([0.0, 0.0], phase=0.0)
    np.testing.assert_allclose(velocity, [2.0, 0.0], atol=1.0e-5)


def test_nominal_field_preserves_center_timing() -> None:
    command = np.array(
        [[2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [2.0, 4.0, 2.0]]
    )
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_support_scale=0.05,
        cross_track_gain=0.2,
    )
    position = np.array([0.0, 0.0])
    rollout = []
    for _ in range(6):
        step = executor.step(position)
        position = step.action
        rollout.append(position.copy())
    expected = np.stack(
        [interpolate_center_plan(plan, phase) for phase in range(1, 7)]
    )
    np.testing.assert_allclose(rollout, expected, atol=1.0e-4)


def test_field_recovers_longitudinal_lag_without_changing_nominal_timing() -> None:
    command = np.array([[2.0, 2.0, 0.0], [2.0, 4.0, 0.0]])
    plan = compile_gaussian_center_plan(command, start_position=[0.0, 0.0])
    executor = OrderedGaussianCenterFieldExecutor(
        plan,
        phase_support_scale=0.05,
        cross_track_gain=0.0,
        reference_tracking_gain=1.0,
        min_phase_rate=1.0,
    )
    executor.phase = 1.0
    step = executor.step([0.5, 0.0])
    np.testing.assert_allclose(step.action, [2.0, 0.0], atol=1.0e-5)
    np.testing.assert_allclose(step.phase, 2.0)
