from __future__ import annotations

import numpy as np

from ns_gaussian_ds_policy.endpoint_field import (
    EndpointFieldConfig,
    EndpointGatedFieldExecutor,
    build_action_integrated_relative_center_chain,
    build_command_endpoint_field_labels,
    build_endpoint_field_labels,
    compile_endpoint_field_plan,
    quaternion_multiply,
    relative_rotation_vector,
    relative_center_chain_to_incremental_actions,
    rotvec_to_quaternion,
    split_command_endpoint_field_actions,
)


IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])


def test_action_window_builds_cumulative_centers_from_current_observation() -> None:
    actions = np.zeros((5, 7), dtype=np.float64)
    actions[1, 0] = 0.2
    actions[1, 5] = 0.2
    actions[2, 1] = 0.4
    actions[2, 5] = 0.2
    actions[:, 6] = [-1.0, -1.0, 1.0, 1.0, 1.0]

    centers = build_action_integrated_relative_center_chain(
        actions,
        n_obs_steps=2,
        source_indices=[10, 11, 12, 12, 12],
    )

    np.testing.assert_allclose(centers[0, :6], 0.0)
    np.testing.assert_allclose(centers[1, :3], [0.2, 0.0, 0.0])
    np.testing.assert_allclose(centers[2, :3], [0.2, 0.4, 0.0])
    np.testing.assert_allclose(centers[3:, :3], [[0.2, 0.4, 0.0]] * 2)
    np.testing.assert_allclose(centers[1:3, 5], [0.2, 0.4], atol=1.0e-7)
    np.testing.assert_allclose(centers[:, 6], actions[:, 6])


def test_relative_center_chain_differences_back_to_source_motion() -> None:
    actions = np.zeros((4, 7), dtype=np.float64)
    actions[:, :3] = [
        [0.2, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [-0.1, 0.0, 0.0],
        [0.0, 0.0, 0.3],
    ]
    actions[:, 3:6] = [
        [0.0, 0.0, 0.2],
        [0.0, 0.0, 0.2],
        [0.1, 0.0, 0.0],
        [0.0, -0.1, 0.0],
    ]
    actions[:, 6] = [-1.0, -1.0, 1.0, 1.0]
    centers = build_action_integrated_relative_center_chain(
        actions,
        n_obs_steps=1,
    )

    reconstructed = relative_center_chain_to_incremental_actions(centers)

    np.testing.assert_allclose(reconstructed, actions, atol=1.0e-6)
    plan = compile_endpoint_field_plan(
        reconstructed,
        [0.4, -0.2, 0.1],
        IDENTITY_QUATERNION,
    )
    np.testing.assert_allclose(
        plan.position_targets - plan.start_position,
        centers[:, :3] * 0.05,
        atol=1.0e-7,
    )


def test_eight_selected_delta_rows_compile_to_eight_future_pose_centers() -> None:
    positions = np.stack(
        [np.arange(12, dtype=np.float64) * 0.01, np.zeros(12), np.zeros(12)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (12, 1))
    raw_actions = np.zeros((12, 7), dtype=np.float64)
    labels, _ = build_endpoint_field_labels(
        positions,
        quaternions,
        raw_actions,
        [12],
    )

    # Diffusion Policy uses start=n_obs_steps-1=1 and returns eight rows.
    plan = compile_endpoint_field_plan(
        labels[1:9],
        positions[1],
        quaternions[1],
    )
    np.testing.assert_allclose(plan.position_targets, positions[2:10])


def test_command_field_labels_keep_geometry_and_source_command_separate() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    quaternions = np.tile(IDENTITY_QUATERNION, (2, 1))
    raw_actions = np.zeros((2, 12), dtype=np.float64)
    raw_actions[:, 0] = [0.6, 0.4]
    raw_actions[:, 6] = [-1.0, 1.0]
    raw_actions[:, 7:] = [0.0, 0.0, 0.0, 0.0, -1.0]

    labels, targets = build_command_endpoint_field_labels(
        positions,
        quaternions,
        raw_actions,
        [2],
    )
    field, command, auxiliary = split_command_endpoint_field_actions(
        labels,
        environment_action_dim=12,
    )

    assert labels.shape == (2, 18)
    np.testing.assert_array_equal(targets, [1, 1])
    np.testing.assert_allclose(field[:, 0], [0.2, 0.0])
    np.testing.assert_allclose(field[:, 6], [-1.0, 1.0])
    np.testing.assert_allclose(command[:, 0], [0.6, 0.4])
    np.testing.assert_allclose(auxiliary, raw_actions[:, 7:])


def test_rotation_delta_round_trip_uses_world_frame() -> None:
    start = rotvec_to_quaternion([0.2, -0.1, 0.0])
    delta = np.array([0.0, 0.3, -0.2])
    target = quaternion_multiply(rotvec_to_quaternion(delta), start)
    np.testing.assert_allclose(relative_rotation_vector(start, target), delta, atol=1.0e-7)


def test_compiler_accumulates_position_and_rotation_centers() -> None:
    config = EndpointFieldConfig(position_action_scale=0.05, rotation_action_scale=0.5)
    commands = np.array(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0],
            [0.0, 0.4, 0.0, 0.0, 0.0, 0.2, 1.0],
        ]
    )
    plan = compile_endpoint_field_plan(
        commands,
        [0.5, 0.0, 0.2],
        IDENTITY_QUATERNION,
        config,
    )
    np.testing.assert_allclose(
        plan.position_targets,
        [[0.51, 0.0, 0.2], [0.51, 0.02, 0.2]],
    )
    np.testing.assert_allclose(
        relative_rotation_vector(IDENTITY_QUATERNION, plan.orientation_targets[-1]),
        [0.0, 0.0, 0.2],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(plan.gripper_targets, [-1.0, 1.0])
    np.testing.assert_allclose(plan.segment_lengths, [0.01, 0.02])
    np.testing.assert_allclose(plan.segment_directions, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(plan.segment_rotation_vectors, [[0.0, 0.0, 0.1]] * 2)
    assert np.all(np.linalg.eigvalsh(plan.segment_precisions) > 0.0)


def _apply_ideal_action(
    position: np.ndarray,
    quaternion: np.ndarray,
    action: np.ndarray,
    config: EndpointFieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    next_position = (
        position + action[:3] * float(config.position_action_scale)
    )
    next_quaternion = quaternion_multiply(
        rotvec_to_quaternion(
            action[3:6] * float(config.rotation_action_scale)
        ),
        quaternion,
    )
    return next_position, next_quaternion


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    distances = []
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        ratio = (
            0.0
            if denominator <= 1.0e-12
            else float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
        )
        distances.append(float(np.linalg.norm(point - (start + ratio * segment))))
    return min(distances)


def test_all_unfinished_centers_participate_in_ordered_gaussian_mixture() -> None:
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)
    weights = executor.mixture_weights([0.0, 0.0, 0.0])

    assert weights.shape == (8,)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_local_segment_expert_advances_and_removes_cross_track_error() -> None:
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[0, 0] = 0.2
    commands[1, 1] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)

    local = executor._position_segment_field(
        np.array([0.003, 0.004, 0.0]),
        segment_index=0,
    )

    assert local[0] > 0.0
    assert local[1] < 0.0
    np.testing.assert_allclose(local[2], 0.0)


def test_mixed_corner_field_is_dominated_by_the_outgoing_segment() -> None:
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[0, 0] = 0.2
    commands[1, 1] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)

    step = executor.preview_action([0.0, 0.0, 0.0], IDENTITY_QUATERNION)

    assert step.action[0] > 0.0
    assert abs(float(step.action[1])) < 0.25 * float(step.action[0])


def test_rotation_segment_expert_advances_and_recovers_axis_error() -> None:
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 5] = 0.4
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)
    perturbed = quaternion_multiply(
        rotvec_to_quaternion([0.1, 0.0, 0.0]),
        IDENTITY_QUATERNION,
    )

    local = executor._rotation_segment_field(perturbed, segment_index=0)

    assert local[0] < 0.0
    assert local[2] > 0.0


def test_shared_se3_phase_waits_for_rotation_only_segment() -> None:
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[0, 5] = 0.4
    commands[1, 0] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)

    weights = executor.mixture_weights(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )

    assert executor.continuous_phase == 0.0
    assert weights[0] > weights[1]


def test_preview_action_does_not_mutate_segment_phase() -> None:
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    executor = EndpointGatedFieldExecutor(plan)

    preview = executor.preview_action(
        [0.009, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )

    assert preview.center_transition_alpha > 0.0
    assert executor.continuous_phase == 0.0


def test_spatial_progress_continuously_interpolates_gripper_command() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        gripper_close_transition_replan=False,
    )
    commands = np.zeros((3, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = [-1.0, 1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    step = executor.step([0.005, 0.0, 0.0], IDENTITY_QUATERNION)

    np.testing.assert_allclose(executor.continuous_phase, 0.5)
    np.testing.assert_allclose(step.action[6], 0.0, atol=1.0e-7)


def test_spatial_gripper_progress_freezes_outside_trajectory_tube() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        gripper_close_transition_replan=False,
    )
    commands = np.zeros((3, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = [-1.0, 1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    on_path = executor.step([0.005, 0.0, 0.0], IDENTITY_QUATERNION)
    off_path = executor.step([0.009, 0.02, 0.0], IDENTITY_QUATERNION)

    np.testing.assert_allclose(executor.continuous_phase, 0.5)
    np.testing.assert_allclose(off_path.action[6], on_path.action[6])


def test_spatial_gripper_progress_does_not_reverse_on_backward_motion() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        gripper_close_transition_replan=False,
    )
    commands = np.zeros((3, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = [-1.0, 1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    forward = executor.step([0.007, 0.0, 0.0], IDENTITY_QUATERNION)
    backward = executor.step([0.002, 0.0, 0.0], IDENTITY_QUATERNION)

    np.testing.assert_allclose(executor.continuous_phase, 0.7)
    np.testing.assert_allclose(backward.action[6], forward.action[6])


def test_oracle_eight_center_field_reconstructs_corner_in_eight_steps() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=8,
        max_plan_steps=200,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:4, 0] = 0.2
    commands[4:, 1] = 0.2
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    position = np.zeros(3, dtype=np.float64)
    quaternion = IDENTITY_QUATERNION.copy()
    rollout = [position.copy()]

    for _ in range(8):
        step = executor.step(position, quaternion)
        assert not step.plan_complete
        assert not step.replan_required
        position, quaternion = _apply_ideal_action(
            position,
            quaternion,
            step.action,
            config,
        )
        rollout.append(position.copy())

    done = executor.step(position, quaternion)
    assert done.plan_complete
    assert np.linalg.norm(position - plan.position_targets[-1]) <= config.position_tolerance
    assert max(
        _distance_to_polyline(point, plan.position_centers)
        for point in rollout
    ) < 5.0e-4


def test_executor_does_not_advance_when_command_is_blocked() -> None:
    config = EndpointFieldConfig(
        position_tolerance=0.001,
        rotation_tolerance=0.01,
        max_plan_steps=4,
    )
    plan = compile_endpoint_field_plan(
        [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    first = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    second = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert first.target_index == second.target_index == 0
    np.testing.assert_allclose(first.action, second.action)
    assert not first.plan_complete


def test_closed_gripper_contact_can_complete_nearby_unreachable_target() -> None:
    config = EndpointFieldConfig(
        position_tolerance=0.002,
        rotation_tolerance=0.01,
        gripper_contact_position_tolerance=0.005,
    )
    plan = compile_endpoint_field_plan(
        [[0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    blocked_position = np.array([0.0, 0.0, -0.002])

    executor.step(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        [0.04, -0.04],
    )
    still_open = executor.step(
        blocked_position,
        IDENTITY_QUATERNION,
        [0.04, -0.04],
    )
    assert not still_open.plan_complete

    contact = executor.step(
        blocked_position,
        IDENTITY_QUATERNION,
        [0.02, -0.02],
    )
    assert contact.plan_complete


def test_gripper_close_transition_replans_before_eighth_center() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=8,
        gripper_contact_position_tolerance=0.005,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = -1.0
    commands[3:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    position = np.zeros(3, dtype=np.float64)
    for target_index in range(3):
        executor.step(position, IDENTITY_QUATERNION, [0.04, -0.04])
        position = plan.position_targets[target_index].copy()
    close = executor.step(position, IDENTITY_QUATERNION, [0.04, -0.04])
    assert close.target_index == 3
    assert close.action[6] == 1.0

    grasped = executor.step(position, IDENTITY_QUATERNION, [0.02, -0.02])
    assert grasped.plan_complete
    assert not grasped.replan_required
    assert grasped.replan_reason == "gripper_close_transition"
    assert executor.target_index == 3
    assert np.linalg.norm(position - plan.position_targets[3]) > 0.005


def test_plan_starting_closed_does_not_immediately_replan() -> None:
    config = EndpointFieldConfig(replan_after_targets=8)
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    first = executor.step(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        [0.02, -0.02],
    )
    assert not first.plan_complete
    assert first.action[6] == 1.0


def test_intermediate_center_advances_but_only_final_center_completes_plan() -> None:
    config = EndpointFieldConfig(position_tolerance=0.001, rotation_tolerance=0.01)
    plan = compile_endpoint_field_plan(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    first = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert first.target_index == 0

    second = executor.step([0.01, 0.0, 0.0], IDENTITY_QUATERNION)
    assert second.target_index == 1
    assert not second.plan_complete
    assert second.action[1] > 0.0

    done = executor.step([0.01, 0.01, 0.0], IDENTITY_QUATERNION)
    assert done.plan_complete


def test_rotation_error_blocks_phase_advance_and_is_recovered() -> None:
    config = EndpointFieldConfig(position_tolerance=0.001, rotation_tolerance=0.01)
    commands = np.array(
        [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0],
        ],
        dtype=np.float64,
    )
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)

    blocked = executor.step([0.01, 0.0, 0.0], IDENTITY_QUATERNION)
    assert blocked.target_index == 0
    assert blocked.action[5] > 0.0

    aligned = executor.step(
        [0.01, 0.0, 0.0],
        plan.orientation_targets[0],
    )
    assert aligned.target_index == 1


def test_lateral_perturbation_is_recovered_without_policy_replan() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=8,
        max_plan_steps=200,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    position = np.zeros(3, dtype=np.float64)
    quaternion = IDENTITY_QUATERNION.copy()

    for _ in range(2):
        step = executor.step(position, quaternion)
        position, quaternion = _apply_ideal_action(
            position,
            quaternion,
            step.action,
            config,
        )
    position[1] += 0.02
    recovery = executor.step(position, quaternion)
    assert recovery.action[1] < 0.0
    assert not recovery.replan_required

    for _ in range(20):
        position, quaternion = _apply_ideal_action(
            position,
            quaternion,
            recovery.action,
            config,
        )
        recovery = executor.step(position, quaternion)
        if recovery.plan_complete:
            break
        assert not recovery.replan_required
    assert recovery.plan_complete
    assert np.linalg.norm(position - plan.position_targets[-1]) <= config.position_tolerance


def test_executor_requests_replan_after_whole_plan_timeout() -> None:
    config = EndpointFieldConfig(max_plan_steps=2)
    plan = compile_endpoint_field_plan(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    assert not executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION).replan_required
    assert not executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION).replan_required
    timed_out = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert timed_out.replan_required
    assert timed_out.replan_reason == "hard_timeout"
    assert timed_out.target_index == 0


def test_executor_replans_only_inside_final_center_tolerance() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=3,
        max_plan_steps=200,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)
    position = np.zeros(3, dtype=np.float64)
    quaternion = IDENTITY_QUATERNION.copy()

    for _ in range(2):
        step = executor.step(position, quaternion)
        assert not step.plan_complete
        assert not step.replan_required
        position, quaternion = _apply_ideal_action(
            position,
            quaternion,
            step.action,
            config,
        )

    third = executor.step(position, quaternion)
    assert not third.plan_complete
    position = plan.position_targets[2] - np.array(
        [0.5 * config.position_tolerance, 0.0, 0.0]
    )
    done = executor.step(position, quaternion)
    assert done.plan_complete
    assert not done.replan_required
    assert done.target_index == 3
    assert np.linalg.norm(position - plan.position_targets[2]) <= config.position_tolerance


def test_external_hold_keeps_the_same_field_until_hard_timeout() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=3,
        max_plan_steps=200,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.4
    plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    for _ in range(100):
        held = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
        assert not held.plan_complete
        assert not held.replan_required
        assert held.action[0] > 0.0

    assert executor.target_index == 0
    assert executor.false_attractor_steps == 0


def test_false_attractor_replans_only_after_persistent_low_field_speed() -> None:
    config = EndpointFieldConfig(
        max_plan_steps=200,
        false_attractor_patience=3,
        false_attractor_position_speed_threshold=1.0,
        false_attractor_rotation_speed_threshold=1.0,
    )
    plan = compile_endpoint_field_plan(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    for _ in range(2):
        step = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
        assert not step.replan_required

    stalled = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert stalled.replan_required
    assert stalled.replan_reason == "false_attractor"
    np.testing.assert_allclose(stalled.action, 0.0)


def test_field_speed_scale_changes_execution_not_centers_or_gripper() -> None:
    commands = np.array(
        [[0.2, 0.0, 0.0, 0.0, 0.0, 0.2, -1.0]],
        dtype=np.float64,
    )
    base_config = EndpointFieldConfig(field_speed_scale=1.0)
    fast_config = EndpointFieldConfig(field_speed_scale=1.5)
    base_plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        base_config,
    )
    fast_plan = compile_endpoint_field_plan(
        commands,
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        fast_config,
    )

    np.testing.assert_allclose(fast_plan.position_targets, base_plan.position_targets)
    np.testing.assert_allclose(
        fast_plan.orientation_targets,
        base_plan.orientation_targets,
    )
    base_step = EndpointGatedFieldExecutor(base_plan, base_config).step(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )
    fast_step = EndpointGatedFieldExecutor(fast_plan, fast_config).step(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(fast_step.action[:6], 1.5 * base_step.action[:6])
    assert fast_step.action[6] == base_step.action[6] == -1.0


def test_field_speed_scale_preserves_action_limits() -> None:
    config = EndpointFieldConfig(field_speed_scale=2.0)
    plan = compile_endpoint_field_plan(
        [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    step = EndpointGatedFieldExecutor(plan, config).step(
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[[0, 3]], 1.0)
    assert np.all(np.abs(step.action[:3]) <= config.max_position_action)
    assert np.all(np.abs(step.action[3:6]) <= config.max_rotation_action)


def test_stationary_targets_still_issue_gripper_commands() -> None:
    config = EndpointFieldConfig(position_tolerance=0.001, rotation_tolerance=0.01)
    plan = compile_endpoint_field_plan(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [0.0, 0.0, 0.0],
        IDENTITY_QUATERNION,
        config,
    )
    executor = EndpointGatedFieldExecutor(plan, config)

    close = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert close.target_index == 0
    assert close.action[6] == -1.0
    assert not close.plan_complete

    open_gripper = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert open_gripper.target_index == 1
    assert open_gripper.action[6] == 1.0
    assert not open_gripper.plan_complete

    done = executor.step([0.0, 0.0, 0.0], IDENTITY_QUATERNION)
    assert done.plan_complete
