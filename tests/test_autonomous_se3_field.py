from __future__ import annotations

import numpy as np

from ns_gaussian_ds_policy.autonomous_se3_field import (
    AutonomousSE3FieldExecutor,
    ControllerAwareSE3FieldExecutor,
    PhaseConditionedPoseActionFieldExecutor,
    PoseIndexedActionFieldExecutor,
    _calibrate_affine_intercepts,
    _pairwise_relative_rotation_vectors,
    partition_se3_center_chain,
)
from ns_gaussian_ds_policy.endpoint_field import (
    EndpointFieldConfig,
    compile_endpoint_field_plan,
    quaternion_multiply,
    relative_rotation_vector,
    rotvec_to_quaternion,
)


IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])


def test_support_calibration_truncates_near_singular_gating_modes() -> None:
    epsilon = 1.0e-12
    gating = np.array(
        [
            [0.5 + epsilon, 0.5 - epsilon],
            [0.5 - epsilon, 0.5 + epsilon],
        ]
    )
    offsets = np.zeros((2, 2, 3), dtype=np.float64)
    matrices = np.zeros((2, 3, 3), dtype=np.float64)
    targets = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])

    intercepts, max_error, condition, rank = _calibrate_affine_intercepts(
        gating,
        offsets,
        matrices,
        targets,
        rcond=1.0e-3,
    )

    assert condition > 1.0e10
    assert rank == 1
    assert np.max(np.abs(intercepts)) < 1.0
    assert max_error <= 0.2 + 1.0e-12


def test_pairwise_relative_rotation_vectors_match_scalar_reference() -> None:
    supports = np.stack(
        [
            IDENTITY_QUATERNION,
            rotvec_to_quaternion([0.2, -0.1, 0.3]),
            -rotvec_to_quaternion([-0.4, 0.2, 0.1]),
        ]
    )
    targets = np.stack(
        [
            rotvec_to_quaternion([0.1, 0.2, -0.3]),
            -rotvec_to_quaternion([0.5, -0.2, 0.4]),
        ]
    )

    actual = _pairwise_relative_rotation_vectors(supports, targets)
    expected = np.asarray(
        [
            [relative_rotation_vector(support, target) for support in supports]
            for target in targets
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def _corner_commands() -> np.ndarray:
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:4, 0] = 0.1
    commands[4:, 1] = 0.1
    commands[:, 5] = 0.04
    commands[:, 6] = -1.0
    return commands


def _integrate_action(
    position: np.ndarray,
    quaternion: np.ndarray,
    action: np.ndarray,
    config: EndpointFieldConfig,
) -> tuple[np.ndarray, np.ndarray]:
    position = (
        position
        + np.asarray(action[:3], dtype=np.float64)
        * float(config.position_action_scale)
    )
    quaternion = quaternion_multiply(
        rotvec_to_quaternion(
            np.asarray(action[3:6], dtype=np.float64)
            * float(config.rotation_action_scale)
        ),
        quaternion,
    )
    return position, quaternion


def test_autonomous_pose_field_is_independent_of_progress_history() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    position = np.array([0.006, 0.003, 0.0])
    quaternion = rotvec_to_quaternion([0.0, 0.0, 0.03])

    first = executor.preview_action(position, quaternion).action[:6]
    executor.target_index = 7
    executor.gripper_phase = 7.0
    executor.continuous_phase = 7.0
    second = executor.preview_action(position, quaternion).action[:6]

    np.testing.assert_allclose(first, second, atol=1.0e-7)


def test_autonomous_se3_experts_remain_strictly_contractive() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    assert executor.chunk_ranges == ((0, 8),)
    for chunk in executor.chunks:
        for matrices in (chunk.position_matrices, chunk.rotation_matrices):
            symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
            assert np.max(np.linalg.eigvalsh(symmetric)) <= -0.1 + 1.0e-8


def test_command_progression_changes_speed_without_changing_center_geometry() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    commands = np.zeros((plan.num_targets, 6), dtype=np.float64)
    commands[:4, 0] = 0.4
    commands[4:, 1] = 0.4
    geometry_executor = AutonomousSE3FieldExecutor(plan, config)
    command_executor = AutonomousSE3FieldExecutor(
        plan,
        config,
        nominal_commands=commands,
    )

    geometry_action = geometry_executor.preview_action(
        np.zeros(3),
        IDENTITY_QUATERNION,
    ).action
    command_action = command_executor.preview_action(
        np.zeros(3),
        IDENTITY_QUATERNION,
    ).action

    np.testing.assert_allclose(
        command_executor.plan.position_targets,
        geometry_executor.plan.position_targets,
    )
    assert command_executor.compile_metadata["progression_source"] == "command_projected"
    assert command_action[0] > geometry_action[0]
    assert command_action[0] > 0.0


def test_controller_aware_field_preserves_policy_actions_on_reference() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = _corner_commands()
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = ControllerAwareSE3FieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )

    for index in range(plan.num_targets):
        action = executor.preview_action(
            plan.position_centers[index],
            plan.orientation_centers[index],
        ).action
        np.testing.assert_allclose(
            action[:6],
            commands[index, :6],
            atol=1.0e-6,
        )

    assert executor.compile_metadata["on_path_action"] == (
        "raw_policy_command"
    )


def test_controller_aware_field_suppresses_progress_and_recovers_off_path() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = _corner_commands()
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = ControllerAwareSE3FieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )
    displaced = np.array([0.0, 0.05, 0.0])

    action = executor.preview_action(
        displaced,
        IDENTITY_QUATERNION,
    ).action

    assert executor.last_feedforward_weight < 1.0e-3
    assert action[1] < -0.9
    assert abs(float(action[0])) < 1.0e-3


def test_controller_aware_field_does_not_skip_short_first_command() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[0, 0] = 0.02
    commands[1:, 1] = 0.4
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    assert plan.segment_lengths[0] < config.position_tolerance
    executor = ControllerAwareSE3FieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )

    action = executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
    ).action

    np.testing.assert_allclose(action[:2], [0.02, 0.0], atol=1.0e-6)


def test_pose_indexed_action_field_exactly_interpolates_support_actions() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = _corner_commands()
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PoseIndexedActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )

    for index in range(plan.num_targets):
        action = executor.preview_action(
            plan.position_centers[index],
            plan.orientation_centers[index],
        ).action
        np.testing.assert_allclose(
            action[:6],
            commands[index, :6],
            atol=1.0e-6,
        )

    assert executor.compile_metadata["action_binding"] == "pre_action_pose"
    assert executor.compile_metadata["progression_source"] == (
        "autonomous_pose_gating"
    )


def test_pose_indexed_action_field_recovers_without_suppressing_progress() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PoseIndexedActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )

    action = executor.preview_action(
        np.array([0.0, 0.05, 0.0]),
        IDENTITY_QUATERNION,
    ).action

    assert action[0] > 0.1
    assert action[1] < -0.9


def test_pose_indexed_action_field_repeats_blocked_contact_command() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[-1, 2] = -0.4
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PoseIndexedActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )
    blocked_pose = plan.position_centers[-2]
    blocked_orientation = plan.orientation_centers[-2]

    first = executor.preview_action(blocked_pose, blocked_orientation).action
    second = executor.preview_action(blocked_pose, blocked_orientation).action

    np.testing.assert_allclose(first[:6], commands[-1, :6], atol=1.0e-6)
    np.testing.assert_allclose(second[:6], commands[-1, :6], atol=1.0e-6)


def test_pose_indexed_field_issues_terminal_gripper_then_replans() -> None:
    config = EndpointFieldConfig(
        gripper_contact_progress=False,
        gripper_close_transition_replan=False,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PoseIndexedActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
        terminal_gripper_command=1.0,
    )

    first = executor.step(
        plan.position_centers[0],
        plan.orientation_centers[0],
    )
    second = executor.step(
        plan.position_centers[1],
        plan.orientation_centers[1],
    )
    gripper = executor.step(
        plan.position_centers[2],
        plan.orientation_centers[2],
    )
    completed = executor.step(
        plan.position_centers[2],
        plan.orientation_centers[2],
    )

    np.testing.assert_allclose(first.action[:6], commands[0, :6])
    np.testing.assert_allclose(second.action[:6], commands[1, :6])
    np.testing.assert_allclose(gripper.action[:6], 0.0)
    assert gripper.action[6] == 1.0
    assert not gripper.plan_complete
    assert completed.plan_complete
    assert completed.replan_reason == "gripper_only_transition"


def test_phase_conditioned_field_preserves_ordered_action_and_gripper_sequence() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = np.linspace(0.05, 0.4, 8)
    commands[:, 5] = np.linspace(-0.2, 0.2, 8)
    commands[:3, 6] = -1.0
    commands[3:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
    )

    # A contact-constrained arm can remain at an earlier point on the pose
    # curve. Ordered action phase must still reach the close command.
    emitted = np.stack(
        [
            executor.step(np.zeros(3), IDENTITY_QUATERNION).action
            for _ in range(8)
        ]
    )

    np.testing.assert_allclose(emitted, commands, atol=1.0e-7)
    assert executor.action_index == 8
    completed = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    assert completed.plan_complete
    assert completed.replan_reason == "ordered_action_horizon"


def test_phase_gated_field_stride_preserves_original_eight_step_horizon() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        replan_after_targets=24,
    )
    source = np.zeros((8, 7), dtype=np.float64)
    source[:, 0] = np.linspace(0.05, 0.4, 8)
    source[:, 6] = -1.0
    dense_geometry = np.repeat(source, 3, axis=0)
    dense_geometry[:, :6] /= 3.0
    dense_commands = np.repeat(source[:, :6], 3, axis=0)
    plan = compile_endpoint_field_plan(
        dense_geometry,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=dense_commands,
        config=config,
        nominal_execution="gaussian_field",
        phase_stride=3,
    )

    emitted = []
    for source_index in range(8):
        support_index = 3 * source_index
        emitted.append(
            executor.step(
                plan.position_centers[support_index],
                plan.orientation_centers[support_index],
            ).action
        )

    np.testing.assert_allclose(
        np.stack(emitted)[:, :6],
        source[:, :6],
        atol=1.0e-7,
    )
    assert executor.action_index == 24
    completed = executor.step(
        plan.position_centers[-1],
        plan.orientation_centers[-1],
    )
    assert completed.plan_complete
    assert completed.replan_reason == "ordered_action_horizon"


def test_phase_conditioned_field_freezes_only_outside_pose_tube() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
        position_tube_radius=0.05,
    )

    nominal = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    np.testing.assert_allclose(nominal.action, commands[0], atol=1.0e-7)
    assert executor.action_index == 1

    displaced = np.array([0.0, 0.2, 0.0])
    recovery = executor.step(displaced, IDENTITY_QUATERNION)
    assert executor.last_phase_frozen
    assert executor.action_index == 1
    assert recovery.action[1] < -0.5
    assert recovery.action[6] == commands[0, 6]

    resumed = executor.step(
        plan.position_centers[1],
        plan.orientation_centers[1],
    )
    assert not executor.last_phase_frozen
    np.testing.assert_allclose(resumed.action, commands[1], atol=1.0e-7)
    assert executor.action_index == 2


def test_phase_gated_field_uses_field_inside_tube_and_freezes_phase_outside() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[0, 6] = -1.0
    commands[1:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
        position_tube_radius=0.05,
        nominal_execution="gaussian_field",
    )

    inside_tube = np.array([0.0, 0.02, 0.0])
    nominal = executor.step(inside_tube, IDENTITY_QUATERNION)

    assert not executor.last_phase_frozen
    assert executor.action_index == 1
    assert nominal.action[1] < -0.1
    assert nominal.action[6] == -1.0
    assert executor.compile_metadata["nominal_execution"] == (
        "cached_gaussian_pose_action_field"
    )

    outside_tube = np.array([0.0, 0.2, 0.0])
    recovery = executor.step(outside_tube, IDENTITY_QUATERNION)

    assert executor.last_phase_frozen
    assert executor.action_index == 1
    assert recovery.action[1] < -0.5
    assert recovery.action[6] == 1.0

    resumed = executor.step(inside_tube, IDENTITY_QUATERNION)
    assert not executor.last_phase_frozen
    assert executor.action_index == 2
    assert resumed.action[6] == 1.0


def test_phase_conditioned_recovery_uses_current_phase_gripper_command() -> None:
    config = EndpointFieldConfig(gripper_command_mode="spatial_progress")
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[0, 6] = -1.0
    commands[1:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
        position_tube_radius=0.05,
    )

    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    recovery = executor.step(
        np.array([0.0, 0.2, 0.0]),
        IDENTITY_QUATERNION,
    )

    assert executor.last_phase_frozen
    assert executor.action_index == 1
    assert recovery.action[6] == 1.0


def test_phase_conditioned_terminal_gripper_command_replans_after_one_step() -> None:
    config = EndpointFieldConfig(
        gripper_contact_progress=False,
        gripper_close_transition_replan=False,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.2
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = PhaseConditionedPoseActionFieldExecutor(
        plan,
        controller_actions=commands[:, :6],
        config=config,
        terminal_gripper_command=1.0,
    )

    first = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    second = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    gripper = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    completed = executor.step(np.zeros(3), IDENTITY_QUATERNION)

    np.testing.assert_allclose(first.action, commands[0], atol=1.0e-7)
    np.testing.assert_allclose(second.action, commands[1], atol=1.0e-7)
    np.testing.assert_allclose(gripper.action[:6], 0.0, atol=1.0e-7)
    assert gripper.action[6] == 1.0
    assert not gripper.plan_complete
    assert completed.plan_complete
    assert completed.replan_reason == "gripper_only_transition"


def test_clf_projection_is_inactive_inside_trajectory_tube() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        clf_projection_enabled=True,
        clf_position_tube_radius=0.01,
    )
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    position = np.array([0.002, 0.005, 0.0])
    original = np.array([0.001, 0.002, 0.0])

    projected = executor._clf_project_position_delta(position, original)

    np.testing.assert_allclose(projected, original)
    assert not executor.last_clf_applied


def test_clf_projection_corrects_only_insufficient_return_outside_tube() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        clf_projection_enabled=True,
        clf_position_tube_radius=0.01,
        clf_position_contraction_rate=0.1,
    )
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    position = np.array([0.0, 0.05, 0.0])
    closest = executor._closest_chunk_position(position)
    offset = position - closest
    outward = np.array([0.0, 0.01, 0.0])

    projected = executor._clf_project_position_delta(position, outward)

    assert executor.last_clf_applied
    assert np.dot(offset, projected) <= (
        -config.clf_position_contraction_rate * np.dot(offset, offset)
        + 1.0e-12
    )

    inward = -0.2 * offset
    unchanged = executor._clf_project_position_delta(position, inward)
    np.testing.assert_allclose(unchanged, inward)
    assert not executor.last_clf_applied


def test_smooth_tube_projection_is_continuous_at_boundary() -> None:
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        clf_projection_enabled=True,
        clf_projection_mode="smooth_tube",
        clf_position_tube_radius=0.01,
        clf_position_contraction_rate=0.1,
    )
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    outward = np.array([0.0, 0.01, 0.0])

    boundary = np.array([0.0, 0.01, 0.0])
    np.testing.assert_allclose(
        executor._clf_project_position_delta(boundary, outward),
        outward,
    )
    outside = np.array([0.0, 0.03, 0.0])
    closest = executor._closest_chunk_position(outside)
    offset = outside - closest
    projected = executor._clf_project_position_delta(outside, outward)

    required = (
        config.clf_position_contraction_rate
        * (np.linalg.norm(offset) - config.clf_position_tube_radius)
        * np.linalg.norm(offset)
    )
    assert np.dot(offset, projected) <= -required + 1.0e-12
    assert executor.last_clf_applied


def test_direction_preserving_saturation_uses_uniform_vector_scale() -> None:
    commands = np.zeros((1, 7), dtype=np.float64)
    config = EndpointFieldConfig(
        preserve_field_direction_on_saturation=True,
        max_position_action=1.0,
        position_action_scale=0.05,
    )
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    action = executor._field_action(
        np.array([0.1, 0.05, 0.0]),
        np.zeros(3),
        -1.0,
    )

    np.testing.assert_allclose(action[:3], [1.0, 0.5, 0.0])


def test_autonomous_se3_field_reconstructs_corner_without_phase() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        max_plan_steps=100,
    )
    plan = compile_endpoint_field_plan(
        _corner_commands(),
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    position = np.zeros(3)
    quaternion = IDENTITY_QUATERNION.copy()

    for _ in range(100):
        step = executor.step(
            position,
            quaternion,
            np.array([0.04, -0.04]),
        )
        if step.plan_complete:
            break
        assert not step.replan_required
        position, quaternion = _integrate_action(
            position,
            quaternion,
            step.action,
            config,
        )
    else:
        raise AssertionError("Autonomous SE(3) field did not complete")

    assert np.linalg.norm(position - plan.position_targets[-1]) <= 0.002


def test_se3_self_intersection_is_split_into_autonomous_chunks() -> None:
    config = EndpointFieldConfig(replan_after_targets=4)
    commands = np.zeros((4, 7), dtype=np.float64)
    points = np.array(
        [
            [0.01, 0.01, 0.0],
            [0.0, 0.01, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    centers = np.concatenate([np.zeros((1, 3)), points], axis=0)
    commands[:, :3] = np.diff(centers, axis=0) / config.position_action_scale
    commands[:, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )

    chunks = partition_se3_center_chain(plan, target_count=4)

    assert len(chunks) >= 2
    assert chunks[0][0] == 0
    assert chunks[-1][1] == 4
    assert chunks[0][1] == chunks[1][0]


def test_monotonic_lift_is_not_split_by_local_bridge_proximity() -> None:
    config = EndpointFieldConfig(replan_after_targets=8)
    increments = np.array(
        [
            [-0.00100, 0.00074, 0.00069],
            [-0.00093, 0.00046, 0.00191],
            [-0.00077, 0.00031, 0.00270],
            [-0.00042, 0.00024, 0.00454],
            [0.00017, 0.00037, 0.00742],
            [0.00027, 0.00067, 0.01036],
            [0.00009, 0.00074, 0.01153],
            [0.00046, 0.00052, 0.01139],
        ],
        dtype=np.float64,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, :3] = increments / config.position_action_scale
    commands[:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )

    chunks = partition_se3_center_chain(plan, target_count=8)

    assert np.all(np.diff(plan.position_centers[:, 2]) > 0.0)
    assert chunks == ((0, 8),)


def test_short_local_curve_is_not_treated_as_a_nonlocal_revisit() -> None:
    config = EndpointFieldConfig(replan_after_targets=8)
    centers = np.array(
        [
            [0.0115, -0.0004, 0.8391],
            [0.0110, 0.0006, 0.8388],
            [0.0106, 0.0016, 0.8390],
            [0.0100, 0.0022, 0.8396],
            [0.0094, 0.0027, 0.8408],
            [0.0086, 0.0035, 0.8426],
            [0.0080, 0.0045, 0.8451],
            [0.0075, 0.0059, 0.8487],
            [0.0069, 0.0074, 0.8549],
        ],
        dtype=np.float64,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, :3] = np.diff(centers, axis=0) / config.position_action_scale
    commands[:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        centers[0],
        IDENTITY_QUATERNION,
        config,
    )

    chunks = partition_se3_center_chain(plan, target_count=8)

    assert chunks == ((0, 8),)


def test_actual_gripper_closure_immediately_ends_autonomous_plan() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        gripper_close_transition_replan=True,
    )
    commands = _corner_commands()
    commands[:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    first = executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        np.array([0.04, -0.04]),
    )
    closed = executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        np.array([0.02, -0.02]),
    )

    assert not first.plan_complete
    assert closed.plan_complete
    assert not closed.replan_required
    assert closed.replan_reason == "gripper_close_transition"


def test_position_progress_closes_gripper_while_rotation_lags() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=2,
        position_tolerance=0.002,
        rotation_tolerance=0.05,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[:, 5] = 0.2
    commands[:, 6] = [-1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    position = np.array([0.0075, 0.0, 0.0])

    lagging = executor.step(position, IDENTITY_QUATERNION)
    assert np.isclose(executor.gripper_phase, 1.0)
    assert executor.last_geometric_on_path
    assert np.isclose(lagging.action[6], 1.0)


def test_terminal_pose_holds_until_final_gripper_command_is_measured() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        position_tolerance=0.002,
        rotation_tolerance=0.01,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.1
    commands[0, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])
    closed_qpos = np.array([0.02, -0.02])

    executor.step(np.zeros(3), IDENTITY_QUATERNION, open_qpos)
    waiting = executor.step(
        plan.position_targets[-1],
        plan.orientation_targets[-1],
        open_qpos,
    )
    closed = executor.step(
        plan.position_targets[-1],
        plan.orientation_targets[-1],
        closed_qpos,
    )

    assert not waiting.plan_complete
    np.testing.assert_allclose(waiting.action[:6], 0.0, atol=1.0e-8)
    assert np.isclose(waiting.action[6], 1.0)
    assert closed.plan_complete
    assert closed.replan_reason == "gripper_close_transition"


def test_blocked_gripper_contact_completes_nearby_autonomous_target() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        position_tolerance=0.002,
        rotation_tolerance=0.01,
        gripper_contact_position_tolerance=0.005,
        gripper_contact_stall_patience=3,
        gripper_contact_min_aperture_delta=0.002,
        gripper_contact_aperture_step_threshold=1.0e-4,
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.1
    commands[0, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])
    moving_qpos = np.array([0.034, -0.034])
    blocked_qpos = moving_qpos.copy()

    first = executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        open_qpos,
    )
    moving = executor.step(
        np.array([0.001, 0.0, 0.0]),
        IDENTITY_QUATERNION,
        moving_qpos,
    )
    settling = [
        executor.step(
            np.array([0.001, 0.0, 0.0]),
            IDENTITY_QUATERNION,
            blocked_qpos,
        )
        for _ in range(3)
    ]
    contact = settling[-1]

    assert not first.plan_complete
    assert not moving.plan_complete
    assert all(not step.plan_complete for step in settling[:-1])
    assert contact.plan_complete
    assert not contact.replan_required
    assert contact.replan_reason == "gripper_contact_progress"
    assert np.isclose(contact.position_error, 0.004)
    assert np.isclose(executor.last_gripper_closure_delta, 0.006)


def test_blocked_gripper_contact_completes_at_intermediate_close_event() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=8,
        gripper_command_mode="spatial_progress",
        gripper_contact_stall_patience=3,
        gripper_contact_min_aperture_delta=0.002,
        gripper_contact_aperture_step_threshold=1.0e-4,
    )
    commands = np.zeros((8, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[:, 6] = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])
    moving_qpos = np.array([0.034, -0.034])
    close_position = plan.position_centers[2]

    executor.step(np.zeros(3), IDENTITY_QUATERNION, open_qpos)
    closing = executor.step(close_position, IDENTITY_QUATERNION, open_qpos)
    executor.step(close_position, IDENTITY_QUATERNION, moving_qpos)
    settling = [
        executor.step(close_position, IDENTITY_QUATERNION, moving_qpos)
        for _ in range(3)
    ]

    assert np.isclose(closing.action[6], 1.0)
    assert settling[-1].plan_complete
    assert settling[-1].replan_reason == "gripper_contact_progress"
    assert settling[-1].position_error > config.gripper_contact_position_tolerance
    assert executor.gripper_execution_state == executor._GRIPPER_TRACKING_CLOSED


def test_plan_starting_closed_does_not_repeat_close_transition() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=2,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((2, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[:, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    first = executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        np.array([0.02, -0.02]),
    )

    assert not first.plan_complete
    assert first.replan_reason is None
    assert executor.gripper_execution_state == executor._GRIPPER_TRACKING_CLOSED


def test_active_closure_preserves_continuous_interpolated_command() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=3,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((3, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[:, 6] = [-1.0, 1.0, 1.0]
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])

    executor.step(np.zeros(3), IDENTITY_QUATERNION, open_qpos)
    closing = executor.step(
        np.array([0.004, 0.0, 0.0]),
        IDENTITY_QUATERNION,
        open_qpos,
    )
    still_closing = executor.step(
        np.array([0.0045, 0.0, 0.0]),
        IDENTITY_QUATERNION,
        open_qpos,
    )

    assert np.isclose(closing.action[6], 0.6)
    assert np.isclose(still_closing.action[6], 0.8)
    assert executor.gripper_execution_state == executor._GRIPPER_CLOSING


def test_close_event_cancels_if_interpolation_returns_to_open() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=3,
        gripper_command_mode="spatial_progress",
    )
    commands = np.zeros((3, 7), dtype=np.float64)
    commands[:, 0] = 0.1
    commands[:, 6] = [-1.0, 1.0, -1.0]
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])

    executor.step(np.zeros(3), IDENTITY_QUATERNION, open_qpos)
    closing = executor.step(
        plan.position_centers[1],
        IDENTITY_QUATERNION,
        open_qpos,
    )
    opening = executor.step(
        plan.position_centers[2],
        IDENTITY_QUATERNION,
        open_qpos,
    )

    assert np.isclose(closing.action[6], 1.0)
    assert np.isclose(opening.action[6], -1.0)
    assert executor.gripper_execution_state == executor._GRIPPER_TRACKING_OPEN


def test_stationary_open_gripper_is_not_misclassified_as_contact() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        gripper_contact_position_tolerance=0.005,
        gripper_contact_stall_patience=2,
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.1
    commands[0, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    unchanged_qpos = np.array([0.04, -0.04])

    steps = [
        executor.step(
            np.array([0.001, 0.0, 0.0]),
            IDENTITY_QUATERNION,
            unchanged_qpos,
        )
        for _ in range(4)
    ]

    assert all(not step.plan_complete for step in steps)
    assert executor.last_gripper_closure_delta == 0.0


def test_contact_tolerance_does_not_loosen_open_gripper_precision() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        position_tolerance=0.002,
        rotation_tolerance=0.01,
        gripper_contact_position_tolerance=0.005,
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.1
    commands[0, 6] = 1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)
    open_qpos = np.array([0.04, -0.04])

    executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        open_qpos,
    )
    near_target = executor.step(
        np.array([0.001, 0.0, 0.0]),
        IDENTITY_QUATERNION,
        open_qpos,
    )

    assert not near_target.plan_complete


def test_terminal_stall_replans_when_active_field_is_blocked_near_goal() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        max_plan_steps=100,
        terminal_stall_radius=0.02,
        terminal_stall_patience=3,
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.2
    commands[0, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    steps = [
        executor.step(np.zeros(3), IDENTITY_QUATERNION)
        for _ in range(4)
    ]

    assert all(not step.replan_required for step in steps[:3])
    assert steps[-1].replan_required
    assert steps[-1].replan_reason == "terminal_stall"
    assert executor.terminal_stall_steps == 3
    assert executor.last_actual_position_step == 0.0
    assert executor.last_commanded_position_step >= 0.001


def test_terminal_stall_does_not_interrupt_recovery_far_from_goal() -> None:
    config = EndpointFieldConfig(
        replan_after_targets=1,
        max_plan_steps=100,
        terminal_stall_radius=0.02,
        terminal_stall_patience=2,
    )
    commands = np.zeros((1, 7), dtype=np.float64)
    commands[0, 0] = 0.8
    commands[0, 6] = -1.0
    plan = compile_endpoint_field_plan(
        commands,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = AutonomousSE3FieldExecutor(plan, config)

    steps = [
        executor.step(np.zeros(3), IDENTITY_QUATERNION)
        for _ in range(5)
    ]

    assert all(not step.replan_required for step in steps)
    assert executor.terminal_stall_steps == 0
