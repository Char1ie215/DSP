from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ns_gaussian_ds_policy.autonomous_se3_field import _pose_at_phase
from ns_gaussian_ds_policy.endpoint_field import (
    EndpointFieldConfig,
    compile_endpoint_field_plan,
    quaternion_multiply,
    relative_rotation_vector,
    rotvec_to_quaternion,
)
from ns_gaussian_ds_policy.libero_hybrid_ds import (
    LiberoPairedBSplineHybridDSAdapter,
    LiberoPairedPoseActionHybridDSAdapter,
    LiberoPairedPoseBSplineRawActionHybridDSAdapter,
    ProjectedPhaseHybridExecutor,
)
from ns_gaussian_ds_policy.libero_paired_curve import (
    LiberoPairedBSplinePoseMedoidPolicy,
    PairedPoseEnsembleSupport,
    build_libero_paired_pose_bspline_raw_action_labels,
)


IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])
ROOT = Path(__file__).resolve().parents[1]


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def get_policy_name(self):
        return "fake"


class _FakeSamplePolicy(_FakePolicy):
    def __init__(self, controls: np.ndarray) -> None:
        super().__init__()
        self.controls = torch.as_tensor(controls, dtype=torch.float32)
        self.predict_calls = 0

    def reset(self):
        return None

    def get_observation_ports(self):
        return (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        )

    def get_observation_modalities(self):
        return {}

    def get_observation_encoder(self):
        return None

    def predict_action(self, obs_dict, **kwargs):
        del kwargs
        self.predict_calls += 1
        batch_size = int(next(iter(obs_dict.values())).shape[0])
        return {"action": self.controls[:batch_size].clone()}


def _observations(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "robot0_eef_pos": torch.zeros(
            (batch_size, 1, 3),
            dtype=torch.float32,
        ),
        "robot0_eef_quat": torch.as_tensor(
            np.tile(IDENTITY_QUATERNION, (batch_size, 1, 1)),
            dtype=torch.float32,
        ),
        "robot0_gripper_qpos": torch.as_tensor(
            np.tile([[0.04, -0.04]], (batch_size, 1, 1)),
            dtype=torch.float32,
        ),
    }


def test_raw_action_hybrid_uses_exact_commands_at_native_phases() -> None:
    steps = 24
    positions = np.stack(
        [
            np.arange(steps, dtype=np.float64) * 0.001,
            np.zeros(steps),
            np.zeros(steps),
        ],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (steps, 1))
    actions = np.zeros((steps, 7), dtype=np.float32)
    actions[:, :6] = np.arange(steps * 6, dtype=np.float32).reshape(
        steps, 6
    ) / 200.0
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_pose_bspline_raw_action_labels(
        positions,
        quaternions,
        actions,
        [steps],
        source_horizon=16,
        num_control_points=16,
    )
    adapter = LiberoPairedPoseBSplineRawActionHybridDSAdapter(
        _FakeSamplePolicy(controls[:1]),
        source_horizon=16,
        plan_steps=8,
        support_subdivisions=2,
        terminal_stall_replan=False,
    )
    decoded = adapter._decode_prediction(torch.from_numpy(controls[:1]))
    executor = adapter._compile_executor(
        decoded.numpy()[0],
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )

    assert isinstance(executor, ProjectedPhaseHybridExecutor)
    assert executor.phase_stride == 2
    np.testing.assert_allclose(
        executor.controller_actions[::2],
        actions[:8, :6],
        atol=1.0e-7,
    )
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "paired_pose_bspline_raw_action_hybrid_ds"
    assert metrics["action_representation"] == "exact_native_steps"


def test_raw_pose_action_hybrid_uses_both_sequences_without_decoding() -> None:
    paired = np.zeros((1, 8, 13), dtype=np.float32)
    paired[0, :, 0] = 0.02
    paired[0, :, 6:12] = np.arange(48, dtype=np.float32).reshape(8, 6) / 50.0
    paired[0, :, 12] = -1.0
    adapter = LiberoPairedPoseActionHybridDSAdapter(
        _FakeSamplePolicy(paired),
        source_horizon=16,
        plan_steps=8,
        support_subdivisions=1,
        terminal_stall_replan=False,
        require_state_innovation=False,
    )

    decoded = adapter._decode_prediction(torch.from_numpy(paired))
    np.testing.assert_array_equal(decoded.numpy(), paired)
    executor = adapter._compile_executor(
        decoded.numpy()[0],
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )

    assert isinstance(executor, ProjectedPhaseHybridExecutor)
    assert executor.phase_stride == 1
    assert executor.require_state_innovation is False
    np.testing.assert_array_equal(executor.controller_actions, paired[0, :, 6:12])
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "paired_pose_action_hybrid_ds"
    assert metrics["pose_representation"] == (
        "measured_native_timestep_increments"
    )
    assert metrics["action_representation"] == "exact_native_steps"
    assert metrics["require_state_innovation"] is False


def _straight_executor(
    *,
    outer_radius: float = 0.005,
    reentry_radius: float = 0.001,
    max_recovery_steps: int = 16,
    recovery_field_source: str = "action",
    recovery_target_mode: str = "projected",
    recovery_anchor_mode: str = "predicted_pose",
    observed_anchor_reentry_radius: float = 0.01,
    observed_anchor_stable_steps: int = 2,
    controller_axis: int = 0,
    phase_stride: int = 1,
    freeze_recovery_phase: bool = False,
    autonomous_chunk_reentry: bool = False,
    replan_on_recovery_reentry: bool = False,
    rotation_clf_enabled: bool = False,
    recovery_progress_patience: int = 0,
    gripper_change_phase: int | None = 1,
    gripper_targets: np.ndarray | None = None,
    recovery_phase_advance_budget: float | None = None,
    field_speed_gain: float = 1.0,
    support_confidence_outer_score: float = 0.05,
    support_confidence_inner_score: float = 0.1,
    recovery_trigger_mode: str = "support_only",
    require_state_innovation: bool = True,
    disturbance_position_innovation_threshold: float = 0.04,
    disturbance_rotation_innovation_threshold: float = 0.35,
    disturbance_response_mode: str = "recover_then_replan",
    nominal_fusion_mode: str = "native_action",
    transverse_fusion_inner_ratio: float = 0.5,
    transverse_fusion_outer_ratio: float = 1.0,
    transverse_fusion_max_weight: float = 1.0,
) -> tuple[ProjectedPhaseHybridExecutor, np.ndarray]:
    clf_radius = min(
        0.001,
        0.5 * reentry_radius if freeze_recovery_phase else 0.001,
    )
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        replan_after_targets=6,
        clf_projection_enabled=True,
        clf_position_tube_radius=clf_radius,
        clf_position_contraction_rate=0.2,
    )
    geometry = np.zeros((6, 7), dtype=np.float64)
    geometry[:, 0] = 0.2
    geometry[:, 6] = -1.0
    if gripper_targets is not None:
        geometry[:, 6] = np.asarray(
            gripper_targets,
            dtype=np.float64,
        )
    elif gripper_change_phase is not None:
        geometry[int(gripper_change_phase) :, 6] = 1.0
    controller_actions = np.zeros((6, 6), dtype=np.float64)
    controller_actions[:, controller_axis] = np.linspace(0.1, 0.6, 6)
    plan = compile_endpoint_field_plan(
        geometry,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = ProjectedPhaseHybridExecutor(
        plan,
        controller_actions=controller_actions,
        config=config,
        position_tube_radius=outer_radius,
        rotation_tube_radius=0.2,
        position_reentry_radius=reentry_radius,
        rotation_reentry_radius=0.1,
        phase_projection_lookahead=2.0,
        recovery_phase_advance_budget=recovery_phase_advance_budget,
        max_recovery_steps=max_recovery_steps,
        recovery_field_source=recovery_field_source,
        recovery_target_mode=recovery_target_mode,
        recovery_anchor_mode=recovery_anchor_mode,
        observed_anchor_reentry_radius=observed_anchor_reentry_radius,
        observed_anchor_stable_steps=observed_anchor_stable_steps,
        freeze_recovery_phase=freeze_recovery_phase,
        autonomous_chunk_reentry=autonomous_chunk_reentry,
        replan_on_recovery_reentry=replan_on_recovery_reentry,
        rotation_clf_enabled=rotation_clf_enabled,
        rotation_clf_tube_radius=0.05,
        recovery_progress_patience=recovery_progress_patience,
        field_speed_gain=field_speed_gain,
        support_confidence_outer_score=(
            support_confidence_outer_score
        ),
        support_confidence_inner_score=(
            support_confidence_inner_score
        ),
        recovery_trigger_mode=recovery_trigger_mode,
        require_state_innovation=require_state_innovation,
        disturbance_position_innovation_threshold=(
            disturbance_position_innovation_threshold
        ),
        disturbance_rotation_innovation_threshold=(
            disturbance_rotation_innovation_threshold
        ),
        disturbance_response_mode=disturbance_response_mode,
        nominal_fusion_mode=nominal_fusion_mode,
        transverse_fusion_inner_ratio=transverse_fusion_inner_ratio,
        transverse_fusion_outer_ratio=transverse_fusion_outer_ratio,
        transverse_fusion_max_weight=transverse_fusion_max_weight,
        phase_stride=phase_stride,
    )
    return executor, controller_actions


def _configure_confident_pose_support(
    executor: ProjectedPhaseHybridExecutor,
    *,
    transverse_variance: float = 4.0e-4,
) -> None:
    phase_count = executor.target_count + 1
    covariance = np.diag(
        [0.0, transverse_variance, transverse_variance]
    )
    support = PairedPoseEnsembleSupport(
        selected_sample_index=0,
        cluster_members=tuple(range(8)),
        cluster_confident=True,
        position_covariances=np.tile(
            covariance[None],
            (phase_count, 1, 1),
        ),
        position_spread_radii=np.full(
            phase_count,
            2.0 * np.sqrt(transverse_variance),
        ),
        rotation_spread_radii=np.full(phase_count, 0.05),
        position_distance_scale=1.0,
        rotation_distance_scale=1.0,
    )
    executor.configure_pose_ensemble_support(
        support,
        adaptive_field_support=True,
        adaptive_tube=True,
    )


def test_nominal_execution_uses_exact_native_action() -> None:
    executor, controller_actions = _straight_executor()

    step = executor.step(np.zeros(3), IDENTITY_QUATERNION)

    np.testing.assert_allclose(step.action[:6], controller_actions[0])
    assert step.action[6] == -1.0
    assert executor.action_phase == 1.0
    assert not executor.last_phase_frozen


def test_transverse_fusion_is_exact_native_without_confident_support() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.5,
        recovery_field_source="geometry",
        nominal_fusion_mode="transverse_minimal",
    )

    step = executor.step(
        np.array([0.0, 0.02, 0.0]),
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[:6], controller_actions[0])
    assert step.action[6] == -1.0
    assert executor.transverse_fusion_checks == 1
    assert executor.transverse_fusion_support_fallback_steps == 1
    assert executor.transverse_fusion_active_steps == 0


def test_transverse_fusion_preserves_tangent_rotation_and_gripper() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.5,
        recovery_field_source="geometry",
        nominal_fusion_mode="transverse_minimal",
    )
    _configure_confident_pose_support(executor)
    precision = executor._interpolate_phase_matrix(
        executor.position_tube_precisions,
        0.0,
    )
    contour = executor.position_tube_mahalanobis_radius
    offset_y = 0.8 * contour / np.sqrt(precision[1, 1])

    step = executor.step(
        np.array([0.0, offset_y, 0.0]),
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[0], controller_actions[0, 0])
    np.testing.assert_allclose(step.action[3:6], controller_actions[0, 3:6])
    assert step.action[1] < 0.0
    assert step.action[6] == -1.0
    assert executor.transverse_fusion_active_steps == 1
    assert 0.0 < executor.last_transverse_fusion_weight < 1.0
    assert executor.last_transverse_fusion_ratio == pytest.approx(0.8)


def test_transverse_fusion_is_zero_in_dense_support_core() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.5,
        recovery_field_source="geometry",
        nominal_fusion_mode="transverse_minimal",
    )
    _configure_confident_pose_support(executor)

    step = executor.step(np.zeros(3), IDENTITY_QUATERNION)

    np.testing.assert_allclose(step.action[:6], controller_actions[0])
    assert executor.transverse_fusion_checks == 1
    assert executor.transverse_fusion_active_steps == 0
    assert executor.last_transverse_fusion_weight == 0.0


def test_outward_only_fusion_preserves_inward_native_action() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.5,
        recovery_field_source="geometry",
        controller_axis=1,
        nominal_fusion_mode="transverse_outward_only",
    )
    controller_actions[:, 1] *= -1.0
    executor.controller_actions[:, 1] *= -1.0
    _configure_confident_pose_support(executor)
    precision = executor._interpolate_phase_matrix(
        executor.position_tube_precisions,
        0.0,
    )
    contour = executor.position_tube_mahalanobis_radius
    offset_y = 0.8 * contour / np.sqrt(precision[1, 1])

    step = executor.step(
        np.array([0.0, offset_y, 0.0]),
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[:6], controller_actions[0])
    assert executor.transverse_fusion_checks == 1
    assert executor.transverse_fusion_active_steps == 0
    assert executor.last_transverse_fusion_weight == 0.0


def test_outward_only_fusion_removes_only_outward_radial_motion() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.5,
        recovery_field_source="geometry",
        controller_axis=1,
        nominal_fusion_mode="transverse_outward_only",
        transverse_fusion_max_weight=1.0,
    )
    _configure_confident_pose_support(executor)
    precision = executor._interpolate_phase_matrix(
        executor.position_tube_precisions,
        0.0,
    )
    contour = executor.position_tube_mahalanobis_radius
    offset_y = contour / np.sqrt(precision[1, 1])

    step = executor.step(
        np.array([0.0, offset_y, 0.0]),
        IDENTITY_QUATERNION,
    )

    assert controller_actions[0, 1] > 0.0
    assert step.action[1] <= 1.0e-7
    np.testing.assert_allclose(step.action[0], controller_actions[0, 0])
    np.testing.assert_allclose(step.action[2:6], controller_actions[0, 2:6])
    assert executor.transverse_fusion_active_steps == 1
    assert executor.last_transverse_fusion_weight == pytest.approx(1.0)


def test_observed_anchor_translates_locked_prefix_to_measured_pose() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        recovery_anchor_mode="observed_perturbation_start",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        gripper_change_phase=5,
    )
    _configure_confident_pose_support(executor)
    executor.action_phase = 2.0
    reference, _ = _pose_at_phase(
        executor.plan,
        executor.action_phase,
        target_count=executor.target_count,
    )
    anchor = reference + np.array([0.0, 0.015, 0.0])
    assert executor.register_observed_recovery_anchor(
        anchor,
        IDENTITY_QUATERNION,
    )
    disturbed = anchor + np.array([0.0, 0.08, 0.0])
    phase, position_error, rotation_error = (
        executor._projection_for_recovery(
            disturbed,
            IDENTITY_QUATERNION,
            entering=True,
        )
    )
    executor._begin_recovery(position_error, rotation_error, phase)

    assert executor.recovery_prefix_chunk is not None
    np.testing.assert_allclose(
        executor.recovery_prefix_chunk.support_positions[-1],
        anchor,
        atol=1.0e-8,
    )
    field = executor._issue_field(
        disturbed,
        IDENTITY_QUATERNION,
        mutate_state=False,
    )
    assert field.action[1] < 0.0
    assert executor._inside_locked_prefix_terminal_support(
        anchor,
        IDENTITY_QUATERNION,
        record=False,
    )
    assert not executor._inside_locked_prefix_terminal_support(
        disturbed,
        IDENTITY_QUATERNION,
        record=False,
    )


def test_observed_anchor_rotates_locked_prefix_to_measured_pose() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        recovery_anchor_mode="observed_perturbation_start",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        gripper_change_phase=5,
    )
    _configure_confident_pose_support(executor)
    executor.action_phase = 2.0
    reference, _ = _pose_at_phase(
        executor.plan,
        executor.action_phase,
        target_count=executor.target_count,
    )
    anchor_quaternion = rotvec_to_quaternion(
        np.array([0.0, 0.0, 0.2])
    )
    assert executor.register_observed_recovery_anchor(
        reference,
        anchor_quaternion,
    )
    disturbed = reference + np.array([0.0, 0.08, 0.0])
    phase, position_error, rotation_error = (
        executor._projection_for_recovery(
            disturbed,
            IDENTITY_QUATERNION,
            entering=True,
        )
    )
    executor._begin_recovery(position_error, rotation_error, phase)

    _, rotation_delta, _ = executor._locked_prefix_full_field(
        disturbed,
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(
        rotation_delta,
        executor.rotation_recovery_gain
        * relative_rotation_vector(
            IDENTITY_QUATERNION,
            anchor_quaternion,
        ),
    )


@pytest.mark.parametrize("require_innovation", [True, False])
def test_disturbance_returns_to_last_safe_pose_then_replans(require_innovation) -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        recovery_anchor_mode="observed_perturbation_start",
        observed_anchor_reentry_radius=0.005,
        observed_anchor_stable_steps=2,
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        disturbance_position_innovation_threshold=0.01,
        require_state_innovation=require_innovation,
        gripper_change_phase=5,
    )
    _configure_confident_pose_support(executor)
    open_gripper = np.array([0.04, -0.04], dtype=np.float64)

    executor.step(
        executor.plan.position_centers[0],
        IDENTITY_QUATERNION,
        open_gripper,
    )
    last_safe = executor.plan.position_centers[1].copy()
    executor.step(last_safe, IDENTITY_QUATERNION, open_gripper)
    assert executor.action_phase == 2.0

    disturbed = last_safe + np.array([0.0, 0.08, 0.0])
    recovery = executor.step(
        disturbed,
        IDENTITY_QUATERNION,
        open_gripper,
    )

    assert executor.recovering
    assert executor.recovery_locked_phase == 1.0
    assert executor.action_phase == 2.0
    assert executor.automatic_recovery_anchor_registrations == 1
    np.testing.assert_allclose(
        executor.active_recovery_anchor_position,
        last_safe,
    )
    assert recovery.action[6] == 0.0

    first_arrival = executor.step(
        last_safe,
        IDENTITY_QUATERNION,
        open_gripper,
    )
    assert not first_arrival.replan_required
    assert executor.recovering
    assert executor.observed_anchor_inside_steps == 1
    assert executor.action_phase == 2.0
    assert first_arrival.action[6] == 0.0

    second_arrival = executor.step(
        last_safe,
        IDENTITY_QUATERNION,
        open_gripper,
    )
    assert second_arrival.replan_required
    assert second_arrival.replan_reason == "recovery_reentry"
    assert not executor.recovering
    assert executor.action_phase == 2.0
    assert executor.recovery_phase_advances == 0
    assert executor.observed_anchor_stability_wait_steps == 1


@pytest.mark.parametrize("require_innovation", [True, False])
def test_no_fusion_ab_changes_only_outside_support_gate(require_innovation) -> None:
    executor, actions = _straight_executor(
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        recovery_anchor_mode="observed_perturbation_start",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        require_state_innovation=require_innovation,
        disturbance_position_innovation_threshold=0.1,
        nominal_fusion_mode="native_action",
        max_recovery_steps=0,
    )
    _configure_confident_pose_support(executor, transverse_variance=1.0e-6)
    safe = executor.plan.position_centers[0].copy()
    first = executor.step(safe, IDENTITY_QUATERNION)
    np.testing.assert_allclose(first.action[:6], actions[0])
    outside = safe + np.array([0.005, 0.008, 0.0])
    step = executor.step(outside, IDENTITY_QUATERNION)
    assert executor.last_position_innovation < 0.1
    assert executor.recovering == (not require_innovation)
    assert executor.transverse_fusion_active_steps == 0
    assert executor.state_innovation_triggers == 0
    if require_innovation:
        np.testing.assert_allclose(step.action[:6], actions[1])
    else:
        np.testing.assert_allclose(executor.active_recovery_anchor_position, safe)
        assert executor.recovery_locked_phase == 0.0
        assert step.action[1] < 0.0
        assert executor.recovery_phase_advances == 0


def test_no_innovation_gate_without_confident_tube_does_not_invent_exit() -> None:
    executor, _ = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
    )
    executor.require_state_innovation = False
    executor.pose_ensemble_cluster_confident = False
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    step = executor.step(np.array([0.005, 0.20, 0.0]), IDENTITY_QUATERNION)
    assert not executor.recovering
    assert not step.replan_required


def test_field_speed_gain_scales_field_but_not_native_action() -> None:
    baseline, controller_actions = _straight_executor(
        recovery_field_source="geometry",
    )
    accelerated, _ = _straight_executor(
        recovery_field_source="geometry",
        field_speed_gain=4.0,
    )

    native = accelerated.step(np.zeros(3), IDENTITY_QUATERNION)
    np.testing.assert_allclose(native.action[:6], controller_actions[0])

    position = baseline.plan.position_centers[1].copy()
    baseline_field = baseline._issue_field(
        position,
        IDENTITY_QUATERNION,
        mutate_state=False,
        phase=1.0,
    )
    accelerated_field = accelerated._issue_field(
        position,
        IDENTITY_QUATERNION,
        mutate_state=False,
        phase=1.0,
    )

    np.testing.assert_allclose(
        accelerated_field.action[:6],
        4.0 * baseline_field.action[:6],
        atol=1.0e-6,
    )
    assert accelerated_field.action[6] == baseline_field.action[6]
    assert accelerated.compile_metadata["field_speed_gain"] == 4.0


def test_autonomous_chunk_reentry_rejects_frozen_phase_mode() -> None:
    try:
        _straight_executor(
            freeze_recovery_phase=True,
            autonomous_chunk_reentry=True,
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("incompatible recovery modes were accepted")


def test_next_action_phase_recovery_requires_frozen_phase() -> None:
    try:
        _straight_executor(recovery_target_mode="next_action_phase")
    except ValueError as error:
        assert "requires freeze_recovery_phase" in str(error)
    else:
        raise AssertionError("unfrozen next-action recovery was accepted")


def test_same_phase_attractor_requires_frozen_phase() -> None:
    try:
        _straight_executor(recovery_target_mode="same_phase_attractor")
    except ValueError as error:
        assert "requires freeze_recovery_phase" in str(error)
    else:
        raise AssertionError("unfrozen same-phase recovery was accepted")


def test_support_confidence_requires_autonomous_phase_observer() -> None:
    try:
        _straight_executor(recovery_target_mode="support_confidence")
    except ValueError as error:
        assert "requires autonomous_chunk_reentry" in str(error)
    else:
        raise AssertionError("support confidence accepted no field observer")


def test_support_confidence_uses_raw_gaussian_not_normalized_weight() -> None:
    executor, _ = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)

    center_score, center_phase = executor._gaussian_support_score(
        executor.plan.position_centers[0],
        IDENTITY_QUATERNION,
        recovery=False,
        lower_phase=0.0,
        upper_phase=0.0,
        record=False,
    )
    far_score, _ = executor._gaussian_support_score(
        np.array([0.0, 0.25, 0.0]),
        IDENTITY_QUATERNION,
        recovery=False,
        lower_phase=0.0,
        upper_phase=0.0,
        record=False,
    )

    assert center_phase == 0.0
    assert np.isclose(center_score, 1.0)
    assert far_score < 1.0e-6
    np.testing.assert_allclose(
        np.sum(executor.mixture_weights(np.array([0.0, 0.25, 0.0]))),
        1.0,
    )


def test_support_confidence_preserves_action_phase_then_requeries() -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        max_recovery_steps=0,
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)

    nominal = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    np.testing.assert_allclose(nominal.action[:6], controller_actions[0])
    assert executor.action_phase == 1.0

    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.25, 0.0]
    )
    recovery = executor.step(displaced, IDENTITY_QUATERNION)
    assert executor.recovering
    assert executor.action_phase == 1.0
    assert executor.phase_reprojections == 0
    assert recovery.action[6] == 0.0
    assert recovery.action[1] < 0.0

    reentry = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
    )
    assert reentry.replan_required
    assert reentry.replan_reason == "recovery_reentry"
    assert not executor.recovering
    assert executor.action_phase == 1.0
    assert executor.phase_reprojections == 0
    assert executor.compile_metadata["support_confidence_metric"] == (
        "max_unnormalized_position_gaussian"
    )
    assert executor.compile_metadata["support_confidence_action_rule"] == (
        "native_inside_or_inward_else_field"
    )
    assert executor.compile_metadata[
        "support_confidence_gripper_event_barrier"
    ] == "issue_native_transition_repeat_until_support_reentry"


def test_reentry_defers_fresh_plan_during_pending_gripper_transition() -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        max_recovery_steps=0,
        gripper_targets=np.array([1, 1, 1, 1, -1, -1]),
    )
    _configure_confident_pose_support(executor)
    open_gripper = np.array([0.04, -0.04], dtype=np.float64)

    executor.step(
        np.zeros(3),
        IDENTITY_QUATERNION,
        open_gripper,
    )
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.25, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION, open_gripper)
    assert executor.recovering

    reentry = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
        open_gripper,
    )

    assert not reentry.replan_required
    np.testing.assert_allclose(reentry.action[:6], controller_actions[1])
    assert reentry.action[6] == 1.0
    assert executor.recovery_reentry_replan_gripper_deferrals == 1


def test_support_confidence_reentry_resumes_preserved_action_phase() -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=False,
        recovery_field_source="geometry",
        max_recovery_steps=0,
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.25, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)

    resumed = executor.step(
        executor.plan.position_centers[2],
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(resumed.action[:6], controller_actions[1])
    assert not resumed.replan_required
    assert not executor.recovering
    assert executor.last_issued_phase == 1.0
    assert executor.action_phase == 2.0
    assert executor.recovery_phase == 2.0
    assert executor.phase_reprojections == 0
    assert executor.compile_metadata["reentry_action"] == (
        "cached_native_action_at_preserved_phase"
    )


def _locked_prefix_executor(
    *,
    replan_on_reentry: bool,
    gripper_change_phase: int | None = None,
    recovery_trigger_mode: str = "support_only",
    disturbance_position_innovation_threshold: float = 0.04,
) -> tuple[ProjectedPhaseHybridExecutor, np.ndarray]:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence_prefix",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=replan_on_reentry,
        recovery_field_source="geometry",
        max_recovery_steps=0,
        gripper_change_phase=gripper_change_phase,
        recovery_trigger_mode=recovery_trigger_mode,
        disturbance_position_innovation_threshold=(
            disturbance_position_innovation_threshold
        ),
    )
    _configure_confident_pose_support(executor)
    return executor, controller_actions


def test_state_innovation_gate_preserves_native_action_outside_support(
) -> None:
    executor, controller_actions = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        disturbance_position_innovation_threshold=0.1,
    )
    _configure_confident_pose_support(
        executor,
        transverse_variance=1.0e-6,
    )
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    expected_motion = np.array([0.005, 0.08, 0.0])
    step = executor.step(expected_motion, IDENTITY_QUATERNION)

    np.testing.assert_allclose(step.action[:6], controller_actions[1])
    assert not executor.recovering
    assert executor.action_phase == 2.0
    assert executor.state_innovation_checks == 1
    assert executor.state_innovation_triggers == 0
    assert executor.state_innovation_native_bypasses == 1
    assert executor.last_position_innovation == pytest.approx(0.08)


def test_state_innovation_gate_enters_locked_field_after_transverse_push(
) -> None:
    executor, _ = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
    )
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    pushed = np.array([0.005, 0.20, 0.0])
    recovery = executor.step(pushed, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.action_phase == 1.0
    assert executor.recovery_locked_phase == 1.0
    assert executor.state_innovation_checks == 1
    assert executor.state_innovation_triggers == 1
    assert executor.state_innovation_native_bypasses == 0
    assert recovery.action[1] < 0.0

    reentry = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
    )
    assert reentry.replan_required
    assert reentry.replan_reason == "recovery_reentry"


def test_state_innovation_immediate_replan_never_issues_field() -> None:
    executor, _ = _straight_executor(
        recovery_target_mode="support_confidence_prefix",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        max_recovery_steps=0,
        recovery_trigger_mode="support_and_state_innovation",
        disturbance_response_mode="immediate_replan",
    )
    _configure_confident_pose_support(executor)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    pushed = np.array([0.005, 0.20, 0.0])
    response = executor.step(pushed, IDENTITY_QUATERNION)

    assert response.replan_required
    assert response.replan_reason == "disturbance_immediate_replan"
    assert not executor.recovering
    assert executor.state_innovation_triggers == 1
    assert executor.recovery_entries == 0
    assert executor.consecutive_recovery_steps == 0
    assert executor.total_steps == 1
    assert executor.action_phase == 1.0
    assert executor.compile_metadata["disturbance_response_mode"] == (
        "immediate_replan"
    )


def test_state_innovation_gate_uses_phase_local_support_for_entry() -> None:
    config = EndpointFieldConfig(
        gripper_command_mode="spatial_progress",
        replan_after_targets=24,
        clf_projection_enabled=True,
        clf_position_tube_radius=0.001,
        clf_position_contraction_rate=0.2,
    )
    geometry = np.zeros((24, 7), dtype=np.float64)
    geometry[:, 6] = -1.0
    geometry[0, 0] = 0.2
    geometry[1:21, 1] = 0.2
    plan = compile_endpoint_field_plan(
        geometry,
        np.zeros(3),
        IDENTITY_QUATERNION,
        config,
    )
    executor = ProjectedPhaseHybridExecutor(
        plan,
        controller_actions=np.zeros((24, 6), dtype=np.float64),
        config=config,
        position_tube_radius=0.005,
        rotation_tube_radius=0.2,
        position_reentry_radius=0.001,
        rotation_reentry_radius=0.1,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        support_confidence_outer_score=0.05,
        support_confidence_inner_score=0.1,
        recovery_trigger_mode="support_and_state_innovation",
        disturbance_position_innovation_threshold=0.04,
    )
    _configure_confident_pose_support(
        executor,
        transverse_variance=1.0e-6,
    )
    executor.step(plan.position_centers[0], IDENTITY_QUATERNION)
    aliased_future_center = plan.position_centers[21]

    assert executor._inside_gaussian_outer_support(
        aliased_future_center,
        IDENTITY_QUATERNION,
        record=False,
    )
    assert not executor._inside_sampled_current_native_support(
        aliased_future_center,
        IDENTITY_QUATERNION,
        record=False,
    )

    recovery = executor.step(
        aliased_future_center,
        IDENTITY_QUATERNION,
    )

    assert executor.recovering
    assert executor.recovery_locked_phase == 1.0
    assert executor.state_innovation_triggers == 1
    assert recovery.action[1] < 0.0
    assert executor.compile_metadata["support_confidence_entry_scope"] == (
        "last_native_finite_pose_segment"
    )
    assert executor.compile_metadata["support_confidence_source"] == (
        "pose_ensemble_only"
    )


def test_state_innovation_switch_uses_sample_spread_not_dynamics_covariance(
) -> None:
    executor, _ = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
    )
    _configure_confident_pose_support(
        executor,
        transverse_variance=1.0e-6,
    )

    assert executor.pose_sample_position_outer_radii is not None
    assert executor.position_tube_radii is not None
    np.testing.assert_allclose(
        executor.pose_sample_position_outer_radii,
        0.002,
    )
    assert np.max(executor.position_tube_radii) > 0.002
    assert executor.compile_metadata["support_confidence_metric"] == (
        "finite_segment_pose_distance_over_k_sample_spread"
    )
    assert executor.compile_metadata["dynamics_gaussian_support_source"] == (
        "compiler_prior_plus_pose_ensemble_covariance"
    )


def test_state_innovation_gate_detects_rotation_outside_sample_support() -> None:
    executor, _ = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
        disturbance_position_innovation_threshold=1.0,
    )
    executor.disturbance_rotation_innovation_threshold = 0.1
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    rotated = rotvec_to_quaternion(np.array([0.0, 0.0, 0.4]))
    recovery = executor.step(executor.plan.position_centers[1], rotated)

    assert executor.recovering
    assert executor.state_innovation_triggers == 1
    assert executor.last_rotation_innovation > 0.1
    assert np.linalg.norm(recovery.action[3:6]) > 0.0


def test_state_innovation_without_confident_cluster_requests_replan() -> None:
    executor, _ = _locked_prefix_executor(
        replan_on_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
    )
    phase_count = executor.target_count + 1
    support = PairedPoseEnsembleSupport(
        selected_sample_index=0,
        cluster_members=tuple(range(8)),
        cluster_confident=False,
        position_covariances=np.zeros((phase_count, 3, 3)),
        position_spread_radii=np.full(phase_count, 0.02),
        rotation_spread_radii=np.full(phase_count, 0.1),
        position_distance_scale=1.0,
        rotation_distance_scale=1.0,
    )
    executor.configure_pose_ensemble_support(
        support,
        adaptive_field_support=True,
        adaptive_tube=True,
    )
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    step = executor.step(np.array([0.005, 0.20, 0.0]), IDENTITY_QUATERNION)

    assert step.replan_required
    assert step.replan_reason == (
        "disturbance_without_confident_pose_support"
    )
    assert not executor.recovering
    assert executor.state_innovation_triggers == 1
    assert executor.support_confidence_direct_fallback_steps == 2


def test_motion_segment_innovation_ignores_slow_aligned_contact_motion(
) -> None:
    residual = ProjectedPhaseHybridExecutor._motion_segment_innovation(
        np.array([0.001, 0.0, 0.0]),
        np.array([0.005, 0.0, 0.0]),
        maximum_expected_progress=1.5,
    )

    assert residual == pytest.approx(0.0)


def test_state_innovation_hybrid_keeps_full_native_gripper_horizon() -> None:
    adapter = LiberoPairedBSplineHybridDSAdapter(
        _FakePolicy(),
        plan_steps=4,
        source_horizon=4,
        support_subdivisions=1,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence_prefix",
        freeze_recovery_phase=True,
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_trigger_mode="support_and_state_innovation",
    )
    paired = np.zeros((4, 13), dtype=np.float32)
    paired[:, 0] = 0.1
    paired[:, 6] = 0.2
    paired[:, 12] = [-1.0, 1.0, 1.0, 1.0]

    executor = adapter._compile_executor(
        paired,
        np.zeros(3),
        IDENTITY_QUATERNION,
    )

    assert not adapter.truncate_at_gripper_only_transition
    assert executor.target_count == 4
    assert executor.terminal_gripper_command is None
    np.testing.assert_allclose(
        executor.plan.gripper_targets,
        paired[:, 12],
    )

    executor.position_innovation_samples.extend([0.001, 0.002, 0.003])
    executor.rotation_innovation_samples.extend([0.01, 0.02, 0.03])
    executor.state_innovation_checks = 3
    executor.max_position_innovation = 0.003
    executor.max_rotation_innovation = 0.03
    adapter._executors = [executor]
    metrics = adapter.get_execution_metrics()
    assert metrics["position_innovation_p50"] == pytest.approx(0.002)
    assert metrics["position_innovation_p90"] == pytest.approx(0.0028)
    assert metrics["position_innovation_p99"] == pytest.approx(0.00298)
    assert metrics["rotation_innovation_p50"] == pytest.approx(0.02)


def test_locked_prefix_recovers_from_off_support_future_displacement() -> None:
    executor, _ = _locked_prefix_executor(replan_on_reentry=True)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    future = executor.plan.position_centers[4] + np.array(
        [0.0, 0.25, 0.0]
    )
    recovery = executor.step(future, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.action_phase == 1.0
    assert executor.recovery_locked_phase == 1.0
    assert executor.recovery_phase == 1.0
    assert executor.recovery_prefix_start == 0
    assert executor.recovery_prefix_end == 1
    assert executor.recovery_prefix_chunk is not None
    assert executor.recovery_prefix_chunk.center_end == 1
    assert np.max(executor.recovery_prefix_chunk.support_positions[:, 0]) <= (
        executor.plan.position_centers[1, 0] + 1.0e-12
    )
    assert recovery.action[0] < 0.0


def test_locked_prefix_has_a_terminal_attractor_at_the_locked_phase() -> None:
    executor, _ = _locked_prefix_executor(replan_on_reentry=True)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    executor.step(
        executor.plan.position_centers[4] + np.array([0.0, 0.25, 0.0]),
        IDENTITY_QUATERNION,
    )

    terminal = executor._locked_prefix_full_field(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(terminal[0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(terminal[1], 0.0, atol=1.0e-8)
    assert executor.recovery_prefix_compiles == 1
    assert executor.compile_metadata["recovery_execution"] == (
        "locked_safe_prefix_gaussian_ds_to_terminal_phase"
    )


def test_locked_prefix_only_replans_at_its_terminal_support() -> None:
    executor, _ = _locked_prefix_executor(replan_on_reentry=True)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    executor.step(
        executor.plan.position_centers[4] + np.array([0.0, 0.25, 0.0]),
        IDENTITY_QUATERNION,
    )

    early = executor.step(
        executor.plan.position_centers[0],
        IDENTITY_QUATERNION,
    )
    assert executor.recovering
    assert not early.replan_required

    terminal = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
    )
    assert terminal.replan_required
    assert terminal.replan_reason == "recovery_reentry"
    assert not executor.recovering
    assert executor.action_phase == 1.0


def test_locked_prefix_cached_resume_preserves_upcoming_gripper_event() -> None:
    executor, controller_actions = _locked_prefix_executor(
        replan_on_reentry=False,
        gripper_change_phase=2,
    )
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    executor.step(executor.plan.position_centers[1], IDENTITY_QUATERNION)

    displaced = executor.plan.position_centers[4] + np.array(
        [0.0, 0.25, 0.0]
    )
    recovery = executor.step(displaced, IDENTITY_QUATERNION)
    assert executor.recovering
    assert executor.action_phase == 2.0
    assert recovery.action[6] == 0.0

    resumed = executor.step(
        executor.plan.position_centers[2],
        IDENTITY_QUATERNION,
    )
    np.testing.assert_allclose(resumed.action[:6], controller_actions[2])
    assert resumed.action[6] == 1.0
    assert executor.last_issued_phase == 2.0
    assert executor.action_phase == 3.0


def test_support_confidence_checks_the_complete_ordered_chunk() -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)

    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)
    step = executor.step(
        executor.plan.position_centers[4],
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[:6], controller_actions[1])
    assert not executor.recovering
    assert executor.action_phase == 2.0


def test_support_confidence_keeps_outside_action_that_points_inward() -> None:
    executor, controller_actions = _straight_executor(
        controller_axis=1,
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)
    executor.step(executor.plan.position_centers[0], IDENTITY_QUATERNION)

    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, -0.25, 0.0]
    )
    step = executor.step(displaced, IDENTITY_QUATERNION)

    np.testing.assert_allclose(step.action[:6], controller_actions[1])
    assert not executor.recovering
    assert executor.action_phase == 2.0
    assert executor.support_confidence_inward_native_steps == 1


def test_support_confidence_nominal_gripper_event_keeps_native_arm() -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        gripper_targets=np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    _configure_confident_pose_support(executor)
    closed_qpos = np.array([0.02, -0.02])

    executor.step(
        executor.plan.position_centers[0],
        IDENTITY_QUATERNION,
        closed_qpos,
    )
    transition = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
        closed_qpos,
    )
    continued = executor.step(
        executor.plan.position_centers[2],
        IDENTITY_QUATERNION,
        closed_qpos,
    )

    np.testing.assert_allclose(transition.action[:6], controller_actions[1])
    np.testing.assert_allclose(continued.action[:6], controller_actions[2])
    assert transition.action[6] == -1.0
    assert continued.action[6] == -1.0
    assert executor.action_phase == 3.0
    assert executor.gripper_transition_priority_issues == 0
    assert executor.gripper_transition_guard_steps == 0


def test_support_confidence_field_does_not_interrupt_open_gripper_event(
) -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=False,
        recovery_field_source="geometry",
        gripper_targets=np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    _configure_confident_pose_support(executor)
    closed_qpos = np.array([0.02, -0.02])

    first = executor.step(
        executor.plan.position_centers[0],
        IDENTITY_QUATERNION,
        closed_qpos,
    )
    np.testing.assert_allclose(first.action[:6], controller_actions[0])
    assert first.action[6] == 1.0

    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.25, 0.0]
    )
    transition = executor.step(displaced, IDENTITY_QUATERNION, closed_qpos)
    np.testing.assert_allclose(transition.action[:6], controller_actions[1])
    assert transition.action[6] == -1.0
    assert executor.action_phase == 2.0
    assert executor.gripper_transition_priority_issues == 1
    assert not executor.recovering

    guarded_field = executor.step(
        displaced,
        IDENTITY_QUATERNION,
        closed_qpos,
    )
    assert guarded_field.action[1] < 0.0
    assert guarded_field.action[6] == -1.0
    assert executor.action_phase == 2.0
    assert executor.gripper_transition_guard_steps == 1
    assert executor.gripper_transition_open_guard_steps == 1
    assert executor.gripper_transition_close_guard_steps == 0
    assert executor.recovering

    reentry = executor.step(
        executor.plan.position_centers[2],
        IDENTITY_QUATERNION,
        closed_qpos,
    )
    np.testing.assert_allclose(reentry.action[:6], controller_actions[2])
    assert reentry.action[6] == -1.0
    assert not reentry.replan_required
    assert executor.action_phase == 3.0
    assert not executor.recovering


def test_support_confidence_falls_back_to_exact_native_action_without_cluster(
) -> None:
    executor, controller_actions = _straight_executor(
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )

    step = executor.step(
        np.array([0.0, 0.25, 0.0]),
        IDENTITY_QUATERNION,
    )

    np.testing.assert_allclose(step.action[:6], controller_actions[0])
    assert not executor.recovering
    assert executor.support_confidence_direct_fallback_steps == 1


def test_same_phase_attractor_freezes_action_phase_and_resumes_it() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.005,
        reentry_radius=0.001,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="same_phase_attractor",
        freeze_recovery_phase=True,
        phase_stride=2,
        gripper_change_phase=2,
    )

    first = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    assert first.action[6] == -1.0
    assert executor.action_phase == 2.0

    displaced = np.array([0.012, 0.02, 0.0], dtype=np.float64)
    recovery = executor.step(displaced, IDENTITY_QUATERNION)
    frozen_phase = float(executor.recovery_phase)
    reference_position, _ = _pose_at_phase(
        executor.plan,
        frozen_phase,
        target_count=executor.target_count,
    )

    assert executor.recovering
    assert frozen_phase == 2.0
    assert executor.action_phase == 2.0
    assert recovery.action[6] == 0.0
    assert recovery.action[0] > 0.0
    np.testing.assert_allclose(recovery.action[1], -0.4)

    at_target = executor._issue_field(
        reference_position,
        IDENTITY_QUATERNION,
        mutate_state=False,
        phase=frozen_phase,
    )
    np.testing.assert_allclose(at_target.action[:6], 0.0, atol=1.0e-7)

    future_displaced = np.array([0.018, 0.02, 0.0], dtype=np.float64)
    second_recovery = executor.step(
        future_displaced,
        IDENTITY_QUATERNION,
    )
    assert executor.recovery_phase == frozen_phase
    assert executor.action_phase == 2.0
    assert second_recovery.action[0] > 0.0
    assert second_recovery.action[1] < 0.0

    resumed = executor.step(
        reference_position + np.array([0.0, 0.0005, 0.0]),
        IDENTITY_QUATERNION,
    )
    expected_arm = controller_actions[int(frozen_phase)]
    np.testing.assert_allclose(resumed.action[:6], expected_arm)
    assert not executor.recovering
    assert executor.last_issued_phase == frozen_phase
    assert executor.action_phase == frozen_phase + 2.0
    assert executor.compile_metadata["recovery_execution"] == (
        "frozen_same_phase_linear_attractor"
    )
    assert executor.compile_metadata["effective_recovery_speed_gain"] == 1.0


def test_next_action_phase_recovery_preserves_target_and_gripper_event() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.005,
        reentry_radius=0.001,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="next_action_phase",
        freeze_recovery_phase=True,
        phase_stride=2,
        gripper_change_phase=2,
    )

    first = executor.step(np.zeros(3), IDENTITY_QUATERNION)
    assert first.action[6] == -1.0
    assert executor.action_phase == 2.0

    far_future = executor.plan.position_centers[5] + np.array(
        [0.0, 0.02, 0.0]
    )
    recovery = executor.step(far_future, IDENTITY_QUATERNION)
    assert executor.recovering
    assert executor.recovery_phase == 2.0
    assert executor.action_phase == 2.0
    assert recovery.action[6] == 0.0
    assert executor.last_gripper_command == -1.0

    still_future = executor.plan.position_centers[4] + np.array(
        [0.0, 0.02, 0.0]
    )
    executor.step(still_future, IDENTITY_QUATERNION)
    assert executor.recovery_phase == 2.0
    assert executor.action_phase == 2.0

    resumed = executor.step(
        executor.plan.position_centers[2],
        IDENTITY_QUATERNION,
    )
    np.testing.assert_allclose(resumed.action[:6], controller_actions[2])
    assert resumed.action[6] == 1.0
    assert not executor.recovering
    assert executor.last_issued_phase == 2.0
    assert executor.action_phase == 4.0
    assert executor.compile_metadata["recovery_target_mode"] == (
        "next_action_phase"
    )


def test_pose_ensemble_support_changes_field_not_selected_action() -> None:
    controls = np.zeros((3, 4, 13), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    controls[0, :, 0] = 0.4 * phase
    controls[1, :, 0] = 0.42 * phase
    controls[1, :, 1] = 0.02 * phase
    controls[2, :, 1] = 2.0 * phase
    controls[0, :, 6] = 0.25
    controls[1, :, 6] = 0.85
    controls[2, :, 6] = -0.75
    controls[:, :, 12] = -1.0
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakeSamplePolicy(controls),
        samples=3,
        source_horizon=4,
        degree=3,
        support_subdivisions=1,
    )
    adapter = LiberoPairedBSplineHybridDSAdapter(
        ensemble,
        plan_steps=4,
        source_horizon=4,
        degree=3,
        support_subdivisions=1,
        recovery_field_source="geometry",
        freeze_recovery_phase=True,
        phase_position_tube_radius=0.04,
        phase_position_reentry_radius=0.02,
        tube_radius=0.005,
        adaptive_pose_support_enabled=True,
        adaptive_pose_tube_enabled=True,
        terminal_stall_replan=False,
    )

    action = adapter.predict_action(_observations())["action"]

    np.testing.assert_allclose(action[0, 0, 0].item(), 0.25, atol=1.0e-6)
    executor = adapter._executors[0]
    assert isinstance(executor, ProjectedPhaseHybridExecutor)
    assert executor.pose_ensemble_support_applied
    assert executor.pose_ensemble_cluster_size == 2
    assert executor.position_tube_radii is not None
    assert executor.position_reentry_radii is not None
    np.testing.assert_allclose(
        executor.position_reentry_radii,
        0.5 * executor.position_tube_radii,
    )
    metrics = adapter.get_execution_metrics()
    assert metrics["pose_ensemble_action_aggregation"] == (
        "none_selected_pair_exact"
    )
    assert metrics["adaptive_pose_tube_enabled"] is True
    assert metrics["adaptive_pose_tube_metric"] == (
        "transverse_mahalanobis"
    )


def test_unconfident_pose_ensemble_keeps_original_compiler() -> None:
    executor, _ = _straight_executor()
    original_precisions = executor.plan.segment_precisions.copy()
    phase_count = executor.target_count + 1
    support = PairedPoseEnsembleSupport(
        selected_sample_index=0,
        cluster_members=(0, 1, 2),
        cluster_confident=False,
        position_covariances=np.tile(
            np.eye(3, dtype=np.float64)[None],
            (phase_count, 1, 1),
        ),
        position_spread_radii=np.ones(phase_count, dtype=np.float64),
        rotation_spread_radii=np.ones(phase_count, dtype=np.float64),
        position_distance_scale=1.0,
        rotation_distance_scale=1.0,
    )

    executor.configure_pose_ensemble_support(
        support,
        adaptive_field_support=True,
        adaptive_tube=True,
    )

    assert not executor.pose_ensemble_support_applied
    assert not executor.pose_ensemble_cluster_confident
    assert executor.position_tube_radii is None
    np.testing.assert_allclose(
        executor.plan.segment_precisions,
        original_precisions,
    )


def test_adaptive_tube_uses_transverse_mahalanobis_error() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.02,
    )
    phase_count = executor.target_count + 1
    support = PairedPoseEnsembleSupport(
        selected_sample_index=0,
        cluster_members=(0, 1, 2),
        cluster_confident=True,
        position_covariances=np.zeros(
            (phase_count, 3, 3),
            dtype=np.float64,
        ),
        position_spread_radii=np.full(
            phase_count,
            0.005,
            dtype=np.float64,
        ),
        rotation_spread_radii=np.zeros(
            phase_count,
            dtype=np.float64,
        ),
        position_distance_scale=1.0,
        rotation_distance_scale=1.0,
    )
    executor.configure_pose_ensemble_support(
        support,
        adaptive_field_support=True,
        adaptive_tube=True,
    )

    phase = 1.0
    reference = executor.plan.position_centers[1]
    longitudinal = reference + np.array([0.10, 0.0, 0.0])
    transverse = reference + np.array([0.0, 0.05, 0.0])

    assert executor._inside_outer_tube(
        0.10,
        0.0,
        phase,
        position=longitudinal,
    )
    assert not executor._inside_outer_tube(
        0.05,
        0.0,
        phase,
        position=transverse,
    )
    assert executor.compile_metadata["position_tube_mode"] == (
        "phase_adaptive_transverse_mahalanobis"
    )


def test_dense_pose_core_defines_independent_reentry_ellipse() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.02,
    )
    phase_count = executor.target_count + 1
    full_covariance = np.diag([0.0, 4.0e-4, 4.0e-4])
    core_covariance = np.diag([0.0, 1.0e-6, 1.0e-6])
    support = PairedPoseEnsembleSupport(
        selected_sample_index=0,
        cluster_members=tuple(range(8)),
        cluster_confident=True,
        position_covariances=np.tile(
            full_covariance[None],
            (phase_count, 1, 1),
        ),
        position_spread_radii=np.full(phase_count, 0.03),
        rotation_spread_radii=np.zeros(phase_count),
        position_distance_scale=1.0,
        rotation_distance_scale=1.0,
        position_dense_core_covariances=np.tile(
            core_covariance[None],
            (phase_count + 2, 1, 1),
        ),
        position_dense_core_spread_radii=np.full(phase_count + 2, 0.002),
        dense_core_fraction=0.75,
    )
    executor.configure_pose_ensemble_support(
        support,
        adaptive_field_support=True,
        adaptive_tube=True,
        support_confidence=0.95,
        dense_core_confidence=0.9,
    )

    assert executor.position_reentry_precisions is not None
    assert executor.position_tube_radii is not None
    assert executor.position_reentry_radii is not None
    assert np.all(
        executor.position_reentry_radii < executor.position_tube_radii
    )
    phase = 1.0
    reference = executor.plan.position_centers[1]
    offset = 0.5 * (
        executor.position_reentry_radii[1]
        + executor.position_tube_radii[1]
    )
    point = reference + np.array([0.0, offset, 0.0])
    assert executor._inside_outer_tube(
        offset,
        0.0,
        phase,
        position=point,
    )
    assert not executor._inside_reentry_tube(
        offset,
        0.0,
        phase,
        position=point,
    )
    assert executor.compile_metadata["position_reentry_mode"] == (
        "robust_dense_core_mahalanobis"
    )


def test_recovery_uses_full_field_and_neutral_gripper_increment() -> None:
    executor, _ = _straight_executor()
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )

    step = executor.step(displaced, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.last_phase_frozen
    assert executor.action_phase == 1.0
    assert step.action[0] > 0.0
    assert step.action[1] < 0.0
    assert step.action[6] == 0.0
    assert executor.last_gripper_command == -1.0
    assert executor.compile_metadata["gripper_during_recovery"] == (
        "neutral_increment_hold"
    )


def test_reentry_reprojects_phase_and_interpolates_native_action() -> None:
    executor, controller_actions = _straight_executor(
        gripper_change_phase=4
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)
    reentry_phase = 1.4
    reentry_position = (
        0.6 * executor.plan.position_centers[1]
        + 0.4 * executor.plan.position_centers[2]
    )

    step = executor.step(reentry_position, IDENTITY_QUATERNION)

    expected = 0.6 * controller_actions[1] + 0.4 * controller_actions[2]
    np.testing.assert_allclose(step.action[:6], expected, atol=1.0e-7)
    assert np.isclose(executor.last_issued_phase, reentry_phase)
    assert np.isclose(executor.action_phase, reentry_phase + 1.0)
    assert step.action[6] == -1.0
    assert not executor.recovering
    assert not executor.last_phase_frozen
    assert executor.phase_reprojections == 1


def test_reentry_can_request_fresh_visual_plan_without_old_action() -> None:
    executor, _ = _straight_executor(
        gripper_change_phase=4,
        replan_on_recovery_reentry=True,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)
    reentry_position = executor.plan.position_centers[1]

    step = executor.step(reentry_position, IDENTITY_QUATERNION)

    np.testing.assert_allclose(step.action, 0.0)
    assert not executor.recovering
    assert executor.recovery_reentries == 1
    assert step.replan_required
    assert not step.plan_complete
    assert step.replan_reason == "recovery_reentry"
    assert executor.action_phase == 1.0
    assert executor.last_gripper_command == -1.0
    assert executor.compile_metadata["reentry_action"] == (
        "fresh_visual_plan_unless_gripper_transition_pending"
    )


def test_adapter_queries_and_emits_fresh_action_on_reentry() -> None:
    controls = np.zeros((1, 4, 13), dtype=np.float32)
    controls[0, :, 0] = np.linspace(0.0, 0.4, 4)
    controls[0, :, 6] = 0.25
    controls[0, :, 12] = -1.0
    base = _FakeSamplePolicy(controls)
    adapter = LiberoPairedBSplineHybridDSAdapter(
        base,
        plan_steps=4,
        source_horizon=4,
        degree=3,
        support_subdivisions=1,
        phase_position_tube_radius=0.005,
        phase_position_reentry_radius=0.001,
        phase_max_recovery_steps=0,
        recovery_field_source="geometry",
        freeze_recovery_phase=True,
        replan_on_recovery_reentry=True,
        tube_radius=0.0005,
        terminal_stall_replan=False,
    )
    obs = _observations()
    adapter.predict_action(obs)
    executor = adapter._executors[0]
    assert isinstance(executor, ProjectedPhaseHybridExecutor)

    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )
    obs["robot0_eef_pos"][0, 0] = torch.as_tensor(
        displaced,
        dtype=torch.float32,
    )
    adapter.predict_action(obs)
    assert executor.recovering
    assert base.predict_calls == 1

    phase = float(executor.recovery_phase)
    lower = min(int(np.floor(phase)), executor.target_count - 1)
    upper = min(lower + 1, executor.target_count)
    fraction = phase - lower
    reentry = (
        (1.0 - fraction) * executor.plan.position_centers[lower]
        + fraction * executor.plan.position_centers[upper]
    )
    obs["robot0_eef_pos"][0, 0] = torch.as_tensor(
        reentry,
        dtype=torch.float32,
    )
    action = adapter.predict_action(obs)["action"]

    assert base.predict_calls == 2
    np.testing.assert_allclose(action[0, 0, 0].item(), 0.25)
    metrics = adapter.get_execution_metrics()
    assert metrics["policy_env_queries"] == 2
    assert metrics["recovery_reentry_replans"] == 1
    assert metrics["replan_reasons"] == {"recovery_reentry": 1}
    assert metrics["field_recovery_steps"] == 1
    assert metrics["gripper_during_recovery"] == "neutral_increment_hold"


def test_recovery_phase_advances_from_captured_forward_motion() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.05, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)
    initial_phase = executor.recovery_phase
    forward_captured = (
        0.5 * executor.plan.position_centers[1]
        + 0.5 * executor.plan.position_centers[2]
        + np.array([0.0, 0.03, 0.0])
    )

    step = executor.step(forward_captured, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.recovery_phase > initial_phase
    assert executor.recovery_phase <= (
        initial_phase + executor.phase_projection_lookahead
    )
    assert executor.action_phase == 1.0
    assert executor.recovery_phase_advances == 1
    assert not step.replan_required

    for phase in (2.0, 3.0, 4.0):
        position = (
            executor.plan.position_centers[int(phase)]
            + np.array([0.0, 0.03, 0.0])
        )
        executor.step(position, IDENTITY_QUATERNION)

    assert executor.recovery_phase <= (
        executor.recovery_entry_phase + executor.phase_stride
    )
    assert executor.compile_metadata[
        "recovery_phase_advance_budget"
    ] == 1.0


def test_recovery_phase_does_not_advance_outside_capture_tube() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        gripper_change_phase=None,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.05, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)
    initial_phase = executor.recovery_phase
    forward_uncaptured = (
        0.5 * executor.plan.position_centers[1]
        + 0.5 * executor.plan.position_centers[2]
        + np.array([0.0, 0.05, 0.0])
    )

    executor.step(forward_uncaptured, IDENTITY_QUATERNION)

    assert executor.recovery_phase == initial_phase
    assert executor.recovery_phase_advances == 0
    assert executor.recovery_phase_guard_blocks == 1


def test_zero_recovery_phase_budget_allows_continuous_progress() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="action",
        gripper_change_phase=None,
        recovery_phase_advance_budget=0.0,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    executor.step(
        executor.plan.position_centers[1] + np.array([0.0, 0.05, 0.0]),
        IDENTITY_QUATERNION,
    )
    entry_phase = executor.recovery_entry_phase
    for phase in (1.5, 2.5):
        lower = int(np.floor(phase))
        alpha = phase - lower
        position = (
            (1.0 - alpha) * executor.plan.position_centers[lower]
            + alpha * executor.plan.position_centers[lower + 1]
            + np.array([0.0, 0.03, 0.0])
        )
        executor.step(position, IDENTITY_QUATERNION)

    assert executor.recovery_phase > entry_phase + executor.phase_stride
    assert executor.compile_metadata[
        "recovery_phase_advance_budget"
    ] == 0.0


def test_recovery_phase_stops_at_gripper_event() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.01,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        gripper_change_phase=1,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.05, 0.0]
    )
    field_step = executor.step(displaced, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.recovery_phase == 1.0
    assert field_step.action[6] == 0.0
    assert executor.last_gripper_command == -1.0

    future_displaced = executor.plan.position_centers[2] + np.array(
        [0.0, 0.03, 0.0]
    )
    executor.step(future_displaced, IDENTITY_QUATERNION)
    assert executor.recovery_phase == 1.0
    assert executor.recovery_gripper_barrier_hits >= 1

    event_step = executor.step(
        executor.plan.position_centers[1],
        IDENTITY_QUATERNION,
    )
    assert not executor.recovering
    assert event_step.action[6] == 1.0


def test_projection_stays_in_ordered_local_window() -> None:
    executor, _ = _straight_executor()
    position = executor.plan.position_centers[4]

    phase, _, _ = executor._ordered_local_projection(
        position,
        IDENTITY_QUATERNION,
        lower_phase=1.0,
        upper_phase=2.0,
    )

    assert 1.0 <= phase <= 2.0


def test_preview_does_not_advance_or_rephase_action_clock() -> None:
    executor, _ = _straight_executor()
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    phase_before = executor.action_phase
    midpoint = 0.5 * (
        executor.plan.position_centers[0]
        + executor.plan.position_centers[1]
    )

    executor.preview_action(midpoint, IDENTITY_QUATERNION)

    assert executor.action_phase == phase_before


def test_action_mode_reprojection_uses_outer_tube_hysteresis() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.005,
        reentry_radius=0.001,
        gripper_change_phase=None,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    position = executor.plan.position_centers[0] + np.array(
        [0.0, 0.003, 0.0]
    )

    step = executor.step(position, IDENTITY_QUATERNION)

    assert not executor.recovering
    assert executor.recovery_entries == 0
    np.testing.assert_allclose(step.action[:6], controller_actions[0])


def test_local_clf_removes_outward_velocity_beyond_its_tube() -> None:
    executor, _ = _straight_executor()
    phase = 1.0
    reference = executor.plan.position_centers[1]
    position = reference + np.array([0.0, 0.02, 0.0])
    outward = np.array([0.01, 0.10, 0.0])

    projected = executor._local_clf_project_position_delta(
        position,
        outward,
        phase,
    )

    offset = position - reference
    assert np.dot(offset, projected) < 0.0
    assert executor.last_clf_applied


def test_rotation_clf_removes_outward_rotation() -> None:
    executor, _ = _straight_executor(rotation_clf_enabled=True)
    phase = 1.0
    quaternion = rotvec_to_quaternion(np.array([0.0, 0.0, 0.3]))
    outward = np.array([0.0, 0.0, 0.1])

    projected = executor._local_clf_project_rotation_delta(
        quaternion,
        outward,
        phase,
    )

    error = relative_rotation_vector(
        quaternion,
        IDENTITY_QUATERNION,
    )
    assert np.dot(error, projected) > 0.0
    assert executor.last_rotation_clf_applied


def test_frozen_recovery_phase_does_not_chase_future_supports() -> None:
    executor, _ = _straight_executor(freeze_recovery_phase=True)
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)
    frozen_phase = executor.recovery_phase

    future_displaced = executor.plan.position_centers[3] + np.array(
        [0.0, 0.02, 0.0]
    )
    step = executor.step(future_displaced, IDENTITY_QUATERNION)

    assert executor.recovering
    assert executor.recovery_phase == frozen_phase
    assert executor.action_phase == 1.0
    assert not step.replan_required


def test_autonomous_chunk_field_uses_all_event_bounded_supports() -> None:
    executor, _ = _straight_executor(
        autonomous_chunk_reentry=True,
        recovery_field_source="geometry",
        gripper_change_phase=3,
    )
    assert executor.autonomous_recovery_chunks == {}
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )
    executor.step(displaced, IDENTITY_QUATERNION)

    assert executor.recovering
    assert list(executor.autonomous_recovery_chunks) == [0]
    assert executor.autonomous_recovery_chunk_compiles == 1
    assert (
        executor.autonomous_recovery_interval_start,
        executor.autonomous_recovery_interval_end,
    ) == (0.0, 3.0)
    _, _, autonomous_weights = executor._autonomous_chunk_full_field(
        displaced,
        IDENTITY_QUATERNION,
        executor.recovery_phase,
    )
    _, _, local_weights = executor._local_full_field(
        displaced,
        IDENTITY_QUATERNION,
        executor.recovery_phase,
    )

    assert len(autonomous_weights) == 4
    assert np.count_nonzero(autonomous_weights) == 4
    assert np.count_nonzero(local_weights) <= 3
    assert executor.compile_metadata["autonomous_recovery_ranges"] == [
        [0, 3],
        [3, 6],
    ]


def test_autonomous_chunk_reentry_resynchronizes_interpolated_action() -> None:
    executor, controller_actions = _straight_executor(
        outer_radius=0.005,
        reentry_radius=0.001,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        autonomous_chunk_reentry=True,
        gripper_change_phase=None,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    executor.step(
        executor.plan.position_centers[1] + np.array([0.0, 0.02, 0.0]),
        IDENTITY_QUATERNION,
    )
    assert executor.recovering

    executor.step(
        executor.plan.position_centers[3] + np.array([0.0, 0.01, 0.0]),
        IDENTITY_QUATERNION,
    )
    assert executor.recovering
    assert executor.recovery_phase >= 3.0

    reentry_phase = 3.4
    reentry_position = (
        0.6 * executor.plan.position_centers[3]
        + 0.4 * executor.plan.position_centers[4]
    )
    step = executor.step(reentry_position, IDENTITY_QUATERNION)

    expected = 0.6 * controller_actions[3] + 0.4 * controller_actions[4]
    np.testing.assert_allclose(step.action[:6], expected, atol=1.0e-7)
    assert not executor.recovering
    assert np.isclose(executor.last_issued_phase, reentry_phase)
    assert np.isclose(executor.action_phase, reentry_phase + 1.0)
    assert executor.phase_reprojection_forward_count == 1
    assert executor.phase_reprojection_backward_count == 0


def test_autonomous_chunk_reentry_stops_before_gripper_event() -> None:
    executor, _ = _straight_executor(
        outer_radius=0.005,
        reentry_radius=0.001,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        autonomous_chunk_reentry=True,
        gripper_change_phase=3,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    executor.step(
        executor.plan.position_centers[1] + np.array([0.0, 0.02, 0.0]),
        IDENTITY_QUATERNION,
    )

    executor.step(
        executor.plan.position_centers[4] + np.array([0.0, 0.01, 0.0]),
        IDENTITY_QUATERNION,
    )
    assert executor.recovering
    assert executor.recovery_phase == 3.0

    event_step = executor.step(
        executor.plan.position_centers[3],
        IDENTITY_QUATERNION,
    )
    assert not executor.recovering
    assert event_step.action[6] == 1.0


def test_recovery_progress_watchdog_uses_state_not_elapsed_time() -> None:
    executor, _ = _straight_executor(
        freeze_recovery_phase=True,
        max_recovery_steps=0,
        recovery_progress_patience=1,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )

    first = executor.step(displaced, IDENTITY_QUATERNION)
    second = executor.step(displaced, IDENTITY_QUATERNION)

    assert not first.replan_required
    assert second.replan_required
    assert second.replan_reason == "recovery_no_progress"


def test_frozen_se3_field_recovers_large_pose_displacement_without_timeout(
) -> None:
    executor, _ = _straight_executor(
        outer_radius=0.04,
        reentry_radius=0.02,
        max_recovery_steps=0,
        recovery_field_source="geometry",
        freeze_recovery_phase=True,
        rotation_clf_enabled=True,
    )
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    position = executor.plan.position_centers[1] + np.array(
        [0.0, 0.05, 0.0]
    )
    quaternion = rotvec_to_quaternion(np.array([0.0, 0.0, 0.3]))

    for _ in range(300):
        step = executor.step(position, quaternion)
        assert not step.replan_required
        if not executor.recovering:
            break
        position += (
            step.action[:3] * executor.config.position_action_scale
        )
        rotation_delta = (
            step.action[3:6] * executor.config.rotation_action_scale
        )
        quaternion = quaternion_multiply(
            rotvec_to_quaternion(rotation_delta),
            quaternion,
        )
    else:
        raise AssertionError("frozen SE(3) field did not reenter the tube")

    position_error, rotation_error = executor._pose_errors_at_phase(
        position,
        quaternion,
        executor.last_issued_phase,
    )
    assert position_error <= executor.position_reentry_radius
    assert rotation_error <= executor.rotation_reentry_radius
    assert executor.recovery_entries == 1
    assert executor.recovery_reentries == 1


def test_support_confidence_field_returns_from_far_outside_support() -> None:
    executor, _ = _straight_executor(
        max_recovery_steps=0,
        recovery_field_source="geometry",
        recovery_target_mode="support_confidence",
        autonomous_chunk_reentry=True,
        replan_on_recovery_reentry=True,
        rotation_clf_enabled=True,
        gripper_change_phase=None,
    )
    _configure_confident_pose_support(executor)
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    position = executor.plan.position_centers[1] + np.array(
        [0.0, 0.25, 0.0]
    )
    quaternion = rotvec_to_quaternion(np.array([0.0, 0.0, 0.3]))
    initial_score, _ = executor._gaussian_support_score(
        position,
        quaternion,
        recovery=False,
        lower_phase=executor.last_issued_phase,
        upper_phase=executor.action_phase,
        record=False,
    )

    for _ in range(400):
        step = executor.step(position, quaternion)
        if step.replan_required:
            break
        position += step.action[:3] * executor.config.position_action_scale
        rotation_delta = (
            step.action[3:6] * executor.config.rotation_action_scale
        )
        quaternion = quaternion_multiply(
            rotvec_to_quaternion(rotation_delta),
            quaternion,
        )
    else:
        raise AssertionError("support field did not return from 25 cm")

    assert step.replan_reason == "recovery_reentry"
    assert executor.last_support_confidence_score >= (
        executor.support_confidence_inner_score
    )
    assert executor.last_support_confidence_score > initial_score
    assert executor.phase_reprojections == 0


def test_geometry_recovery_uses_pose_tangent_not_misaligned_action() -> None:
    action_executor, _ = _straight_executor(controller_axis=1)
    geometry_executor, _ = _straight_executor(
        controller_axis=1,
        recovery_field_source="geometry",
    )
    phase = 1.0
    position = action_executor.plan.position_centers[1]

    action_delta, _, _ = action_executor._local_full_field(
        position,
        IDENTITY_QUATERNION,
        phase,
    )
    geometry_delta, _, _ = geometry_executor._local_full_field(
        position,
        IDENTITY_QUATERNION,
        phase,
    )

    assert action_delta[1] > abs(action_delta[0])
    assert geometry_delta[0] > abs(geometry_delta[1])
    assert geometry_executor.compile_metadata["recovery_field_source"] == (
        "geometry"
    )


def test_geometry_recovery_preserves_native_cadence_after_densification() -> None:
    executor, _ = _straight_executor(
        recovery_field_source="geometry",
        phase_stride=2,
    )

    assert executor.geometry_nominal_twists is not None
    support_delta = (
        executor.plan.position_centers[2]
        - executor.plan.position_centers[1]
    )
    np.testing.assert_allclose(
        executor.geometry_nominal_twists[1, :3],
        2.0 * support_delta,
    )
    assert executor.compile_metadata["geometry_support_time"] == 0.5
    assert executor.compile_metadata["geometry_velocity_scale"] == 2.0


def test_projected_action_recovery_uses_pose_tangent() -> None:
    executor, _ = _straight_executor(
        recovery_field_source="projected_action",
        controller_axis=1,
    )
    phase = 1.0
    position = executor.plan.position_centers[1]

    delta, _, _ = executor._local_full_field(
        position,
        IDENTITY_QUATERNION,
        phase,
    )

    assert delta[0] > abs(delta[1])
    assert executor.compile_metadata["recovery_field_source"] == (
        "projected_action"
    )


def test_recovery_timeout_requests_visual_replan() -> None:
    executor, _ = _straight_executor(max_recovery_steps=1)
    executor.step(np.zeros(3), IDENTITY_QUATERNION)
    displaced = executor.plan.position_centers[1] + np.array(
        [0.0, 0.02, 0.0]
    )

    first = executor.step(displaced, IDENTITY_QUATERNION)
    second = executor.step(displaced, IDENTITY_QUATERNION)

    assert not first.replan_required
    assert second.replan_required
    assert second.replan_reason == "recovery_horizon"


def test_diffusion_rollout_adapter_constructs_hybrid_executor() -> None:
    module_path = (
        ROOT
        / "src"
        / "dsp_diffusion"
        / "policy"
        / "gaussian_action_adapter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_hybrid_action_adapter",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            policy=SimpleNamespace(
                lazy_eval=False,
                rollout_adapter=SimpleNamespace(
                    type="paired_bspline_hybrid_ds",
                    plan_steps=4,
                    bspline_source_horizon=4,
                    phase_position_tube_radius=0.04,
                    phase_position_reentry_radius=0.02,
                    recovery_field_source="geometry",
                    recovery_target_mode="next_action_phase",
                    freeze_recovery_phase=True,
                    autonomous_chunk_reentry=False,
                    replan_on_recovery_reentry=True,
                    rotation_clf_enabled=True,
                    rotation_clf_tube_radius=0.05,
                    recovery_progress_patience=12,
                ),
            )
        )
    )

    wrapped = module.maybe_wrap_policy_for_rollout(_FakePolicy(), cfg)

    assert isinstance(wrapped, LiberoPairedBSplineHybridDSAdapter)
    assert wrapped.phase_position_tube_radius == 0.04
    assert wrapped.phase_position_reentry_radius == 0.02
    assert wrapped.recovery_phase_advance_budget == 1.0
    assert wrapped.recovery_field_source == "geometry"
    assert wrapped.recovery_target_mode == "next_action_phase"
    assert wrapped.freeze_recovery_phase
    assert not wrapped.autonomous_chunk_reentry
    assert wrapped.replan_on_recovery_reentry
    assert wrapped.rotation_clf_enabled
    assert wrapped.rotation_clf_tube_radius == 0.05
    assert wrapped.recovery_progress_patience == 12

    paired = np.zeros((4, 13), dtype=np.float32)
    paired[:, 0] = 0.2
    paired[:, 6] = 0.1
    paired[:, 12] = -1.0
    executor = wrapped._compile_executor(
        paired,
        np.zeros(3),
        IDENTITY_QUATERNION,
    )
    assert isinstance(executor, ProjectedPhaseHybridExecutor)

