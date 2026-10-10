from __future__ import annotations

import json
from pathlib import Path
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from ns_gaussian_ds_policy.libero_paired_curve import (
    LiberoActionBSplineDirectAdapter,
    LiberoPairedBSplineDirectAdapter,
    LiberoPairedBSplineGeometryDirectAdapter,
    LiberoPairedBSplineFieldAdapter,
    LiberoPairedBSplineIntegratedFieldAdapter,
    LiberoPairedBSplinePoseMedoidPolicy,
    LiberoPairedBSplinePoseFieldAdapter,
    LiberoPairedPoseBSplineRawActionDirectAdapter,
    LiberoPairedPoseBSplineRawActionFieldAdapter,
    LiberoPairedPoseActionDirectAdapter,
    LiberoPairedPoseActionFieldAdapter,
    LiberoPairedPoseActionMedoidPolicy,
    build_libero_action_bspline_labels,
    build_libero_paired_bspline_labels,
    build_libero_paired_pose_bspline_raw_action_labels,
    build_libero_paired_pose_action_labels,
    decode_libero_action_bspline_controls,
    decode_libero_paired_bspline_controls,
    decode_libero_paired_pose_bspline_raw_action,
    find_gripper_only_boundary,
    paired_bspline_basis,
)


IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])
ROOT = Path(__file__).resolve().parents[1]


class _FakePairedPolicy(torch.nn.Module):
    def __init__(self, prediction: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.prediction = torch.as_tensor(prediction, dtype=torch.float32)
        self.predict_calls = 0
        self.reset_calls = 0

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def reset(self):
        self.reset_calls += 1

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

    def get_policy_name(self):
        return "fake_paired"

    def predict_action(self, obs_dict, **kwargs):
        del kwargs
        self.predict_calls += 1
        batch_size = int(next(iter(obs_dict.values())).shape[0])
        prediction = self.prediction
        if prediction.shape[0] == 1 and batch_size > 1:
            prediction = prediction.expand(batch_size, -1, -1)
        return {"action": prediction[:batch_size].clone()}


def _observations(batch_size: int = 1, x: float = 0.0):
    positions = np.tile([[x, 0.0, 0.0]], (batch_size, 1, 1))
    quaternions = np.tile(IDENTITY_QUATERNION, (batch_size, 1, 1))
    gripper_qpos = np.tile([[0.04, -0.04]], (batch_size, 1, 1))
    return {
        "robot0_eef_pos": torch.as_tensor(positions, dtype=torch.float32),
        "robot0_eef_quat": torch.as_tensor(quaternions, dtype=torch.float32),
        "robot0_gripper_qpos": torch.as_tensor(gripper_qpos, dtype=torch.float32),
    }


def test_libero_paired_labels_align_pose_and_native_action() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.01, 0.02, 0.0],
        ]
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (3, 1))
    actions = np.arange(21, dtype=np.float32).reshape(3, 7) / 20.0

    labels, targets, summary = build_libero_paired_pose_action_labels(
        positions,
        quaternions,
        actions,
        [3],
        clip=False,
    )

    assert labels.shape == (3, 13)
    np.testing.assert_array_equal(targets, [1, 2, 2])
    np.testing.assert_allclose(labels[:, :3], [[0.2, 0, 0], [0, 0.4, 0], [0, 0, 0]])
    np.testing.assert_allclose(labels[:, 3:6], 0.0)
    np.testing.assert_allclose(labels[:, 6:12], actions[:, :6])
    np.testing.assert_allclose(labels[:, 12], actions[:, 6])
    assert summary["action_dim"] == 13


def test_paired_bspline_basis_is_clamped_partition_of_unity() -> None:
    basis = paired_bspline_basis(
        np.arange(32, dtype=np.float64),
        num_control_points=8,
        degree=3,
        source_horizon=32,
    )

    assert basis.shape == (32, 8)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(basis[0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(basis[-1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_paired_bspline_labels_share_phase_and_reconstruct_line() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = 0.4
    actions[:, 6] = -1.0

    controls, summary = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    decoded = decode_libero_paired_bspline_controls(controls[:1])

    assert controls.shape == (40, 16, 13)
    np.testing.assert_allclose(controls[:, 0, :6], 0.0, atol=1.0e-7)
    np.testing.assert_allclose(controls[:, 0, 6:12], actions[:, :6], atol=1.0e-7)
    np.testing.assert_allclose(controls[:, 0, 12], actions[:, 6], atol=1.0e-7)
    assert decoded.shape == (1, 32, 13)
    np.testing.assert_allclose(decoded[0, :20, 0], 0.02, atol=2.0e-5)
    np.testing.assert_allclose(decoded[0, :, 6], 0.4, atol=2.0e-5)
    np.testing.assert_allclose(decoded[0, :, 12], -1.0, atol=2.0e-5)
    assert summary["layout"] == (
        "relative_pose_controls6 + native_arm_action_controls6 + "
        "discrete_gripper_controls1"
    )
    assert summary["fit_rmse"] < 0.01


def test_paired_bspline_densification_preserves_integer_phase_samples() -> None:
    positions = np.stack(
        [
            np.arange(40, dtype=np.float64) * 0.001,
            np.sin(np.arange(40, dtype=np.float64) * 0.2) * 0.002,
            np.zeros(40),
        ],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(0.1, 0.5, 40)
    actions[:18, 6] = -1.0
    actions[18:, 6] = 1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )

    coarse = decode_libero_paired_bspline_controls(controls[:1])
    dense = decode_libero_paired_bspline_controls(
        controls[:1],
        support_subdivisions=2,
    )

    assert coarse.shape == (1, 32, 13)
    assert dense.shape == (1, 63, 13)
    np.testing.assert_allclose(
        dense[:, ::2, 6:],
        coarse[:, :, 6:],
        atol=2.0e-6,
    )
    coarse_centers = np.concatenate(
        [np.zeros((1, 1, 3)), np.cumsum(coarse[..., :3], axis=1)],
        axis=1,
    )
    dense_centers = np.concatenate(
        [np.zeros((1, 1, 3)), np.cumsum(dense[..., :3], axis=1)],
        axis=1,
    )
    np.testing.assert_allclose(
        dense_centers[:, :33:2],
        coarse_centers[:, :17],
        atol=2.0e-6,
    )


def test_pose_bspline_raw_action_labels_preserve_every_native_step() -> None:
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
    actions = np.zeros((steps, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(-0.8, 0.8, steps)
    actions[:, 1] = np.sin(np.arange(steps) * 0.3) * 0.5
    actions[:9, 6] = -1.0
    actions[9:, 6] = 1.0

    paired_controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [steps],
        source_horizon=16,
        num_control_points=16,
    )
    controls, summary = build_libero_paired_pose_bspline_raw_action_labels(
        positions,
        quaternions,
        actions,
        [steps],
        source_horizon=16,
        num_control_points=16,
    )

    np.testing.assert_array_equal(controls[..., :6], paired_controls[..., :6])
    np.testing.assert_array_equal(
        controls[0, :, 6:],
        actions[:16].astype(np.float32),
    )
    np.testing.assert_array_equal(
        controls[-1, :, 6:],
        np.repeat(actions[-1:], 16, axis=0).astype(np.float32),
    )
    assert summary["action_representation"] == "exact_native_steps"
    assert summary["native_action_fit_rmse"] == 0.0

    dense = decode_libero_paired_pose_bspline_raw_action(
        controls[:1],
        output_steps=8,
        source_horizon=16,
        support_subdivisions=2,
    )
    assert dense.shape == (1, 15, 13)
    np.testing.assert_allclose(dense[0, ::2, 6:], actions[:8], atol=1.0e-7)


def test_pose_bspline_raw_action_direct_executes_exact_action_prefix() -> None:
    steps = 20
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
    actions[:, 6] = np.where(np.arange(steps) < 5, -1.0, 1.0)
    controls, _ = build_libero_paired_pose_bspline_raw_action_labels(
        positions,
        quaternions,
        actions,
        [steps],
        source_horizon=16,
        num_control_points=16,
    )
    adapter = LiberoPairedPoseBSplineRawActionDirectAdapter(
        _FakePairedPolicy(controls[:1]),
        source_horizon=16,
        execution_steps=8,
    )

    result = adapter.predict_action(_observations())

    np.testing.assert_allclose(
        result["action"].detach().cpu().numpy()[0],
        actions[:8],
        atol=1.0e-7,
    )
    assert (
        adapter.get_execution_metrics()["action_representation"]
        == "exact_native_steps"
    )


def test_action_bspline_labels_reconstruct_native_action_curve() -> None:
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(-0.4, 0.4, 40)
    actions[:, 2] = 0.25
    actions[:18, 6] = -1.0
    actions[18:, 6] = 1.0

    controls, summary = build_libero_action_bspline_labels(actions, [40])
    decoded = decode_libero_action_bspline_controls(controls[:1])

    assert controls.shape == (40, 16, 7)
    np.testing.assert_allclose(controls[:, 0, :6], actions[:, :6], atol=1.0e-7)
    assert decoded.shape == (1, 32, 7)
    np.testing.assert_allclose(decoded[0, :, 2], 0.25, atol=2.0e-5)
    assert set(np.unique(decoded[..., 6])).issubset({-1.0, 1.0})
    assert summary["action_dim"] == 7
    assert summary["layout"] == (
        "native_arm_action_controls6 + discrete_gripper_controls1"
    )
    assert summary["native_action_fit_rmse"] < 0.01


def test_paired_action_half_matches_action_only_bspline_exactly() -> None:
    steps = np.arange(48, dtype=np.float64)
    positions = np.stack(
        [
            0.001 * steps,
            0.002 * np.sin(steps * 0.2),
            0.001 * np.cos(steps * 0.1),
        ],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (len(steps), 1))
    actions = np.zeros((len(steps), 7), dtype=np.float64)
    actions[:, 0] = np.sin(steps * 0.3)
    actions[:, 1] = np.cos(steps * 0.2)
    actions[:, 5] = np.linspace(-0.5, 0.5, len(steps))
    actions[:, 6] = np.where((steps % 17) < 8, -1.0, 1.0)
    episode_ends = [19, 48]

    paired, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        episode_ends,
    )
    action_only, _ = build_libero_action_bspline_labels(
        actions,
        episode_ends,
    )
    paired_action = np.concatenate(
        [paired[..., 6:12], paired[..., 12:13]],
        axis=-1,
    )

    np.testing.assert_array_equal(paired_action, action_only)
    paired_decoded = decode_libero_paired_bspline_controls(paired[:3])
    paired_decoded_action = np.concatenate(
        [paired_decoded[..., 6:12], paired_decoded[..., 12:13]],
        axis=-1,
    )
    action_decoded = decode_libero_action_bspline_controls(action_only[:3])
    np.testing.assert_array_equal(paired_decoded_action, action_decoded)


def test_paired_bspline_decodes_stationary_gripper_event() -> None:
    positions = np.zeros((50, 3), dtype=np.float64)
    positions[:12, 0] = np.arange(12, dtype=np.float64) * 0.001
    positions[12:, 0] = positions[11, 0]
    quaternions = np.tile(IDENTITY_QUATERNION, (50, 1))
    actions = np.zeros((50, 7), dtype=np.float64)
    actions[:15, 6] = -1.0
    actions[15:, 6] = 1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [50],
    )
    decoded = decode_libero_paired_bspline_controls(controls[:1])[0]

    boundary = find_gripper_only_boundary(
        decoded,
        position_action_scale=0.05,
        rotation_action_scale=0.5,
        position_threshold=0.002,
        rotation_threshold=0.05,
    )

    assert boundary is not None
    assert boundary.command == 1.0
    assert np.linalg.norm(decoded[boundary.action_index, :3]) * 0.05 <= 0.002


def test_gripper_only_boundary_requires_small_pose_delta() -> None:
    paired = np.zeros((4, 13), dtype=np.float32)
    paired[:, 12] = [-1.0, -1.0, 1.0, 1.0]
    paired[2, :3] = [0.02, 0.0, 0.0]

    boundary = find_gripper_only_boundary(
        paired,
        position_action_scale=0.05,
        rotation_action_scale=0.5,
        position_threshold=0.002,
        rotation_threshold=0.05,
    )
    assert boundary is not None
    assert boundary.action_index == 2
    assert boundary.command == 1.0

    paired[2, :3] = [0.1, 0.0, 0.0]
    assert find_gripper_only_boundary(
        paired,
        position_action_scale=0.05,
        rotation_action_scale=0.5,
        position_threshold=0.002,
        rotation_threshold=0.05,
    ) is None


def test_direct_adapter_emits_native_action_curve() -> None:
    paired = np.zeros((1, 3, 13), dtype=np.float32)
    paired[0, :, :6] = 0.25
    paired[0, :, 6:12] = np.arange(18, dtype=np.float32).reshape(3, 6)
    paired[0, :, 12] = [-1.0, 1.0, 1.0]
    base = _FakePairedPolicy(paired)
    adapter = LiberoPairedPoseActionDirectAdapter(base)

    result = adapter.predict_action(_observations())

    assert result["action"].shape == (1, 3, 7)
    np.testing.assert_allclose(
        result["action"][0, :, :6].numpy(),
        paired[0, :, 6:12],
    )
    np.testing.assert_allclose(result["action"][0, :, 6].numpy(), paired[0, :, 12])
    assert adapter.get_execution_metrics()["policy_env_queries"] == 1


def test_direct_adapter_limits_raw_action_execution_prefix() -> None:
    paired = np.zeros((1, 16, 13), dtype=np.float32)
    paired[0, :, 6] = np.arange(16, dtype=np.float32)
    paired[0, :, 12] = -1.0
    adapter = LiberoPairedPoseActionDirectAdapter(
        _FakePairedPolicy(paired),
        execution_steps=8,
    )

    result = adapter.predict_action(_observations())

    assert result["action"].shape == (1, 8, 7)
    np.testing.assert_array_equal(
        result["action"][0, :, 0].numpy(),
        np.arange(8, dtype=np.float32),
    )
    assert adapter.get_execution_metrics()["execution_steps"] == 8


def test_bspline_direct_adapter_decodes_native_action_curve() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 1] = -0.3
    actions[:, 6] = 1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplineDirectAdapter(
        _FakePairedPolicy(controls[:1]),
    )

    result = adapter.predict_action(_observations())

    assert result["structured_action"].shape == (1, 16, 13)
    assert result["action"].shape == (1, 16, 7)
    np.testing.assert_allclose(result["action"][0, :, 1].numpy(), -0.3, atol=2.0e-5)
    np.testing.assert_allclose(result["action"][0, :, 6].numpy(), 1.0, atol=2.0e-5)


def test_bspline_geometry_direct_uses_pose_center_differences() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = -0.8
    actions[:, 6] = 1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplineGeometryDirectAdapter(
        _FakePairedPolicy(controls[:1]),
        geometry_command_gain=4.0,
    )

    result = adapter.predict_action(_observations())

    assert result["action"].shape == (1, 16, 7)
    np.testing.assert_allclose(
        result["action"][0, :-1, 0].numpy(),
        4.0 * 0.001 / 0.05,
        atol=2.0e-4,
    )
    assert not np.any(np.isclose(result["action"][0, :, 0].numpy(), -0.8))
    np.testing.assert_allclose(
        result["action"][0, :, 6].numpy(),
        1.0,
        atol=2.0e-5,
    )
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "paired_bspline_geometry_direct"
    assert metrics["arm_source"] == "decoded_pose_center_difference"
    assert metrics["action_bspline_arm_channels_used"] is False
    assert metrics["geometry_command_gain"] == 4.0


def test_pose_medoid_keeps_selected_pair_action_exact() -> None:
    controls = np.zeros((3, 16, 13), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    controls[0, :, 0] = 0.4 * phase
    controls[1, :, 0] = 0.42 * phase
    controls[1, :, 1] = 0.02 * phase
    controls[2, :, 1] = 2.0 * phase
    controls[0, :, 6] = 0.25
    controls[1, :, 6] = 0.85
    controls[2, :, 6] = -0.75
    controls[:, :, 12] = -1.0
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(controls),
        samples=3,
        selection_steps=8,
        cluster_cutoff=3.0,
        minimum_cluster_fraction=0.5,
        dense_core_fraction=0.5,
    )

    result = ensemble.predict_action(_observations())

    np.testing.assert_array_equal(
        result["action"][0].numpy(),
        controls[0],
    )
    support = ensemble.consume_pose_ensemble_support()[0]
    assert support.selected_sample_index == 0
    assert support.cluster_members == (0, 1)
    assert support.cluster_confident
    assert support.position_covariances.shape == (17, 3, 3)
    assert float(np.max(support.position_covariances[:, 1, 1])) > 0.0
    assert support.position_dense_core_covariances is not None
    assert support.position_dense_core_spread_radii is not None
    assert support.dense_core_fraction == 0.5
    metrics = ensemble.get_ensemble_metrics()
    assert metrics["pose_ensemble_action_aggregation"] == (
        "none_selected_pair_exact"
    )
    assert metrics["pose_ensemble_cluster_size_mean"] == 2.0
    assert metrics["pose_ensemble_selection_native_steps"] == 8
    assert metrics["pose_ensemble_selection_decoded_steps"] == 16
    assert metrics["pose_ensemble_dense_core_size_mean"] == 1.0


def test_raw_pose_action_medoid_keeps_every_selected_step_exact() -> None:
    predictions = np.zeros((3, 8, 13), dtype=np.float32)
    predictions[0, :, 0] = 0.10
    predictions[1, :, 0] = 0.11
    predictions[2, :, 1] = 0.75
    predictions[0, :, 6:12] = 0.25
    predictions[1, :, 6:12] = 0.85
    predictions[2, :, 6:12] = -0.75
    predictions[:, :, 12] = -1.0
    ensemble = LiberoPairedPoseActionMedoidPolicy(
        _FakePairedPolicy(predictions),
        samples=3,
        source_horizon=16,
        selection_steps=8,
        support_subdivisions=1,
    )

    result = ensemble.predict_action(_observations())

    np.testing.assert_array_equal(result["action"][0].numpy(), predictions[0])
    support = ensemble.consume_pose_ensemble_support()[0]
    assert support.selected_sample_index == 0
    assert support.position_covariances.shape == (9, 3, 3)
    metrics = ensemble.get_ensemble_metrics()
    assert metrics["pose_ensemble_selection_native_steps"] == 8
    assert metrics["pose_ensemble_selection_decoded_steps"] == 8


def test_raw_pose_action_medoid_rejects_densification() -> None:
    with pytest.raises(ValueError, match="support_subdivisions=1"):
        LiberoPairedPoseActionMedoidPolicy(
            _FakePairedPolicy(np.zeros((2, 8, 13), dtype=np.float32)),
            samples=2,
            source_horizon=16,
            selection_steps=8,
            support_subdivisions=2,
        )


def test_all_sample_medoid_uses_k_stable_scale_and_all_pose_samples() -> None:
    controls = np.zeros((3, 16, 13), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    controls[0, :, 0] = 0.4 * phase
    controls[1, :, 0] = 0.42 * phase
    controls[1, :, 1] = 0.02 * phase
    controls[2, :, 1] = 2.0 * phase
    controls[0, :, 6] = 0.25
    controls[1, :, 6] = 0.85
    controls[2, :, 6] = -0.75
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(controls),
        samples=3,
        selection_steps=8,
        selection_mode="all_samples",
        distance_scale_mode="pairwise_median",
    )

    result = ensemble.predict_action(_observations())

    np.testing.assert_array_equal(result["action"][0].numpy(), controls[1])
    support = ensemble.consume_pose_ensemble_support()[0]
    assert support.selected_sample_index == 1
    assert support.cluster_members == (0, 1, 2)
    assert support.cluster_confident
    assert support.position_distance_scale > 0.0
    metrics = ensemble.get_ensemble_metrics()
    assert metrics["pose_ensemble_selection"] == (
        "pose_only_all_sample_medoid"
    )
    assert metrics["pose_ensemble_selection_mode"] == "all_samples"
    assert metrics["pose_ensemble_distance_scale_mode"] == (
        "pairwise_median"
    )


def test_cluster_medoid_can_estimate_support_from_all_samples() -> None:
    controls = np.zeros((3, 16, 13), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    controls[0, :, 0] = 0.4 * phase
    controls[1, :, 0] = 0.42 * phase
    controls[1, :, 1] = 0.02 * phase
    controls[2, :, 1] = 2.0 * phase
    controls[0, :, 6] = 0.25
    controls[1, :, 6] = 0.85
    controls[2, :, 6] = -0.75
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(controls),
        samples=3,
        selection_steps=8,
        cluster_cutoff=3.0,
        minimum_cluster_fraction=0.5,
        support_members_mode="all_samples",
    )

    result = ensemble.predict_action(_observations())

    np.testing.assert_array_equal(result["action"][0].numpy(), controls[0])
    support = ensemble.consume_pose_ensemble_support()[0]
    assert support.selected_sample_index == 0
    assert support.cluster_members == (0, 1, 2)
    assert support.cluster_confident
    metrics = ensemble.get_ensemble_metrics()
    assert metrics["pose_ensemble_selection"] == (
        "pose_only_dominant_cluster_medoid"
    )
    assert metrics["pose_ensemble_support_members_mode"] == "all_samples"


def test_manifold_kde_selects_real_paired_mode_and_weights_support() -> None:
    controls = np.zeros((5, 16, 13), dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    controls[0, :, 0] = 0.20 * phase
    controls[0, :, 1] = -0.01 * phase
    controls[1, :, 0] = 0.20 * phase
    controls[2, :, 0] = 0.20 * phase
    controls[2, :, 1] = 0.01 * phase
    controls[3, :, 0] = -0.20 * phase
    controls[3, :, 1] = 0.50 * phase
    controls[4, :, 0] = -0.20 * phase
    controls[4, :, 1] = 0.52 * phase
    controls[:, :, 6] = np.asarray(
        [0.10, 0.70, 0.20, -0.60, -0.40],
        dtype=np.float32,
    )[:, None]
    controls[:, :, 12] = -1.0
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(controls),
        samples=5,
        selection_steps=8,
        selection_mode="manifold_kde",
        support_members_mode="kde_responsibility",
        minimum_cluster_fraction=0.4,
        kde_position_bandwidth=0.001,
        kde_rotation_bandwidth=0.25,
        kde_gripper_bandwidth=1.0,
        kde_support_mass=0.9,
    )

    result = ensemble.predict_action(_observations())

    support = ensemble.consume_pose_ensemble_support()[0]
    assert support.selected_sample_index in {0, 1, 2}
    np.testing.assert_array_equal(
        result["action"][0].numpy(),
        controls[support.selected_sample_index],
    )
    assert support.cluster_members == (0, 1, 2)
    assert support.cluster_confident
    assert support.sample_weights is not None
    assert support.sample_weights.shape == (5,)
    assert np.max(support.sample_weights[3:]) < 1.0e-3
    assert support.selected_density is not None
    assert support.effective_sample_size is not None
    assert 2.0 < support.effective_sample_size < 3.1
    assert support.kernel_mass_fraction is not None
    assert float(np.max(support.position_spread_radii)) < 0.01
    metrics = ensemble.get_ensemble_metrics()
    assert metrics["pose_ensemble_selection"] == (
        "pose_gripper_manifold_kde_mode"
    )
    assert metrics["pose_ensemble_support_members_mode"] == (
        "kde_responsibility"
    )
    assert metrics["pose_ensemble_kde_effective_sample_size_mean"] == (
        support.effective_sample_size
    )


def test_manifold_kde_separates_incompatible_gripper_modes() -> None:
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(np.zeros((4, 16, 13), dtype=np.float32)),
        samples=4,
        selection_mode="manifold_kde",
        support_members_mode="kde_responsibility",
        minimum_cluster_fraction=0.5,
        kde_position_bandwidth=1.0,
        kde_rotation_bandwidth=1.0,
        kde_gripper_bandwidth=0.1,
    )
    positions = np.zeros((4, 5, 3), dtype=np.float64)
    positions[:, :, 0] = np.asarray([-0.02, 0.02, 0.04, 0.0])[:, None]
    orientations = np.tile(IDENTITY_QUATERNION, (4, 5, 1))
    grippers = np.ones((4, 4), dtype=np.float64)
    grippers[:3] = -1.0

    members, selected, confident, weights, *_ = (
        ensemble._manifold_kde_selection(
            positions,
            orientations,
            grippers,
        )
    )

    assert selected in {0, 1, 2}
    assert 3 not in members
    assert confident
    assert weights[3] < 1.0e-10


def test_kde_responsibility_requires_kde_selection() -> None:
    with pytest.raises(ValueError, match="requires manifold_kde"):
        LiberoPairedBSplinePoseMedoidPolicy(
            _FakePairedPolicy(np.zeros((4, 16, 13), dtype=np.float32)),
            samples=4,
            support_members_mode="kde_responsibility",
        )


def test_pose_dense_core_rejects_within_cluster_phase_outliers() -> None:
    ensemble = LiberoPairedBSplinePoseMedoidPolicy(
        _FakePairedPolicy(np.zeros((8, 16, 13), dtype=np.float32)),
        samples=8,
        dense_core_fraction=0.75,
    )
    phase_count = 5
    positions = np.zeros((8, phase_count, 3), dtype=np.float64)
    positions[:, :, 0] = np.linspace(0.0, 0.04, phase_count)
    positions[:, :, 1] = np.asarray(
        [0.0, 0.002, -0.002, 0.003, -0.003, 0.004, 0.03, -0.04]
    )[:, None]
    orientations = np.tile(
        IDENTITY_QUATERNION,
        (8, phase_count, 1),
    )

    support = ensemble._pose_support(
        positions,
        orientations,
        list(range(8)),
        0,
        cluster_confident=True,
        position_scale=1.0,
        rotation_scale=1.0,
    )

    assert support.position_dense_core_covariances is not None
    assert support.position_dense_core_spread_radii is not None
    assert np.all(
        support.position_dense_core_covariances[:, 1, 1]
        < support.position_covariances[:, 1, 1]
    )
    assert np.all(
        support.position_dense_core_spread_radii
        < support.position_spread_radii
    )


def test_action_bspline_direct_adapter_decodes_native_action_curve() -> None:
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 1] = -0.3
    actions[:, 6] = 1.0
    controls, _ = build_libero_action_bspline_labels(actions, [40])
    adapter = LiberoActionBSplineDirectAdapter(
        _FakePairedPolicy(controls[:1]),
    )

    result = adapter.predict_action(_observations())

    assert result["structured_action"].shape == (1, 16, 7)
    assert result["action"].shape == (1, 16, 7)
    np.testing.assert_allclose(result["action"][0, :, 1].numpy(), -0.3, atol=2.0e-5)
    np.testing.assert_allclose(result["action"][0, :, 6].numpy(), 1.0, atol=2.0e-5)
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "action_bspline_direct"
    assert metrics["representation"] == "action_bspline"


def test_field_adapter_caches_prediction_and_matches_command_on_support() -> None:
    paired = np.zeros((1, 3, 13), dtype=np.float32)
    paired[0, :, 0] = 0.1
    paired[0, :, 6] = 0.4
    paired[0, :, 12] = -1.0
    base = _FakePairedPolicy(paired)
    adapter = LiberoPairedPoseActionFieldAdapter(
        base,
        plan_steps=3,
        terminal_stall_replan=False,
    )
    obs = _observations(batch_size=2)

    first = adapter.predict_action(obs)["action"]
    second = adapter.predict_action(obs)["action"]

    assert first.shape == (2, 1, 7)
    assert second.shape == (2, 1, 7)
    np.testing.assert_allclose(first[:, 0, 0].numpy(), 0.4, atol=1.0e-5)
    np.testing.assert_allclose(second[:, 0, 0].numpy(), 0.4, atol=1.0e-5)
    assert base.predict_calls == 1
    assert adapter.get_execution_metrics()["policy_env_queries"] == 2


def test_field_adapter_uses_ordered_native_actions_inside_pose_tube() -> None:
    paired = np.zeros((1, 3, 13), dtype=np.float32)
    paired[0, :, 0] = 0.1
    paired[0, :, 6] = [0.1, 0.2, 0.3]
    paired[0, :, 12] = -1.0
    adapter = LiberoPairedPoseActionFieldAdapter(
        _FakePairedPolicy(paired),
        plan_steps=3,
        terminal_stall_replan=False,
    )
    obs = _observations()

    emitted = np.array(
        [adapter.predict_action(obs)["action"][0, 0, 0].item() for _ in range(3)]
    )

    np.testing.assert_allclose(emitted, [0.1, 0.2, 0.3], atol=1.0e-6)
    metrics = adapter.get_execution_metrics()
    assert metrics["phase_conditioned_execution"] is True
    assert metrics["nominal_execution"] == "exact_native_policy_command"
    assert metrics["recovery_execution"] == "cached_gaussian_pose_action_field"
    assert metrics["native_action_steps"] == 3
    assert metrics["field_recovery_steps"] == 0


def test_bspline_field_adapter_uses_same_compiler_after_decode() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = 0.4
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplineFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=16,
        terminal_stall_replan=False,
    )

    action = adapter.predict_action(_observations())["action"]

    assert action.shape == (1, 1, 7)
    np.testing.assert_allclose(action[0, 0, 0].numpy(), 0.4, atol=2.0e-4)
    metrics = adapter.get_execution_metrics()
    assert metrics["representation"] == "paired_bspline"
    assert metrics["nominal_execution"] == "exact_native_policy_command"
    assert metrics["tube_projection_enabled"] is True
    assert metrics["tube_radius"] == 0.01
    assert metrics["phase_position_tube_radius"] == 0.075
    assert metrics["direction_preserving_saturation"] is True


def test_bspline_field_densifies_supports_without_changing_action_cadence() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(0.1, 0.5, 40)
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    coarse_adapter = LiberoPairedBSplineFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=4,
        terminal_stall_replan=False,
        support_subdivisions=1,
    )
    dense_adapter = LiberoPairedBSplineFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=4,
        terminal_stall_replan=False,
        support_subdivisions=2,
    )
    prediction = torch.as_tensor(controls[:1], dtype=torch.float32)
    coarse = coarse_adapter._decode_prediction(prediction).numpy()[0]
    dense = dense_adapter._decode_prediction(prediction).numpy()[0]
    coarse_executor = coarse_adapter._compile_executor(
        coarse,
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )
    dense_executor = dense_adapter._compile_executor(
        dense,
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )

    assert coarse_executor.target_count == 4
    assert dense_executor.target_count == 8
    assert coarse_executor.phase_stride == 1
    assert dense_executor.phase_stride == 2
    np.testing.assert_allclose(
        dense_executor.controller_actions[::2],
        coarse_executor.controller_actions,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        dense_executor.plan.position_centers[::2],
        coarse_executor.plan.position_centers,
        atol=2.0e-6,
    )
    metrics = dense_adapter.get_execution_metrics()
    assert metrics["native_plan_steps"] == 4
    assert metrics["field_support_steps"] == 8
    assert metrics["phase_stride"] == 2
    assert metrics["support_subdivisions"] == 2


def test_raw_action_field_keeps_exact_commands_at_native_phases() -> None:
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
    adapter = LiberoPairedPoseBSplineRawActionFieldAdapter(
        _FakePairedPolicy(controls[:1]),
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

    assert executor.phase_stride == 2
    np.testing.assert_allclose(
        executor.controller_actions[::2],
        actions[:8, :6],
        atol=1.0e-7,
    )
    assert (
        adapter.get_execution_metrics()["action_representation"]
        == "exact_native_steps"
    )


def test_bspline_field_densification_preserves_gripper_event_phase() -> None:
    positions = np.zeros((50, 3), dtype=np.float64)
    positions[:12, 0] = np.arange(12, dtype=np.float64) * 0.001
    positions[12:, 0] = positions[11, 0]
    quaternions = np.tile(IDENTITY_QUATERNION, (50, 1))
    actions = np.zeros((50, 7), dtype=np.float64)
    actions[:12, 6] = -1.0
    actions[12:, 6] = 1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [50],
    )
    prediction = torch.as_tensor(controls[:1], dtype=torch.float32)
    executors = []
    for subdivisions in (1, 2):
        adapter = LiberoPairedBSplineFieldAdapter(
            _FakePairedPolicy(controls[:1]),
            plan_steps=16,
            terminal_stall_replan=False,
            support_subdivisions=subdivisions,
        )
        decoded = adapter._decode_prediction(prediction).numpy()[0]
        executors.append(
            adapter._compile_executor(
                decoded,
                np.zeros(3, dtype=np.float64),
                IDENTITY_QUATERNION,
            )
        )

    coarse, dense = executors
    assert coarse.terminal_gripper_command == 1.0
    assert dense.terminal_gripper_command == 1.0
    assert dense.target_count == 2 * coarse.target_count
    assert dense.phase_stride == 2


def test_bspline_integrated_field_uses_native_actions_as_geometry() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = 0.4
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplineIntegratedFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=4,
        terminal_stall_replan=False,
    )

    decoded = adapter._decode_prediction(
        torch.as_tensor(controls[:1], dtype=torch.float32)
    ).numpy()
    np.testing.assert_allclose(decoded[..., :6], decoded[..., 6:12])

    executor = adapter._compile_executor(
        decoded[0],
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )
    expected_x = np.arange(5, dtype=np.float64) * 0.4 * 0.05
    np.testing.assert_allclose(
        executor.plan.position_centers[:, 0],
        expected_x,
        atol=2.0e-5,
    )
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "paired_bspline_integrated_field"
    assert metrics["geometry_source"] == "integrated_native_action"


def test_bspline_pose_only_field_discards_arm_action_spline() -> None:
    positions = np.stack(
        [np.arange(40, dtype=np.float64) * 0.001, np.zeros(40), np.zeros(40)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 1] = 0.8
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplinePoseFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=4,
        terminal_stall_replan=False,
        support_subdivisions=2,
    )

    decoded = adapter._decode_prediction(
        torch.as_tensor(controls[:1], dtype=torch.float32)
    ).numpy()

    np.testing.assert_allclose(
        decoded[..., 6:12],
        decoded[..., :6] * 2.0,
    )
    assert np.max(np.abs(decoded[..., 7])) < 1.0e-6
    executor = adapter._compile_executor(
        decoded[0],
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )
    assert executor.__class__.__name__ == "PoseIndexedActionFieldExecutor"
    metrics = adapter.get_execution_metrics()
    assert metrics["executor"] == "paired_bspline_pose_only_field"
    assert metrics["nominal_velocity_source"] == "pose_center_difference"
    assert not metrics["arm_action_bspline_used"]
    assert metrics["shared_gripper_schedule_used"]


def test_bspline_integrated_densification_scales_geometry_only() -> None:
    positions = np.zeros((40, 3), dtype=np.float64)
    quaternions = np.tile(IDENTITY_QUATERNION, (40, 1))
    actions = np.zeros((40, 7), dtype=np.float64)
    actions[:, 0] = 0.4
    actions[:, 6] = -1.0
    controls, _ = build_libero_paired_bspline_labels(
        positions,
        quaternions,
        actions,
        [40],
    )
    adapter = LiberoPairedBSplineIntegratedFieldAdapter(
        _FakePairedPolicy(controls[:1]),
        plan_steps=4,
        terminal_stall_replan=False,
        support_subdivisions=2,
    )

    decoded = adapter._decode_prediction(
        torch.as_tensor(controls[:1], dtype=torch.float32)
    ).numpy()

    np.testing.assert_allclose(
        decoded[..., :6],
        decoded[..., 6:12] / 2.0,
    )
    executor = adapter._compile_executor(
        decoded[0],
        np.zeros(3, dtype=np.float64),
        IDENTITY_QUATERNION,
    )
    assert executor.target_count == 8
    assert executor.phase_stride == 2
    np.testing.assert_allclose(
        executor.controller_actions[::2, 0],
        0.4,
        atol=2.0e-5,
    )


def test_libero_field_splits_and_handoffs_at_self_intersection() -> None:
    centers = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.01, 0.00],
            [0.00, 0.01, 0.00],
            [0.01, 0.00, 0.00],
            [0.02, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    paired = np.zeros((4, 13), dtype=np.float32)
    paired[:, :3] = np.diff(centers, axis=0) / 0.05
    paired[:, 6:9] = paired[:, :3]
    paired[:, 12] = -1.0
    adapter = LiberoPairedPoseActionFieldAdapter(
        _FakePairedPolicy(paired[None]),
        plan_steps=4,
        terminal_stall_replan=False,
        tube_projection_enabled=False,
    )
    executor = adapter._compile_executor(
        paired,
        centers[0],
        IDENTITY_QUATERNION,
    )

    assert executor.chunk_ranges == ((0, 1), (1, 4))
    position = centers[0].copy()
    visited_chunks = [executor.current_chunk]
    for _ in range(20):
        step = executor.step(
            position,
            IDENTITY_QUATERNION,
            np.array([0.04, -0.04]),
        )
        if executor.current_chunk != visited_chunks[-1]:
            visited_chunks.append(executor.current_chunk)
        if step.plan_complete:
            break
        assert not step.replan_required
        position += step.action[:3] * 0.05
    else:
        raise AssertionError("Self-intersection chunks did not complete")

    assert visited_chunks == [0, 1]
    np.testing.assert_allclose(position, centers[-1], atol=2.0e-3)


def test_field_adapter_executes_gripper_only_event_before_replan() -> None:
    paired = np.zeros((1, 3, 13), dtype=np.float32)
    paired[0, 0, 0] = 0.1
    paired[0, 0, 6] = 0.2
    paired[0, :, 12] = [-1.0, 1.0, 1.0]
    base = _FakePairedPolicy(paired)
    adapter = LiberoPairedPoseActionFieldAdapter(
        base,
        plan_steps=3,
        terminal_stall_replan=False,
    )

    approach = adapter.predict_action(_observations(x=0.0))["action"]
    gripper_event = adapter.predict_action(_observations(x=0.005))["action"]

    np.testing.assert_allclose(approach[0, 0, 0].numpy(), 0.2, atol=1.0e-5)
    np.testing.assert_allclose(gripper_event[0, 0, :6].numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(gripper_event[0, 0, 6].numpy(), 1.0)
    assert base.predict_calls == 1

    adapter.predict_action(_observations(x=0.005))
    metrics = adapter.get_execution_metrics()
    assert base.predict_calls == 2
    assert metrics["gripper_event_replans"] == 1
    assert metrics["plan_completion_reasons"] == {
        "gripper_only_transition": 1
    }
    assert metrics["replan_reasons"] == {}


def test_field_adapter_counts_hard_timeout_replan_reason() -> None:
    paired = np.zeros((1, 3, 13), dtype=np.float32)
    paired[0, :, 0] = 0.2
    paired[0, :, 6] = 0.2
    paired[0, :, 12] = -1.0
    base = _FakePairedPolicy(paired)
    adapter = LiberoPairedPoseActionFieldAdapter(
        base,
        plan_steps=3,
        max_plan_steps=1,
        terminal_stall_replan=False,
        tube_projection_enabled=False,
    )

    adapter.predict_action(_observations(x=0.0))
    adapter.predict_action(_observations(x=0.0))

    metrics = adapter.get_execution_metrics()
    assert base.predict_calls == 2
    assert metrics["completed_plans"] == 0
    assert metrics["replan_requests"] == 1
    assert metrics["plan_completion_reasons"] == {}
    assert metrics["replan_reasons"] == {"hard_timeout": 1}




def test_lazy_training_skips_rollout_adapter_construction() -> None:
    module_path = (
        ROOT / "src" / "dsp_diffusion" / "policy" /
        "gaussian_action_adapter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_gaussian_action_adapter",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = _FakePairedPolicy(np.zeros((1, 1, 13), dtype=np.float32))
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            policy=SimpleNamespace(
                lazy_eval=True,
                rollout_adapter=SimpleNamespace(type="paired_pose_action_field"),
            )
        )
    )

    assert module.maybe_wrap_policy_for_rollout(policy, cfg) is policy


def test_label_cli_writes_paired_arrays_to_zarr(tmp_path: Path) -> None:
    path = tmp_path / "tiny_libero.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    quaternions = np.tile(IDENTITY_QUATERNION, (2, 1))
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = [-1.0, 1.0]
    data.array("robot0_eef_pos", positions)
    data.array("robot0_eef_quat", quaternions)
    data.array("action", actions)
    meta.array("episode_ends", np.array([2], dtype=np.int64))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_libero_paired_pose_action_labels.py"),
            "--input",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = zarr.open(str(path), mode="r")["data"]
    assert stored["paired_pose_action13_s1"].shape == (2, 13)
    assert stored["paired_pose_action13_s1_target_idx"].shape == (2,)

    rerun = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_libero_paired_pose_action_labels.py"),
            "--input",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "skipping label construction" in rerun.stdout


def test_label_cli_writes_precomputed_bspline_windows(tmp_path: Path) -> None:
    path = tmp_path / "tiny_libero_bspline.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    positions = np.stack(
        [np.arange(12, dtype=np.float64) * 0.001, np.zeros(12), np.zeros(12)],
        axis=1,
    )
    quaternions = np.tile(IDENTITY_QUATERNION, (12, 1))
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    data.array("robot0_eef_pos", positions)
    data.array("robot0_eef_quat", quaternions)
    data.array("action", actions)
    meta.array("episode_ends", np.array([12], dtype=np.int64))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_libero_paired_pose_action_labels.py"),
            "--input",
            str(path),
            "--representation",
            "paired_bspline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = zarr.open(str(path), mode="r")["data"]
    assert stored["paired_bspline13_cp16_h32"].shape == (12, 16, 13)
    assert "paired_bspline13_cp16_h32_target_idx" not in stored

    rerun = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_libero_paired_pose_action_labels.py"),
            "--input",
            str(path),
            "--representation",
            "paired_bspline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "skipping label construction" in rerun.stdout


def test_label_cli_writes_action_bspline_windows(tmp_path: Path) -> None:
    path = tmp_path / "tiny_libero_action_bspline.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    actions = np.zeros((12, 7), dtype=np.float32)
    actions[:, 0] = 0.2
    actions[:, 6] = -1.0
    data.array("action", actions)
    meta.array("episode_ends", np.array([12], dtype=np.int64))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_libero_paired_pose_action_labels.py"),
            "--input",
            str(path),
            "--representation",
            "action_bspline",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    stored = zarr.open(str(path), mode="r")["data"]
    assert stored["action_bspline7_cp16_h32"].shape == (12, 16, 7)
    assert "action_bspline7_cp16_h32_target_idx" not in stored


def test_window_zarr_dataset_returns_current_precomputed_curve(
    tmp_path: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    module_path = (
        ROOT / "src" / "dsp_diffusion" / "dataset" /
        "window_zarr_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_window_zarr_dataset",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "window_dataset.zarr"
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.array("state", np.arange(15, dtype=np.float32).reshape(5, 3))
    curves = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    data.array("paired", curves)
    meta.array("episode_ends", np.array([5], dtype=np.int64))

    dataset = module.WindowZarrDataset(
        zarr_path=str(path),
        obs_keys=["state"],
        action_key="paired",
        n_obs_steps=2,
        n_action_steps=4,
        val_ratio=0.0,
    )

    assert tuple(dataset[0]["action"].shape) == (4, 2)
    np.testing.assert_array_equal(dataset[0]["action"].numpy(), curves[0])
    np.testing.assert_array_equal(dataset[3]["action"].numpy(), curves[3])


def test_libero_rollout_selector_uses_success_then_val_loss(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "select_libero_rollout_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "test_select_libero_rollout_checkpoint",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    evaluation_root = tmp_path / "evaluation"
    candidates = [
        (0, 0.20, 7),
        (200, 0.10, 8),
        (400, 0.08, 8),
    ]
    for index, (epoch, val_loss, successes) in enumerate(candidates):
        checkpoint = checkpoint_dir / f"ep-{epoch:04d}_vl-{val_loss:.6f}.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        result_dir = evaluation_root / f"candidate_{index}"
        result_dir.mkdir(parents=True)
        success_values = {
            f"task/success_{200000 + rollout}": float(rollout < successes)
            for rollout in range(10)
        }
        result = {
            "checkpoint": str(checkpoint),
            "mode": "paired_pose_action_field",
            "num_exp": 1,
            "n_test": 10,
            "n_parallel_envs": 1,
            "mean_success_rate": successes / 10,
            "args": {
                "test_start_seed": 200000,
                "method_profile": "last_safe_v1",
                "eval_rng_mode": "per_episode",
                "eval_seed": 300000,
            },
            "episode_sampling_seeds": list(range(300000, 300010)),
            "episode_outcomes": [
                {"success": rollout < successes}
                for rollout in range(10)
            ],
            "runs": [success_values],
        }
        (result_dir / "eval_log.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )

    report = module.select_checkpoint(
        evaluation_root,
        run_dir,
        expected_mode="paired_pose_action_field",
        expected_method_profile="last_safe_v1",
        expected_eval_rng_mode="per_episode",
        expected_eval_seed=300000,
        expected_n_parallel_envs=1,
        expected_start_seed=200000,
        expected_rollouts=10,
        expected_candidates=3,
    )
    assert report["selected"]["epoch"] == 400
    assert report["selected"]["successes"] == 8
    assert report["selected"]["val_loss"] == 0.08
