from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ns_gaussian_ds_policy.libero_hybrid_perturbation import (
    LiberoHybridPerturbationAdapter,
)


class _FakeExecutor:
    def __init__(self) -> None:
        self.action_phase = 4.0
        self.last_issued_phase = 2.0
        self.recovery_phase = 4.0
        self.recovering = False
        self.target_count = 8
        self.last_gripper_command = -1.0
        self.plan = SimpleNamespace(
            position_centers=np.stack(
                [np.array([0.01 * index, 0.0, 0.0]) for index in range(9)]
            )
        )
        self.observed_recovery_anchor = None
        self.external_gripper_transition_registered = False

    def register_observed_recovery_anchor(self, position, quaternion) -> bool:
        self.observed_recovery_anchor = (
            np.asarray(position).copy(),
            np.asarray(quaternion).copy(),
        )
        return True

    def register_external_gripper_transition(self) -> bool:
        self.external_gripper_transition_registered = True
        return True

    def _next_gripper_event_phase(self, phase: float) -> float:
        del phase
        return 6.0

    def _pose_errors_at_phase(self, position, quaternion, phase):
        del quaternion, phase
        return float(np.linalg.norm(position)), 0.0

    def _position_outer_radius(self, phase):
        del phase
        return 0.04

    def _position_reentry_radius(self, phase):
        del phase
        return 0.02


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._executors = [_FakeExecutor()]
        self.calls = 0
        self.diagnostic_records = []

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        return None

    def predict_action(self, obs_dict, **kwargs):
        del obs_dict, kwargs
        self.calls += 1
        executor = self._executors[0]
        action = torch.zeros((1, 1, 7), dtype=torch.float32)
        if self.calls == 1:
            executor.recovering = True
            executor.recovery_phase = 4.25
            action[0, 0, 6] = 0.0
        elif self.calls == 2:
            executor.recovering = False
            executor.last_issued_phase = 5.0
            action[0, 0, 6] = -1.0
        else:
            executor.last_issued_phase = 6.0
            executor.last_gripper_command = 1.0
            action[0, 0, 6] = 1.0
        return {"action": action}

    def get_execution_metrics(self):
        return {"base_calls": self.calls}

    def append_controller_diagnostic_record(self, record, **kwargs):
        self.diagnostic_records.append((record, kwargs))


class _FakeReplanPolicy(_FakePolicy):
    def predict_action(self, obs_dict, **kwargs):
        result = super().predict_action(obs_dict, **kwargs)
        if self.calls == 2:
            self._executors[0] = _FakeExecutor()
        return result


def _observation(
    position,
    *,
    gripper_aperture: float = 0.04,
) -> dict[str, torch.Tensor]:
    return {
        "robot0_eef_pos": torch.tensor(
            [[position]],
            dtype=torch.float32,
        ),
        "robot0_eef_quat": torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ),
        "robot0_gripper_qpos": torch.tensor(
            [[[gripper_aperture, -gripper_aperture]]],
            dtype=torch.float32,
        ),
    }


def test_perturbation_does_not_advance_policy_or_skip_gripper_event() -> None:
    base = _FakePolicy()
    policy = LiberoHybridPerturbationAdapter(
        base,
        radius_m=0.04,
        reach_fraction=0.9,
    )

    perturb = policy.predict_action(_observation([0.0, 0.0, 0.0]))
    assert base.calls == 0
    assert perturb["action"][0, 0, 1] > 0.0
    assert perturb["action"][0, 0, 6] == 0.0
    assert base.diagnostic_records[0][0]["mode"] == "perturb"
    assert base.diagnostic_records[0][0]["gripper_command"] == 0.0
    assert base.diagnostic_records[0][1] == {
        "env_index": 0,
        "batch_size": 1,
    }
    anchor_position, anchor_quaternion = (
        base._executors[0].observed_recovery_anchor
    )
    np.testing.assert_allclose(anchor_position, np.zeros(3))
    np.testing.assert_allclose(
        anchor_quaternion,
        [0.0, 0.0, 0.0, 1.0],
    )

    recovery = policy.predict_action(_observation([0.0, 0.04, 0.0]))
    assert base.calls == 1
    assert recovery["action"][0, 0, 6] == 0.0

    policy.predict_action(_observation([0.04, 0.0, 0.0]))
    event = policy.predict_action(_observation([0.05, 0.0, 0.0]))
    assert event["action"][0, 0, 6] == 1.0

    record = policy.get_execution_metrics()["controlled_perturbation"][
        "record"
    ]
    assert record["recovery_observed"]
    assert record["recovery_reentered"]
    assert record["first_gripper_transition_phase"] == 6.0
    assert record["gripper_changed_during_perturbation"] is False
    assert (
        record[
            "recovery_reentry_cache_requested_for_gripper_transition"
        ]
        is False
    )
    assert record["gripper_event_skipped"] is False
    assert record["neutral_gripper_action"] == 0.0
    assert record["recovery_anchor_registered"] is True
    assert record["start_position"] == [0.0, 0.0, 0.0]
    assert record["reentry_position_error_to_start_m"] is not None
    assert record["max_gripper_aperture_change"] == 0.0


def test_perturbation_flags_measured_gripper_aperture_change() -> None:
    base = _FakePolicy()
    policy = LiberoHybridPerturbationAdapter(
        base,
        radius_m=0.04,
        reach_fraction=0.9,
    )

    policy.predict_action(_observation([0.0, 0.0, 0.0]))
    policy.predict_action(
        _observation(
            [0.0, 0.04, 0.0],
            gripper_aperture=0.035,
        )
    )

    record = policy.get_execution_metrics()["controlled_perturbation"][
        "record"
    ]
    assert record["gripper_changed_during_perturbation"] is True
    assert record["max_gripper_aperture_change"] == pytest.approx(0.005)
    assert (
        record[
            "recovery_reentry_cache_requested_for_gripper_transition"
        ]
        is True
    )
    assert base._executors[0].external_gripper_transition_registered


def test_perturbation_metrics_archive_each_episode() -> None:
    base = _FakePolicy()
    policy = LiberoHybridPerturbationAdapter(
        base,
        radius_m=0.04,
        reach_fraction=0.9,
    )

    policy.predict_action(_observation([0.0, 0.0, 0.0]))
    policy.predict_action(_observation([0.0, 0.04, 0.0]))
    policy.reset()

    metrics = policy.get_execution_metrics()["controlled_perturbation"]
    assert metrics["episodes_observed"] == 1
    assert metrics["triggered_episodes"] == 1
    assert metrics["completed_perturbations"] == 1
    assert metrics["recovery_observed_episodes"] == 1
    assert len(metrics["episode_records"]) == 1
    assert metrics["episode_records"][0]["perturbation_complete"]


def test_replan_supersedes_old_gripper_event_without_counting_a_skip() -> None:
    base = _FakeReplanPolicy()
    policy = LiberoHybridPerturbationAdapter(
        base,
        radius_m=0.04,
        reach_fraction=0.9,
    )

    policy.predict_action(_observation([0.0, 0.0, 0.0]))
    policy.predict_action(_observation([0.0, 0.04, 0.0]))
    policy.predict_action(_observation([0.04, 0.0, 0.0]))
    policy.predict_action(_observation([0.05, 0.0, 0.0]))

    metrics = policy.get_execution_metrics()["controlled_perturbation"]
    record = metrics["record"]
    assert record["executor_replaced_before_event"] is True
    assert record["gripper_event_superseded_by_replan"] is True
    assert record["first_recovery_phase"] == 4.25
    assert record["first_reentry_phase"] == 4.25
    assert record["gripper_event_skipped"] is None
    assert metrics["gripper_event_skips"] == 0
    assert metrics["gripper_events_superseded_by_replan"] == 1
