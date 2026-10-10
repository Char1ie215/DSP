import pytest
import torch

from ns_gaussian_ds_policy.libero_direct_perturbation import (
    LiberoDirectChunkPerturbationAdapter,
)


class _FakeDirectPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        return None

    def get_observation_ports(self):
        return ["robot0_eef_pos"]

    def get_observation_modalities(self):
        return {}

    def get_observation_encoder(self):
        return None

    def get_policy_name(self):
        return "fake_direct"

    def predict_action(self, obs_dict, **kwargs):
        del obs_dict, kwargs
        self.calls += 1
        action = torch.zeros((1, 8, 7), dtype=torch.float32)
        action[0, :, 0] = torch.arange(1, 9, dtype=torch.float32) / 10.0
        action[0, :6, 6] = -1.0
        action[0, 6:, 6] = 1.0
        return {"action": action}

    def get_execution_metrics(self):
        return {"base_policy_calls": self.calls}


def _observation(x: float = 0.0, y: float = 0.0):
    return {
        "robot0_eef_pos": torch.tensor(
            [[[x, y, 0.0]]],
            dtype=torch.float32,
        ),
        "robot0_gripper_qpos": torch.tensor(
            [[[0.04, -0.04]]],
            dtype=torch.float32,
        ),
    }


def test_stepwise_direct_preserves_chunk_and_policy_query_cadence() -> None:
    base = _FakeDirectPolicy()
    policy = LiberoDirectChunkPerturbationAdapter(base, radius_m=0.0)

    values = []
    for _ in range(8):
        result = policy.predict_action(_observation())
        values.append(float(result["action"][0, 0, 0]))

    assert base.calls == 1
    assert values == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    result = policy.predict_action(_observation())
    assert base.calls == 2
    assert float(result["action"][0, 0, 0]) == pytest.approx(0.1)


def test_direct_perturbation_freezes_index_and_gripper() -> None:
    base = _FakeDirectPolicy()
    policy = LiberoDirectChunkPerturbationAdapter(
        base,
        radius_m=0.20,
        trigger_phase=4,
        reach_fraction=0.99,
        gripper_event_guard=2,
    )

    nominal = [
        policy.predict_action(_observation())
        for _ in range(4)
    ]
    assert [float(row["action"][0, 0, 0]) for row in nominal] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )
    assert base.calls == 1

    perturb = policy.predict_action(_observation())
    assert float(perturb["action"][0, 0, 1]) > 0.0
    assert float(perturb["action"][0, 0, 6]) == 0.0
    assert base.calls == 1

    resumed = policy.predict_action(_observation(y=0.20))
    assert float(resumed["action"][0, 0, 0]) == pytest.approx(0.5)
    assert float(resumed["action"][0, 0, 6]) == -1.0
    assert base.calls == 1

    policy.predict_action(_observation(y=0.20))
    event = policy.predict_action(_observation(y=0.20))
    assert float(event["action"][0, 0, 0]) == pytest.approx(0.7)
    assert float(event["action"][0, 0, 6]) == 1.0

    metrics = policy.get_execution_metrics()
    record = metrics["controlled_perturbation"]["episode_records"][0]
    assert metrics["direct_chunk_policy_batches"] == 1
    assert record["trigger_chunk_index"] == 4
    assert record["expected_gripper_event_index"] == 6
    assert record["first_gripper_transition_index"] == 6
    assert record["gripper_changed_during_perturbation"] is False
    assert record["gripper_event_skipped"] is False
    assert record["neutral_gripper_action"] == 0.0
    assert record["max_gripper_aperture_change"] == 0.0


def test_direct_perturbation_archives_untriggered_episode() -> None:
    base = _FakeDirectPolicy()
    policy = LiberoDirectChunkPerturbationAdapter(
        base,
        radius_m=0.20,
        trigger_phase=7,
    )

    policy.predict_action(_observation())
    policy.reset()

    metrics = policy.get_execution_metrics()["controlled_perturbation"]
    assert metrics["episodes_observed"] == 1
    assert metrics["triggered_episodes"] == 0
    assert metrics["episode_records"] == [None]
