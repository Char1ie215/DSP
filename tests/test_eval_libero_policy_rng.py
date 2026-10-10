from __future__ import annotations

import pytest

from scripts import eval_libero_policy as eval_policy


class _DummyPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> int:
        self.reset_calls += 1
        return self.reset_calls


class _OutcomeCapableRunner:
    def run(self) -> dict[str, float]:
        task_name = "task"
        seed = 0
        return {f"{task_name}/success_{seed}": 1.0}


class _AggregateOnlyRunner:
    def run(self) -> dict[str, float]:
        return {"mean_success_rate": 1.0}


def test_per_episode_rng_reset_uses_consecutive_seeds(monkeypatch) -> None:
    observed_seeds: list[int] = []
    monkeypatch.setattr(eval_policy, "_set_eval_seed", observed_seeds.append)
    policy = _DummyPolicy()

    recorded_seeds = eval_policy._install_per_episode_rng_reset(
        policy,
        eval_seed=410000,
        n_parallel_envs=1,
    )

    assert policy.reset() == 1
    assert policy.reset() == 2
    assert observed_seeds == [410000, 410001]
    assert recorded_seeds == observed_seeds


@pytest.mark.parametrize(
    ("eval_seed", "n_parallel_envs", "message"),
    [
        (None, 1, "requires --eval-seed"),
        (0, 2, "requires --n-parallel-envs 1"),
    ],
)
def test_per_episode_rng_reset_rejects_unmatched_protocols(
    eval_seed: int | None,
    n_parallel_envs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        eval_policy._install_per_episode_rng_reset(
            _DummyPolicy(),
            eval_seed=eval_seed,
            n_parallel_envs=n_parallel_envs,
        )


def test_build_episode_outcome_records_retains_matched_identity() -> None:
    records = eval_policy._build_episode_outcome_records(
        {
            "task_a/success_200000": True,
            "task_a/success_step_200000": 123,
            "task_b/success_200001": 0.0,
            "task_b/success_step_200001": -1,
        },
        task_names=["task_a", "task_b"],
        environment_seeds=[200000, 200001],
        sampling_seeds=[410000, 410001],
        experiment_index=3,
    )

    assert records == [
        {
            "experiment_index": 3,
            "episode_index": 0,
            "task_name": "task_a",
            "environment_seed": 200000,
            "sampling_seed": 410000,
            "success": True,
            "success_step": 123,
        },
        {
            "experiment_index": 3,
            "episode_index": 1,
            "task_name": "task_b",
            "environment_seed": 200001,
            "sampling_seed": 410001,
            "success": False,
            "success_step": -1,
        },
    ]


def test_build_episode_outcome_records_rejects_partial_logs() -> None:
    with pytest.raises(ValueError, match="partial episode-outcome"):
        eval_policy._build_episode_outcome_records(
            {"task_a/success_200000": True},
            task_names=["task_a", "task_b"],
            environment_seeds=[200000, 200001],
            sampling_seeds=[0, 1],
            experiment_index=0,
        )


def test_formal_runner_capability_check_accepts_episode_logging() -> None:
    eval_policy._require_episode_outcome_capable_runner(
        _OutcomeCapableRunner()
    )


def test_formal_runner_capability_check_rejects_aggregate_only() -> None:
    with pytest.raises(RuntimeError, match="does not retain"):
        eval_policy._require_episode_outcome_capable_runner(
            _AggregateOnlyRunner()
        )
