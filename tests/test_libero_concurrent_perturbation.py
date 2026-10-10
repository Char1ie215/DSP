from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ns_gaussian_ds_policy.libero_concurrent_perturbation import (
    ConcurrentPerturbationConfig,
    collect_concurrent_perturbation,
    install_concurrent_perturbation,
)


class _FakeModel:
    site_bodyid = np.asarray([0], dtype=np.int64)

    @staticmethod
    def site_id2name(site_id: int) -> str:
        assert site_id == 0
        return "eef"


class _FakeData:
    def __init__(self) -> None:
        self.site_xpos = np.zeros((1, 3), dtype=np.float64)
        self.xfrc_applied = np.zeros((1, 6), dtype=np.float64)

    @staticmethod
    def get_site_xvelp(site_name: str) -> np.ndarray:
        assert site_name == "eef"
        return np.zeros(3, dtype=np.float64)


class _FakeLiberoEnv:
    def __init__(self, task_name: str = "test_task", seed: int = 7) -> None:
        self.task_name = task_name
        self.seed_value = seed
        self.cur_step = 0
        self.sim = SimpleNamespace(model=_FakeModel(), data=_FakeData())
        raw_env = SimpleNamespace(
            sim=self.sim,
            robots=[SimpleNamespace(eef_site_id=0)],
        )
        self.env = SimpleNamespace(env=raw_env)
        self.received_actions: list[np.ndarray] = []
        self.force_during_steps: list[np.ndarray] = []

    def step(self, action: np.ndarray):
        force = self.sim.data.xfrc_applied[0, :3].copy()
        self.force_during_steps.append(force)
        self.received_actions.append(np.asarray(action).copy())
        self.sim.data.site_xpos[0] += 0.01 * force
        self.cur_step += 1
        return {}, 0.0, False, False, {}

    def reset(self, *args, **kwargs):
        self.cur_step = 0
        return {}, {}


class _FakeOuterEnv:
    def __init__(self, libero: _FakeLiberoEnv) -> None:
        self.env = SimpleNamespace(env=libero)

    @property
    def libero(self) -> _FakeLiberoEnv:
        return self.env.env

    def replace_libero(self, libero: _FakeLiberoEnv) -> None:
        self.env.env = libero

    def step(self, action: np.ndarray):
        return self.libero.step(action)

    def reset(self, *args, **kwargs):
        return self.libero.reset(*args, **kwargs)


def _wrapped_env() -> tuple[_FakeOuterEnv, _FakeLiberoEnv]:
    libero = _FakeLiberoEnv()
    return _FakeOuterEnv(libero), libero


def test_concurrent_wrench_never_replaces_policy_actions() -> None:
    outer, libero = _wrapped_env()
    config = ConcurrentPerturbationConfig(
        start_step=2,
        distance=0.01,
        direction=(1.0, 0.0, 0.0),
        ramp_steps=1,
        hold_steps=1,
        max_push_steps=2,
        stiffness=100.0,
        damping=1.0,
        max_force=10.0,
    )
    installed = install_concurrent_perturbation(outer, config)

    actions = [np.full(7, value, dtype=np.float64) for value in range(4)]
    for action in actions:
        outer.step(action)
        np.testing.assert_allclose(libero.sim.data.xfrc_applied, 0.0)

    np.testing.assert_allclose(libero.received_actions, actions)
    np.testing.assert_allclose(libero.force_during_steps[0], 0.0)
    np.testing.assert_allclose(libero.force_during_steps[1], 0.0)
    assert libero.force_during_steps[2][0] > 0.0
    assert installed["task_name"] == "test_task"

    outer.reset()
    summary = collect_concurrent_perturbation(outer)
    record = summary["episode_records"][0]
    assert summary["protocol"] == "concurrent_environment_wrench"
    assert record["first_force_step"] == 2
    assert record["target_reached"] is True
    assert record["actions_seen"] == len(actions)
    assert record["policy_actions_replaced"] == 0
    assert record["max_signed_displacement_during_push_m"] == pytest.approx(
        0.01
    )
    assert record["max_signed_displacement_over_episode_m"] == pytest.approx(
        0.01
    )


def test_concurrent_wrench_rejects_zero_direction() -> None:
    with pytest.raises(ValueError, match="direction must be nonzero"):
        ConcurrentPerturbationConfig(direction=(0.0, 0.0, 0.0))


def test_integral_force_accumulates_under_persistent_position_error() -> None:
    outer, libero = _wrapped_env()
    config = ConcurrentPerturbationConfig(
        start_step=0,
        distance=1.0,
        direction=(1.0, 0.0, 0.0),
        ramp_steps=1,
        hold_steps=1,
        max_push_steps=4,
        stiffness=1.0,
        integral_gain=2.0,
        integral_error_limit=2.0,
        damping=1.0,
        max_force=10.0,
    )
    install_concurrent_perturbation(outer, config)

    # Keep the measured EEF fixed to emulate a controller opposing the push.
    original_step = libero.env.env.sim.data.site_xpos
    outer.step(np.zeros(7, dtype=np.float64))
    original_step[:] = 0.0
    outer.step(np.zeros(7, dtype=np.float64))

    assert libero.force_during_steps[0][0] == pytest.approx(3.0)
    assert libero.force_during_steps[1][0] == pytest.approx(5.0)
    summary = collect_concurrent_perturbation(outer)
    record = summary["episode_records"][0]
    assert record["max_integral_force_n"] == pytest.approx(4.0)


def test_concurrent_wrench_archives_serial_episodes() -> None:
    outer, libero = _wrapped_env()
    config = ConcurrentPerturbationConfig(
        start_step=0,
        distance=0.01,
        direction=(1.0, 0.0, 0.0),
        ramp_steps=1,
        hold_steps=1,
        max_push_steps=2,
        stiffness=100.0,
        damping=1.0,
        max_force=10.0,
    )
    install_concurrent_perturbation(outer, config)

    for episode in range(2):
        for _ in range(2):
            outer.step(np.full(7, episode, dtype=np.float64))
        if episode == 0:
            outer.reset()

    summary = collect_concurrent_perturbation(outer)
    records = summary["episode_records"]
    assert len(records) == 2
    assert [record["actions_seen"] for record in records] == [2, 2]
    assert all(record["target_reached"] for record in records)
    assert all(record["policy_actions_replaced"] == 0 for record in records)


def test_concurrent_wrench_survives_libero_task_replacement() -> None:
    outer, first = _wrapped_env()
    config = ConcurrentPerturbationConfig(
        start_step=0,
        distance=0.01,
        direction=(1.0, 0.0, 0.0),
        ramp_steps=1,
        hold_steps=1,
        max_push_steps=2,
        stiffness=100.0,
        damping=1.0,
        max_force=10.0,
    )
    install_concurrent_perturbation(outer, config)

    outer.step(np.zeros(7, dtype=np.float64))
    outer.step(np.zeros(7, dtype=np.float64))
    second = _FakeLiberoEnv(task_name="next_task", seed=8)
    outer.replace_libero(second)
    outer.reset()
    outer.step(np.ones(7, dtype=np.float64))
    outer.step(np.ones(7, dtype=np.float64))

    summary = collect_concurrent_perturbation(outer)
    records = summary["episode_records"]
    assert len(records) == 2
    assert first.force_during_steps[0][0] > 0.0
    assert second.force_during_steps[0][0] > 0.0
    assert [record["actions_seen"] for record in records] == [2, 2]
    assert all(record["target_reached"] for record in records)
