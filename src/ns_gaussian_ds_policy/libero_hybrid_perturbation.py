"""Controlled perturbations for diagnosing cached LIBERO hybrid policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class _PerturbationRecord:
    requested_radius_m: float
    trigger_phase: float
    expected_gripper_event_phase: float | None
    held_gripper_command: float
    direction: list[float]
    start_position: list[float]
    start_quaternion: list[float]
    recovery_anchor_registered: bool = False
    neutral_gripper_action: float = 0.0
    start_gripper_aperture: float | None = None
    last_gripper_aperture: float | None = None
    max_gripper_aperture_change: float = 0.0
    perturbation_steps: int = 0
    perturbation_complete: bool = False
    achieved_radius_m: float = 0.0
    achieved_displacement_norm_m: float = 0.0
    recovery_observed: bool = False
    recovery_reentered: bool = False
    first_recovery_phase: float | None = None
    first_reentry_phase: float | None = None
    first_gripper_transition_phase: float | None = None
    first_gripper_transition_command: float | None = None
    gripper_changed_during_perturbation: bool = False
    recovery_reentry_cache_requested_for_gripper_transition: bool = False
    gripper_event_skipped: bool | None = None
    executor_replaced_before_event: bool = False
    gripper_event_superseded_by_replan: bool = False
    reentry_position: list[float] | None = None
    reentry_quaternion: list[float] | None = None
    reentry_position_error_to_start_m: float | None = None
    reentry_rotation_error_to_start_rad: float | None = None


class LiberoHybridPerturbationAdapter(torch.nn.Module):
    """Inject one physical OSC displacement without advancing the policy.

    The wrapper is intentionally restricted to a single environment. During
    the perturbation it does not call ``base_policy.predict_action``. The
    cached hybrid executor therefore keeps its action phase, recovery phase,
    and gripper event state unchanged while the environment is displaced.
    """

    _GRIPPER_APERTURE_CHANGE_TOLERANCE = 1.0e-3

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        radius_m: float,
        trigger_phase: float = 4.0,
        direction_sign: float = 1.0,
        position_action_scale: float = 0.05,
        max_perturbation_steps: int = 12,
        reach_fraction: float = 0.9,
        gripper_event_guard: float = 2.0,
        require_future_gripper_event: bool = True,
    ) -> None:
        super().__init__()
        if float(radius_m) <= 0.0:
            raise ValueError("radius_m must be positive")
        if float(position_action_scale) <= 0.0:
            raise ValueError("position_action_scale must be positive")
        if int(max_perturbation_steps) < 1:
            raise ValueError("max_perturbation_steps must be positive")
        if not 0.0 < float(reach_fraction) <= 1.0:
            raise ValueError("reach_fraction must be in (0,1]")

        self.base_policy = base_policy
        self.radius_m = float(radius_m)
        self.trigger_phase = float(trigger_phase)
        self.direction_sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
        self.position_action_scale = float(position_action_scale)
        self.max_perturbation_steps = int(max_perturbation_steps)
        self.reach_fraction = float(reach_fraction)
        self.gripper_event_guard = float(gripper_event_guard)
        self.require_future_gripper_event = bool(
            require_future_gripper_event
        )

        self._active = False
        self._finished = False
        self._start_position: np.ndarray | None = None
        self._direction: np.ndarray | None = None
        self._executor_identity: int | None = None
        self._record: _PerturbationRecord | None = None
        self._archived_episode_records: list[
            _PerturbationRecord | None
        ] = []
        self._episode_observed = False
        self._last_gripper_target: float | None = None

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        if self._episode_observed:
            self._archived_episode_records.append(self._record)
        self._active = False
        self._finished = False
        self._start_position = None
        self._direction = None
        self._executor_identity = None
        self._record = None
        self._episode_observed = False
        self._last_gripper_target = None
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_perturbed"

    def enable_controller_diagnostics(self, enabled: bool = True) -> None:
        callback = getattr(
            self.base_policy,
            "enable_controller_diagnostics",
            None,
        )
        if callable(callback):
            callback(enabled)

    def consume_controller_diagnostic_traces(self):
        callback = getattr(
            self.base_policy,
            "consume_controller_diagnostic_traces",
            None,
        )
        if not callable(callback):
            return []
        return callback()

    @staticmethod
    def _latest_position(obs_dict: dict[str, Any]) -> np.ndarray:
        value = obs_dict["robot0_eef_pos"]
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        positions = np.asarray(array, dtype=np.float64).reshape(
            len(array), -1, 3
        )
        if len(positions) != 1:
            raise ValueError(
                "Hybrid perturbation diagnostics require one parallel env"
            )
        return positions[0, -1].copy()

    @staticmethod
    def _latest_quaternion(obs_dict: dict[str, Any]) -> np.ndarray:
        value = obs_dict["robot0_eef_quat"]
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        quaternions = np.asarray(array, dtype=np.float64).reshape(
            len(array), -1, 4
        )
        if len(quaternions) != 1:
            raise ValueError(
                "Hybrid perturbation diagnostics require one parallel env"
            )
        return quaternions[0, -1].copy()

    @staticmethod
    def _latest_gripper_aperture(
        obs_dict: dict[str, Any],
    ) -> float | None:
        value = obs_dict.get("robot0_gripper_qpos")
        if value is None:
            return None
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        qpos = np.asarray(array, dtype=np.float64).reshape(len(array), -1)
        if len(qpos) != 1 or qpos.shape[1] < 1:
            return None
        return float(np.max(np.abs(qpos[0, -2:])))

    def _executor(self):
        executors = getattr(self.base_policy, "_executors", ())
        if len(executors) != 1:
            return None
        return executors[0]

    @staticmethod
    def _horizontal_transverse_direction(executor, phase: float) -> np.ndarray:
        centers = np.asarray(
            executor.plan.position_centers,
            dtype=np.float64,
        )
        index = int(np.clip(np.floor(phase), 0, len(centers) - 1))
        lower = max(0, index - 1)
        upper = min(len(centers) - 1, index + 1)
        tangent = centers[upper] - centers[lower]
        tangent_xy = tangent[:2]
        norm = float(np.linalg.norm(tangent_xy))
        if norm <= 1.0e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return np.array(
            [-tangent_xy[1] / norm, tangent_xy[0] / norm, 0.0],
            dtype=np.float64,
        )

    def _can_trigger(self, executor) -> bool:
        if executor is None or bool(getattr(executor, "recovering", False)):
            return False
        phase = float(getattr(executor, "action_phase", 0.0))
        if phase < self.trigger_phase:
            return False
        next_event = float(executor._next_gripper_event_phase(phase))
        target_count = float(executor.target_count)
        has_event = next_event < target_count - 1.0e-9
        if self.require_future_gripper_event and not has_event:
            return False
        return bool(
            not has_event
            or next_event - phase >= self.gripper_event_guard
        )

    def _begin(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        executor,
        gripper_aperture: float | None,
    ) -> None:
        phase = float(executor.action_phase)
        direction = self._horizontal_transverse_direction(executor, phase)
        direction *= self.direction_sign
        next_event = float(executor._next_gripper_event_phase(phase))
        expected_event = (
            next_event
            if next_event < float(executor.target_count) - 1.0e-9
            else None
        )
        held_gripper = float(executor.last_gripper_command)
        self._active = True
        self._start_position = position.copy()
        self._direction = direction
        self._executor_identity = id(executor)
        self._last_gripper_target = held_gripper
        self._record = _PerturbationRecord(
            requested_radius_m=self.radius_m,
            trigger_phase=phase,
            expected_gripper_event_phase=expected_event,
            held_gripper_command=held_gripper,
            direction=direction.tolist(),
            start_position=position.tolist(),
            start_quaternion=quaternion.tolist(),
            start_gripper_aperture=gripper_aperture,
            last_gripper_aperture=gripper_aperture,
        )
        callback = getattr(
            executor,
            "register_observed_recovery_anchor",
            None,
        )
        if callable(callback):
            self._record.recovery_anchor_registered = bool(
                callback(position, quaternion)
            )

    def _update_gripper_aperture(
        self,
        aperture: float | None,
    ) -> None:
        if self._record is None or aperture is None:
            return
        self._record.last_gripper_aperture = float(aperture)
        start = self._record.start_gripper_aperture
        if start is not None:
            change = abs(float(aperture) - float(start))
            self._record.max_gripper_aperture_change = max(
                self._record.max_gripper_aperture_change,
                change,
            )
            if change > self._GRIPPER_APERTURE_CHANGE_TOLERANCE:
                self._record.gripper_changed_during_perturbation = True

    def _update_displacement(self, position: np.ndarray) -> float:
        assert self._record is not None
        assert self._start_position is not None
        assert self._direction is not None
        displacement = position - self._start_position
        directional = max(0.0, float(np.dot(displacement, self._direction)))
        self._record.achieved_radius_m = max(
            self._record.achieved_radius_m,
            directional,
        )
        self._record.achieved_displacement_norm_m = max(
            self._record.achieved_displacement_norm_m,
            float(np.linalg.norm(displacement)),
        )
        return directional

    def _append_perturbation_diagnostic(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_aperture: float | None,
    ) -> None:
        callback = getattr(
            self.base_policy,
            "append_controller_diagnostic_record",
            None,
        )
        executor = self._executor()
        if not callable(callback) or executor is None:
            return
        phase = float(getattr(executor, "action_phase", 0.0))
        position_error, rotation_error = executor._pose_errors_at_phase(
            position,
            quaternion,
            phase,
        )
        callback(
            {
                "mode": "perturb",
                "phase": phase,
                "phase_count": int(executor.target_count),
                "position_error": float(position_error),
                "rotation_error": float(rotation_error),
                "position_tube_radius": float(
                    executor._position_outer_radius(phase)
                ),
                "position_reentry_radius": float(
                    executor._position_reentry_radius(phase)
                ),
                "policy_query": False,
                "recovery_steps": 0,
                "gripper_command": 0.0,
                "gripper_target_command": float(
                    self._record.held_gripper_command
                ),
                "gripper_aperture": gripper_aperture,
                "perturbation_radius": float(
                    self._record.achieved_radius_m
                ),
            },
            env_index=0,
            batch_size=1,
        )

    def _perturbation_action(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_aperture: float | None,
    ) -> dict[str, Any]:
        assert self._record is not None
        assert self._direction is not None
        directional = self._update_displacement(position)
        remaining = max(self.radius_m - directional, 0.0)
        action = np.zeros((1, 1, 7), dtype=np.float32)
        magnitude = float(
            np.clip(remaining / self.position_action_scale, 0.05, 1.0)
        )
        action[0, 0, :3] = self._direction * magnitude
        # Panda gripper actions are incremental. Repeating +/-1 keeps moving
        # the fingers; zero preserves the current actuator target.
        action[0, 0, 6] = self._record.neutral_gripper_action
        self._record.perturbation_steps += 1
        self._update_gripper_aperture(gripper_aperture)
        self._append_perturbation_diagnostic(
            position,
            quaternion,
            gripper_aperture,
        )
        return {
            "action": torch.as_tensor(
                action,
                device=self.device,
                dtype=self.dtype,
            )
        }

    def _observe_base_step(
        self,
        result: dict[str, Any],
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        if self._record is None:
            return
        executor = self._executor()
        if executor is None:
            return
        executor_replaced = bool(
            self._executor_identity is not None
            and id(executor) != self._executor_identity
        )
        if executor_replaced:
            if self._record.first_gripper_transition_phase is None:
                self._record.executor_replaced_before_event = True
                self._record.gripper_event_superseded_by_replan = bool(
                    self._record.expected_gripper_event_phase is not None
                )
            self._executor_identity = id(executor)

        recovering = bool(getattr(executor, "recovering", False))
        if recovering and not self._record.recovery_observed:
            self._record.recovery_observed = True
            self._record.first_recovery_phase = float(
                executor.recovery_phase
            )
        if (
            self._record.recovery_observed
            and not recovering
            and not self._record.recovery_reentered
        ):
            self._record.recovery_reentered = True
            self._record.first_reentry_phase = float(
                self._record.first_recovery_phase
                if executor_replaced
                and self._record.first_recovery_phase is not None
                else executor.last_issued_phase
            )
            start_position = np.asarray(
                self._record.start_position,
                dtype=np.float64,
            )
            start_quaternion = np.asarray(
                self._record.start_quaternion,
                dtype=np.float64,
            )
            normalized_quaternion = quaternion / max(
                float(np.linalg.norm(quaternion)),
                1.0e-12,
            )
            normalized_start = start_quaternion / max(
                float(np.linalg.norm(start_quaternion)),
                1.0e-12,
            )
            cosine = float(
                np.clip(
                    abs(normalized_quaternion @ normalized_start),
                    0.0,
                    1.0,
                )
            )
            self._record.reentry_position = position.tolist()
            self._record.reentry_quaternion = quaternion.tolist()
            self._record.reentry_position_error_to_start_m = float(
                np.linalg.norm(position - start_position)
            )
            self._record.reentry_rotation_error_to_start_rad = float(
                2.0 * np.arccos(cosine)
            )

        gripper_target = float(executor.last_gripper_command)
        if (
            self._record.first_gripper_transition_phase is None
            and self._last_gripper_target is not None
            and np.sign(gripper_target) != np.sign(self._last_gripper_target)
        ):
            phase = float(executor.last_issued_phase)
            self._record.first_gripper_transition_phase = phase
            self._record.first_gripper_transition_command = gripper_target
            expected = self._record.expected_gripper_event_phase
            if not self._record.gripper_event_superseded_by_replan:
                self._record.gripper_event_skipped = bool(
                    expected is not None and phase < expected - 1.0e-6
                )
        self._last_gripper_target = gripper_target

    def predict_action(self, obs_dict, **kwargs):
        self._episode_observed = True
        position = self._latest_position(obs_dict)
        quaternion = self._latest_quaternion(obs_dict)
        gripper_aperture = self._latest_gripper_aperture(obs_dict)
        if self._active:
            directional = self._update_displacement(position)
            self._update_gripper_aperture(gripper_aperture)
            assert self._record is not None
            reached = directional >= self.reach_fraction * self.radius_m
            exhausted = (
                self._record.perturbation_steps
                >= self.max_perturbation_steps
            )
            if not reached and not exhausted:
                return self._perturbation_action(
                    position,
                    quaternion,
                    gripper_aperture,
                )
            self._active = False
            self._finished = True
            self._record.perturbation_complete = True
            if self._record.gripper_changed_during_perturbation:
                executor = self._executor()
                callback = getattr(
                    executor,
                    "register_external_gripper_transition",
                    None,
                )
                if callable(callback):
                    requested = bool(callback())
                    self._record.recovery_reentry_cache_requested_for_gripper_transition = (
                        requested
                    )

        if not self._finished:
            executor = self._executor()
            if self._can_trigger(executor):
                self._begin(
                    position,
                    quaternion,
                    executor,
                    gripper_aperture,
                )
                return self._perturbation_action(
                    position,
                    quaternion,
                    gripper_aperture,
                )

        result = self.base_policy.predict_action(obs_dict, **kwargs)
        self._observe_base_step(result, position, quaternion)
        return result

    def get_execution_metrics(self) -> dict[str, object]:
        callback = getattr(self.base_policy, "get_execution_metrics", None)
        metrics = dict(callback()) if callable(callback) else {}
        episode_records = list(self._archived_episode_records)
        if self._episode_observed:
            episode_records.append(self._record)
        serialized = [
            None if item is None else asdict(item)
            for item in episode_records
        ]
        triggered_records = [
            item for item in serialized if item is not None
        ]
        record = triggered_records[-1] if triggered_records else None
        achieved = [
            float(item["achieved_radius_m"])
            for item in triggered_records
        ]
        metrics["controlled_perturbation"] = {
            "enabled": True,
            "episodes_observed": len(serialized),
            "triggered_episodes": len(triggered_records),
            "completed_perturbations": sum(
                bool(item["perturbation_complete"])
                for item in triggered_records
            ),
            "recovery_observed_episodes": sum(
                bool(item["recovery_observed"])
                for item in triggered_records
            ),
            "recovery_reentered_episodes": sum(
                bool(item["recovery_reentered"])
                for item in triggered_records
            ),
            "gripper_event_skips": sum(
                item["gripper_event_skipped"] is True
                for item in triggered_records
            ),
            "gripper_events_superseded_by_replan": sum(
                bool(item["gripper_event_superseded_by_replan"])
                for item in triggered_records
            ),
            "achieved_radius_m_mean": (
                float(np.mean(achieved)) if achieved else None
            ),
            "achieved_radius_m_min": (
                float(np.min(achieved)) if achieved else None
            ),
            "triggered": bool(triggered_records),
            "active": self._active,
            "complete": bool(
                record is not None and record["perturbation_complete"]
            ),
            "record": record,
            "episode_records": serialized,
        }
        return metrics
