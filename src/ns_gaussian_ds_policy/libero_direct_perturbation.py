"""Stepwise cached execution and controlled perturbations for LIBERO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class _DirectPerturbationRecord:
    requested_radius_m: float
    trigger_chunk_index: int
    expected_gripper_event_index: int | None
    held_gripper_command: float
    direction: list[float]
    neutral_gripper_action: float = 0.0
    start_gripper_aperture: float | None = None
    last_gripper_aperture: float | None = None
    max_gripper_aperture_change: float = 0.0
    perturbation_steps: int = 0
    perturbation_complete: bool = False
    achieved_radius_m: float = 0.0
    achieved_displacement_norm_m: float = 0.0
    first_gripper_transition_index: int | None = None
    first_gripper_transition_command: float | None = None
    gripper_changed_during_perturbation: bool = False
    gripper_event_skipped: bool | None = None


class LiberoDirectChunkPerturbationAdapter(torch.nn.Module):
    """Execute direct action chunks one step at a time.

    The underlying policy is queried only after every cached chunk has been
    consumed. A controlled perturbation pauses the cached action index, sends
    a neutral gripper action, and resumes the exact same chunk afterward.
    """

    def __init__(
        self,
        base_policy: torch.nn.Module,
        *,
        radius_m: float = 0.0,
        trigger_phase: int = 4,
        direction_sign: float = 1.0,
        position_action_scale: float = 0.05,
        max_perturbation_steps: int = 25,
        reach_fraction: float = 0.99,
        gripper_event_guard: int = 2,
        require_future_gripper_event: bool = True,
    ) -> None:
        super().__init__()
        if float(radius_m) < 0.0:
            raise ValueError("radius_m must be nonnegative")
        if int(trigger_phase) < 0:
            raise ValueError("trigger_phase must be nonnegative")
        if float(position_action_scale) <= 0.0:
            raise ValueError("position_action_scale must be positive")
        if int(max_perturbation_steps) < 1:
            raise ValueError("max_perturbation_steps must be positive")
        if not 0.0 < float(reach_fraction) <= 1.0:
            raise ValueError("reach_fraction must be in (0,1]")
        if int(gripper_event_guard) < 0:
            raise ValueError("gripper_event_guard must be nonnegative")

        self.base_policy = base_policy
        self.radius_m = float(radius_m)
        self.trigger_phase = int(trigger_phase)
        self.direction_sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
        self.position_action_scale = float(position_action_scale)
        self.max_perturbation_steps = int(max_perturbation_steps)
        self.reach_fraction = float(reach_fraction)
        self.gripper_event_guard = int(gripper_event_guard)
        self.require_future_gripper_event = bool(
            require_future_gripper_event
        )

        self.policy_batches = 0
        self.native_action_steps = 0
        self._cached_actions: torch.Tensor | None = None
        self._chunk_index = 0
        self._active = False
        self._finished = False
        self._start_position: np.ndarray | None = None
        self._direction: np.ndarray | None = None
        self._record: _DirectPerturbationRecord | None = None
        self._archived_episode_records: list[
            _DirectPerturbationRecord | None
        ] = []
        self._episode_observed = False
        self._last_output_gripper: float | None = None

    @property
    def device(self):
        return self.base_policy.device

    @property
    def dtype(self):
        return self.base_policy.dtype

    def reset(self):
        if self._episode_observed:
            self._archived_episode_records.append(self._record)
        self._cached_actions = None
        self._chunk_index = 0
        self._active = False
        self._finished = False
        self._start_position = None
        self._direction = None
        self._record = None
        self._episode_observed = False
        self._last_output_gripper = None
        return self.base_policy.reset()

    def get_observation_ports(self):
        return self.base_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.base_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.base_policy.get_observation_encoder()

    def get_policy_name(self):
        suffix = "_stepwise"
        if self.radius_m > 0.0:
            suffix += "_perturbed"
        return self.base_policy.get_policy_name() + suffix

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
                "Direct perturbation diagnostics require one parallel env"
            )
        return positions[0, -1].copy()

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

    def _query_chunk(self, obs_dict, **kwargs) -> None:
        result = self.base_policy.predict_action(obs_dict, **kwargs)
        action = result["action"]
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(
                action,
                device=self.device,
                dtype=self.dtype,
            )
        if action.ndim != 3 or action.shape[0] != 1 or action.shape[-1] != 7:
            raise ValueError(
                "Stepwise direct execution requires an action chunk with "
                f"shape (1,T,7), got {tuple(action.shape)}"
            )
        if action.shape[1] < 1:
            raise ValueError("Direct policy returned an empty action chunk")
        self._cached_actions = action
        self._chunk_index = 0
        self.policy_batches += 1

    def _future_gripper_event(self) -> int | None:
        assert self._cached_actions is not None
        actions = self._cached_actions[0].detach().cpu().numpy()
        if self._last_output_gripper is None:
            held = float(actions[self._chunk_index, 6])
        else:
            held = float(self._last_output_gripper)
        held_sign = 1.0 if held >= 0.0 else -1.0
        for index in range(self._chunk_index, len(actions)):
            command = float(actions[index, 6])
            command_sign = 1.0 if command >= 0.0 else -1.0
            if command_sign != held_sign:
                return index
        return None

    def _can_trigger(self) -> tuple[bool, int | None]:
        if self.radius_m <= 0.0 or self._finished or self._active:
            return False, None
        if self._chunk_index < self.trigger_phase:
            return False, None
        event = self._future_gripper_event()
        if self.require_future_gripper_event and event is None:
            return False, event
        if event is not None and (
            event - self._chunk_index < self.gripper_event_guard
        ):
            return False, event
        return True, event

    def _transverse_direction(self) -> np.ndarray:
        assert self._cached_actions is not None
        actions = self._cached_actions[0].detach().cpu().numpy()
        start = self._chunk_index
        end = min(len(actions), start + 3)
        tangent_xy = np.sum(actions[start:end, :2], axis=0)
        norm = float(np.linalg.norm(tangent_xy))
        if norm <= 1.0e-8:
            for offset in range(len(actions)):
                for index in (start + offset, start - offset):
                    if not 0 <= index < len(actions):
                        continue
                    candidate = np.asarray(actions[index, :2], dtype=np.float64)
                    norm = float(np.linalg.norm(candidate))
                    if norm > 1.0e-8:
                        tangent_xy = candidate
                        break
                if norm > 1.0e-8:
                    break
        if norm <= 1.0e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        direction = np.array(
            [-tangent_xy[1] / norm, tangent_xy[0] / norm, 0.0],
            dtype=np.float64,
        )
        return direction * self.direction_sign

    def _begin(
        self,
        position: np.ndarray,
        expected_gripper_event: int | None,
        gripper_aperture: float | None,
    ) -> None:
        assert self._cached_actions is not None
        held = (
            float(self._last_output_gripper)
            if self._last_output_gripper is not None
            else float(self._cached_actions[0, self._chunk_index, 6])
        )
        self._active = True
        self._start_position = position.copy()
        self._direction = self._transverse_direction()
        self._record = _DirectPerturbationRecord(
            requested_radius_m=self.radius_m,
            trigger_chunk_index=self._chunk_index,
            expected_gripper_event_index=expected_gripper_event,
            held_gripper_command=held,
            direction=self._direction.tolist(),
            start_gripper_aperture=gripper_aperture,
            last_gripper_aperture=gripper_aperture,
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
            self._record.max_gripper_aperture_change = max(
                self._record.max_gripper_aperture_change,
                abs(float(aperture) - float(start)),
            )

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

    def _perturbation_action(
        self,
        position: np.ndarray,
        gripper_aperture: float | None,
    ) -> dict[str, Any]:
        assert self._record is not None
        assert self._direction is not None
        directional = self._update_displacement(position)
        remaining = max(self.radius_m - directional, 0.0)
        action = torch.zeros(
            (1, 1, 7),
            device=self.device,
            dtype=self.dtype,
        )
        magnitude = float(
            np.clip(remaining / self.position_action_scale, 0.05, 1.0)
        )
        action[0, 0, :3] = torch.as_tensor(
            self._direction * magnitude,
            device=self.device,
            dtype=self.dtype,
        )
        # The Panda gripper command is incremental, so zero is the only
        # command that does not keep opening or closing during displacement.
        action[0, 0, 6] = self._record.neutral_gripper_action
        self._record.perturbation_steps += 1
        self._update_gripper_aperture(gripper_aperture)
        return {"action": action}

    def _issue_cached_action(self) -> dict[str, Any]:
        assert self._cached_actions is not None
        index = self._chunk_index
        action = self._cached_actions[:, index : index + 1]
        gripper = float(action[0, 0, 6])
        if (
            self._record is not None
            and self._record.first_gripper_transition_index is None
            and self._last_output_gripper is not None
            and np.sign(gripper) != np.sign(self._last_output_gripper)
        ):
            self._record.first_gripper_transition_index = index
            self._record.first_gripper_transition_command = gripper
            expected = self._record.expected_gripper_event_index
            self._record.gripper_event_skipped = bool(
                expected is not None and index < expected
            )
        self._last_output_gripper = gripper
        self._chunk_index += 1
        self.native_action_steps += 1
        return {"action": action}

    def predict_action(self, obs_dict, **kwargs):
        self._episode_observed = True
        position = self._latest_position(obs_dict)
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
                    gripper_aperture,
                )
            self._active = False
            self._finished = True
            self._record.perturbation_complete = True

        if (
            self._cached_actions is None
            or self._chunk_index >= self._cached_actions.shape[1]
        ):
            self._query_chunk(obs_dict, **kwargs)

        should_trigger, expected_event = self._can_trigger()
        if should_trigger:
            self._begin(position, expected_event, gripper_aperture)
            return self._perturbation_action(
                position,
                gripper_aperture,
            )
        return self._issue_cached_action()

    def get_execution_metrics(self) -> dict[str, object]:
        callback = getattr(self.base_policy, "get_execution_metrics", None)
        metrics = dict(callback()) if callable(callback) else {}
        metrics.update(
            {
                "direct_stepwise_execution": True,
                "direct_chunk_policy_batches": int(self.policy_batches),
                "direct_chunk_native_action_steps": int(
                    self.native_action_steps
                ),
            }
        )
        episode_records = list(self._archived_episode_records)
        if self._episode_observed:
            episode_records.append(self._record)
        serialized = [
            None if item is None else asdict(item)
            for item in episode_records
        ]
        triggered = [item for item in serialized if item is not None]
        achieved = [float(item["achieved_radius_m"]) for item in triggered]
        metrics["controlled_perturbation"] = {
            "enabled": self.radius_m > 0.0,
            "episodes_observed": len(serialized),
            "triggered_episodes": len(triggered),
            "completed_perturbations": sum(
                bool(item["perturbation_complete"]) for item in triggered
            ),
            "recovery_observed_episodes": 0,
            "recovery_reentered_episodes": 0,
            "gripper_event_skips": sum(
                item["gripper_event_skipped"] is True for item in triggered
            ),
            "achieved_radius_m_mean": (
                float(np.mean(achieved)) if achieved else None
            ),
            "achieved_radius_m_min": (
                float(np.min(achieved)) if achieved else None
            ),
            "episode_records": serialized,
        }
        return metrics
