"""Environment-side Cartesian perturbations for LIBERO rollouts.

Unlike the older policy adapters, this hook never replaces a policy action.
It applies a bounded external wrench to the EEF body while the normal policy
and executor continue stepping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConcurrentPerturbationConfig:
    start_step: int = 2
    distance: float = 0.20
    direction: tuple[float, float, float] = (0.3882361, -0.9215599, 0.0)
    ramp_steps: int = 12
    hold_steps: int = 1
    max_push_steps: int = 40
    stiffness: float = 600.0
    integral_gain: float = 0.0
    integral_error_limit: float = 1.0
    damping: float = 25.0
    max_force: float = 100.0
    position_tolerance: float = 0.002

    def __post_init__(self) -> None:
        if int(self.start_step) < 0:
            raise ValueError("start_step must be nonnegative")
        if float(self.distance) <= 0.0:
            raise ValueError("distance must be positive")
        direction = np.asarray(self.direction, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction must contain three finite values")
        if float(np.linalg.norm(direction)) <= 1.0e-12:
            raise ValueError("direction must be nonzero")
        for name in ("ramp_steps", "hold_steps", "max_push_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.max_push_steps) < int(self.ramp_steps):
            raise ValueError("max_push_steps must be at least ramp_steps")
        for name in ("stiffness", "integral_error_limit", "damping", "max_force"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.integral_gain) < 0.0:
            raise ValueError("integral_gain must be nonnegative")
        if float(self.position_tolerance) < 0.0:
            raise ValueError("position_tolerance must be nonnegative")


class _ConcurrentPerturbationState:
    def __init__(self, config: ConcurrentPerturbationConfig) -> None:
        self.config = config
        direction = np.asarray(config.direction, dtype=np.float64)
        self.direction = direction / np.linalg.norm(direction)
        self.history: list[dict[str, Any]] = []
        self.reset_episode()

    def reset_episode(self) -> None:
        self.control_steps = 0
        self.start_position: np.ndarray | None = None
        self.release_step: int | None = None
        self.release_reason: str | None = None
        self.target_reached = False
        self.hold_count = 0
        self.force_steps = 0
        self.max_force = 0.0
        self.integrated_position_error = 0.0
        self.max_integral_force = 0.0
        self.max_signed_displacement_during_push = 0.0
        self.max_displacement_norm_during_push = 0.0
        self.max_signed_displacement_over_episode = 0.0
        self.max_displacement_norm_over_episode = 0.0
        self.release_signed_displacement: float | None = None
        self.first_force_step: int | None = None
        self.actions_seen = 0

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def signed_displacement(self, position: np.ndarray) -> float:
        if self.start_position is None:
            return 0.0
        return float(np.dot(position - self.start_position, self.direction))

    def update_displacement(
        self,
        position: np.ndarray,
        *,
        during_push: bool,
    ) -> float:
        if self.start_position is None:
            return 0.0
        signed = self.signed_displacement(position)
        norm = float(np.linalg.norm(position - self.start_position))
        self.max_signed_displacement_over_episode = max(
            self.max_signed_displacement_over_episode, signed
        )
        self.max_displacement_norm_over_episode = max(
            self.max_displacement_norm_over_episode, norm
        )
        if during_push:
            self.max_signed_displacement_during_push = max(
                self.max_signed_displacement_during_push, signed
            )
            self.max_displacement_norm_during_push = max(
                self.max_displacement_norm_during_push, norm
            )
        return signed

    def force_for_step(
        self,
        control_step: int,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        config = self.config
        if control_step < int(config.start_step):
            return np.zeros(3, dtype=np.float64), "waiting"
        if self.release_step is not None:
            return np.zeros(3, dtype=np.float64), "released"
        if self.start_position is None:
            self.start_position = np.asarray(position, dtype=np.float64).copy()
            self.first_force_step = int(control_step)

        elapsed = int(control_step) - int(config.start_step)
        signed = self.update_displacement(position, during_push=True)
        signed_velocity = float(np.dot(velocity, self.direction))
        alpha = self._smoothstep((elapsed + 1) / int(config.ramp_steps))
        target = float(config.distance) * alpha
        reached_now = bool(
            alpha >= 1.0
            and signed >= float(config.distance) - float(config.position_tolerance)
        )
        if reached_now:
            self.target_reached = True
            self.hold_count += 1
        else:
            self.hold_count = 0

        if self.target_reached and self.hold_count >= int(config.hold_steps):
            self.release_step = int(control_step)
            self.release_reason = "target_reached"
            self.release_signed_displacement = signed
            return np.zeros(3, dtype=np.float64), "released"
        if elapsed >= int(config.max_push_steps):
            self.release_step = int(control_step)
            self.release_reason = "push_timeout"
            self.release_signed_displacement = signed
            return np.zeros(3, dtype=np.float64), "released"

        position_error = target - signed
        candidate_integral = float(
            np.clip(
                self.integrated_position_error + position_error,
                -float(config.integral_error_limit),
                float(config.integral_error_limit),
            )
        )
        base_scalar = (
            float(config.stiffness) * position_error
            - float(config.damping) * signed_velocity
        )
        candidate_scalar = (
            base_scalar + float(config.integral_gain) * candidate_integral
        )
        # Do not wind the integrator farther into force saturation. Always
        # allow it to unwind when the position error changes direction.
        if (
            abs(candidate_scalar) <= float(config.max_force)
            or np.sign(position_error) != np.sign(candidate_scalar)
        ):
            self.integrated_position_error = candidate_integral
        integral_force = (
            float(config.integral_gain) * self.integrated_position_error
        )
        scalar = base_scalar + integral_force
        scalar = float(np.clip(scalar, -config.max_force, config.max_force))
        force = self.direction * scalar
        self.force_steps += int(np.linalg.norm(force) > 0.0)
        self.max_force = max(self.max_force, abs(scalar))
        self.max_integral_force = max(
            self.max_integral_force, abs(integral_force)
        )
        return force, "pushing"

    def archive_episode(self) -> None:
        if self.control_steps < 1:
            return
        self.history.append(
            {
                "start_step": int(self.config.start_step),
                "first_force_step": self.first_force_step,
                "release_step": self.release_step,
                "release_reason": self.release_reason,
                "target_reached": bool(self.target_reached),
                "requested_distance_m": float(self.config.distance),
                "direction": self.direction.tolist(),
                "start_position": (
                    None
                    if self.start_position is None
                    else self.start_position.tolist()
                ),
                "max_signed_displacement_during_push_m": float(
                    self.max_signed_displacement_during_push
                ),
                "max_displacement_norm_during_push_m": float(
                    self.max_displacement_norm_during_push
                ),
                "max_signed_displacement_over_episode_m": float(
                    self.max_signed_displacement_over_episode
                ),
                "max_displacement_norm_over_episode_m": float(
                    self.max_displacement_norm_over_episode
                ),
                "release_signed_displacement_m": (
                    None
                    if self.release_signed_displacement is None
                    else float(self.release_signed_displacement)
                ),
                "force_steps": int(self.force_steps),
                "max_force_n": float(self.max_force),
                "integrated_position_error_m_steps": float(
                    self.integrated_position_error
                ),
                "max_integral_force_n": float(self.max_integral_force),
                "control_steps": int(self.control_steps),
                "actions_seen": int(self.actions_seen),
                "policy_actions_replaced": 0,
            }
        )

    def summary(self) -> dict[str, Any]:
        episodes = list(self.history)
        if self.control_steps > 0:
            self.archive_episode()
            episodes = list(self.history)
            self.history.pop()
        return {
            "protocol": "concurrent_environment_wrench",
            "config": asdict(self.config),
            "episode_records": episodes,
        }


def _find_libero_env(outer_env: Any) -> Any:
    current = outer_env
    for _ in range(8):
        if all(hasattr(current, name) for name in ("cur_step", "task_name", "env")):
            inner = getattr(current, "env", None)
            if hasattr(inner, "env") and hasattr(inner.env, "sim"):
                return current
        if not hasattr(current, "env"):
            break
        current = current.env
    raise TypeError("Could not locate LiberoEnv inside the rollout wrappers")


def _simulation_handles(libero_env: Any) -> tuple[Any, int, str, int]:
    robosuite_env = libero_env.env.env
    sim = robosuite_env.sim
    robot = robosuite_env.robots[0]
    eef_site_id = int(robot.eef_site_id)
    eef_site_name = str(sim.model.site_id2name(eef_site_id))
    eef_body_id = int(sim.model.site_bodyid[eef_site_id])
    return sim, eef_site_id, eef_site_name, eef_body_id


def _find_state_holder(outer_env: Any) -> Any | None:
    current = outer_env
    for _ in range(8):
        if hasattr(current, "_concurrent_perturbation_state"):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def install_concurrent_perturbation(
    outer_env: Any,
    config: ConcurrentPerturbationConfig,
) -> dict[str, Any]:
    """Install one environment-side perturbation hook in a worker."""
    if _find_state_holder(outer_env) is not None:
        raise RuntimeError("Concurrent perturbation hook is already installed")

    state = _ConcurrentPerturbationState(config)
    original_outer_step = outer_env.step
    original_outer_reset = outer_env.reset

    def ensure_current_libero_hook() -> Any:
        libero_env = _find_libero_env(outer_env)
        installed_state = getattr(
            libero_env, "_concurrent_perturbation_state", None
        )
        if installed_state is state:
            return libero_env
        if installed_state is not None:
            raise RuntimeError(
                "Current LiberoEnv has a different concurrent perturbation hook"
            )

        original_step = libero_env.step

        def step_with_perturbation(action: np.ndarray) -> Any:
            sim, eef_site_id, eef_site_name, eef_body_id = (
                _simulation_handles(libero_env)
            )
            position = np.asarray(
                sim.data.site_xpos[eef_site_id], dtype=np.float64
            ).copy()
            velocity = np.asarray(
                sim.data.get_site_xvelp(eef_site_name), dtype=np.float64
            ).copy()
            control_step = int(libero_env.cur_step)
            force, force_phase = state.force_for_step(
                control_step, position, velocity
            )
            sim.data.xfrc_applied[eef_body_id] = 0.0
            sim.data.xfrc_applied[eef_body_id, :3] = force
            state.actions_seen += 1
            try:
                result = original_step(action)
            finally:
                sim.data.xfrc_applied[eef_body_id] = 0.0

            post_position = np.asarray(
                sim.data.site_xpos[eef_site_id], dtype=np.float64
            ).copy()
            state.update_displacement(
                post_position,
                during_push=(force_phase == "pushing"),
            )
            if (
                state.release_step is not None
                and state.release_signed_displacement is None
            ):
                state.release_signed_displacement = state.signed_displacement(
                    post_position
                )
            state.control_steps += 1
            return result

        libero_env._concurrent_perturbation_state = state
        libero_env._concurrent_perturbation_original_step = original_step
        libero_env.step = step_with_perturbation
        return libero_env

    def reset_with_perturbation(*args: Any, **kwargs: Any) -> Any:
        state.archive_episode()
        state.reset_episode()
        ensure_current_libero_hook()
        return original_outer_reset(*args, **kwargs)

    def step_with_current_libero(action: np.ndarray) -> Any:
        ensure_current_libero_hook()
        return original_outer_step(action)

    libero_env = ensure_current_libero_hook()
    outer_env._concurrent_perturbation_state = state
    outer_env._concurrent_perturbation_original_step = original_outer_step
    outer_env._concurrent_perturbation_original_reset = original_outer_reset
    outer_env.step = step_with_current_libero
    outer_env.reset = reset_with_perturbation
    return {
        "installed": True,
        "task_name": str(libero_env.task_name),
        "seed": int(getattr(libero_env, "seed_value", -1)),
        "config": asdict(config),
    }


def collect_concurrent_perturbation(outer_env: Any) -> dict[str, Any]:
    holder = _find_state_holder(outer_env)
    if holder is None:
        raise RuntimeError("Concurrent perturbation hook is not installed")
    state = holder._concurrent_perturbation_state
    return state.summary()
