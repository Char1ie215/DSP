"""Fixed-radius geometric recovery comparators for paired policy outputs.

These are controlled baselines, not reproductions of published algorithms.
Only the selected trajectory is used; no Gaussian support or innovation gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ns_gaussian_ds_policy.endpoint_field import (
    normalize_quaternion, quaternion_multiply, relative_rotation_vector,
    rotvec_to_quaternion,
)

RECOVERY_MODES = ("pd_return", "distance_field")


@dataclass(frozen=True)
class RecoveryConfig:
    tube_radius: float = 0.03
    rotation_radius: float = 0.2
    reentry_radius: float = 0.005
    rotation_reentry_radius: float = 0.1
    stable_steps: int = 2
    kp: float = 4.0  # 1/s: desired Cartesian velocity per pose error
    kd: float = 0.5  # damping on measured velocity, not error differences
    control_dt: float = 0.05
    position_action_scale: float = 0.05
    rotation_action_scale: float = 0.5
    max_position_action: float = 1.0
    max_rotation_action: float = 0.5

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) for v in values.values()):
            raise ValueError("Recovery settings must be finite")
        if any(v <= 0 for k, v in values.items() if k != "kd") or self.kd < 0:
            raise ValueError("Recovery settings must be positive (kd may be zero)")
        if self.reentry_radius >= self.tube_radius:
            raise ValueError("Reentry radius must be smaller than trigger radius")
        if self.rotation_reentry_radius >= self.rotation_radius:
            raise ValueError("Rotation reentry radius must be smaller than trigger radius")


def add_recovery_arguments(parser):
    for name in ("tube_radius", "rotation_radius", "reentry_radius",
                 "rotation_reentry_radius", "kp", "kd", "control_dt"):
        parser.add_argument("--recovery-" + name.replace("_", "-"),
                            type=float, default=getattr(RecoveryConfig(), name))


def recovery_kwargs(args):
    return {name: getattr(args, "recovery_" + name) for name in (
        "tube_radius", "rotation_radius", "reentry_radius",
        "rotation_reentry_radius", "kp", "kd", "control_dt")}


def project_polyline(position, positions, quaternions, start, stop, quaternion=None):
    """Nearest point on segments [start, stop), with SO(3) interpolation.

    Ties select the earliest segment. A zero-length segment is well-defined.
    """
    a = positions[start:stop]
    delta = positions[start + 1:stop + 1] - a
    denom = np.einsum("ij,ij->i", delta, delta)
    alpha = np.clip(np.einsum("ij,ij->i", position - a, delta)
                    / np.maximum(denom, 1e-15), 0, 1)
    if quaternion is not None:
        for i in np.flatnonzero(denom < 1e-15):
            j = start + i
            arc = relative_rotation_vector(quaternions[j], quaternions[j + 1])
            traveled = relative_rotation_vector(quaternions[j], quaternion)
            alpha[i] = np.clip(np.dot(arc, traveled) / max(np.dot(arc, arc), 1e-15), 0, 1)
    points = a + alpha[:, None] * delta
    i = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    j = start + i
    rotation = relative_rotation_vector(quaternions[j], quaternions[j + 1])
    q = quaternion_multiply(rotvec_to_quaternion(alpha[i] * rotation), quaternions[j])
    return points[i], q, float(j + alpha[i])


def _bounded(vector, limit):
    return vector * min(1.0, limit / max(float(np.linalg.norm(vector)), 1e-15))


class GeometricRecoveryPlan:
    def __init__(self, paired, position, quaternion, mode, config):
        self.config, self.mode = config, mode
        paired = np.asarray(paired, dtype=float)
        if mode not in RECOVERY_MODES:
            raise ValueError(f"Unknown recovery mode: {mode}")
        if paired.ndim != 2 or paired.shape[1] != 13 or not len(paired):
            raise ValueError("Expected a nonempty (T,13) decoded paired trajectory")
        if not np.isfinite(paired).all():
            raise ValueError("Non-finite paired trajectory")
        self.actions = paired[:, 6:13].copy()
        self.positions = np.vstack((position, position + np.cumsum(
            paired[:, :3] * config.position_action_scale, axis=0)))
        q = normalize_quaternion(quaternion)
        orientations = [q]
        for delta in paired[:, 3:6]:
            q = quaternion_multiply(rotvec_to_quaternion(
                delta * config.rotation_action_scale), q)
            orientations.append(q)
        self.orientations = np.asarray(orientations)
        self.cursor = 0
        self.prefix_start = 0
        self.recovering = False
        self.stable = 0
        self.last_safe = (np.asarray(position).copy(), orientations[0].copy())
        self.previous = None
        self.anchor = None
        self.locked_stop = None
        self.metrics = Counter()
        self.events = []

    def step(self, position, quaternion):
        """Return native action, or None to request a fresh visual plan."""
        c = self.config
        p, q = np.asarray(position, dtype=float), normalize_quaternion(quaternion)
        velocity = np.zeros(3)
        angular_velocity = np.zeros(3)
        if self.previous is not None:
            velocity = (p - self.previous[0]) / c.control_dt
            angular_velocity = relative_rotation_vector(self.previous[1], q) / c.control_dt
        self.previous = (p.copy(), q.copy())
        # Compare measured feedback against the last issued native segment,
        # including its endpoints, before discarding an exhausted action chunk.
        segment = max(0, self.cursor - 1)
        if not self.recovering:
            ref, rot, _ = project_polyline(p, self.positions, self.orientations,
                                           segment, segment + 1, q)
            if (np.linalg.norm(p - ref) > c.tube_radius
                    or np.linalg.norm(relative_rotation_vector(q, rot)) > c.rotation_radius):
                self.recovering = True
                self.anchor = self.last_safe
                self.locked_stop = segment + 1
                self.metrics["recovery_entries"] += 1
                self.events.append({"event": "entry", "phase": self.cursor,
                                    "position": p.tolist(), "target": self.anchor[0].tolist(),
                                    "prefix_start": self.prefix_start,
                                    "prefix_stop": self.locked_stop})
            else:
                self.last_safe = (p.copy(), q.copy())

        if self.recovering:
            if self.mode == "pd_return":
                ref, rot = self.anchor
            else:
                # -grad(1/2 distance(x, prefix)^2) = projection(x) - x.
                # The projection is onto the locked prefix, never future phases.
                ref, rot, _ = project_polyline(p, self.positions, self.orientations,
                                               self.prefix_start, self.locked_stop, q)
            error = ref - p
            rotation_error = relative_rotation_vector(q, rot)
            inside = (np.linalg.norm(error) <= c.reentry_radius
                      and np.linalg.norm(rotation_error) <= c.rotation_reentry_radius)
            self.stable = self.stable + 1 if inside else 0
            if self.stable >= c.stable_steps:
                self.metrics["recovery_reentries"] += 1
                self.events.append({"event": "reentry", "phase": self.cursor,
                                    "position_error": float(np.linalg.norm(error)),
                                    "rotation_error": float(np.linalg.norm(rotation_error))})
                return None
            damping = c.kd if self.mode == "pd_return" else 0.0
            xyz = (c.kp * error - damping * velocity) * c.control_dt
            rotvec = (c.kp * rotation_error - damping * angular_velocity) * c.control_dt
            self.metrics["recovery_steps"] += 1
            # Zero is the existing runner's incremental-gripper hold convention.
            return np.r_[_bounded(xyz / c.position_action_scale, c.max_position_action),
                         _bounded(rotvec / c.rotation_action_scale, c.max_rotation_action), 0.0]

        if self.cursor >= len(self.actions):
            return None
        action = self.actions[self.cursor].copy()
        if self.cursor and (action[6] > 0) != (self.actions[self.cursor - 1, 6] > 0):
            self.prefix_start = self.cursor
        self.cursor += 1
        self.metrics["native_action_steps"] += 1
        return action


class PairedGeometricRecoveryPolicy(torch.nn.Module):
    """Reuse the exact paired selector/decoder; replace only the executor."""
    def __init__(self, direct_policy, mode, config=None):
        super().__init__()
        self.direct_policy = direct_policy
        self.mode = mode
        self.config = config or RecoveryConfig()
        self.plans = []
        self.metrics = Counter()
        self.events = []

    @property
    def device(self):
        return self.direct_policy.device

    @property
    def dtype(self):
        return self.direct_policy.dtype

    def get_policy_name(self):
        return self.direct_policy.get_policy_name() + "_" + self.mode

    def get_observation_ports(self):
        return self.direct_policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.direct_policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.direct_policy.get_observation_encoder()

    def _archive(self, plan):
        if plan is not None:
            self.metrics.update(plan.metrics)
            self.events.extend(plan.events)

    def reset(self):
        for plan in self.plans:
            self._archive(plan)
        self.plans = []
        return self.direct_policy.reset()

    def get_execution_metrics(self):
        counts = self.metrics.copy()
        for plan in self.plans:
            if plan is not None:
                counts.update(plan.metrics)
        return {**self.direct_policy.get_execution_metrics(), **counts,
                "executor": self.mode, "gaussian_support_used": False,
                "freeze_phase": True, "replan_on_reentry": True,
                "recovery_config": asdict(self.config),
                "recovery_events": self.events + [e for p in self.plans
                                                   if p is not None for e in p.events]}

    def predict_action(self, obs_dict, **kwargs):
        from ns_gaussian_ds_policy.libero_paired_curve import LiberoPairedPoseActionFieldAdapter
        latest = LiberoPairedPoseActionFieldAdapter._latest_numpy
        positions = latest(obs_dict, "robot0_eef_pos", 3)
        orientations = latest(obs_dict, "robot0_eef_quat", 4)
        if len(self.plans) != len(positions):
            for plan in self.plans:
                self._archive(plan)
            self.plans = [None] * len(positions)
        actions, query = [], []
        for i, plan in enumerate(self.plans):
            action = None if plan is None else plan.step(positions[i], orientations[i])
            actions.append(action)
            if action is None:
                query.append(i)
        if query:
            subset = {k: v[query] if isinstance(v, (torch.Tensor, np.ndarray)) else v
                      for k, v in obs_dict.items()}
            result = self.direct_policy.predict_action(subset, **kwargs)
            decoded = self.direct_policy._decode_prediction(result["structured_action"])
            decoded = decoded.detach().cpu().numpy()[:, :self.direct_policy.execution_steps]
            for row, i in enumerate(query):
                self._archive(self.plans[i])
                plan = GeometricRecoveryPlan(decoded[row], positions[i], orientations[i],
                                             self.mode, self.config)
                self.plans[i] = plan
                actions[i] = plan.step(positions[i], orientations[i])
        return {"action": torch.as_tensor(np.stack(actions)[:, None],
                                          device=self.device, dtype=self.dtype)}
