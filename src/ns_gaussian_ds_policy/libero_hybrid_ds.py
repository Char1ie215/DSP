from __future__ import annotations

from dataclasses import replace

import numpy as np

from ns_gaussian_ds_policy.autonomous_se3_field import (
    AutonomousSE3FitConfig,
    PhaseConditionedPoseActionFieldExecutor,
    _compile_chunk,
    _compile_pose_indexed_action_chunk,
    _lookahead_velocity,
    _mixture_output,
    _pose_at_phase,
)
from ns_gaussian_ds_policy.endpoint_field import (
    FIELD_ACTION_DIM,
    EndpointFieldConfig,
    EndpointFieldPlan,
    EndpointFieldStep,
    normalize_quaternion,
    relative_rotation_vector,
)
from ns_gaussian_ds_policy.libero_paired_curve import (
    LiberoPairedBSplineFieldAdapter,
    PairedPoseEnsembleSupport,
    decode_libero_paired_pose_bspline_raw_action,
)


class ProjectedPhaseHybridExecutor(
    PhaseConditionedPoseActionFieldExecutor
):
    """Use native actions in the tube and the full DS outside it.

    Native execution advances an action clock. If the measured pose leaves the
    expected pose range, a monotone local projection initializes a state-based
    phase observer. The complete Gaussian field then controls the robot until
    it re-enters the pose tube. At re-entry, execution either resumes from an
    interpolated action or requests a fresh visual plan, depending on the
    configured experiment.
    """

    _SUPPORT_CONFIDENCE_MODES = frozenset(
        {"support_confidence", "support_confidence_prefix"}
    )

    def __init__(
        self,
        plan: EndpointFieldPlan,
        controller_actions: np.ndarray,
        config: EndpointFieldConfig | None = None,
        fit_config: AutonomousSE3FitConfig | None = None,
        *,
        position_tube_radius: float = 0.04,
        rotation_tube_radius: float = 0.2,
        position_reentry_radius: float = 0.02,
        rotation_reentry_radius: float = 0.1,
        phase_projection_lookahead: float | None = None,
        recovery_phase_advance_budget: float | None = None,
        max_recovery_steps: int = 16,
        recovery_field_source: str = "action",
        recovery_target_mode: str = "projected",
        recovery_anchor_mode: str = "predicted_pose",
        observed_anchor_reentry_radius: float = 0.01,
        observed_anchor_stable_steps: int = 2,
        freeze_recovery_phase: bool = False,
        autonomous_chunk_reentry: bool = False,
        replan_on_recovery_reentry: bool = False,
        rotation_clf_enabled: bool = False,
        rotation_clf_tube_radius: float = 0.05,
        rotation_clf_contraction_rate: float | None = None,
        recovery_progress_patience: int = 0,
        recovery_progress_min_delta: float = 1.0e-3,
        field_speed_gain: float = 1.0,
        support_confidence_outer_score: float = 0.05,
        support_confidence_inner_score: float = 0.1,
        recovery_trigger_mode: str = "support_only",
        require_state_innovation: bool = True,
        disturbance_position_innovation_threshold: float = 0.04,
        disturbance_rotation_innovation_threshold: float = 0.35,
        disturbance_expected_motion_multiplier: float = 1.5,
        disturbance_response_mode: str = "recover_then_replan",
        nominal_fusion_mode: str = "native_action",
        transverse_fusion_inner_ratio: float = 0.5,
        transverse_fusion_outer_ratio: float = 1.0,
        transverse_fusion_max_weight: float = 1.0,
        gating_precision_scale: float = 25.0,
        position_recovery_gain: float = 1.0,
        rotation_recovery_gain: float = 1.0,
        terminal_gripper_command: float | None = None,
        phase_stride: int = 1,
    ) -> None:
        super().__init__(
            plan,
            controller_actions=controller_actions,
            config=config,
            fit_config=fit_config,
            position_tube_radius=position_tube_radius,
            rotation_tube_radius=rotation_tube_radius,
            gating_precision_scale=gating_precision_scale,
            position_recovery_gain=position_recovery_gain,
            rotation_recovery_gain=rotation_recovery_gain,
            terminal_gripper_command=terminal_gripper_command,
            nominal_execution="native_action",
            phase_stride=phase_stride,
        )
        for name, value, outer in (
            (
                "position_reentry_radius",
                position_reentry_radius,
                position_tube_radius,
            ),
            (
                "rotation_reentry_radius",
                rotation_reentry_radius,
                rotation_tube_radius,
            ),
        ):
            if not 0.0 < float(value) <= float(outer):
                raise ValueError(f"{name} must be in (0, outer_radius]")
        if int(max_recovery_steps) < 0:
            raise ValueError("max_recovery_steps must be non-negative")
        frozen_target_modes = {
            "next_action_phase",
            "same_phase_attractor",
            "support_confidence_prefix",
        }
        if recovery_target_mode not in {
            "projected",
            "support_confidence",
            *frozen_target_modes,
        }:
            raise ValueError(
                "recovery_target_mode must be 'projected', "
                "'support_confidence', 'support_confidence_prefix', "
                "'next_action_phase', or 'same_phase_attractor'"
            )
        if recovery_anchor_mode not in {
            "predicted_pose",
            "observed_perturbation_start",
        }:
            raise ValueError(
                "recovery_anchor_mode must be 'predicted_pose' or "
                "'observed_perturbation_start'"
            )
        if (
            recovery_anchor_mode == "observed_perturbation_start"
            and recovery_target_mode != "support_confidence_prefix"
        ):
            raise ValueError(
                "observed perturbation anchors require "
                "support_confidence_prefix recovery"
            )
        if (
            not np.isfinite(float(observed_anchor_reentry_radius))
            or float(observed_anchor_reentry_radius) <= 0.0
        ):
            raise ValueError(
                "observed_anchor_reentry_radius must be finite and positive"
            )
        if int(observed_anchor_stable_steps) <= 0:
            raise ValueError(
                "observed_anchor_stable_steps must be positive"
            )
        if (
            recovery_target_mode in frozen_target_modes
            and not bool(freeze_recovery_phase)
        ):
            raise ValueError(
                f"{recovery_target_mode} recovery requires "
                "freeze_recovery_phase"
            )
        if (
            bool(freeze_recovery_phase)
            and bool(autonomous_chunk_reentry)
            and recovery_target_mode != "support_confidence_prefix"
        ):
            raise ValueError(
                "freeze_recovery_phase and autonomous_chunk_reentry are "
                "mutually exclusive"
            )
        if recovery_target_mode == "support_confidence":
            if bool(freeze_recovery_phase):
                raise ValueError(
                    "support_confidence recovery cannot freeze its field "
                    "observer phase"
                )
            if not bool(autonomous_chunk_reentry):
                raise ValueError(
                    "support_confidence recovery requires "
                    "autonomous_chunk_reentry"
                )
        if recovery_target_mode == "support_confidence_prefix":
            if not bool(freeze_recovery_phase):
                raise ValueError(
                    "support_confidence_prefix recovery requires a frozen "
                    "logical action phase"
                )
            if not bool(autonomous_chunk_reentry):
                raise ValueError(
                    "support_confidence_prefix recovery requires safe chunk "
                    "partitioning"
                )
            if recovery_field_source != "geometry":
                raise ValueError(
                    "support_confidence_prefix recovery requires the "
                    "geometry field source"
                )
        if (
            recovery_trigger_mode == "support_and_state_innovation"
            and recovery_target_mode not in self._SUPPORT_CONFIDENCE_MODES
        ):
            raise ValueError(
                "support_and_state_innovation requires a support-confidence "
                "recovery target mode"
            )
        if recovery_field_source not in {
            "action",
            "geometry",
            "projected_action",
        }:
            raise ValueError(
                "recovery_field_source must be 'action', 'geometry', or "
                "'projected_action'"
            )
        lookahead = (
            float(self.phase_stride)
            if phase_projection_lookahead is None
            else float(phase_projection_lookahead)
        )
        if lookahead <= 0.0:
            raise ValueError("phase_projection_lookahead must be positive")
        phase_advance_budget = (
            float(self.phase_stride)
            if recovery_phase_advance_budget is None
            else float(recovery_phase_advance_budget)
        )
        if phase_advance_budget < 0.0:
            raise ValueError(
                "recovery_phase_advance_budget must be non-negative"
            )
        rotation_contraction = (
            float(self.config.clf_position_contraction_rate)
            if rotation_clf_contraction_rate is None
            else float(rotation_clf_contraction_rate)
        )
        if float(rotation_clf_tube_radius) < 0.0:
            raise ValueError("rotation_clf_tube_radius must be non-negative")
        if rotation_contraction <= 0.0:
            raise ValueError(
                "rotation_clf_contraction_rate must be positive"
            )
        if int(recovery_progress_patience) < 0:
            raise ValueError(
                "recovery_progress_patience must be non-negative"
            )
        if float(recovery_progress_min_delta) < 0.0:
            raise ValueError(
                "recovery_progress_min_delta must be non-negative"
            )
        if (
            not np.isfinite(float(field_speed_gain))
            or float(field_speed_gain) <= 0.0
        ):
            raise ValueError("field_speed_gain must be finite and positive")
        outer_score = float(support_confidence_outer_score)
        inner_score = float(support_confidence_inner_score)
        if not 0.0 < outer_score < inner_score <= 1.0:
            raise ValueError(
                "support confidence scores must satisfy "
                "0 < outer < inner <= 1"
            )
        if recovery_trigger_mode not in {
            "support_only",
            "support_and_state_innovation",
        }:
            raise ValueError(
                "recovery_trigger_mode must be 'support_only' or "
                "'support_and_state_innovation'"
            )
        for name, value in (
            (
                "disturbance_position_innovation_threshold",
                disturbance_position_innovation_threshold,
            ),
            (
                "disturbance_rotation_innovation_threshold",
                disturbance_rotation_innovation_threshold,
            ),
        ):
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(float(disturbance_expected_motion_multiplier))
            or float(disturbance_expected_motion_multiplier) < 1.0
        ):
            raise ValueError(
                "disturbance_expected_motion_multiplier must be finite "
                "and at least 1"
            )
        if disturbance_response_mode not in {
            "recover_then_replan",
            "immediate_replan",
        }:
            raise ValueError(
                "disturbance_response_mode must be "
                "'recover_then_replan' or 'immediate_replan'"
            )
        if (
            disturbance_response_mode == "immediate_replan"
            and recovery_trigger_mode != "support_and_state_innovation"
        ):
            raise ValueError(
                "immediate_replan requires "
                "support_and_state_innovation recovery triggering"
            )
        if nominal_fusion_mode not in {
            "native_action",
            "transverse_minimal",
            "transverse_outward_only",
        }:
            raise ValueError(
                "nominal_fusion_mode must be 'native_action', "
                "'transverse_minimal', or 'transverse_outward_only'"
            )
        fusion_inner = float(transverse_fusion_inner_ratio)
        fusion_outer = float(transverse_fusion_outer_ratio)
        fusion_max_weight = float(transverse_fusion_max_weight)
        if not 0.0 <= fusion_inner < fusion_outer:
            raise ValueError(
                "transverse fusion ratios must satisfy 0 <= inner < outer"
            )
        if not 0.0 <= fusion_max_weight <= 1.0:
            raise ValueError(
                "transverse_fusion_max_weight must be in [0,1]"
            )
        if (
            bool(freeze_recovery_phase)
            and bool(self.config.clf_projection_enabled)
            and self.config.clf_projection_mode == "smooth_tube"
            and float(self.config.clf_position_tube_radius)
            >= float(position_reentry_radius)
        ):
            raise ValueError(
                "frozen recovery requires the position CLF radius to be "
                "strictly inside the position reentry radius"
            )
        if (
            bool(freeze_recovery_phase)
            and bool(rotation_clf_enabled)
            and self.config.clf_projection_mode == "smooth_tube"
            and float(rotation_clf_tube_radius)
            >= float(rotation_reentry_radius)
        ):
            raise ValueError(
                "frozen recovery requires the rotation CLF radius to be "
                "strictly inside the rotation reentry radius"
            )

        self.position_reentry_radius = float(position_reentry_radius)
        self.rotation_reentry_radius = float(rotation_reentry_radius)
        self.phase_projection_lookahead = lookahead
        self.recovery_phase_advance_budget = phase_advance_budget
        self.max_recovery_steps = int(max_recovery_steps)
        self.recovery_field_source = str(recovery_field_source)
        self.recovery_target_mode = str(recovery_target_mode)
        self.recovery_anchor_mode = str(recovery_anchor_mode)
        self.observed_anchor_reentry_radius = float(
            observed_anchor_reentry_radius
        )
        self.observed_anchor_stable_steps = int(
            observed_anchor_stable_steps
        )
        self.freeze_recovery_phase = bool(freeze_recovery_phase)
        self.autonomous_chunk_reentry = bool(autonomous_chunk_reentry)
        self.replan_on_recovery_reentry = bool(
            replan_on_recovery_reentry
        )
        self.rotation_clf_enabled = bool(rotation_clf_enabled)
        self.rotation_clf_tube_radius = float(rotation_clf_tube_radius)
        self.rotation_clf_contraction_rate = rotation_contraction
        self.recovery_progress_patience = int(recovery_progress_patience)
        self.recovery_progress_min_delta = float(
            recovery_progress_min_delta
        )
        self.field_speed_gain = float(field_speed_gain)
        self.support_confidence_outer_score = outer_score
        self.support_confidence_inner_score = inner_score
        self.recovery_trigger_mode = str(recovery_trigger_mode)
        self.require_state_innovation = bool(require_state_innovation)
        self.disturbance_position_innovation_threshold = float(
            disturbance_position_innovation_threshold
        )
        self.disturbance_rotation_innovation_threshold = float(
            disturbance_rotation_innovation_threshold
        )
        self.disturbance_expected_motion_multiplier = float(
            disturbance_expected_motion_multiplier
        )
        self.disturbance_response_mode = str(disturbance_response_mode)
        self.nominal_fusion_mode = str(nominal_fusion_mode)
        self.transverse_fusion_inner_ratio = fusion_inner
        self.transverse_fusion_outer_ratio = fusion_outer
        self.transverse_fusion_max_weight = fusion_max_weight
        self.geometry_nominal_twists: np.ndarray | None = None
        self.position_tube_radii: np.ndarray | None = None
        self.position_reentry_radii: np.ndarray | None = None
        self.position_clf_radii: np.ndarray | None = None
        self.position_tube_precisions: np.ndarray | None = None
        self.position_reentry_precisions: np.ndarray | None = None
        self.position_tube_mahalanobis_radius = 1.0
        self.position_reentry_mahalanobis_radius = 0.5
        self.position_clf_mahalanobis_radius = 0.25
        self.pose_ensemble_support_applied = False
        self.pose_ensemble_cluster_size = 0
        self.pose_ensemble_cluster_confident = False
        self.pose_ensemble_support_confidence = 0.0
        self.pose_ensemble_dense_core_fraction: float | None = None
        self.pose_ensemble_dense_core_confidence: float | None = None
        self.pose_sample_position_outer_radii: np.ndarray | None = None
        self.pose_sample_position_reentry_radii: np.ndarray | None = None
        self.pose_sample_rotation_outer_radii: np.ndarray | None = None
        self.pose_sample_rotation_reentry_radii: np.ndarray | None = None
        self.pose_sample_spread_quantile: float | None = None
        self.last_pose_sample_support_ratio = 0.0
        self.maximum_pose_sample_support_ratio = 0.0
        self._recompile_recovery_chunks()

        self.action_phase = 0.0
        self.last_issued_phase = 0.0
        self.recovery_phase = 0.0
        self.recovery_entry_phase = 0.0
        self.recovering = False
        self.consecutive_recovery_steps = 0
        self.phase_reprojections = 0
        self.phase_reprojection_sum = 0.0
        self.max_phase_reprojection = 0.0
        self.last_rotation_clf_applied = False
        self.last_rotation_clf_distance = 0.0
        self.last_rotation_clf_correction_norm = 0.0
        self.recovery_best_error = float("inf")
        self.recovery_no_progress_steps = 0
        self.recovery_entries = 0
        self.recovery_reentries = 0
        self.recovery_reentry_replan_gripper_deferrals = 0
        self.recovery_no_progress_replans = 0
        self.completed_recovery_steps = 0
        self.max_completed_recovery_steps = 0
        self.recovery_phase_advances = 0
        self.recovery_phase_advance_sum = 0.0
        self.max_recovery_phase_advance = 0.0
        self.recovery_phase_guard_blocks = 0
        self.recovery_gripper_barrier_hits = 0
        self.phase_reprojection_forward_count = 0
        self.phase_reprojection_forward_sum = 0.0
        self.max_phase_reprojection_forward = 0.0
        self.phase_reprojection_backward_count = 0
        self.phase_reprojection_backward_sum = 0.0
        self.max_phase_reprojection_backward = 0.0
        self.autonomous_recovery_interval_index = 0
        self.autonomous_recovery_interval_start = 0.0
        self.autonomous_recovery_interval_end = 0.0
        self.autonomous_recovery_chunk_compiles = 0
        self.last_support_confidence_score = 1.0
        self.minimum_support_confidence_score = 1.0
        self.support_confidence_checks = 0
        self.support_confidence_score_sum = 0.0
        self.support_confidence_direct_fallback_steps = 0
        self.support_confidence_inward_native_steps = 0
        self.last_native_observation_position: np.ndarray | None = None
        self.last_native_observation_quaternion: np.ndarray | None = None
        self.last_native_expected_position_delta: np.ndarray | None = None
        self.last_native_expected_rotation_delta: np.ndarray | None = None
        self.last_safe_observation_position: np.ndarray | None = None
        self.last_safe_observation_quaternion: np.ndarray | None = None
        self.last_safe_observation_phase: float | None = None
        self.state_innovation_checks = 0
        self.state_innovation_triggers = 0
        self.state_innovation_native_bypasses = 0
        self.last_position_innovation = 0.0
        self.last_rotation_innovation = 0.0
        self.max_position_innovation = 0.0
        self.max_rotation_innovation = 0.0
        self.position_innovation_samples: list[float] = []
        self.rotation_innovation_samples: list[float] = []
        self.recovery_locked_phase = 0.0
        self.recovery_prefix_chunk = None
        self.recovery_prefix_start = 0
        self.recovery_prefix_end = 0
        self.recovery_prefix_compiles = 0
        self.recovery_prefix_supports_sum = 0
        self.recovery_prefix_terminal_phase_tolerance = (
            0.5 * float(self.phase_stride)
        )
        self.pending_recovery_anchor_position: np.ndarray | None = None
        self.pending_recovery_anchor_quaternion: np.ndarray | None = None
        self.pending_recovery_anchor_phase: float | None = None
        self.active_recovery_anchor_position: np.ndarray | None = None
        self.active_recovery_anchor_quaternion: np.ndarray | None = None
        self.active_recovery_anchor_offset: np.ndarray | None = None
        self.automatic_recovery_anchor_registrations = 0
        self.observed_anchor_inside_steps = 0
        self.observed_anchor_stability_wait_steps = 0
        self.external_gripper_transition_since_anchor = False
        self.gripper_command_issued = False
        self.gripper_transition_priority_issues = 0
        self.gripper_transition_guard_steps = 0
        self.gripper_transition_open_guard_steps = 0
        self.gripper_transition_close_guard_steps = 0
        self.transverse_fusion_checks = 0
        self.transverse_fusion_active_steps = 0
        self.transverse_fusion_support_fallback_steps = 0
        self.transverse_fusion_weight_sum = 0.0
        self.transverse_fusion_max_weight_applied = 0.0
        self.transverse_fusion_correction_norm_sum = 0.0
        self.transverse_fusion_max_correction_norm = 0.0
        self.last_transverse_fusion_ratio = 0.0
        self.last_transverse_fusion_weight = 0.0
        self.last_transverse_fusion_correction_norm = 0.0

    def _recompile_recovery_chunks(self) -> None:
        if self.recovery_field_source == "action":
            self.geometry_nominal_twists = None
        else:
            if self.recovery_field_source == "geometry":
                self.geometry_nominal_twists = np.zeros(
                    (self.plan.num_targets, 6),
                    dtype=np.float64,
                )
                for phase in range(self.target_count):
                    position_delta, rotation_delta = _lookahead_velocity(
                        self.plan,
                        float(phase),
                        target_count=self.target_count,
                    )
                    # Densification inserts supports inside one native
                    # controller interval, so restore displacement per step.
                    self.geometry_nominal_twists[phase, :3] = (
                        float(self.phase_stride) * position_delta
                    )
                    self.geometry_nominal_twists[phase, 3:6] = (
                        float(self.phase_stride) * rotation_delta
                    )
            else:
                self.geometry_nominal_twists = None

        self.chunks = tuple(
            self._compile_recovery_range(start, end)
            for start, end in self.chunk_ranges
        )
        if self.autonomous_chunk_reentry:
            self.autonomous_recovery_ranges = (
                self._autonomous_recovery_ranges()
            )
            self.autonomous_recovery_chunks = {}
        else:
            self.autonomous_recovery_ranges = ()
            self.autonomous_recovery_chunks = {}
        self.current_chunk = 0

    def _compile_recovery_range(self, start: int, end: int):
        if self.recovery_field_source == "action":
            return _compile_pose_indexed_action_chunk(
                self.plan,
                center_start=start,
                center_end=end,
                controller_twists=self.controller_twists,
                fit_config=self.fit_config,
                gating_precision_scale=self.gating_precision_scale,
                position_recovery_gain=self.position_recovery_gain,
                rotation_recovery_gain=self.rotation_recovery_gain,
            )
        nominal_twists = (
            self.geometry_nominal_twists
            if self.recovery_field_source == "geometry"
            else self.controller_twists
        )
        return _compile_chunk(
            self.plan,
            center_start=start,
            center_end=end,
            target_count=self.target_count,
            field_config=self.config,
            fit_config=self.fit_config,
            nominal_twists=nominal_twists,
        )

    def _compile_locked_recovery_prefix(self) -> None:
        """Compile the active safe prefix with the locked phase as terminal."""

        locked = float(self.recovery_locked_phase)
        rounded = int(round(locked))
        if abs(locked - float(rounded)) > 1.0e-6:
            raise RuntimeError(
                "support-confidence prefix recovery requires an integral "
                "native action phase"
            )
        self._select_autonomous_recovery_interval(locked)
        start = int(self.autonomous_recovery_interval_start)
        end = int(
            np.clip(
                rounded,
                start,
                int(self.autonomous_recovery_interval_end),
            )
        )
        self.recovery_prefix_start = start
        self.recovery_prefix_end = end
        if end <= start:
            self.recovery_prefix_chunk = None
            return

        chunk = _compile_chunk(
            self.plan,
            center_start=start,
            center_end=end,
            target_count=end,
            field_config=self.config,
            fit_config=self.fit_config,
            nominal_twists=self.geometry_nominal_twists,
        )
        if self.active_recovery_anchor_offset is not None:
            chunk = replace(
                chunk,
                support_positions=(
                    chunk.support_positions
                    + self.active_recovery_anchor_offset[None]
                ),
            )
        self.recovery_prefix_chunk = chunk
        self.recovery_prefix_compiles += 1
        self.recovery_prefix_supports_sum += (
            self.recovery_prefix_chunk.num_supports
        )

    def register_observed_recovery_anchor(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        phase: float | None = None,
    ) -> bool:
        """Register a measured EEF pose as the next recovery terminal."""

        if self.recovery_anchor_mode != "observed_perturbation_start":
            return False
        self.pending_recovery_anchor_position = np.asarray(
            position,
            dtype=np.float64,
        ).reshape(3).copy()
        self.pending_recovery_anchor_quaternion = normalize_quaternion(
            quaternion
        )
        self.pending_recovery_anchor_phase = float(
            self.action_phase if phase is None else phase
        )
        return True

    def _record_last_safe_recovery_state(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """Remember the last measured state accepted by pose support."""

        if self.recovery_anchor_mode != "observed_perturbation_start":
            return
        self.last_safe_observation_position = np.asarray(
            position,
            dtype=np.float64,
        ).reshape(3).copy()
        self.last_safe_observation_quaternion = normalize_quaternion(
            quaternion
        )
        self.last_safe_observation_phase = float(self.action_phase)

    def _register_last_safe_recovery_anchor(self) -> bool:
        """Promote the last in-support measurement after a disturbance."""

        if self.recovery_anchor_mode != "observed_perturbation_start":
            return False
        if self._observed_recovery_anchor() is not None:
            return True
        if (
            self.last_safe_observation_position is None
            or self.last_safe_observation_quaternion is None
            or self.last_safe_observation_phase is None
        ):
            return False
        registered = self.register_observed_recovery_anchor(
            self.last_safe_observation_position,
            self.last_safe_observation_quaternion,
            phase=self.last_safe_observation_phase,
        )
        if registered:
            self.automatic_recovery_anchor_registrations += 1
        return registered

    def register_external_gripper_transition(self) -> bool:
        """Preserve cached event semantics after an externally driven move."""

        if self.recovery_anchor_mode != "observed_perturbation_start":
            return False
        if self.pending_recovery_anchor_position is None:
            return False
        self.external_gripper_transition_since_anchor = True
        return True

    def _activate_observed_recovery_anchor(self) -> None:
        self.active_recovery_anchor_position = None
        self.active_recovery_anchor_quaternion = None
        self.active_recovery_anchor_offset = None
        if self.recovery_anchor_mode != "observed_perturbation_start":
            return
        if (
            self.pending_recovery_anchor_position is None
            or self.pending_recovery_anchor_quaternion is None
            or self.pending_recovery_anchor_phase is None
            or abs(
                self.pending_recovery_anchor_phase
                - self.recovery_locked_phase
            )
            > 1.0e-6
        ):
            self.pending_recovery_anchor_position = None
            self.pending_recovery_anchor_quaternion = None
            self.pending_recovery_anchor_phase = None
            return
        reference_position, _ = _pose_at_phase(
            self.plan,
            self.recovery_locked_phase,
            target_count=self.target_count,
        )
        self.active_recovery_anchor_position = (
            self.pending_recovery_anchor_position.copy()
        )
        self.active_recovery_anchor_quaternion = (
            self.pending_recovery_anchor_quaternion.copy()
        )
        self.active_recovery_anchor_offset = (
            self.active_recovery_anchor_position - reference_position
        )
        self.pending_recovery_anchor_position = None
        self.pending_recovery_anchor_quaternion = None
        self.pending_recovery_anchor_phase = None

    def _observed_recovery_anchor(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if (
            self.active_recovery_anchor_position is not None
            and self.active_recovery_anchor_quaternion is not None
        ):
            return (
                self.active_recovery_anchor_position,
                self.active_recovery_anchor_quaternion,
            )
        if (
            self.pending_recovery_anchor_position is not None
            and self.pending_recovery_anchor_quaternion is not None
            and self.pending_recovery_anchor_phase is not None
        ):
            return (
                self.pending_recovery_anchor_position,
                self.pending_recovery_anchor_quaternion,
            )
        return None

    def _observed_recovery_anchor_errors(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[float, float] | None:
        anchor = self._observed_recovery_anchor()
        if anchor is None:
            return None
        anchor_position, anchor_quaternion = anchor
        return (
            float(np.linalg.norm(position - anchor_position)),
            float(
                np.linalg.norm(
                    relative_rotation_vector(quaternion, anchor_quaternion)
                )
            ),
        )

    def _inside_observed_recovery_anchor(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> bool | None:
        errors = self._observed_recovery_anchor_errors(position, quaternion)
        if errors is None:
            return None
        position_error, rotation_error = errors
        return bool(
            position_error <= self.observed_anchor_reentry_radius
            and rotation_error <= self.rotation_reentry_radius
        )

    def _uses_support_confidence(self) -> bool:
        return self.recovery_target_mode in self._SUPPORT_CONFIDENCE_MODES

    def _uses_locked_support_prefix(self) -> bool:
        return self.recovery_target_mode == "support_confidence_prefix"

    def _autonomous_recovery_ranges(self) -> tuple[tuple[int, int], ...]:
        """Split ordered chunks at discrete gripper transitions."""

        transitions = {
            phase
            for phase in range(1, self.target_count)
            if self._gripper_target_is_closed(
                self.plan.gripper_targets[phase]
            )
            != self._gripper_target_is_closed(
                self.plan.gripper_targets[phase - 1]
            )
        }
        ranges: list[tuple[int, int]] = []
        for chunk_start, chunk_end in self.chunk_ranges:
            boundaries = [int(chunk_start)]
            boundaries.extend(
                phase
                for phase in sorted(transitions)
                if chunk_start < phase < chunk_end
            )
            boundaries.append(int(chunk_end))
            ranges.extend(
                (start, end)
                for start, end in zip(boundaries[:-1], boundaries[1:])
                if end > start
            )
        return tuple(ranges)

    @staticmethod
    def _positive_definite_precision(covariance: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 1.0e-10)
        return (
            eigenvectors
            @ np.diag(1.0 / eigenvalues)
            @ eigenvectors.T
        )

    def configure_pose_ensemble_support(
        self,
        support: PairedPoseEnsembleSupport,
        *,
        adaptive_field_support: bool,
        adaptive_tube: bool,
        support_confidence: float = 0.95,
        reentry_fraction: float = 0.5,
        dense_core_confidence: float | None = None,
    ) -> None:
        """Apply pose-sample support without modifying native actions."""

        if not 0.0 < float(support_confidence) < 1.0:
            raise ValueError("support_confidence must be in (0,1)")
        if not 0.0 < float(reentry_fraction) < 1.0:
            raise ValueError("reentry_fraction must be in (0,1)")
        if (
            dense_core_confidence is not None
            and not 0.0 < float(dense_core_confidence) < 1.0
        ):
            raise ValueError("dense_core_confidence must be in (0,1)")
        covariances = np.asarray(
            support.position_covariances,
            dtype=np.float64,
        )
        radii = np.asarray(
            support.position_spread_radii,
            dtype=np.float64,
        )
        rotation_radii = np.asarray(
            support.rotation_spread_radii,
            dtype=np.float64,
        )
        required = self.target_count + 1
        if covariances.shape[0] < required or covariances.shape[1:] != (3, 3):
            raise ValueError(
                "pose ensemble covariances must cover every phase center"
            )
        if radii.shape[0] < required:
            raise ValueError("pose ensemble radii must cover every phase center")
        if rotation_radii.shape[0] < required:
            raise ValueError(
                "pose ensemble rotation radii must cover every phase center"
            )

        self.pose_ensemble_cluster_size = len(support.cluster_members)
        self.pose_ensemble_cluster_confident = bool(
            support.cluster_confident
        )
        self.pose_ensemble_support_confidence = float(support_confidence)
        self.pose_ensemble_dense_core_fraction = support.dense_core_fraction
        self.pose_ensemble_dense_core_confidence = (
            None
            if dense_core_confidence is None
            else float(dense_core_confidence)
        )
        self.pose_sample_spread_quantile = float(support.spread_quantile)
        if not support.cluster_confident:
            # A fragmented pose ensemble does not define one trustworthy tube.
            # Keep the selected paired plan, but retain the original compiler.
            self.pose_ensemble_support_applied = False
            return

        position_outer = np.maximum(radii[:required], 0.0)
        rotation_outer = np.maximum(rotation_radii[:required], 0.0)
        dense_radii = support.position_dense_core_spread_radii
        if dense_radii is not None:
            dense_radii = np.asarray(dense_radii, dtype=np.float64)
            if dense_radii.shape[0] < required:
                raise ValueError(
                    "dense-core pose radii must cover every phase center"
                )
            position_reentry = np.minimum(
                position_outer,
                np.maximum(dense_radii[:required], 0.0),
            )
        else:
            position_reentry = float(reentry_fraction) * position_outer
        self.pose_sample_position_outer_radii = np.ascontiguousarray(
            position_outer
        )
        self.pose_sample_position_reentry_radii = np.ascontiguousarray(
            position_reentry
        )
        self.pose_sample_rotation_outer_radii = np.ascontiguousarray(
            rotation_outer
        )
        self.pose_sample_rotation_reentry_radii = np.ascontiguousarray(
            float(reentry_fraction) * rotation_outer
        )

        prior_covariances = np.stack(
            [
                np.linalg.inv(precision)
                for precision in self.plan.segment_precisions
            ],
            axis=0,
        )
        combined_covariances = prior_covariances + covariances[
            : self.target_count
        ]
        if adaptive_field_support:
            precisions = np.stack(
                [
                    self._positive_definite_precision(covariance)
                    for covariance in combined_covariances
                ],
                axis=0,
            )
            self.plan = replace(
                self.plan,
                segment_precisions=np.ascontiguousarray(precisions),
            )
            self._recompile_recovery_chunks()

        if adaptive_tube:
            contour = float(
                np.sqrt(-2.0 * np.log(1.0 - float(support_confidence)))
            )
            outer = np.zeros((required,), dtype=np.float64)
            outer_transverse_covariances = np.zeros(
                (required, 2, 2),
                dtype=np.float64,
            )
            tube_precisions = np.zeros(
                (required, 3, 3),
                dtype=np.float64,
            )
            for phase in range(required):
                segment = min(phase, self.target_count - 1)
                direction = self.plan.segment_directions[segment]
                covariance = (
                    prior_covariances[segment]
                    + covariances[phase]
                )
                basis = self._transverse_basis(direction)
                transverse = basis.T @ covariance @ basis
                eigenvalues, eigenvectors = np.linalg.eigh(
                    0.5 * (transverse + transverse.T)
                )
                empirical_variance = (
                    float(radii[phase]) / max(contour, 1.0e-12)
                ) ** 2
                eigenvalues = np.maximum(
                    eigenvalues,
                    max(empirical_variance, 1.0e-10),
                )
                transverse = (
                    eigenvectors
                    @ np.diag(eigenvalues)
                    @ eigenvectors.T
                )
                outer_transverse_covariances[phase] = transverse
                tube_precisions[phase] = (
                    basis @ np.linalg.inv(transverse) @ basis.T
                )
                outer[phase] = contour * np.sqrt(
                    float(np.max(eigenvalues))
                )
            self.position_tube_radii = outer
            self.position_tube_precisions = tube_precisions
            self.position_tube_mahalanobis_radius = contour
            dense_covariances = support.position_dense_core_covariances
            dense_radii = support.position_dense_core_spread_radii
            use_dense_core = (
                dense_core_confidence is not None
                and dense_covariances is not None
                and dense_radii is not None
            )
            if use_dense_core:
                dense_covariances = np.asarray(
                    dense_covariances,
                    dtype=np.float64,
                )
                dense_radii = np.asarray(dense_radii, dtype=np.float64)
                if (
                    dense_covariances.shape[0] < required
                    or dense_covariances.shape[1:] != (3, 3)
                    or dense_radii.shape[0] < required
                ):
                    raise ValueError(
                        "dense-core pose support must cover every phase center"
                    )
                core_contour = float(
                    np.sqrt(
                        -2.0
                        * np.log(1.0 - float(dense_core_confidence))
                    )
                )
                reentry_radii = np.zeros((required,), dtype=np.float64)
                reentry_precisions = np.zeros(
                    (required, 3, 3),
                    dtype=np.float64,
                )
                for phase in range(required):
                    segment = min(phase, self.target_count - 1)
                    direction = self.plan.segment_directions[segment]
                    basis = self._transverse_basis(direction)
                    covariance = (
                        prior_covariances[segment]
                        + dense_covariances[phase]
                    )
                    transverse = basis.T @ covariance @ basis
                    eigenvalues, eigenvectors = np.linalg.eigh(
                        0.5 * (transverse + transverse.T)
                    )
                    empirical_variance = (
                        float(dense_radii[phase])
                        / max(core_contour, 1.0e-12)
                    ) ** 2
                    eigenvalues = np.maximum(
                        eigenvalues,
                        max(empirical_variance, 1.0e-10),
                    )
                    transverse = (
                        eigenvectors
                        @ np.diag(eigenvalues)
                        @ eigenvectors.T
                    )
                    core_shape = (core_contour**2) * transverse
                    outer_shape = (contour**2) * (
                        outer_transverse_covariances[phase]
                    )
                    outer_values, outer_vectors = np.linalg.eigh(
                        0.5 * (outer_shape + outer_shape.T)
                    )
                    outer_inverse_sqrt = (
                        outer_vectors
                        @ np.diag(
                            1.0 / np.sqrt(np.maximum(outer_values, 1.0e-12))
                        )
                        @ outer_vectors.T
                    )
                    relative = (
                        outer_inverse_sqrt
                        @ core_shape
                        @ outer_inverse_sqrt
                    )
                    max_ratio = float(
                        np.max(np.linalg.eigvalsh(0.5 * (relative + relative.T)))
                    )
                    if max_ratio > 1.0:
                        core_shape = core_shape / max_ratio
                    reentry_precisions[phase] = (
                        basis @ np.linalg.inv(core_shape) @ basis.T
                    )
                    reentry_radii[phase] = np.sqrt(
                        float(np.max(np.linalg.eigvalsh(core_shape)))
                    )
                self.position_reentry_radii = reentry_radii
                self.position_reentry_precisions = reentry_precisions
                self.position_reentry_mahalanobis_radius = 1.0
            else:
                self.position_reentry_radii = (
                    float(reentry_fraction) * outer
                )
                self.position_reentry_precisions = None
                self.position_reentry_mahalanobis_radius = (
                    float(reentry_fraction) * contour
                )
            self.position_clf_radii = 0.5 * self.position_reentry_radii
            self.position_clf_mahalanobis_radius = (
                0.5 * self.position_reentry_mahalanobis_radius
            )

        self.pose_ensemble_support_applied = bool(
            adaptive_field_support or adaptive_tube
        )

    @staticmethod
    def _transverse_basis(direction: np.ndarray) -> np.ndarray:
        tangent = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-12:
            tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent = tangent / norm
        axis = np.eye(3, dtype=np.float64)[
            int(np.argmin(np.abs(tangent)))
        ]
        first = np.cross(tangent, axis)
        first /= max(float(np.linalg.norm(first)), 1.0e-12)
        second = np.cross(tangent, first)
        second /= max(float(np.linalg.norm(second)), 1.0e-12)
        return np.stack([first, second], axis=1)

    @staticmethod
    def _interpolate_phase_value(
        values: np.ndarray | None,
        phase: float,
        fallback: float,
    ) -> float:
        if values is None:
            return float(fallback)
        clipped = float(np.clip(phase, 0.0, len(values) - 1))
        lower = int(np.floor(clipped))
        upper = min(lower + 1, len(values) - 1)
        alpha = clipped - lower
        return float(
            (1.0 - alpha) * values[lower] + alpha * values[upper]
        )

    @staticmethod
    def _interpolate_phase_matrix(
        values: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        clipped = float(np.clip(phase, 0.0, len(values) - 1))
        lower = int(np.floor(clipped))
        upper = min(lower + 1, len(values) - 1)
        alpha = clipped - lower
        return (
            (1.0 - alpha) * values[lower]
            + alpha * values[upper]
        )

    def _position_mahalanobis_distance(
        self,
        position: np.ndarray,
        phase: float,
        *,
        precisions: np.ndarray | None = None,
    ) -> float | None:
        precision_values = (
            self.position_tube_precisions
            if precisions is None
            else precisions
        )
        if precision_values is None:
            return None
        reference, _ = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        offset = np.asarray(position, dtype=np.float64).reshape(3) - reference
        precision = self._interpolate_phase_matrix(
            precision_values,
            phase,
        )
        squared = float(offset @ precision @ offset)
        return float(np.sqrt(max(squared, 0.0)))

    def _position_outer_radius(self, phase: float) -> float:
        return self._interpolate_phase_value(
            self.position_tube_radii,
            phase,
            self.position_tube_radius,
        )

    def _position_reentry_radius(self, phase: float) -> float:
        return self._interpolate_phase_value(
            self.position_reentry_radii,
            phase,
            self.position_reentry_radius,
        )

    def _position_clf_radius(self, phase: float) -> float:
        return self._interpolate_phase_value(
            self.position_clf_radii,
            phase,
            self.config.clf_position_tube_radius,
        )

    def _support_entry_position_radius(self, phase: float) -> float:
        if (
            self.recovery_trigger_mode == "support_and_state_innovation"
            and self.pose_sample_position_outer_radii is not None
        ):
            return max(
                self._interpolate_phase_value(
                    self.pose_sample_position_outer_radii,
                    phase,
                    0.0,
                ),
                float(self.config.position_tolerance),
            )
        return self._position_outer_radius(phase)

    def _support_reentry_position_radius(self, phase: float) -> float:
        if (
            self.recovery_trigger_mode == "support_and_state_innovation"
            and self.pose_sample_position_reentry_radii is not None
        ):
            return max(
                self._interpolate_phase_value(
                    self.pose_sample_position_reentry_radii,
                    phase,
                    0.0,
                ),
                float(self.config.position_tolerance),
            )
        return self._position_reentry_radius(phase)

    @property
    def compile_metadata(self) -> dict[str, object]:
        metadata = dict(super().compile_metadata)
        metadata.update(
            {
                "executor": "projected_phase_hybrid_ds",
                "progression_source": (
                    "native_action_clock_with_locked_prefix_recovery"
                    if self._uses_locked_support_prefix()
                    else (
                        "native_action_clock_or_support_field_observer"
                        if self.recovery_target_mode == "support_confidence"
                        else (
                            "action_clock_or_autonomous_chunk_projection"
                            if self.autonomous_chunk_reentry
                            else (
                                "action_clock_or_frozen_pose_phase"
                                if self.freeze_recovery_phase
                                else (
                                    "action_clock_or_captured_monotone_pose_projection"
                                )
                            )
                        )
                    )
                ),
                "recovery_execution": (
                    "locked_safe_prefix_gaussian_ds_to_terminal_phase"
                    if self._uses_locked_support_prefix()
                    else (
                        "support_confidence_autonomous_gaussian_ds_with_se3_clf"
                        if self.recovery_target_mode == "support_confidence"
                        else (
                            "frozen_same_phase_linear_attractor"
                            if self.recovery_target_mode
                            == "same_phase_attractor"
                            else (
                                "autonomous_event_chunk_gaussian_ds_with_se3_clf"
                                if self.autonomous_chunk_reentry
                                else "local_full_gaussian_ds_with_se3_clf"
                            )
                        )
                    )
                ),
                "reentry_action": (
                    "fresh_visual_plan_unless_gripper_transition_pending"
                    if self.replan_on_recovery_reentry
                    else (
                        "cached_native_action_at_preserved_phase"
                        if self._uses_support_confidence()
                        else "interpolated_native_action"
                    )
                ),
                "position_reentry_radius": self.position_reentry_radius,
                "rotation_reentry_radius": self.rotation_reentry_radius,
                "phase_projection_lookahead": (
                    self.phase_projection_lookahead
                ),
                "recovery_phase_advance_budget": (
                    self.recovery_phase_advance_budget
                ),
                "max_recovery_steps": self.max_recovery_steps,
                "recovery_field_source": self.recovery_field_source,
                "recovery_target_mode": self.recovery_target_mode,
                "recovery_anchor_mode": self.recovery_anchor_mode,
                "observed_anchor_reentry_radius": (
                    self.observed_anchor_reentry_radius
                ),
                "observed_anchor_stable_steps": (
                    self.observed_anchor_stable_steps
                ),
                "automatic_recovery_anchor_registrations": int(
                    self.automatic_recovery_anchor_registrations
                ),
                "observed_anchor_stability_wait_steps": int(
                    self.observed_anchor_stability_wait_steps
                ),
                "freeze_recovery_phase": self.freeze_recovery_phase,
                "autonomous_chunk_reentry": (
                    self.autonomous_chunk_reentry
                ),
                "replan_on_recovery_reentry": (
                    self.replan_on_recovery_reentry
                ),
                "autonomous_recovery_ranges": [
                    list(value)
                    for value in self.autonomous_recovery_ranges
                ],
                "rotation_clf_enabled": self.rotation_clf_enabled,
                "rotation_clf_tube_radius": (
                    self.rotation_clf_tube_radius
                ),
                "rotation_clf_contraction_rate": (
                    self.rotation_clf_contraction_rate
                ),
                "recovery_progress_patience": (
                    self.recovery_progress_patience
                ),
                "recovery_progress_min_delta": (
                    self.recovery_progress_min_delta
                ),
                "field_speed_gain": self.field_speed_gain,
                "support_confidence_metric": (
                    (
                        "finite_segment_pose_distance_over_k_sample_spread"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "max_unnormalized_position_gaussian"
                    )
                    if self._uses_support_confidence()
                    else None
                ),
                "support_confidence_source": (
                    (
                        "pose_ensemble_only"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "compiled_dynamics_gaussian"
                    )
                    if self._uses_support_confidence()
                    else None
                ),
                "support_confidence_outer_ratio": (
                    1.0
                    if self.recovery_trigger_mode
                    == "support_and_state_innovation"
                    else None
                ),
                "support_confidence_outer_score": (
                    self.support_confidence_outer_score
                ),
                "support_confidence_inner_score": (
                    self.support_confidence_inner_score
                ),
                "support_confidence_cluster_fallback": (
                    "native_action_only"
                    if self._uses_support_confidence()
                    else None
                ),
                "support_confidence_gripper_event_barrier": (
                    "hold_unissued_event_at_locked_terminal"
                    if self._uses_locked_support_prefix()
                    else (
                        "issue_native_transition_repeat_until_support_reentry"
                        if self.recovery_target_mode == "support_confidence"
                        else None
                    )
                ),
                "support_confidence_action_rule": (
                    (
                        (
                            "native_unless_outside_support_and_state_innovation"
                            if self.require_state_innovation
                            else "native_unless_outside_confident_sampled_support"
                        )
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "native_inside_or_inward_else_field"
                    )
                    if self._uses_support_confidence()
                    else None
                ),
                "support_confidence_entry_scope": (
                    (
                        "last_native_finite_pose_segment"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "gripper_safe_chunk"
                    )
                    if self._uses_support_confidence()
                    else None
                ),
                "recovery_trigger_mode": self.recovery_trigger_mode,
                "require_state_innovation": self.require_state_innovation,
                "disturbance_response_mode": (
                    self.disturbance_response_mode
                ),
                "nominal_fusion_mode": self.nominal_fusion_mode,
                "nominal_fusion_geometry": (
                    "native_tangent_plus_confidence_gated_field_transverse"
                    if self.nominal_fusion_mode == "transverse_minimal"
                    else (
                        "native_action_with_outward_radial_component_shield"
                        if self.nominal_fusion_mode
                        == "transverse_outward_only"
                        else "exact_native_action"
                    )
                ),
                "transverse_fusion_inner_ratio": (
                    self.transverse_fusion_inner_ratio
                ),
                "transverse_fusion_outer_ratio": (
                    self.transverse_fusion_outer_ratio
                ),
                "transverse_fusion_max_weight": (
                    self.transverse_fusion_max_weight
                ),
                "transverse_fusion_rotation": "exact_native_action",
                "transverse_fusion_gripper": "exact_native_action",
                "disturbance_position_innovation_threshold": (
                    self.disturbance_position_innovation_threshold
                ),
                "disturbance_rotation_innovation_threshold": (
                    self.disturbance_rotation_innovation_threshold
                ),
                "disturbance_expected_motion_multiplier": (
                    self.disturbance_expected_motion_multiplier
                ),
                "recovery_prefix_start": self.recovery_prefix_start,
                "recovery_prefix_end": self.recovery_prefix_end,
                "recovery_prefix_compiles": self.recovery_prefix_compiles,
                "recovery_prefix_supports_sum": (
                    self.recovery_prefix_supports_sum
                ),
                "recovery_prefix_terminal_phase_tolerance": (
                    self.recovery_prefix_terminal_phase_tolerance
                ),
                "effective_recovery_speed_gain": (
                    1.0
                    if self.recovery_target_mode
                    == "same_phase_attractor"
                    else self.field_speed_gain
                ),
                "geometry_support_time": (
                    None
                    if self.recovery_field_source != "geometry"
                    else 1.0 / float(self.phase_stride)
                ),
                "geometry_velocity_scale": (
                    None
                    if self.recovery_field_source != "geometry"
                    else float(self.phase_stride)
                ),
                "gripper_during_recovery": "neutral_increment_hold",
                "recovery_phase_event_barrier": "next_gripper_transition",
                "pose_ensemble_support_applied": (
                    self.pose_ensemble_support_applied
                ),
                "pose_ensemble_cluster_size": (
                    self.pose_ensemble_cluster_size
                ),
                "pose_ensemble_cluster_confident": (
                    self.pose_ensemble_cluster_confident
                ),
                "pose_ensemble_support_confidence": (
                    self.pose_ensemble_support_confidence
                ),
                "pose_ensemble_dense_core_fraction": (
                    self.pose_ensemble_dense_core_fraction
                ),
                "pose_ensemble_dense_core_confidence": (
                    self.pose_ensemble_dense_core_confidence
                ),
                "pose_sample_spread_quantile": (
                    self.pose_sample_spread_quantile
                ),
                "pose_sample_position_outer_radius_min": (
                    float(np.min(self.pose_sample_position_outer_radii))
                    if self.pose_sample_position_outer_radii is not None
                    else None
                ),
                "pose_sample_position_outer_radius_mean": (
                    float(np.mean(self.pose_sample_position_outer_radii))
                    if self.pose_sample_position_outer_radii is not None
                    else None
                ),
                "pose_sample_position_outer_radius_max": (
                    float(np.max(self.pose_sample_position_outer_radii))
                    if self.pose_sample_position_outer_radii is not None
                    else None
                ),
                "pose_sample_position_reentry_radius_mean": (
                    float(np.mean(self.pose_sample_position_reentry_radii))
                    if self.pose_sample_position_reentry_radii is not None
                    else None
                ),
                "dynamics_gaussian_support_source": (
                    "compiler_prior_plus_pose_ensemble_covariance"
                    if self.pose_ensemble_support_applied
                    else "compiler_prior"
                ),
                "position_reentry_mode": (
                    "robust_dense_core_mahalanobis"
                    if self.position_reentry_precisions is not None
                    else "scaled_outer_mahalanobis"
                ),
                "position_tube_mode": (
                    "phase_adaptive_transverse_mahalanobis"
                    if self.position_tube_precisions is not None
                    else "fixed_radius"
                ),
                "phase_position_tube_radius_min": (
                    float(np.min(self.position_tube_radii))
                    if self.position_tube_radii is not None
                    else self.position_tube_radius
                ),
                "phase_position_tube_radius_mean": (
                    float(np.mean(self.position_tube_radii))
                    if self.position_tube_radii is not None
                    else self.position_tube_radius
                ),
                "phase_position_tube_radius_max": (
                    float(np.max(self.position_tube_radii))
                    if self.position_tube_radii is not None
                    else self.position_tube_radius
                ),
            }
        )
        return metadata

    def _sync_action_index(self) -> None:
        self.action_index = min(
            int(np.floor(self.action_phase)),
            self.target_count,
        )
        self.continuous_phase = float(self.action_phase)

    def _select_chunk_for_phase(self, phase: float) -> None:
        self._select_chunk_for_action(
            min(int(np.floor(float(phase))), self.target_count - 1)
        )

    def _select_autonomous_recovery_interval(self, phase: float) -> None:
        phase = float(np.clip(phase, 0.0, self.target_count))
        next_event = self._next_gripper_event_phase(phase)
        tolerance = 1.0e-9
        if next_event <= phase + tolerance:
            candidates = [
                index
                for index, (start, end) in enumerate(
                    self.autonomous_recovery_ranges
                )
                if start <= phase + tolerance
                and abs(float(end) - next_event) <= tolerance
            ]
        else:
            candidates = [
                index
                for index, (start, end) in enumerate(
                    self.autonomous_recovery_ranges
                )
                if start <= phase + tolerance
                and phase <= end + tolerance
                and end <= next_event + tolerance
            ]
        if not candidates:
            candidates = [
                index
                for index, (start, end) in enumerate(
                    self.autonomous_recovery_ranges
                )
                if start <= phase + tolerance and phase <= end + tolerance
            ]
        if not candidates:
            raise RuntimeError(
                f"No autonomous recovery interval contains phase {phase}"
            )
        index = max(
            candidates,
            key=lambda value: self.autonomous_recovery_ranges[value][0],
        )
        start, end = self.autonomous_recovery_ranges[index]
        self.autonomous_recovery_interval_index = int(index)
        self.autonomous_recovery_interval_start = float(start)
        self.autonomous_recovery_interval_end = float(end)
        if index not in self.autonomous_recovery_chunks:
            self.autonomous_recovery_chunks[index] = (
                self._compile_recovery_range(start, end)
            )
            self.autonomous_recovery_chunk_compiles += 1

    def _support_confidence_available(self) -> bool:
        return bool(
            self.pose_ensemble_support_applied
            and self.pose_ensemble_cluster_confident
        )

    def _pose_sample_confidence_available(self) -> bool:
        return bool(
            self.pose_ensemble_cluster_confident
            and self.pose_sample_position_outer_radii is not None
            and self.pose_sample_position_reentry_radii is not None
            and self.pose_sample_rotation_outer_radii is not None
            and self.pose_sample_rotation_reentry_radii is not None
        )

    def _record_support_confidence_score(self, score: float) -> None:
        value = float(np.clip(score, 0.0, 1.0))
        self.last_support_confidence_score = value
        self.minimum_support_confidence_score = min(
            self.minimum_support_confidence_score,
            value,
        )
        self.support_confidence_checks += 1
        self.support_confidence_score_sum += value

    def _sampled_pose_support_ratio(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        lower_phase: float,
        upper_phase: float,
        reentry: bool,
        record: bool,
    ) -> tuple[float, float]:
        """Measure pose distance against the finite K-sample support tube."""

        if not self._pose_sample_confidence_available():
            return 0.0, float(self.action_phase)
        position_radii = (
            self.pose_sample_position_reentry_radii
            if reentry
            else self.pose_sample_position_outer_radii
        )
        rotation_radii = (
            self.pose_sample_rotation_reentry_radii
            if reentry
            else self.pose_sample_rotation_outer_radii
        )
        assert position_radii is not None
        assert rotation_radii is not None

        maximum_phase = float(self.target_count)
        lower = float(np.clip(lower_phase, 0.0, maximum_phase))
        upper = float(np.clip(upper_phase, 0.0, maximum_phase))
        if upper < lower:
            lower, upper = upper, lower
        knots = [lower]
        for phase in range(int(np.floor(lower)) + 1, int(np.ceil(upper))):
            if lower < float(phase) < upper:
                knots.append(float(phase))
        if upper > lower + 1.0e-12:
            knots.append(upper)

        measured_position = np.asarray(position, dtype=np.float64).reshape(3)
        measured_quaternion = normalize_quaternion(quaternion)
        best_ratio = float("inf")
        best_phase = lower
        intervals = list(zip(knots[:-1], knots[1:], strict=True))
        if not intervals:
            intervals = [(lower, lower)]
        for phase_start, phase_end in intervals:
            start_position, _ = _pose_at_phase(
                self.plan,
                phase_start,
                target_count=self.target_count,
            )
            end_position, _ = _pose_at_phase(
                self.plan,
                phase_end,
                target_count=self.target_count,
            )
            segment = end_position - start_position
            segment_squared = float(segment @ segment)
            if segment_squared <= 1.0e-12:
                fraction = 0.0
            else:
                fraction = float(
                    np.clip(
                        (measured_position - start_position) @ segment
                        / segment_squared,
                        0.0,
                        1.0,
                    )
                )
            phase = phase_start + fraction * (phase_end - phase_start)
            reference_position, reference_quaternion = _pose_at_phase(
                self.plan,
                phase,
                target_count=self.target_count,
            )
            position_error = float(
                np.linalg.norm(measured_position - reference_position)
            )
            rotation_error = float(
                np.linalg.norm(
                    relative_rotation_vector(
                        measured_quaternion,
                        reference_quaternion,
                    )
                )
            )
            position_radius = max(
                self._interpolate_phase_value(position_radii, phase, 0.0),
                float(self.config.position_tolerance),
                1.0e-6,
            )
            rotation_radius = max(
                self._interpolate_phase_value(rotation_radii, phase, 0.0),
                float(self.config.rotation_tolerance),
                1.0e-6,
            )
            ratio = max(
                position_error / position_radius,
                rotation_error / rotation_radius,
            )
            if ratio < best_ratio:
                best_ratio = ratio
                best_phase = phase

        if record:
            self.last_pose_sample_support_ratio = best_ratio
            self.maximum_pose_sample_support_ratio = max(
                self.maximum_pose_sample_support_ratio,
                best_ratio,
            )
            self._record_support_confidence_score(
                float(np.exp(max(-0.5 * best_ratio**2, -80.0)))
            )
        return best_ratio, best_phase

    def _gaussian_support_score(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        recovery: bool,
        lower_phase: float | None = None,
        upper_phase: float | None = None,
        record: bool,
    ) -> tuple[float, float]:
        """Return max unnormalized position support and its ordered phase.

        The K-sample support model estimates positional covariance. Rotation is
        still handled by the SE(3) recovery field and CLF, but it must not be
        folded into a confidence score that was never calibrated from the
        sampled orientation distribution.
        """

        if not self._support_confidence_available():
            return 1.0, float(self.action_phase)
        if recovery:
            chunk = self.autonomous_recovery_chunks[
                self.autonomous_recovery_interval_index
            ]
        else:
            self._select_chunk_for_phase(self.action_phase)
            chunk = self.current_field

        _, position_offsets, _ = (
            self._chunk_gating_and_offsets(position, quaternion, chunk)
        )
        position_squared = np.einsum(
            "ki,kij,kj->k",
            position_offsets[0],
            chunk.position_precisions,
            position_offsets[0],
        )
        squared_distance = position_squared
        support_phases = np.arange(
            chunk.center_start,
            chunk.center_start + len(chunk.support_positions),
            dtype=np.float64,
        )
        if lower_phase is not None and upper_phase is not None:
            lower = min(float(lower_phase), float(upper_phase))
            upper = max(float(lower_phase), float(upper_phase))
            mask = (
                (support_phases >= lower - 1.0e-9)
                & (support_phases <= upper + 1.0e-9)
            )
            if not np.any(mask):
                midpoint = 0.5 * (lower + upper)
                mask[np.argmin(np.abs(support_phases - midpoint))] = True
            squared_distance = np.where(mask, squared_distance, np.inf)

        nearest = int(np.argmin(squared_distance))
        minimum = float(max(squared_distance[nearest], 0.0))
        score = float(np.exp(max(-0.5 * minimum, -80.0)))
        if record:
            self._record_support_confidence_score(score)
        return score, float(support_phases[nearest])

    def _inside_gaussian_outer_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        # Confidence belongs to the complete ordered, gripper-safe support
        # chunk. Only the recovery field is truncated at the locked phase.
        self._select_autonomous_recovery_interval(self.action_phase)
        score, _ = self._gaussian_support_score(
            position,
            quaternion,
            recovery=True,
            record=record,
        )
        return bool(score >= self.support_confidence_outer_score)

    def _inside_current_native_segment_outer_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        """Test support only along the segment expected after the last action."""

        self._select_autonomous_recovery_interval(self.action_phase)
        lower_phase = max(
            0.0,
            float(self.action_phase) - float(self.phase_stride),
        )
        upper_phase = max(
            lower_phase,
            float(self.action_phase) - 1.0,
        )
        score, _ = self._gaussian_support_score(
            position,
            quaternion,
            recovery=True,
            lower_phase=lower_phase,
            upper_phase=upper_phase,
            record=record,
        )
        return bool(score >= self.support_confidence_outer_score)

    def _inside_sampled_current_native_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        lower_phase = max(
            0.0,
            float(self.action_phase) - float(self.phase_stride),
        )
        ratio, _ = self._sampled_pose_support_ratio(
            position,
            quaternion,
            lower_phase=lower_phase,
            upper_phase=float(self.action_phase),
            reentry=False,
            record=record,
        )
        return bool(ratio <= 1.0)

    def _inside_gaussian_inner_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        score, _ = self._gaussian_support_score(
            position,
            quaternion,
            recovery=True,
            record=record,
        )
        return bool(score >= self.support_confidence_inner_score)

    def _inside_locked_prefix_terminal_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        anchored = self._inside_observed_recovery_anchor(
            position,
            quaternion,
        )
        if anchored is not None:
            return anchored
        score, _ = self._gaussian_support_score(
            position,
            quaternion,
            recovery=True,
            lower_phase=self.recovery_locked_phase,
            upper_phase=self.recovery_locked_phase,
            record=record,
        )
        _, rotation_error = self._pose_errors_at_phase(
            position,
            quaternion,
            self.recovery_locked_phase,
        )
        projected_phase, _, _ = self._ordered_local_projection(
            position,
            quaternion,
            lower_phase=float(self.recovery_prefix_start),
            upper_phase=self.recovery_locked_phase,
        )
        return bool(
            score >= self.support_confidence_inner_score
            and rotation_error <= self.rotation_reentry_radius
            and projected_phase
            >= (
                self.recovery_locked_phase
                - self.recovery_prefix_terminal_phase_tolerance
                + 1.0e-9
            )
        )

    def _inside_sampled_locked_terminal_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        record: bool,
    ) -> bool:
        anchored = self._inside_observed_recovery_anchor(
            position,
            quaternion,
        )
        if anchored is not None:
            return anchored
        ratio, _ = self._sampled_pose_support_ratio(
            position,
            quaternion,
            lower_phase=self.recovery_locked_phase,
            upper_phase=self.recovery_locked_phase,
            reentry=True,
            record=record,
        )
        projected_phase, _, _ = self._ordered_local_projection(
            position,
            quaternion,
            lower_phase=float(self.recovery_prefix_start),
            upper_phase=self.recovery_locked_phase,
        )
        return bool(
            ratio <= 1.0
            and projected_phase
            >= (
                self.recovery_locked_phase
                - self.recovery_prefix_terminal_phase_tolerance
                + 1.0e-9
            )
        )

    def _native_action_improves_support(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> bool:
        """Keep an out-of-support action when it already points inward."""

        action = self._native_action_at_phase(
            self.action_phase,
            None,
            mutate_gripper=False,
        )
        candidate_position = (
            np.asarray(position, dtype=np.float64).reshape(3)
            + np.asarray(action[:3], dtype=np.float64)
            * float(self.config.position_action_scale)
        )
        candidate_score, _ = self._gaussian_support_score(
            candidate_position,
            quaternion,
            recovery=True,
            record=False,
        )
        return bool(
            candidate_score
            > self.last_support_confidence_score * (1.0 + 1.0e-6)
        )

    def _pose_errors_at_phase(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        phase: float,
    ) -> tuple[float, float]:
        reference_position, reference_orientation = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        return (
            float(np.linalg.norm(position - reference_position)),
            float(
                np.linalg.norm(
                    relative_rotation_vector(
                        quaternion,
                        reference_orientation,
                    )
                )
            ),
        )

    def _inside_outer_tube(
        self,
        position_error: float,
        rotation_error: float,
        phase: float,
        *,
        position: np.ndarray | None = None,
    ) -> bool:
        mahalanobis = (
            None
            if position is None
            else self._position_mahalanobis_distance(position, phase)
        )
        position_inside = (
            float(position_error) <= self._position_outer_radius(phase)
            if mahalanobis is None
            else mahalanobis <= self.position_tube_mahalanobis_radius
        )
        return bool(
            position_inside
            and rotation_error <= self.rotation_tube_radius
        )

    def _next_gripper_event_phase(self, phase: float) -> float:
        held_closed = self._gripper_target_is_closed(
            self.last_gripper_command
        )
        start = max(int(np.ceil(float(phase) - 1.0e-9)), 0)
        for index in range(start, self.target_count):
            target_closed = self._gripper_target_is_closed(
                self.plan.gripper_targets[index]
            )
            if target_closed != held_closed:
                return float(index)
        return float(self.target_count)

    def _forward_motion_supports_phase_advance(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        phase: float,
        upper_phase: float,
    ) -> bool:
        if self.previous_position is None or self.previous_quaternion is None:
            return False
        probe_phase = min(float(upper_phase), float(phase) + 1.0)
        if probe_phase <= float(phase) + 1.0e-9:
            return False

        reference_position, reference_orientation = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        next_position, next_orientation = _pose_at_phase(
            self.plan,
            probe_phase,
            target_count=self.target_count,
        )
        translation_tangent = next_position - reference_position
        measured_translation = position - self.previous_position
        translation_progress = float(
            np.dot(measured_translation, translation_tangent)
        )

        rotation_tangent = relative_rotation_vector(
            reference_orientation,
            next_orientation,
        )
        measured_rotation = relative_rotation_vector(
            self.previous_quaternion,
            quaternion,
        )
        rotation_progress = float(
            np.dot(measured_rotation, rotation_tangent)
        )
        return bool(
            translation_progress > 1.0e-12
            or rotation_progress > 1.0e-12
        )

    def _dynamic_recovery_projection(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        lower_phase: float,
        requested_upper_phase: float,
        require_progress_evidence: bool,
    ) -> tuple[float, float, float, bool, bool]:
        lower = float(lower_phase)
        gripper_limit = self._next_gripper_event_phase(lower)
        upper = min(float(requested_upper_phase), gripper_limit)
        barrier_hit = bool(
            gripper_limit < float(requested_upper_phase) - 1.0e-9
        )
        phase, position_error, rotation_error = (
            self._ordered_local_projection(
                position,
                quaternion,
                lower_phase=lower,
                upper_phase=upper,
            )
        )
        guard_blocked = False
        if phase > lower + 1.0e-9 and require_progress_evidence:
            captured = self._inside_outer_tube(
                position_error,
                rotation_error,
                phase,
                position=position,
            )
            moving_forward = self._forward_motion_supports_phase_advance(
                position,
                quaternion,
                phase=lower,
                upper_phase=upper,
            )
            if not (captured and moving_forward):
                phase = lower
                position_error, rotation_error = self._pose_errors_at_phase(
                    position,
                    quaternion,
                    phase,
                )
                guard_blocked = True
        return (
            float(phase),
            float(position_error),
            float(rotation_error),
            guard_blocked,
            barrier_hit,
        )

    def _ordered_local_projection(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        lower_phase: float,
        upper_phase: float,
    ) -> tuple[float, float, float]:
        lower = float(np.clip(lower_phase, 0.0, self.target_count))
        upper = float(np.clip(upper_phase, lower, self.target_count))
        if upper <= lower + 1.0e-12:
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                lower,
            )
            return lower, position_error, rotation_error

        start_segment = min(int(np.floor(lower)), self.target_count - 1)
        end_segment = min(int(np.ceil(upper)), self.target_count)
        candidates: list[tuple[float, float, float]] = []
        for segment_index in range(start_segment, end_segment):
            alpha, _, _ = self._segment_projection(
                position,
                quaternion,
                segment_index,
            )
            phase = float(
                np.clip(segment_index + alpha, lower, upper)
            )
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                phase,
            )
            candidates.append((phase, position_error, rotation_error))
        if not candidates:
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                lower,
            )
            return lower, position_error, rotation_error
        return min(
            candidates,
            key=lambda value: (
                value[1],
                value[2],
                abs(value[0] - lower),
            ),
        )

    def _native_action_at_phase(
        self,
        phase: float,
        gripper_qpos: np.ndarray | None,
        *,
        mutate_gripper: bool,
    ) -> np.ndarray:
        clipped = float(
            np.clip(phase, 0.0, max(self.target_count - 1, 0))
        )
        index = min(int(np.floor(clipped)), self.target_count - 1)
        next_index = min(index + 1, self.target_count - 1)
        alpha = float(clipped - index)
        arm = (
            (1.0 - alpha) * self.controller_actions[index]
            + alpha * self.controller_actions[next_index]
        )
        gripper = float(self.plan.gripper_targets[index])
        if mutate_gripper:
            gripper = self._set_gripper_command_state(
                gripper,
                gripper_qpos,
            )
        return np.concatenate([arm, [gripper]]).astype(np.float32)

    def _upcoming_gripper_transition(self) -> bool:
        if (
            not self.gripper_command_issued
            or self.action_phase >= float(self.target_count)
        ):
            return False
        index = min(
            int(np.floor(float(self.action_phase))),
            self.target_count - 1,
        )
        upcoming = float(self.plan.gripper_targets[index])
        return bool(
            self._gripper_target_is_closed(upcoming)
            != self._gripper_target_is_closed(self.last_gripper_command)
        )

    def _issued_gripper_transition_pending(
        self,
        gripper_qpos: np.ndarray | None,
    ) -> bool:
        if not self.gripper_command_issued or gripper_qpos is None:
            return False
        if self._gripper_target_is_closed(self.last_gripper_command):
            return bool(
                self.gripper_execution_state == self._GRIPPER_CLOSING
            )
        return not self._gripper_target_reached(
            self.last_gripper_command,
            gripper_qpos,
        )

    def _local_clf_project_position_delta(
        self,
        position: np.ndarray,
        position_delta: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        self.last_clf_applied = False
        self.last_clf_correction_norm = 0.0
        if not bool(self.config.clf_projection_enabled):
            self.last_clf_distance = 0.0
            return position_delta

        reference_position, _ = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        offset = position - reference_position
        if self.position_tube_precisions is not None:
            precision = self._interpolate_phase_matrix(
                self.position_tube_precisions,
                phase,
            )
            gradient = precision @ offset
            squared_distance = float(offset @ gradient)
            distance = float(np.sqrt(max(squared_distance, 0.0)))
            self.last_clf_distance = distance
            tube_radius = self.position_clf_mahalanobis_radius
            gradient_norm_squared = float(np.dot(gradient, gradient))
            if (
                distance <= tube_radius
                or gradient_norm_squared <= 1.0e-12
            ):
                return position_delta
            contraction = float(
                self.config.clf_position_contraction_rate
            )
            if self.config.clf_projection_mode == "smooth_tube":
                required = (
                    contraction * (distance - tube_radius) * distance
                )
            else:
                required = contraction * squared_distance
            violation = float(np.dot(gradient, position_delta) + required)
            if violation <= 0.0:
                return position_delta
            correction = (
                violation / gradient_norm_squared
            ) * gradient
            self.last_clf_applied = True
            self.last_clf_correction_norm = float(
                np.linalg.norm(correction)
            )
            return position_delta - correction

        squared_distance = float(np.dot(offset, offset))
        distance = float(np.sqrt(squared_distance))
        self.last_clf_distance = distance
        tube_radius = self._position_clf_radius(phase)
        if distance <= tube_radius or squared_distance <= 1.0e-12:
            return position_delta

        contraction = float(self.config.clf_position_contraction_rate)
        if self.config.clf_projection_mode == "smooth_tube":
            required = contraction * (distance - tube_radius) * distance
        else:
            required = contraction * squared_distance
        violation = float(np.dot(offset, position_delta) + required)
        if violation <= 0.0:
            return position_delta

        correction = (violation / squared_distance) * offset
        self.last_clf_applied = True
        self.last_clf_correction_norm = float(np.linalg.norm(correction))
        return position_delta - correction

    def _local_clf_project_rotation_delta(
        self,
        quaternion: np.ndarray,
        rotation_delta: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        self.last_rotation_clf_applied = False
        self.last_rotation_clf_correction_norm = 0.0
        if not self.rotation_clf_enabled:
            self.last_rotation_clf_distance = 0.0
            return rotation_delta

        _, reference_orientation = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        error = relative_rotation_vector(
            quaternion,
            reference_orientation,
        )
        squared_distance = float(np.dot(error, error))
        distance = float(np.sqrt(squared_distance))
        self.last_rotation_clf_distance = distance
        if (
            distance <= self.rotation_clf_tube_radius
            or squared_distance <= 1.0e-12
        ):
            return rotation_delta

        if self.config.clf_projection_mode == "smooth_tube":
            required = (
                self.rotation_clf_contraction_rate
                * (distance - self.rotation_clf_tube_radius)
                * distance
            )
        else:
            required = self.rotation_clf_contraction_rate * squared_distance
        violation = required - float(np.dot(error, rotation_delta))
        if violation <= 0.0:
            return rotation_delta

        correction = (violation / squared_distance) * error
        self.last_rotation_clf_applied = True
        self.last_rotation_clf_correction_norm = float(
            np.linalg.norm(correction)
        )
        return rotation_delta + correction

    def _local_full_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        phase: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk = self.current_field
        gating, position_offsets, rotation_offsets = (
            self._chunk_gating_and_offsets(position, quaternion, chunk)
        )
        support_phases = np.arange(
            chunk.center_start,
            chunk.center_start + chunk.num_supports,
            dtype=np.float64,
        )
        support_radius = max(1.0, float(self.phase_stride))
        local_mask = np.abs(support_phases - float(phase)) <= support_radius
        if not np.any(local_mask):
            local_mask[np.argmin(np.abs(support_phases - phase))] = True
        local_gating = gating * local_mask[None]
        normalizer = np.sum(local_gating, axis=1, keepdims=True)
        local_gating = local_gating / np.maximum(normalizer, 1.0e-12)

        position_delta = _mixture_output(
            local_gating,
            position_offsets,
            chunk.position_velocities,
            chunk.position_matrices,
        )[0]
        rotation_delta = _mixture_output(
            local_gating,
            rotation_offsets,
            chunk.rotation_velocities,
            chunk.rotation_matrices,
        )[0]
        position_delta = self._local_clf_project_position_delta(
            position,
            position_delta,
            phase,
        )
        rotation_delta = self._local_clf_project_rotation_delta(
            quaternion,
            rotation_delta,
            phase,
        )
        return position_delta, rotation_delta, local_gating[0]

    def _same_phase_attractor_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        phase: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Contract to one frozen pose without nominal phase progression."""

        reference_position, reference_orientation = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        position_delta = self.position_recovery_gain * (
            reference_position - position
        )
        rotation_delta = self.rotation_recovery_gain * (
            relative_rotation_vector(quaternion, reference_orientation)
        )
        self._select_chunk_for_phase(phase)
        gating, _, _ = self._chunk_gating_and_offsets(
            position,
            quaternion,
            self.current_field,
        )
        return position_delta, rotation_delta, gating[0]

    def _autonomous_chunk_full_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        observer_phase: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk = self.autonomous_recovery_chunks[
            self.autonomous_recovery_interval_index
        ]
        gating, position_offsets, rotation_offsets = (
            self._chunk_gating_and_offsets(position, quaternion, chunk)
        )
        position_delta = _mixture_output(
            gating,
            position_offsets,
            chunk.position_velocities,
            chunk.position_matrices,
        )[0]
        rotation_delta = _mixture_output(
            gating,
            rotation_offsets,
            chunk.rotation_velocities,
            chunk.rotation_matrices,
        )[0]
        position_delta = self._local_clf_project_position_delta(
            position,
            position_delta,
            observer_phase,
        )
        rotation_delta = self._local_clf_project_rotation_delta(
            quaternion,
            rotation_delta,
            observer_phase,
        )
        return position_delta, rotation_delta, gating[0]

    def _locked_prefix_full_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chunk = self.recovery_prefix_chunk
        if chunk is None:
            anchor = self._observed_recovery_anchor()
            if anchor is not None:
                anchor_position, anchor_quaternion = anchor
                self._select_chunk_for_phase(self.recovery_locked_phase)
                gating, _, _ = self._chunk_gating_and_offsets(
                    position,
                    quaternion,
                    self.current_field,
                )
                return (
                    self.position_recovery_gain
                    * (anchor_position - position),
                    self.rotation_recovery_gain
                    * relative_rotation_vector(
                        quaternion,
                        anchor_quaternion,
                    ),
                    gating[0],
                )
            return self._same_phase_attractor_field(
                position,
                quaternion,
                self.recovery_locked_phase,
            )
        gating, position_offsets, rotation_offsets = (
            self._chunk_gating_and_offsets(position, quaternion, chunk)
        )
        position_delta = _mixture_output(
            gating,
            position_offsets,
            chunk.position_velocities,
            chunk.position_matrices,
        )[0]
        rotation_delta = _mixture_output(
            gating,
            rotation_offsets,
            chunk.rotation_velocities,
            chunk.rotation_matrices,
        )[0]
        clf_position = position
        if self.active_recovery_anchor_offset is not None:
            clf_position = position - self.active_recovery_anchor_offset
        position_delta = self._local_clf_project_position_delta(
            clf_position,
            position_delta,
            self.recovery_locked_phase,
        )
        rotation_delta = self._local_clf_project_rotation_delta(
            quaternion,
            rotation_delta,
            self.recovery_locked_phase,
        )
        if self.active_recovery_anchor_quaternion is not None:
            rotation_delta = self.rotation_recovery_gain * (
                relative_rotation_vector(
                    quaternion,
                    self.active_recovery_anchor_quaternion,
                )
            )
        return position_delta, rotation_delta, gating[0]

    def _set_tube_state(
        self,
        phase: float,
        position_error: float,
        rotation_error: float,
    ) -> None:
        self.last_tube_phase = float(phase)
        self.last_tube_position_error = float(position_error)
        self.last_tube_rotation_error = float(rotation_error)
        self.last_geometric_phase = float(phase)

    def _record_reprojection(self, old_phase: float, new_phase: float) -> None:
        delta = float(new_phase) - float(old_phase)
        shift = abs(delta)
        if shift <= 1.0e-9:
            return
        self.phase_reprojections += 1
        self.phase_reprojection_sum += shift
        self.max_phase_reprojection = max(
            self.max_phase_reprojection,
            shift,
        )
        if delta > 0.0:
            self.phase_reprojection_forward_count += 1
            self.phase_reprojection_forward_sum += delta
            self.max_phase_reprojection_forward = max(
                self.max_phase_reprojection_forward,
                delta,
            )
        else:
            magnitude = -delta
            self.phase_reprojection_backward_count += 1
            self.phase_reprojection_backward_sum += magnitude
            self.max_phase_reprojection_backward = max(
                self.max_phase_reprojection_backward,
                magnitude,
            )

    @staticmethod
    def _motion_segment_innovation(
        actual_delta: np.ndarray,
        expected_delta: np.ndarray,
        *,
        maximum_expected_progress: float,
    ) -> float:
        """Distance from measured motion to plausible commanded progress."""

        actual = np.asarray(actual_delta, dtype=np.float64).reshape(3)
        expected = np.asarray(expected_delta, dtype=np.float64).reshape(3)
        expected_squared = float(expected @ expected)
        if expected_squared <= 1.0e-12:
            return float(np.linalg.norm(actual))
        progress = float(actual @ expected) / expected_squared
        progress = float(
            np.clip(progress, 0.0, float(maximum_expected_progress))
        )
        return float(np.linalg.norm(actual - progress * expected))

    def _state_innovation_detected(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> bool:
        """Detect motion that cannot be explained by the last native action."""

        if (
            self.last_native_observation_position is None
            or self.last_native_observation_quaternion is None
            or self.last_native_expected_position_delta is None
            or self.last_native_expected_rotation_delta is None
        ):
            self.last_position_innovation = 0.0
            self.last_rotation_innovation = 0.0
            return False

        actual_position_delta = (
            np.asarray(position, dtype=np.float64).reshape(3)
            - self.last_native_observation_position
        )
        actual_rotation_delta = relative_rotation_vector(
            self.last_native_observation_quaternion,
            quaternion,
        )
        position_innovation = self._motion_segment_innovation(
            actual_position_delta,
            self.last_native_expected_position_delta,
            maximum_expected_progress=(
                self.disturbance_expected_motion_multiplier
            ),
        )
        rotation_innovation = self._motion_segment_innovation(
            actual_rotation_delta,
            self.last_native_expected_rotation_delta,
            maximum_expected_progress=(
                self.disturbance_expected_motion_multiplier
            ),
        )
        self.state_innovation_checks += 1
        self.last_position_innovation = position_innovation
        self.last_rotation_innovation = rotation_innovation
        self.position_innovation_samples.append(position_innovation)
        self.rotation_innovation_samples.append(rotation_innovation)
        self.max_position_innovation = max(
            self.max_position_innovation,
            position_innovation,
        )
        self.max_rotation_innovation = max(
            self.max_rotation_innovation,
            rotation_innovation,
        )
        return bool(
            position_innovation
            > self.disturbance_position_innovation_threshold
            or rotation_innovation
            > self.disturbance_rotation_innovation_threshold
        )

    def _phase_tangent(self, phase: float) -> np.ndarray | None:
        position_delta, _ = _lookahead_velocity(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        norm = float(np.linalg.norm(position_delta))
        if norm <= 1.0e-12 and float(phase) > 0.0:
            current, _ = _pose_at_phase(
                self.plan,
                phase,
                target_count=self.target_count,
            )
            previous, _ = _pose_at_phase(
                self.plan,
                max(0.0, float(phase) - 1.0),
                target_count=self.target_count,
            )
            position_delta = current - previous
            norm = float(np.linalg.norm(position_delta))
        if norm <= 1.0e-12:
            return None
        return position_delta / norm

    def _transverse_support_ratio(
        self,
        position: np.ndarray,
        phase: float,
        tangent: np.ndarray,
    ) -> float | None:
        """Return normalized transverse distance to the K-sample support."""

        if not self._support_confidence_available():
            return None
        mahalanobis = self._position_mahalanobis_distance(position, phase)
        if mahalanobis is not None:
            return mahalanobis / max(
                float(self.position_tube_mahalanobis_radius),
                1.0e-12,
            )
        if self.pose_sample_position_outer_radii is None:
            return None
        reference, _ = _pose_at_phase(
            self.plan,
            phase,
            target_count=self.target_count,
        )
        offset = np.asarray(position, dtype=np.float64).reshape(3) - reference
        transverse = offset - float(offset @ tangent) * tangent
        radius = max(
            self._interpolate_phase_value(
                self.pose_sample_position_outer_radii,
                phase,
                0.0,
            ),
            float(self.config.position_tolerance),
            1.0e-6,
        )
        return float(np.linalg.norm(transverse)) / radius

    def _bounded_transverse_weight(
        self,
        native_position_action: np.ndarray,
        transverse_correction: np.ndarray,
        requested_weight: float,
    ) -> float:
        """Reduce the blend weight without changing the native tangent."""

        native = np.asarray(
            native_position_action,
            dtype=np.float64,
        ).reshape(3)
        correction = np.asarray(
            transverse_correction,
            dtype=np.float64,
        ).reshape(3)
        limit = float(self.config.max_position_action)
        if np.any(np.abs(native) > limit + 1.0e-9):
            return 0.0
        weight = float(np.clip(requested_weight, 0.0, 1.0))
        for value, delta in zip(native, correction, strict=True):
            if delta > 1.0e-12:
                weight = min(weight, (limit - value) / delta)
            elif delta < -1.0e-12:
                weight = min(weight, (-limit - value) / delta)
        return float(np.clip(weight, 0.0, requested_weight))

    def _transverse_fused_action(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        phase: float,
        native_action: np.ndarray,
    ) -> np.ndarray:
        """Apply only the field component transverse to the pose spline."""

        self.last_transverse_fusion_ratio = 0.0
        self.last_transverse_fusion_weight = 0.0
        self.last_transverse_fusion_correction_norm = 0.0
        if self.nominal_fusion_mode not in {
            "transverse_minimal",
            "transverse_outward_only",
        }:
            return native_action

        self.transverse_fusion_checks += 1
        tangent = self._phase_tangent(phase)
        if tangent is None:
            self.transverse_fusion_support_fallback_steps += 1
            return native_action
        ratio = self._transverse_support_ratio(position, phase, tangent)
        if ratio is None or not np.isfinite(ratio):
            self.transverse_fusion_support_fallback_steps += 1
            return native_action
        self.last_transverse_fusion_ratio = float(ratio)

        interval = (
            self.transverse_fusion_outer_ratio
            - self.transverse_fusion_inner_ratio
        )
        unit = float(
            np.clip(
                (ratio - self.transverse_fusion_inner_ratio) / interval,
                0.0,
                1.0,
            )
        )
        requested_weight = self.transverse_fusion_max_weight * (
            unit * unit * (3.0 - 2.0 * unit)
        )
        if requested_weight <= 1.0e-12:
            return native_action

        self._select_chunk_for_phase(phase)
        position_delta, _, _ = self._local_full_field(
            position,
            quaternion,
            phase,
        )
        field_position_action = self._field_action(
            self.field_speed_gain * position_delta,
            np.zeros(3, dtype=np.float64),
            0.0,
        )[:3].astype(np.float64)
        native_position_action = np.asarray(
            native_action[:3],
            dtype=np.float64,
        )
        native_transverse = native_position_action - (
            float(native_position_action @ tangent) * tangent
        )
        field_transverse = field_position_action - (
            float(field_position_action @ tangent) * tangent
        )
        if self.nominal_fusion_mode == "transverse_outward_only":
            reference, _ = _pose_at_phase(
                self.plan,
                phase,
                target_count=self.target_count,
            )
            offset = np.asarray(position, dtype=np.float64).reshape(3) - (
                reference
            )
            radial = offset - float(offset @ tangent) * tangent
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm <= 1.0e-12:
                return native_action
            radial /= radial_norm
            native_outward = float(native_transverse @ radial)
            if native_outward <= 1.0e-12:
                return native_action
            field_outward = float(field_transverse @ radial)
            target_outward = min(field_outward, 0.0)
            correction = (
                target_outward - native_outward
            ) * radial
        else:
            correction = field_transverse - native_transverse
        weight = self._bounded_transverse_weight(
            native_position_action,
            correction,
            requested_weight,
        )
        if weight <= 1.0e-12:
            return native_action

        fused = np.asarray(native_action, dtype=np.float32).copy()
        applied = weight * correction
        fused[:3] = (native_position_action + applied).astype(np.float32)
        correction_norm = float(np.linalg.norm(applied))
        self.transverse_fusion_active_steps += 1
        self.transverse_fusion_weight_sum += weight
        self.transverse_fusion_max_weight_applied = max(
            self.transverse_fusion_max_weight_applied,
            weight,
        )
        self.transverse_fusion_correction_norm_sum += correction_norm
        self.transverse_fusion_max_correction_norm = max(
            self.transverse_fusion_max_correction_norm,
            correction_norm,
        )
        self.last_transverse_fusion_weight = weight
        self.last_transverse_fusion_correction_norm = correction_norm
        return fused

    def _issue_native(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        gripper_qpos: np.ndarray | None,
        *,
        phase: float,
        mutate_gripper: bool,
    ) -> EndpointFieldStep:
        issued_phase = float(phase)
        if mutate_gripper:
            self.action_phase = issued_phase
            self._sync_action_index()
        self._select_chunk_for_phase(issued_phase)
        self.target_index = min(
            int(np.floor(issued_phase)),
            self.target_count - 1,
        )
        action = self._native_action_at_phase(
            issued_phase,
            gripper_qpos,
            mutate_gripper=mutate_gripper,
        )
        if mutate_gripper:
            action = self._transverse_fused_action(
                position,
                quaternion,
                issued_phase,
                action,
            )
        weights = self.mixture_weights(position, quaternion)
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        if mutate_gripper:
            self.last_gripper_command = float(action[-1])
            self.gripper_command_issued = True
            self._record_action_state(position, quaternion, action)
            self.last_native_observation_position = position.copy()
            self.last_native_observation_quaternion = quaternion.copy()
            self.last_native_expected_position_delta = (
                np.asarray(action[:3], dtype=np.float64)
                * float(self.config.position_action_scale)
            )
            self.last_native_expected_rotation_delta = (
                np.asarray(action[3:6], dtype=np.float64)
                * float(self.config.rotation_action_scale)
            )
            self.total_steps += 1
            self.steps_on_target += 1
            self.last_issued_phase = issued_phase
            self.action_phase = issued_phase + float(self.phase_stride)
            self._sync_action_index()
            self.gripper_phase = issued_phase
            self.consecutive_recovery_steps = 0
        self.last_phase_frozen = False
        self.last_geometric_on_path = True
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=float(np.max(weights)),
            plan_complete=False,
            replan_required=False,
        )

    def _issue_field(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        mutate_state: bool,
        phase: float | None = None,
        gripper_command_override: float | None = None,
    ) -> EndpointFieldStep:
        field_phase = (
            self.recovery_phase if phase is None else float(phase)
        )
        if self._uses_locked_support_prefix():
            position_delta, rotation_delta, weights = (
                self._locked_prefix_full_field(position, quaternion)
            )
        elif self.recovery_target_mode == "same_phase_attractor":
            position_delta, rotation_delta, weights = (
                self._same_phase_attractor_field(
                    position,
                    quaternion,
                    field_phase,
                )
            )
        elif self.autonomous_chunk_reentry:
            position_delta, rotation_delta, weights = (
                self._autonomous_chunk_full_field(
                    position,
                    quaternion,
                    field_phase,
                )
            )
        else:
            self._select_chunk_for_phase(field_phase)
            position_delta, rotation_delta, weights = self._local_full_field(
                position,
                quaternion,
                field_phase,
            )
        speed_gain = (
            1.0
            if self.recovery_target_mode == "same_phase_attractor"
            else self.field_speed_gain
        )
        action = self._field_action(
            speed_gain * position_delta,
            speed_gain * rotation_delta,
            (
                0.0
                if gripper_command_override is None
                else float(gripper_command_override)
            ),
        )
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        if mutate_state:
            self._record_action_state(position, quaternion, action)
            self.total_steps += 1
            self.steps_on_target += 1
            self.consecutive_recovery_steps += 1
            if gripper_command_override is not None:
                self.gripper_transition_guard_steps += 1
                if self._gripper_target_is_closed(
                    gripper_command_override
                ):
                    self.gripper_transition_close_guard_steps += 1
                else:
                    self.gripper_transition_open_guard_steps += 1
        self.last_phase_frozen = True
        self.last_geometric_on_path = False
        return self._make_step(
            action=action,
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=float(np.max(weights)),
            plan_complete=False,
            replan_required=False,
        )

    def _projection_for_recovery(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        entering: bool,
    ) -> tuple[float, float, float]:
        if self._uses_locked_support_prefix():
            if entering:
                pending_phase = self.pending_recovery_anchor_phase
                self.recovery_locked_phase = float(
                    self.action_phase
                    if pending_phase is None
                    else pending_phase
                )
            phase = float(self.recovery_locked_phase)
            anchor_errors = self._observed_recovery_anchor_errors(
                position,
                quaternion,
            )
            if anchor_errors is None:
                position_error, rotation_error = self._pose_errors_at_phase(
                    position,
                    quaternion,
                    phase,
                )
            else:
                position_error, rotation_error = anchor_errors
            self.recovery_phase = phase
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if entering and self.recovery_target_mode == "support_confidence":
            lower = min(float(self.last_issued_phase), self.action_phase)
            upper = max(float(self.last_issued_phase), self.action_phase)
            phase, position_error, rotation_error = (
                self._ordered_local_projection(
                    position,
                    quaternion,
                    lower_phase=lower,
                    upper_phase=upper,
                )
            )
            self.recovery_phase = float(phase)
            self._select_chunk_for_phase(phase)
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if not entering and self.recovery_target_mode == "support_confidence":
            old_phase = float(self.recovery_phase)
            upper_phase = min(
                self.autonomous_recovery_interval_end,
                old_phase + float(self.phase_stride),
            )
            phase, position_error, rotation_error = (
                self._ordered_local_projection(
                    position,
                    quaternion,
                    lower_phase=old_phase,
                    upper_phase=upper_phase,
                )
            )
            phase = max(old_phase, float(phase))
            if phase > old_phase + 1.0e-9:
                advance = phase - old_phase
                self.recovery_phase_advances += 1
                self.recovery_phase_advance_sum += advance
                self.max_recovery_phase_advance = max(
                    self.max_recovery_phase_advance,
                    advance,
                )
            self.recovery_phase = phase
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if entering and self.recovery_target_mode == "next_action_phase":
            phase = min(
                float(self.target_count),
                max(0.0, float(self.action_phase)),
            )
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                phase,
            )
            self.recovery_phase = phase
            self._select_chunk_for_phase(phase)
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if entering and self.recovery_target_mode == "same_phase_attractor":
            phase = min(
                float(self.target_count),
                max(0.0, float(self.action_phase)),
            )
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                phase,
            )
            self.recovery_phase = float(phase)
            self._select_chunk_for_phase(phase)
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if not entering and self.autonomous_chunk_reentry:
            old_phase = float(self.recovery_phase)
            phase, position_error, rotation_error = (
                self._ordered_local_projection(
                    position,
                    quaternion,
                    lower_phase=old_phase,
                    upper_phase=self.autonomous_recovery_interval_end,
                )
            )
            phase = max(old_phase, float(phase))
            if phase > old_phase + 1.0e-9:
                advance = phase - old_phase
                self.recovery_phase_advances += 1
                self.recovery_phase_advance_sum += advance
                self.max_recovery_phase_advance = max(
                    self.max_recovery_phase_advance,
                    advance,
                )
            self.recovery_phase = phase
            self._set_tube_state(
                phase,
                position_error,
                rotation_error,
            )
            return phase, position_error, rotation_error

        if not entering and self.freeze_recovery_phase:
            phase = float(self.recovery_phase)
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                phase,
            )
            self._set_tube_state(phase, position_error, rotation_error)
            return phase, position_error, rotation_error

        if entering:
            lower = float(self.last_issued_phase)
            requested_upper = min(
                float(self.target_count),
                max(self.action_phase, lower)
                + self.phase_projection_lookahead,
            )
        else:
            lower = float(self.recovery_phase)
            budget_upper = (
                float(self.target_count)
                if self.recovery_phase_advance_budget == 0.0
                else (
                    self.recovery_entry_phase
                    + self.recovery_phase_advance_budget
                )
            )
            requested_upper = min(
                float(self.target_count),
                lower + self.phase_projection_lookahead,
                budget_upper,
            )
        (
            phase,
            position_error,
            rotation_error,
            guard_blocked,
            barrier_hit,
        ) = self._dynamic_recovery_projection(
            position,
            quaternion,
            lower_phase=lower,
            requested_upper_phase=requested_upper,
            require_progress_evidence=not entering,
        )
        self.recovery_phase_guard_blocks += int(guard_blocked)
        self.recovery_gripper_barrier_hits += int(barrier_hit)
        if not entering:
            phase = max(float(self.recovery_phase), float(phase))
            position_error, rotation_error = self._pose_errors_at_phase(
                position,
                quaternion,
                phase,
            )
            advance = phase - float(self.recovery_phase)
            if advance > 1.0e-9:
                self.recovery_phase_advances += 1
                self.recovery_phase_advance_sum += advance
                self.max_recovery_phase_advance = max(
                    self.max_recovery_phase_advance,
                    advance,
                )
        self.recovery_phase = float(phase)
        self._select_chunk_for_phase(self.recovery_phase)
        self._set_tube_state(phase, position_error, rotation_error)
        return phase, position_error, rotation_error

    def _inside_reentry_tube(
        self,
        position_error: float,
        rotation_error: float,
        phase: float,
        *,
        position: np.ndarray | None = None,
    ) -> bool:
        mahalanobis = (
            None
            if position is None
            else self._position_mahalanobis_distance(
                position,
                phase,
                precisions=self.position_reentry_precisions,
            )
        )
        position_inside = (
            float(position_error) <= self._position_reentry_radius(phase)
            if mahalanobis is None
            else mahalanobis <= self.position_reentry_mahalanobis_radius
        )
        return bool(
            position_inside
            and rotation_error <= self.rotation_reentry_radius
        )

    def _recovery_error_score(
        self,
        position_error: float,
        rotation_error: float,
        phase: float,
    ) -> float:
        position_scale = max(
            self._position_reentry_radius(phase),
            1.0e-12,
        )
        rotation_scale = max(self.rotation_reentry_radius, 1.0e-12)
        return float(
            (position_error / position_scale) ** 2
            + (rotation_error / rotation_scale) ** 2
        )

    def _begin_recovery(
        self,
        position_error: float,
        rotation_error: float,
        phase: float,
    ) -> None:
        self.recovering = True
        self.recovery_entry_phase = float(self.recovery_phase)
        if self._uses_locked_support_prefix():
            self.recovery_locked_phase = float(phase)
            self.recovery_phase = float(self.recovery_locked_phase)
            self.recovery_entry_phase = float(self.recovery_locked_phase)
            self._activate_observed_recovery_anchor()
            self._compile_locked_recovery_prefix()
            self.observed_anchor_inside_steps = 0
        if self.autonomous_chunk_reentry:
            self._select_autonomous_recovery_interval(
                self.recovery_entry_phase
            )
        self.recovery_entries += 1
        self.recovery_best_error = self._recovery_error_score(
            position_error,
            rotation_error,
            phase,
        )
        self.recovery_no_progress_steps = 0

    def _update_recovery_progress(
        self,
        position_error: float,
        rotation_error: float,
        phase: float,
    ) -> None:
        score = self._recovery_error_score(
            position_error,
            rotation_error,
            phase,
        )
        if score <= (
            self.recovery_best_error - self.recovery_progress_min_delta
        ):
            self.recovery_best_error = score
            self.recovery_no_progress_steps = 0
            return
        self.recovery_no_progress_steps += 1

    def _replan_step(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        *,
        reason: str = "recovery_horizon",
    ) -> EndpointFieldStep:
        position_error, rotation_error = self._final_errors(
            position,
            quaternion,
        )
        return self._make_step(
            action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
            position_error=position_error,
            rotation_error=rotation_error,
            coverage=0.0,
            plan_complete=False,
            replan_required=True,
            replan_reason=reason,
        )

    def preview_action(self, position, quaternion) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        if self.action_phase >= float(self.target_count):
            self._sync_action_index()
            return super().preview_action(position, quaternion)
        if self.recovering:
            if self._uses_locked_support_prefix():
                phase = float(self.recovery_locked_phase)
                position_error, rotation_error = self._pose_errors_at_phase(
                    position,
                    quaternion,
                    phase,
                )
                if (
                    self._inside_sampled_locked_terminal_support(
                        position,
                        quaternion,
                        record=False,
                    )
                    if self.recovery_trigger_mode
                    == "support_and_state_innovation"
                    else self._inside_locked_prefix_terminal_support(
                        position,
                        quaternion,
                        record=False,
                    )
                ):
                    return self._issue_native(
                        position,
                        quaternion,
                        None,
                        phase=self.action_phase,
                        mutate_gripper=False,
                    )
                return self._issue_field(
                    position,
                    quaternion,
                    mutate_state=False,
                    phase=phase,
                )
            if self.autonomous_chunk_reentry:
                phase, position_error, rotation_error = (
                    self._ordered_local_projection(
                        position,
                        quaternion,
                        lower_phase=self.recovery_phase,
                        upper_phase=(
                            self.autonomous_recovery_interval_end
                        ),
                    )
                )
            elif self.freeze_recovery_phase:
                phase = float(self.recovery_phase)
                position_error, rotation_error = self._pose_errors_at_phase(
                    position,
                    quaternion,
                    phase,
                )
            else:
                budget_upper = (
                    float(self.target_count)
                    if self.recovery_phase_advance_budget == 0.0
                    else (
                        self.recovery_entry_phase
                        + self.recovery_phase_advance_budget
                    )
                )
                (
                    phase,
                    position_error,
                    rotation_error,
                    _,
                    _,
                ) = self._dynamic_recovery_projection(
                    position,
                    quaternion,
                    lower_phase=self.recovery_phase,
                    requested_upper_phase=min(
                        self.target_count,
                        self.recovery_phase
                        + self.phase_projection_lookahead,
                        budget_upper,
                    ),
                    require_progress_evidence=True,
                )
            if self._inside_reentry_tube(
                position_error,
                rotation_error,
                phase,
                position=position,
            ):
                return self._issue_native(
                    position,
                    quaternion,
                    None,
                    phase=phase,
                    mutate_gripper=False,
                )
            return self._issue_field(
                position,
                quaternion,
                mutate_state=False,
                phase=phase,
            )

        position_error, rotation_error = self._pose_errors_at_phase(
            position,
            quaternion,
            self.action_phase,
        )
        if self._inside_outer_tube(
            position_error,
            rotation_error,
            self.action_phase,
            position=position,
        ):
            return self._issue_native(
                position,
                quaternion,
                None,
                phase=self.action_phase,
                mutate_gripper=False,
            )
        if self.recovery_target_mode == "next_action_phase":
            phase = min(
                float(self.target_count),
                max(0.0, float(self.action_phase)),
            )
            projected_position_error, projected_rotation_error = (
                self._pose_errors_at_phase(position, quaternion, phase)
            )
        elif self.recovery_target_mode == "same_phase_attractor":
            phase = min(
                float(self.target_count),
                max(0.0, float(self.action_phase)),
            )
            projected_position_error, projected_rotation_error = (
                self._pose_errors_at_phase(
                    position,
                    quaternion,
                    phase,
                )
            )
        else:
            (
                phase,
                projected_position_error,
                projected_rotation_error,
                _,
                _,
            ) = self._dynamic_recovery_projection(
                position,
                quaternion,
                lower_phase=self.last_issued_phase,
                requested_upper_phase=min(
                    self.target_count,
                    self.action_phase + self.phase_projection_lookahead,
                ),
                require_progress_evidence=False,
            )
        if self._inside_outer_tube(
            projected_position_error,
            projected_rotation_error,
            phase,
            position=position,
        ):
            return self._issue_native(
                position,
                quaternion,
                None,
                phase=phase,
                mutate_gripper=False,
            )
        return self._issue_field(
            position,
            quaternion,
            mutate_state=False,
            phase=phase,
        )

    def step(
        self,
        position,
        quaternion,
        gripper_qpos=None,
    ) -> EndpointFieldStep:
        position = np.asarray(position, dtype=np.float64).reshape(3)
        quaternion = normalize_quaternion(quaternion)
        self._sync_action_index()

        if self.action_phase >= float(self.target_count):
            self.action_index = self.target_count
            return super().step(position, quaternion, gripper_qpos)

        gripper_completion = self._observe_gripper_closure(gripper_qpos)
        if (
            gripper_completion is not None
            and self.terminal_gripper_command is None
        ):
            position_error, rotation_error = self._final_errors(
                position,
                quaternion,
            )
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=1.0,
                plan_complete=True,
                replan_required=False,
                replan_reason=gripper_completion,
            )
        if self.total_steps >= int(self.config.max_plan_steps):
            position_error, rotation_error = self._final_errors(
                position,
                quaternion,
            )
            return self._make_step(
                action=np.zeros((FIELD_ACTION_DIM,), dtype=np.float32),
                position_error=position_error,
                rotation_error=rotation_error,
                coverage=0.0,
                plan_complete=False,
                replan_required=True,
                replan_reason="hard_timeout",
            )

        if self.recovering:
            gripper_transition_pending = bool(
                self._uses_support_confidence()
                and self._issued_gripper_transition_pending(gripper_qpos)
            )
            phase, position_error, rotation_error = (
                self._projection_for_recovery(
                    position,
                    quaternion,
                    entering=False,
                )
            )
            if self._uses_locked_support_prefix():
                if (
                    self.recovery_trigger_mode
                    == "support_and_state_innovation"
                ):
                    inside_reentry = (
                        self._inside_sampled_locked_terminal_support(
                            position,
                            quaternion,
                            record=True,
                        )
                    )
                else:
                    inside_reentry = (
                        self._inside_locked_prefix_terminal_support(
                            position,
                            quaternion,
                            record=True,
                        )
                    )
            elif self.recovery_target_mode == "support_confidence":
                inside_reentry = self._inside_gaussian_inner_support(
                    position,
                    quaternion,
                    record=True,
                )
            else:
                inside_reentry = self._inside_reentry_tube(
                    position_error,
                    rotation_error,
                    phase,
                    position=position,
                )
            if self.active_recovery_anchor_position is not None:
                if inside_reentry:
                    self.observed_anchor_inside_steps += 1
                else:
                    self.observed_anchor_inside_steps = 0
                if (
                    inside_reentry
                    and self.observed_anchor_inside_steps
                    < self.observed_anchor_stable_steps
                ):
                    self.observed_anchor_stability_wait_steps += 1
                    inside_reentry = False
            if inside_reentry:
                old_phase = float(self.action_phase)
                self.recovering = False
                self.recovery_reentries += 1
                self.completed_recovery_steps += (
                    self.consecutive_recovery_steps
                )
                self.max_completed_recovery_steps = max(
                    self.max_completed_recovery_steps,
                    self.consecutive_recovery_steps,
                )
                self.recovery_no_progress_steps = 0
                self.observed_anchor_inside_steps = 0
                if not self._uses_support_confidence():
                    self._record_reprojection(old_phase, phase)
                preserve_cached_gripper_transition = bool(
                    gripper_transition_pending
                    or self.external_gripper_transition_since_anchor
                )
                if (
                    self.replan_on_recovery_reentry
                    and not preserve_cached_gripper_transition
                ):
                    self.external_gripper_transition_since_anchor = False
                    self.active_recovery_anchor_position = None
                    self.active_recovery_anchor_quaternion = None
                    self.active_recovery_anchor_offset = None
                    return self._replan_step(
                        position,
                        quaternion,
                        reason="recovery_reentry",
                    )
                if (
                    self.replan_on_recovery_reentry
                    and preserve_cached_gripper_transition
                ):
                    self.recovery_reentry_replan_gripper_deferrals += 1
                self.external_gripper_transition_since_anchor = False
                step = self._issue_native(
                    position,
                    quaternion,
                    gripper_qpos,
                    phase=(
                        self.action_phase
                        if self._uses_support_confidence()
                        else phase
                    ),
                    mutate_gripper=True,
                )
                self.active_recovery_anchor_position = None
                self.active_recovery_anchor_quaternion = None
                self.active_recovery_anchor_offset = None
                return step
            self._update_recovery_progress(
                position_error,
                rotation_error,
                phase,
            )
            if (
                self.recovery_progress_patience > 0
                and self.recovery_no_progress_steps
                >= self.recovery_progress_patience
            ):
                self.recovery_no_progress_replans += 1
                return self._replan_step(
                    position,
                    quaternion,
                    reason="recovery_no_progress",
                )
            if (
                self.max_recovery_steps > 0
                and self.consecutive_recovery_steps
                >= self.max_recovery_steps
            ):
                return self._replan_step(position, quaternion)
            return self._issue_field(
                position,
                quaternion,
                mutate_state=True,
                gripper_command_override=(
                    self.last_gripper_command
                    if gripper_transition_pending
                    else None
                ),
            )

        if self._uses_support_confidence():
            confidence_available = (
                self._pose_sample_confidence_available()
                if self.recovery_trigger_mode
                == "support_and_state_innovation"
                else self._support_confidence_available()
            )
            if not confidence_available:
                self.support_confidence_direct_fallback_steps += 1
                if (
                    self.recovery_trigger_mode
                    == "support_and_state_innovation"
                ):
                    if self.require_state_innovation and self._state_innovation_detected(
                        position, quaternion,
                    ):
                        self.state_innovation_triggers += 1
                        if (
                            self.disturbance_response_mode
                            == "immediate_replan"
                        ):
                            return self._replan_step(
                                position,
                                quaternion,
                                reason="disturbance_immediate_replan",
                            )
                        self._register_last_safe_recovery_anchor()
                        if (
                            self.recovery_anchor_mode
                            == "observed_perturbation_start"
                            and self._observed_recovery_anchor() is not None
                        ):
                            phase, position_error, rotation_error = (
                                self._projection_for_recovery(
                                    position,
                                    quaternion,
                                    entering=True,
                                )
                            )
                            self._begin_recovery(
                                position_error,
                                rotation_error,
                                phase,
                            )
                            return self._issue_field(
                                position,
                                quaternion,
                                mutate_state=True,
                                gripper_command_override=(
                                    self.last_gripper_command
                                    if self._issued_gripper_transition_pending(
                                        gripper_qpos
                                    )
                                    else None
                                ),
                            )
                        return self._replan_step(
                            position,
                            quaternion,
                            reason=(
                                "disturbance_without_confident_pose_support"
                            ),
                        )
                    self.state_innovation_native_bypasses += 1
                return self._issue_native(
                    position,
                    quaternion,
                    gripper_qpos,
                    phase=self.action_phase,
                    mutate_gripper=True,
                )
            if self.recovery_trigger_mode == "support_and_state_innovation":
                inside_outer_support = (
                    self._inside_sampled_current_native_support(
                        position,
                        quaternion,
                        record=True,
                    )
                )
            else:
                inside_outer_support = self._inside_gaussian_outer_support(
                    position,
                    quaternion,
                    record=True,
                )
            if inside_outer_support:
                self._record_last_safe_recovery_state(
                    position,
                    quaternion,
                )
                return self._issue_native(
                    position,
                    quaternion,
                    gripper_qpos,
                    phase=self.action_phase,
                    mutate_gripper=True,
                )

            if (
                self.recovery_trigger_mode
                == "support_and_state_innovation"
            ):
                innovation = self._state_innovation_detected(
                    position,
                    quaternion,
                )
                if self.require_state_innovation and not innovation:
                    self.state_innovation_native_bypasses += 1
                    return self._issue_native(
                        position,
                        quaternion,
                        gripper_qpos,
                        phase=self.action_phase,
                        mutate_gripper=True,
                    )
                if innovation:
                    self.state_innovation_triggers += 1
                if (
                    self.disturbance_response_mode == "immediate_replan"
                ):
                    return self._replan_step(
                        position,
                        quaternion,
                        reason="disturbance_immediate_replan",
                    )
                self._register_last_safe_recovery_anchor()
            else:
                if self._native_action_improves_support(
                    position,
                    quaternion,
                ):
                    self.support_confidence_inward_native_steps += 1
                    return self._issue_native(
                        position,
                        quaternion,
                        gripper_qpos,
                        phase=self.action_phase,
                        mutate_gripper=True,
                    )

            if (
                self.recovery_trigger_mode == "support_only"
                and not self._uses_locked_support_prefix()
                and self._upcoming_gripper_transition()
            ):
                self.gripper_transition_priority_issues += 1
                return self._issue_native(
                    position,
                    quaternion,
                    gripper_qpos,
                    phase=self.action_phase,
                    mutate_gripper=True,
                )

            phase, position_error, rotation_error = (
                self._projection_for_recovery(
                    position,
                    quaternion,
                    entering=True,
                )
            )
            self._begin_recovery(
                position_error,
                rotation_error,
                phase,
            )
            return self._issue_field(
                position,
                quaternion,
                mutate_state=True,
                gripper_command_override=(
                    self.last_gripper_command
                    if self._issued_gripper_transition_pending(
                        gripper_qpos
                    )
                    else None
                ),
            )

        position_error, rotation_error = self._pose_errors_at_phase(
            position,
            quaternion,
            self.action_phase,
        )
        self._set_tube_state(
            self.action_phase,
            position_error,
            rotation_error,
        )
        if self._inside_outer_tube(
            position_error,
            rotation_error,
            self.action_phase,
            position=position,
        ):
            return self._issue_native(
                position,
                quaternion,
                gripper_qpos,
                phase=self.action_phase,
                mutate_gripper=True,
            )

        phase, projected_position_error, projected_rotation_error = (
            self._projection_for_recovery(
                position,
                quaternion,
                entering=True,
            )
        )
        if self._inside_outer_tube(
            projected_position_error,
            projected_rotation_error,
            phase,
            position=position,
        ):
            old_phase = float(self.action_phase)
            self._record_reprojection(old_phase, phase)
            return self._issue_native(
                position,
                quaternion,
                gripper_qpos,
                phase=phase,
                mutate_gripper=True,
            )

        self._begin_recovery(
            projected_position_error,
            projected_rotation_error,
            phase,
        )
        if (
            self.max_recovery_steps > 0
            and self.consecutive_recovery_steps >= self.max_recovery_steps
        ):
            return self._replan_step(position, quaternion)
        return self._issue_field(
            position,
            quaternion,
            mutate_state=True,
        )


class LiberoPairedBSplineHybridDSAdapter(
    LiberoPairedBSplineFieldAdapter
):
    """Compile paired B-splines into the projected-phase hybrid executor."""

    def __init__(
        self,
        base_policy,
        *,
        phase_position_reentry_radius: float = 0.02,
        phase_rotation_reentry_radius: float = 0.1,
        phase_projection_lookahead: float | None = None,
        recovery_phase_advance_budget: float | None = None,
        phase_max_recovery_steps: int = 16,
        recovery_field_source: str = "action",
        recovery_target_mode: str = "projected",
        recovery_anchor_mode: str = "predicted_pose",
        observed_anchor_reentry_radius: float = 0.01,
        observed_anchor_stable_steps: int = 2,
        freeze_recovery_phase: bool = False,
        autonomous_chunk_reentry: bool = False,
        replan_on_recovery_reentry: bool = False,
        rotation_clf_enabled: bool = False,
        rotation_clf_tube_radius: float = 0.05,
        rotation_clf_contraction_rate: float | None = None,
        recovery_progress_patience: int = 0,
        recovery_progress_min_delta: float = 1.0e-3,
        field_speed_gain: float = 1.0,
        support_confidence_outer_score: float = 0.05,
        support_confidence_inner_score: float = 0.1,
        recovery_trigger_mode: str = "support_only",
        require_state_innovation: bool = True,
        disturbance_position_innovation_threshold: float = 0.04,
        disturbance_rotation_innovation_threshold: float = 0.35,
        disturbance_expected_motion_multiplier: float = 1.5,
        disturbance_response_mode: str = "recover_then_replan",
        nominal_fusion_mode: str = "native_action",
        transverse_fusion_inner_ratio: float = 0.5,
        transverse_fusion_outer_ratio: float = 1.0,
        transverse_fusion_max_weight: float = 1.0,
        adaptive_pose_support_enabled: bool = False,
        adaptive_pose_tube_enabled: bool = False,
        pose_support_confidence: float = 0.95,
        pose_reentry_fraction: float = 0.5,
        pose_dense_core_confidence: float | None = None,
        **kwargs,
    ) -> None:
        self.phase_position_reentry_radius = float(
            phase_position_reentry_radius
        )
        self.phase_rotation_reentry_radius = float(
            phase_rotation_reentry_radius
        )
        self.phase_projection_lookahead = phase_projection_lookahead
        self.recovery_phase_advance_budget = (
            None
            if recovery_phase_advance_budget is None
            else float(recovery_phase_advance_budget)
        )
        self.phase_max_recovery_steps = int(phase_max_recovery_steps)
        self.recovery_field_source = str(recovery_field_source)
        self.recovery_target_mode = str(recovery_target_mode)
        self.recovery_anchor_mode = str(recovery_anchor_mode)
        self.observed_anchor_reentry_radius = float(
            observed_anchor_reentry_radius
        )
        self.observed_anchor_stable_steps = int(
            observed_anchor_stable_steps
        )
        self.freeze_recovery_phase = bool(freeze_recovery_phase)
        self.autonomous_chunk_reentry = bool(autonomous_chunk_reentry)
        self.replan_on_recovery_reentry = bool(
            replan_on_recovery_reentry
        )
        self.rotation_clf_enabled = bool(rotation_clf_enabled)
        self.rotation_clf_tube_radius = float(rotation_clf_tube_radius)
        self.rotation_clf_contraction_rate = (
            None
            if rotation_clf_contraction_rate is None
            else float(rotation_clf_contraction_rate)
        )
        self.recovery_progress_patience = int(recovery_progress_patience)
        self.recovery_progress_min_delta = float(
            recovery_progress_min_delta
        )
        self.field_speed_gain = float(field_speed_gain)
        self.support_confidence_outer_score = float(
            support_confidence_outer_score
        )
        self.support_confidence_inner_score = float(
            support_confidence_inner_score
        )
        self.recovery_trigger_mode = str(recovery_trigger_mode)
        self.require_state_innovation = bool(require_state_innovation)
        self.disturbance_position_innovation_threshold = float(
            disturbance_position_innovation_threshold
        )
        self.disturbance_rotation_innovation_threshold = float(
            disturbance_rotation_innovation_threshold
        )
        self.disturbance_expected_motion_multiplier = float(
            disturbance_expected_motion_multiplier
        )
        self.disturbance_response_mode = str(disturbance_response_mode)
        self.nominal_fusion_mode = str(nominal_fusion_mode)
        self.transverse_fusion_inner_ratio = float(
            transverse_fusion_inner_ratio
        )
        self.transverse_fusion_outer_ratio = float(
            transverse_fusion_outer_ratio
        )
        self.transverse_fusion_max_weight = float(
            transverse_fusion_max_weight
        )
        self.adaptive_pose_support_enabled = bool(
            adaptive_pose_support_enabled
        )
        self.adaptive_pose_tube_enabled = bool(
            adaptive_pose_tube_enabled
        )
        self.pose_support_confidence = float(pose_support_confidence)
        self.pose_reentry_fraction = float(pose_reentry_fraction)
        self.pose_dense_core_confidence = (
            None
            if pose_dense_core_confidence is None
            else float(pose_dense_core_confidence)
        )
        if not 0.0 < self.pose_support_confidence < 1.0:
            raise ValueError("pose_support_confidence must be in (0,1)")
        if not 0.0 < self.pose_reentry_fraction < 1.0:
            raise ValueError("pose_reentry_fraction must be in (0,1)")
        if not (
            0.0
            < self.support_confidence_outer_score
            < self.support_confidence_inner_score
            <= 1.0
        ):
            raise ValueError(
                "support confidence scores must satisfy "
                "0 < outer < inner <= 1"
            )
        if self.recovery_trigger_mode not in {
            "support_only",
            "support_and_state_innovation",
        }:
            raise ValueError(
                "recovery_trigger_mode must be 'support_only' or "
                "'support_and_state_innovation'"
            )
        for name, value in (
            (
                "disturbance_position_innovation_threshold",
                self.disturbance_position_innovation_threshold,
            ),
            (
                "disturbance_rotation_innovation_threshold",
                self.disturbance_rotation_innovation_threshold,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.disturbance_expected_motion_multiplier)
            or self.disturbance_expected_motion_multiplier < 1.0
        ):
            raise ValueError(
                "disturbance_expected_motion_multiplier must be finite "
                "and at least 1"
            )
        if self.disturbance_response_mode not in {
            "recover_then_replan",
            "immediate_replan",
        }:
            raise ValueError(
                "disturbance_response_mode must be "
                "'recover_then_replan' or 'immediate_replan'"
            )
        if (
            self.disturbance_response_mode == "immediate_replan"
            and self.recovery_trigger_mode
            != "support_and_state_innovation"
        ):
            raise ValueError(
                "immediate_replan requires "
                "support_and_state_innovation recovery triggering"
            )
        if (
            self.pose_dense_core_confidence is not None
            and not 0.0 < self.pose_dense_core_confidence < 1.0
        ):
            raise ValueError("pose_dense_core_confidence must be in (0,1)")
        kwargs.pop("phase_conditioned_execution", None)
        if self.recovery_trigger_mode == "support_and_state_innovation":
            kwargs["truncate_at_gripper_only_transition"] = False
        super().__init__(
            base_policy,
            phase_conditioned_execution=True,
            **kwargs,
        )
        if self.recovery_phase_advance_budget is None:
            self.recovery_phase_advance_budget = float(self.phase_stride)
        if self.rotation_clf_contraction_rate is None:
            self.rotation_clf_contraction_rate = float(
                self.field_config.clf_position_contraction_rate
            )
        self.representation = "paired_bspline_hybrid_ds"
        self._archived_phase_reprojections = 0
        self._archived_phase_reprojection_sum = 0.0
        self._archived_max_phase_reprojection = 0.0
        self._archived_recovery_entries = 0
        self._archived_recovery_reentries = 0
        self._archived_recovery_reentry_replan_gripper_deferrals = 0
        self._archived_recovery_no_progress_replans = 0
        self._archived_completed_recovery_steps = 0
        self._archived_max_completed_recovery_steps = 0
        self._archived_unresolved_recoveries = 0
        self._archived_recovery_phase_advances = 0
        self._archived_recovery_phase_advance_sum = 0.0
        self._archived_max_recovery_phase_advance = 0.0
        self._archived_recovery_phase_guard_blocks = 0
        self._archived_recovery_gripper_barrier_hits = 0
        self._archived_phase_reprojection_forward_count = 0
        self._archived_phase_reprojection_forward_sum = 0.0
        self._archived_max_phase_reprojection_forward = 0.0
        self._archived_phase_reprojection_backward_count = 0
        self._archived_phase_reprojection_backward_sum = 0.0
        self._archived_max_phase_reprojection_backward = 0.0
        self._archived_autonomous_recovery_chunk_compiles = 0
        self._archived_recovery_prefix_compiles = 0
        self._archived_recovery_prefix_supports_sum = 0
        self._archived_automatic_recovery_anchor_registrations = 0
        self._archived_observed_anchor_stability_wait_steps = 0
        self._archived_support_confidence_checks = 0
        self._archived_support_confidence_score_sum = 0.0
        self._archived_minimum_support_confidence_score = 1.0
        self._archived_support_confidence_direct_fallback_steps = 0
        self._archived_support_confidence_inward_native_steps = 0
        self._archived_state_innovation_checks = 0
        self._archived_state_innovation_triggers = 0
        self._archived_state_innovation_native_bypasses = 0
        self._archived_max_position_innovation = 0.0
        self._archived_max_rotation_innovation = 0.0
        self._archived_position_innovation_samples: list[float] = []
        self._archived_rotation_innovation_samples: list[float] = []
        self._archived_gripper_transition_priority_issues = 0
        self._archived_gripper_transition_guard_steps = 0
        self._archived_gripper_transition_open_guard_steps = 0
        self._archived_gripper_transition_close_guard_steps = 0
        self._archived_transverse_fusion_checks = 0
        self._archived_transverse_fusion_active_steps = 0
        self._archived_transverse_fusion_support_fallback_steps = 0
        self._archived_transverse_fusion_weight_sum = 0.0
        self._archived_transverse_fusion_max_weight_applied = 0.0
        self._archived_transverse_fusion_correction_norm_sum = 0.0
        self._archived_transverse_fusion_max_correction_norm = 0.0
        self._controller_trace_chunks: list[list[list[dict[str, object]]]] = []
        self._controller_trace_active: list[list[dict[str, object]]] | None = None
        self._controller_trace_queried_indices: set[int] = set()
        self._controller_diagnostics_enabled = False
        self._adaptive_position_tube_values: list[float] = []
        self._adaptive_position_reentry_values: list[float] = []
        self._pose_sample_position_outer_values: list[float] = []
        self._pose_sample_position_reentry_values: list[float] = []

    def enable_controller_diagnostics(self, enabled: bool = True) -> None:
        self._controller_diagnostics_enabled = bool(enabled)
        if not self._controller_diagnostics_enabled:
            self._controller_trace_chunks = []
            self._controller_trace_active = None

    def _archive_controller_trace(self) -> None:
        if self._controller_trace_active is None:
            return
        if any(self._controller_trace_active):
            self._controller_trace_chunks.append(
                self._controller_trace_active
            )
        self._controller_trace_active = None

    def consume_controller_diagnostic_traces(
        self,
    ) -> list[list[list[dict[str, object]]]]:
        """Return completed rollout traces and reset the trace buffer."""

        self._archive_controller_trace()
        traces = self._controller_trace_chunks
        self._controller_trace_chunks = []
        return traces

    def append_controller_diagnostic_record(
        self,
        record: dict[str, object],
        *,
        env_index: int,
        batch_size: int,
    ) -> None:
        """Append a wrapper-owned environment step to the aligned trace."""

        if not self._controller_diagnostics_enabled:
            return
        if (
            self._controller_trace_active is None
            or len(self._controller_trace_active) != int(batch_size)
        ):
            self._archive_controller_trace()
            self._controller_trace_active = [
                [] for _ in range(int(batch_size))
            ]
        if not 0 <= int(env_index) < int(batch_size):
            raise IndexError("controller diagnostic env_index is out of range")
        self._controller_trace_active[int(env_index)].append(dict(record))

    def _archive_executor(self, executor) -> None:
        if not isinstance(executor, ProjectedPhaseHybridExecutor):
            return
        self._archived_phase_reprojections += executor.phase_reprojections
        self._archived_phase_reprojection_sum += (
            executor.phase_reprojection_sum
        )
        self._archived_max_phase_reprojection = max(
            self._archived_max_phase_reprojection,
            executor.max_phase_reprojection,
        )
        self._archived_recovery_entries += executor.recovery_entries
        self._archived_recovery_reentries += executor.recovery_reentries
        self._archived_recovery_reentry_replan_gripper_deferrals += (
            executor.recovery_reentry_replan_gripper_deferrals
        )
        self._archived_recovery_no_progress_replans += (
            executor.recovery_no_progress_replans
        )
        self._archived_completed_recovery_steps += (
            executor.completed_recovery_steps
        )
        self._archived_max_completed_recovery_steps = max(
            self._archived_max_completed_recovery_steps,
            executor.max_completed_recovery_steps,
        )
        self._archived_unresolved_recoveries += int(executor.recovering)
        self._archived_recovery_phase_advances += (
            executor.recovery_phase_advances
        )
        self._archived_recovery_phase_advance_sum += (
            executor.recovery_phase_advance_sum
        )
        self._archived_max_recovery_phase_advance = max(
            self._archived_max_recovery_phase_advance,
            executor.max_recovery_phase_advance,
        )
        self._archived_recovery_phase_guard_blocks += (
            executor.recovery_phase_guard_blocks
        )
        self._archived_recovery_gripper_barrier_hits += (
            executor.recovery_gripper_barrier_hits
        )
        self._archived_phase_reprojection_forward_count += (
            executor.phase_reprojection_forward_count
        )
        self._archived_phase_reprojection_forward_sum += (
            executor.phase_reprojection_forward_sum
        )
        self._archived_max_phase_reprojection_forward = max(
            self._archived_max_phase_reprojection_forward,
            executor.max_phase_reprojection_forward,
        )
        self._archived_phase_reprojection_backward_count += (
            executor.phase_reprojection_backward_count
        )
        self._archived_phase_reprojection_backward_sum += (
            executor.phase_reprojection_backward_sum
        )
        self._archived_max_phase_reprojection_backward = max(
            self._archived_max_phase_reprojection_backward,
            executor.max_phase_reprojection_backward,
        )
        self._archived_autonomous_recovery_chunk_compiles += (
            executor.autonomous_recovery_chunk_compiles
        )
        self._archived_recovery_prefix_compiles += (
            executor.recovery_prefix_compiles
        )
        self._archived_recovery_prefix_supports_sum += (
            executor.recovery_prefix_supports_sum
        )
        self._archived_automatic_recovery_anchor_registrations += (
            executor.automatic_recovery_anchor_registrations
        )
        self._archived_observed_anchor_stability_wait_steps += (
            executor.observed_anchor_stability_wait_steps
        )
        self._archived_support_confidence_checks += (
            executor.support_confidence_checks
        )
        self._archived_support_confidence_score_sum += (
            executor.support_confidence_score_sum
        )
        self._archived_minimum_support_confidence_score = min(
            self._archived_minimum_support_confidence_score,
            executor.minimum_support_confidence_score,
        )
        self._archived_support_confidence_direct_fallback_steps += (
            executor.support_confidence_direct_fallback_steps
        )
        self._archived_support_confidence_inward_native_steps += (
            executor.support_confidence_inward_native_steps
        )
        self._archived_state_innovation_checks += (
            executor.state_innovation_checks
        )
        self._archived_state_innovation_triggers += (
            executor.state_innovation_triggers
        )
        self._archived_state_innovation_native_bypasses += (
            executor.state_innovation_native_bypasses
        )
        self._archived_max_position_innovation = max(
            self._archived_max_position_innovation,
            executor.max_position_innovation,
        )
        self._archived_max_rotation_innovation = max(
            self._archived_max_rotation_innovation,
            executor.max_rotation_innovation,
        )
        self._archived_position_innovation_samples.extend(
            executor.position_innovation_samples
        )
        self._archived_rotation_innovation_samples.extend(
            executor.rotation_innovation_samples
        )
        self._archived_gripper_transition_priority_issues += (
            executor.gripper_transition_priority_issues
        )
        self._archived_gripper_transition_guard_steps += (
            executor.gripper_transition_guard_steps
        )
        self._archived_gripper_transition_open_guard_steps += (
            executor.gripper_transition_open_guard_steps
        )
        self._archived_gripper_transition_close_guard_steps += (
            executor.gripper_transition_close_guard_steps
        )
        self._archived_transverse_fusion_checks += (
            executor.transverse_fusion_checks
        )
        self._archived_transverse_fusion_active_steps += (
            executor.transverse_fusion_active_steps
        )
        self._archived_transverse_fusion_support_fallback_steps += (
            executor.transverse_fusion_support_fallback_steps
        )
        self._archived_transverse_fusion_weight_sum += (
            executor.transverse_fusion_weight_sum
        )
        self._archived_transverse_fusion_max_weight_applied = max(
            self._archived_transverse_fusion_max_weight_applied,
            executor.transverse_fusion_max_weight_applied,
        )
        self._archived_transverse_fusion_correction_norm_sum += (
            executor.transverse_fusion_correction_norm_sum
        )
        self._archived_transverse_fusion_max_correction_norm = max(
            self._archived_transverse_fusion_max_correction_norm,
            executor.transverse_fusion_max_correction_norm,
        )

    def reset(self):
        self._archive_controller_trace()
        for executor in self._executors:
            self._archive_executor(executor)
        return super().reset()

    def _query_and_compile(
        self,
        obs_dict,
        positions: np.ndarray,
        quaternions: np.ndarray,
        env_indices: list[int],
        **kwargs,
    ) -> None:
        if self._controller_diagnostics_enabled:
            self._controller_trace_queried_indices.update(env_indices)
        for env_index in env_indices:
            if env_index < len(self._executors):
                self._archive_executor(self._executors[env_index])
        super()._query_and_compile(
            obs_dict,
            positions,
            quaternions,
            env_indices,
            **kwargs,
        )
        consume_support = getattr(
            self.base_policy,
            "consume_pose_ensemble_support",
            None,
        )
        if not callable(consume_support):
            return
        supports = consume_support()
        if len(supports) != len(env_indices):
            raise RuntimeError(
                "Pose ensemble returned "
                f"{len(supports)} supports for {len(env_indices)} envs"
            )
        for row, env_index in enumerate(env_indices):
            executor = self._executors[env_index]
            if not isinstance(executor, ProjectedPhaseHybridExecutor):
                raise RuntimeError("Hybrid adapter compiled the wrong executor")
            executor.configure_pose_ensemble_support(
                supports[row],
                adaptive_field_support=(
                    self.adaptive_pose_support_enabled
                ),
                adaptive_tube=self.adaptive_pose_tube_enabled,
                support_confidence=self.pose_support_confidence,
                reentry_fraction=self.pose_reentry_fraction,
                dense_core_confidence=self.pose_dense_core_confidence,
            )
            if executor.position_tube_radii is not None:
                self._adaptive_position_tube_values.extend(
                    executor.position_tube_radii.tolist()
                )
            if executor.position_reentry_radii is not None:
                self._adaptive_position_reentry_values.extend(
                    executor.position_reentry_radii.tolist()
                )
            if executor.pose_sample_position_outer_radii is not None:
                self._pose_sample_position_outer_values.extend(
                    executor.pose_sample_position_outer_radii.tolist()
                )
            if executor.pose_sample_position_reentry_radii is not None:
                self._pose_sample_position_reentry_values.extend(
                    executor.pose_sample_position_reentry_radii.tolist()
                )

    def predict_action(self, obs_dict, **kwargs):
        if not self._controller_diagnostics_enabled:
            return super().predict_action(obs_dict, **kwargs)
        self._controller_trace_queried_indices = set()
        result = super().predict_action(obs_dict, **kwargs)
        actions = result["action"]
        action_array = (
            actions.detach().cpu().numpy()
            if hasattr(actions, "detach")
            else np.asarray(actions)
        )
        gripper_qpos = self._latest_numpy(
            obs_dict,
            "robot0_gripper_qpos",
            2,
        )
        batch_size = len(self._executors)
        if (
            self._controller_trace_active is None
            or len(self._controller_trace_active) != batch_size
        ):
            self._archive_controller_trace()
            self._controller_trace_active = [
                [] for _ in range(batch_size)
            ]

        assert self._controller_trace_active is not None
        for env_index, executor in enumerate(self._executors):
            if not isinstance(executor, ProjectedPhaseHybridExecutor):
                continue
            field_active = bool(executor.last_phase_frozen)
            phase = (
                executor.recovery_phase
                if field_active
                else executor.last_issued_phase
            )
            self._controller_trace_active[env_index].append(
                {
                    "mode": "field" if field_active else "action",
                    "phase": float(phase),
                    "phase_count": int(executor.target_count),
                    "position_error": float(
                        executor.last_tube_position_error
                    ),
                    "rotation_error": float(
                        executor.last_tube_rotation_error
                    ),
                    "position_tube_radius": float(
                        executor._support_entry_position_radius(phase)
                    ),
                    "position_reentry_radius": float(
                        executor._support_reentry_position_radius(phase)
                    ),
                    "support_confidence_score": float(
                        executor.last_support_confidence_score
                    ),
                    "support_confidence_outer_score": float(
                        executor.support_confidence_outer_score
                    ),
                    "support_confidence_inner_score": float(
                        executor.support_confidence_inner_score
                    ),
                    "pose_sample_support_ratio": float(
                        executor.last_pose_sample_support_ratio
                    ),
                    "transverse_fusion_ratio": float(
                        executor.last_transverse_fusion_ratio
                    ),
                    "transverse_fusion_weight": float(
                        executor.last_transverse_fusion_weight
                    ),
                    "transverse_fusion_correction_norm": float(
                        executor.last_transverse_fusion_correction_norm
                    ),
                    "policy_query": bool(
                        env_index
                        in self._controller_trace_queried_indices
                    ),
                    "recovery_steps": int(
                        executor.consecutive_recovery_steps
                    ),
                    "gripper_command": float(
                        action_array[env_index, 0, 6]
                    ),
                    "gripper_target_command": float(
                        executor.last_gripper_command
                    ),
                    "gripper_aperture": float(
                        np.max(np.abs(gripper_qpos[env_index]))
                    ),
                }
            )
        return result

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_paired_bspline_hybrid_ds"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        active = [
            executor
            for executor in self._executors
            if isinstance(executor, ProjectedPhaseHybridExecutor)
        ]
        reprojections = self._archived_phase_reprojections + sum(
            executor.phase_reprojections for executor in active
        )
        shift_sum = self._archived_phase_reprojection_sum + sum(
            executor.phase_reprojection_sum for executor in active
        )
        max_shift = max(
            self._archived_max_phase_reprojection,
            max(
                (
                    executor.max_phase_reprojection
                    for executor in active
                ),
                default=0.0,
            ),
        )
        recovery_entries = self._archived_recovery_entries + sum(
            executor.recovery_entries for executor in active
        )
        recovery_reentries = self._archived_recovery_reentries + sum(
            executor.recovery_reentries for executor in active
        )
        recovery_reentry_replan_gripper_deferrals = (
            self._archived_recovery_reentry_replan_gripper_deferrals
            + sum(
                executor.recovery_reentry_replan_gripper_deferrals
                for executor in active
            )
        )
        no_progress_replans = (
            self._archived_recovery_no_progress_replans
            + sum(
                executor.recovery_no_progress_replans
                for executor in active
            )
        )
        completed_recovery_steps = (
            self._archived_completed_recovery_steps
            + sum(
                executor.completed_recovery_steps for executor in active
            )
        )
        max_completed_recovery_steps = max(
            self._archived_max_completed_recovery_steps,
            max(
                (
                    executor.max_completed_recovery_steps
                    for executor in active
                ),
                default=0,
            ),
        )
        unresolved_recoveries = self._archived_unresolved_recoveries + sum(
            int(executor.recovering) for executor in active
        )
        recovery_phase_advances = (
            self._archived_recovery_phase_advances
            + sum(executor.recovery_phase_advances for executor in active)
        )
        recovery_phase_advance_sum = (
            self._archived_recovery_phase_advance_sum
            + sum(
                executor.recovery_phase_advance_sum for executor in active
            )
        )
        max_recovery_phase_advance = max(
            self._archived_max_recovery_phase_advance,
            max(
                (
                    executor.max_recovery_phase_advance
                    for executor in active
                ),
                default=0.0,
            ),
        )
        recovery_phase_guard_blocks = (
            self._archived_recovery_phase_guard_blocks
            + sum(
                executor.recovery_phase_guard_blocks for executor in active
            )
        )
        recovery_gripper_barrier_hits = (
            self._archived_recovery_gripper_barrier_hits
            + sum(
                executor.recovery_gripper_barrier_hits for executor in active
            )
        )
        forward_reprojections = (
            self._archived_phase_reprojection_forward_count
            + sum(
                executor.phase_reprojection_forward_count
                for executor in active
            )
        )
        forward_reprojection_sum = (
            self._archived_phase_reprojection_forward_sum
            + sum(
                executor.phase_reprojection_forward_sum
                for executor in active
            )
        )
        max_forward_reprojection = max(
            self._archived_max_phase_reprojection_forward,
            max(
                (
                    executor.max_phase_reprojection_forward
                    for executor in active
                ),
                default=0.0,
            ),
        )
        backward_reprojections = (
            self._archived_phase_reprojection_backward_count
            + sum(
                executor.phase_reprojection_backward_count
                for executor in active
            )
        )
        backward_reprojection_sum = (
            self._archived_phase_reprojection_backward_sum
            + sum(
                executor.phase_reprojection_backward_sum
                for executor in active
            )
        )
        max_backward_reprojection = max(
            self._archived_max_phase_reprojection_backward,
            max(
                (
                    executor.max_phase_reprojection_backward
                    for executor in active
                ),
                default=0.0,
            ),
        )
        autonomous_recovery_chunk_compiles = (
            self._archived_autonomous_recovery_chunk_compiles
            + sum(
                executor.autonomous_recovery_chunk_compiles
                for executor in active
            )
        )
        recovery_prefix_compiles = (
            self._archived_recovery_prefix_compiles
            + sum(executor.recovery_prefix_compiles for executor in active)
        )
        recovery_prefix_supports_sum = (
            self._archived_recovery_prefix_supports_sum
            + sum(
                executor.recovery_prefix_supports_sum
                for executor in active
            )
        )
        automatic_recovery_anchor_registrations = (
            self._archived_automatic_recovery_anchor_registrations
            + sum(
                executor.automatic_recovery_anchor_registrations
                for executor in active
            )
        )
        observed_anchor_stability_wait_steps = (
            self._archived_observed_anchor_stability_wait_steps
            + sum(
                executor.observed_anchor_stability_wait_steps
                for executor in active
            )
        )
        support_confidence_checks = (
            self._archived_support_confidence_checks
            + sum(executor.support_confidence_checks for executor in active)
        )
        support_confidence_score_sum = (
            self._archived_support_confidence_score_sum
            + sum(
                executor.support_confidence_score_sum
                for executor in active
            )
        )
        minimum_support_confidence_score = min(
            self._archived_minimum_support_confidence_score,
            min(
                (
                    executor.minimum_support_confidence_score
                    for executor in active
                ),
                default=1.0,
            ),
        )
        support_confidence_direct_fallback_steps = (
            self._archived_support_confidence_direct_fallback_steps
            + sum(
                executor.support_confidence_direct_fallback_steps
                for executor in active
            )
        )
        support_confidence_inward_native_steps = (
            self._archived_support_confidence_inward_native_steps
            + sum(
                executor.support_confidence_inward_native_steps
                for executor in active
            )
        )
        state_innovation_checks = (
            self._archived_state_innovation_checks
            + sum(executor.state_innovation_checks for executor in active)
        )
        state_innovation_triggers = (
            self._archived_state_innovation_triggers
            + sum(executor.state_innovation_triggers for executor in active)
        )
        state_innovation_native_bypasses = (
            self._archived_state_innovation_native_bypasses
            + sum(
                executor.state_innovation_native_bypasses
                for executor in active
            )
        )
        max_position_innovation = max(
            self._archived_max_position_innovation,
            max(
                (executor.max_position_innovation for executor in active),
                default=0.0,
            ),
        )
        max_rotation_innovation = max(
            self._archived_max_rotation_innovation,
            max(
                (executor.max_rotation_innovation for executor in active),
                default=0.0,
            ),
        )
        position_innovation_samples = (
            self._archived_position_innovation_samples
            + [
                sample
                for executor in active
                for sample in executor.position_innovation_samples
            ]
        )
        rotation_innovation_samples = (
            self._archived_rotation_innovation_samples
            + [
                sample
                for executor in active
                for sample in executor.rotation_innovation_samples
            ]
        )

        def innovation_quantile(samples: list[float], q: float) -> float:
            if not samples:
                return 0.0
            return float(np.quantile(np.asarray(samples), q))

        gripper_transition_priority_issues = (
            self._archived_gripper_transition_priority_issues
            + sum(
                executor.gripper_transition_priority_issues
                for executor in active
            )
        )
        gripper_transition_guard_steps = (
            self._archived_gripper_transition_guard_steps
            + sum(
                executor.gripper_transition_guard_steps
                for executor in active
            )
        )
        gripper_transition_open_guard_steps = (
            self._archived_gripper_transition_open_guard_steps
            + sum(
                executor.gripper_transition_open_guard_steps
                for executor in active
            )
        )
        gripper_transition_close_guard_steps = (
            self._archived_gripper_transition_close_guard_steps
            + sum(
                executor.gripper_transition_close_guard_steps
                for executor in active
            )
        )
        transverse_fusion_checks = (
            self._archived_transverse_fusion_checks
            + sum(executor.transverse_fusion_checks for executor in active)
        )
        transverse_fusion_active_steps = (
            self._archived_transverse_fusion_active_steps
            + sum(
                executor.transverse_fusion_active_steps
                for executor in active
            )
        )
        transverse_fusion_support_fallback_steps = (
            self._archived_transverse_fusion_support_fallback_steps
            + sum(
                executor.transverse_fusion_support_fallback_steps
                for executor in active
            )
        )
        transverse_fusion_weight_sum = (
            self._archived_transverse_fusion_weight_sum
            + sum(
                executor.transverse_fusion_weight_sum
                for executor in active
            )
        )
        transverse_fusion_max_weight_applied = max(
            self._archived_transverse_fusion_max_weight_applied,
            max(
                (
                    executor.transverse_fusion_max_weight_applied
                    for executor in active
                ),
                default=0.0,
            ),
        )
        transverse_fusion_correction_norm_sum = (
            self._archived_transverse_fusion_correction_norm_sum
            + sum(
                executor.transverse_fusion_correction_norm_sum
                for executor in active
            )
        )
        transverse_fusion_max_correction_norm = max(
            self._archived_transverse_fusion_max_correction_norm,
            max(
                (
                    executor.transverse_fusion_max_correction_norm
                    for executor in active
                ),
                default=0.0,
            ),
        )
        metrics.update(
            {
                "executor": "paired_bspline_hybrid_ds",
                "phase_position_reentry_radius": (
                    self.phase_position_reentry_radius
                ),
                "phase_rotation_reentry_radius": (
                    self.phase_rotation_reentry_radius
                ),
                "phase_projection_lookahead": (
                    self.phase_projection_lookahead
                ),
                "recovery_phase_advance_budget": (
                    self.recovery_phase_advance_budget
                ),
                "phase_max_recovery_steps": (
                    self.phase_max_recovery_steps
                ),
                "recovery_field_source": self.recovery_field_source,
                "recovery_target_mode": self.recovery_target_mode,
                "recovery_anchor_mode": self.recovery_anchor_mode,
                "observed_anchor_reentry_radius": (
                    self.observed_anchor_reentry_radius
                ),
                "observed_anchor_stable_steps": (
                    self.observed_anchor_stable_steps
                ),
                "automatic_recovery_anchor_registrations": int(
                    automatic_recovery_anchor_registrations
                ),
                "observed_anchor_stability_wait_steps": int(
                    observed_anchor_stability_wait_steps
                ),
                "freeze_recovery_phase": self.freeze_recovery_phase,
                "autonomous_chunk_reentry": (
                    self.autonomous_chunk_reentry
                ),
                "replan_on_recovery_reentry": (
                    self.replan_on_recovery_reentry
                ),
                "recovery_execution": (
                    "locked_safe_prefix_gaussian_ds_to_terminal_phase"
                    if self.recovery_target_mode
                    == "support_confidence_prefix"
                    else (
                        "support_confidence_autonomous_gaussian_ds_with_se3_clf"
                        if self.recovery_target_mode == "support_confidence"
                        else (
                            "frozen_same_phase_linear_attractor"
                            if self.recovery_target_mode
                            == "same_phase_attractor"
                            else (
                                "autonomous_event_chunk_gaussian_ds_with_se3_clf"
                                if self.autonomous_chunk_reentry
                                else metrics.get("recovery_execution")
                            )
                        )
                    )
                ),
                "gripper_during_recovery": (
                    "repeat_pending_binary_event_else_neutral"
                    if self.recovery_target_mode
                    in ProjectedPhaseHybridExecutor._SUPPORT_CONFIDENCE_MODES
                    else "neutral_increment_hold"
                ),
                "autonomous_recovery_chunk_compiles": int(
                    autonomous_recovery_chunk_compiles
                ),
                "recovery_prefix_compiles": int(
                    recovery_prefix_compiles
                ),
                "recovery_prefix_supports_sum": int(
                    recovery_prefix_supports_sum
                ),
                "rotation_clf_enabled": self.rotation_clf_enabled,
                "rotation_clf_tube_radius": (
                    self.rotation_clf_tube_radius
                ),
                "rotation_clf_contraction_rate": (
                    self.rotation_clf_contraction_rate
                ),
                "recovery_progress_patience": (
                    self.recovery_progress_patience
                ),
                "recovery_progress_min_delta": (
                    self.recovery_progress_min_delta
                ),
                "field_speed_gain": self.field_speed_gain,
                "nominal_fusion_mode": self.nominal_fusion_mode,
                "transverse_fusion_inner_ratio": (
                    self.transverse_fusion_inner_ratio
                ),
                "transverse_fusion_outer_ratio": (
                    self.transverse_fusion_outer_ratio
                ),
                "transverse_fusion_max_weight": (
                    self.transverse_fusion_max_weight
                ),
                "transverse_fusion_checks": int(
                    transverse_fusion_checks
                ),
                "transverse_fusion_active_steps": int(
                    transverse_fusion_active_steps
                ),
                "transverse_fusion_active_fraction": (
                    float(
                        transverse_fusion_active_steps
                        / transverse_fusion_checks
                    )
                    if transverse_fusion_checks
                    else 0.0
                ),
                "transverse_fusion_support_fallback_steps": int(
                    transverse_fusion_support_fallback_steps
                ),
                "transverse_fusion_weight_mean_active": (
                    float(
                        transverse_fusion_weight_sum
                        / transverse_fusion_active_steps
                    )
                    if transverse_fusion_active_steps
                    else 0.0
                ),
                "transverse_fusion_weight_max": float(
                    transverse_fusion_max_weight_applied
                ),
                "transverse_fusion_correction_action_norm_mean_active": (
                    float(
                        transverse_fusion_correction_norm_sum
                        / transverse_fusion_active_steps
                    )
                    if transverse_fusion_active_steps
                    else 0.0
                ),
                "transverse_fusion_correction_action_norm_max": float(
                    transverse_fusion_max_correction_norm
                ),
                "support_confidence_metric": (
                    (
                        "finite_segment_pose_distance_over_k_sample_spread"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "max_unnormalized_position_gaussian"
                    )
                    if self.recovery_target_mode
                    in ProjectedPhaseHybridExecutor._SUPPORT_CONFIDENCE_MODES
                    else None
                ),
                "support_confidence_source": (
                    (
                        "pose_ensemble_only"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "compiled_dynamics_gaussian"
                    )
                    if self.recovery_target_mode
                    in ProjectedPhaseHybridExecutor._SUPPORT_CONFIDENCE_MODES
                    else None
                ),
                "support_confidence_outer_ratio": (
                    1.0
                    if self.recovery_trigger_mode
                    == "support_and_state_innovation"
                    else None
                ),
                "support_confidence_outer_score": (
                    self.support_confidence_outer_score
                ),
                "support_confidence_inner_score": (
                    self.support_confidence_inner_score
                ),
                "support_confidence_action_rule": (
                    (
                        (
                            "native_unless_outside_support_and_state_innovation"
                            if self.require_state_innovation
                            else "native_unless_outside_confident_sampled_support"
                        )
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "native_inside_or_inward_else_field"
                    )
                    if self.recovery_target_mode
                    in ProjectedPhaseHybridExecutor._SUPPORT_CONFIDENCE_MODES
                    else None
                ),
                "support_confidence_entry_scope": (
                    (
                        "last_native_finite_pose_segment"
                        if self.recovery_trigger_mode
                        == "support_and_state_innovation"
                        else "gripper_safe_chunk"
                    )
                    if self.recovery_target_mode
                    in ProjectedPhaseHybridExecutor._SUPPORT_CONFIDENCE_MODES
                    else None
                ),
                "recovery_trigger_mode": self.recovery_trigger_mode,
                "require_state_innovation": self.require_state_innovation,
                "disturbance_response_mode": (
                    self.disturbance_response_mode
                ),
                "disturbance_position_innovation_threshold": (
                    self.disturbance_position_innovation_threshold
                ),
                "disturbance_rotation_innovation_threshold": (
                    self.disturbance_rotation_innovation_threshold
                ),
                "disturbance_expected_motion_multiplier": (
                    self.disturbance_expected_motion_multiplier
                ),
                "state_innovation_checks": int(state_innovation_checks),
                "state_innovation_triggers": int(state_innovation_triggers),
                "state_innovation_native_bypasses": int(
                    state_innovation_native_bypasses
                ),
                "max_position_innovation": float(max_position_innovation),
                "max_rotation_innovation": float(max_rotation_innovation),
                "position_innovation_p50": innovation_quantile(
                    position_innovation_samples,
                    0.50,
                ),
                "position_innovation_p90": innovation_quantile(
                    position_innovation_samples,
                    0.90,
                ),
                "position_innovation_p99": innovation_quantile(
                    position_innovation_samples,
                    0.99,
                ),
                "rotation_innovation_p50": innovation_quantile(
                    rotation_innovation_samples,
                    0.50,
                ),
                "rotation_innovation_p90": innovation_quantile(
                    rotation_innovation_samples,
                    0.90,
                ),
                "rotation_innovation_p99": innovation_quantile(
                    rotation_innovation_samples,
                    0.99,
                ),
                "support_confidence_checks": int(
                    support_confidence_checks
                ),
                "support_confidence_score_mean": (
                    float(
                        support_confidence_score_sum
                        / support_confidence_checks
                    )
                    if support_confidence_checks
                    else 1.0
                ),
                "support_confidence_score_min": float(
                    minimum_support_confidence_score
                ),
                "support_confidence_direct_fallback_steps": int(
                    support_confidence_direct_fallback_steps
                ),
                "support_confidence_inward_native_steps": int(
                    support_confidence_inward_native_steps
                ),
                "recovery_prefix_terminal_phase_tolerance": (
                    0.5 * float(self.phase_stride)
                    if self.recovery_target_mode
                    == "support_confidence_prefix"
                    else None
                ),
                "gripper_transition_priority_issues": int(
                    gripper_transition_priority_issues
                ),
                "gripper_transition_guard_steps": int(
                    gripper_transition_guard_steps
                ),
                "gripper_transition_open_guard_steps": int(
                    gripper_transition_open_guard_steps
                ),
                "gripper_transition_close_guard_steps": int(
                    gripper_transition_close_guard_steps
                ),
                "effective_recovery_speed_gain": (
                    1.0
                    if self.recovery_target_mode
                    == "same_phase_attractor"
                    else self.field_speed_gain
                ),
                "adaptive_pose_support_enabled": (
                    self.adaptive_pose_support_enabled
                ),
                "adaptive_pose_tube_enabled": (
                    self.adaptive_pose_tube_enabled
                ),
                "adaptive_pose_tube_metric": (
                    "transverse_mahalanobis"
                    if self.adaptive_pose_tube_enabled
                    else "fixed_euclidean"
                ),
                "pose_support_confidence": (
                    self.pose_support_confidence
                ),
                "pose_reentry_fraction": self.pose_reentry_fraction,
                "pose_dense_core_confidence": (
                    self.pose_dense_core_confidence
                ),
                "recovery_entries": int(recovery_entries),
                "recovery_reentries": int(recovery_reentries),
                "recovery_reentry_replans": int(
                    self.replan_reasons.get("recovery_reentry", 0)
                ),
                "recovery_reentry_replan_gripper_deferrals": int(
                    recovery_reentry_replan_gripper_deferrals
                ),
                "recovery_no_progress_replans": int(
                    no_progress_replans
                ),
                "mean_completed_recovery_steps": (
                    float(completed_recovery_steps / recovery_reentries)
                    if recovery_reentries
                    else 0.0
                ),
                "max_completed_recovery_steps": int(
                    max_completed_recovery_steps
                ),
                "unresolved_recoveries": int(unresolved_recoveries),
                "recovery_phase_advances": int(recovery_phase_advances),
                "mean_recovery_phase_advance": (
                    float(
                        recovery_phase_advance_sum
                        / recovery_phase_advances
                    )
                    if recovery_phase_advances
                    else 0.0
                ),
                "max_recovery_phase_advance": float(
                    max_recovery_phase_advance
                ),
                "recovery_phase_guard_blocks": int(
                    recovery_phase_guard_blocks
                ),
                "recovery_gripper_barrier_hits": int(
                    recovery_gripper_barrier_hits
                ),
                "phase_reprojections": int(reprojections),
                "mean_phase_reprojection": (
                    float(shift_sum / reprojections)
                    if reprojections
                    else 0.0
                ),
                "max_phase_reprojection": float(max_shift),
                "phase_reprojection_forward_count": int(
                    forward_reprojections
                ),
                "mean_phase_reprojection_forward": (
                    float(forward_reprojection_sum / forward_reprojections)
                    if forward_reprojections
                    else 0.0
                ),
                "max_phase_reprojection_forward": float(
                    max_forward_reprojection
                ),
                "phase_reprojection_backward_count": int(
                    backward_reprojections
                ),
                "mean_phase_reprojection_backward": (
                    float(
                        backward_reprojection_sum
                        / backward_reprojections
                    )
                    if backward_reprojections
                    else 0.0
                ),
                "max_phase_reprojection_backward": float(
                    max_backward_reprojection
                ),
            }
        )
        if self._adaptive_position_tube_values:
            values = np.asarray(
                self._adaptive_position_tube_values,
                dtype=np.float64,
            )
            metrics.update(
                {
                    "adaptive_position_tube_radius_mean": float(
                        np.mean(values)
                    ),
                    "adaptive_position_tube_radius_p50": float(
                        np.quantile(values, 0.5)
                    ),
                    "adaptive_position_tube_radius_p90": float(
                        np.quantile(values, 0.9)
                    ),
                    "adaptive_position_tube_radius_max": float(
                        np.max(values)
                    ),
                }
            )
        if self._adaptive_position_reentry_values:
            values = np.asarray(
                self._adaptive_position_reentry_values,
                dtype=np.float64,
            )
            metrics["adaptive_position_reentry_radius_mean"] = float(
                np.mean(values)
            )
        if self._pose_sample_position_outer_values:
            values = np.asarray(
                self._pose_sample_position_outer_values,
                dtype=np.float64,
            )
            metrics.update(
                {
                    "pose_sample_position_outer_radius_mean": float(
                        np.mean(values)
                    ),
                    "pose_sample_position_outer_radius_p50": float(
                        np.quantile(values, 0.5)
                    ),
                    "pose_sample_position_outer_radius_p90": float(
                        np.quantile(values, 0.9)
                    ),
                    "pose_sample_position_outer_radius_max": float(
                        np.max(values)
                    ),
                }
            )
        if self._pose_sample_position_reentry_values:
            values = np.asarray(
                self._pose_sample_position_reentry_values,
                dtype=np.float64,
            )
            metrics["pose_sample_position_reentry_radius_mean"] = float(
                np.mean(values)
            )
        return metrics

    def _phase_executor_class(self):
        return ProjectedPhaseHybridExecutor

    def _phase_executor_kwargs(self) -> dict[str, object]:
        return {
            "position_tube_radius": self.phase_position_tube_radius,
            "rotation_tube_radius": self.phase_rotation_tube_radius,
            "position_reentry_radius": (
                self.phase_position_reentry_radius
            ),
            "rotation_reentry_radius": (
                self.phase_rotation_reentry_radius
            ),
            "phase_projection_lookahead": (
                self.phase_projection_lookahead
            ),
            "recovery_phase_advance_budget": (
                self.recovery_phase_advance_budget
            ),
            "max_recovery_steps": self.phase_max_recovery_steps,
            "recovery_field_source": self.recovery_field_source,
            "recovery_target_mode": self.recovery_target_mode,
            "recovery_anchor_mode": self.recovery_anchor_mode,
            "observed_anchor_reentry_radius": (
                self.observed_anchor_reentry_radius
            ),
            "observed_anchor_stable_steps": (
                self.observed_anchor_stable_steps
            ),
            "freeze_recovery_phase": self.freeze_recovery_phase,
            "autonomous_chunk_reentry": self.autonomous_chunk_reentry,
            "replan_on_recovery_reentry": (
                self.replan_on_recovery_reentry
            ),
            "rotation_clf_enabled": self.rotation_clf_enabled,
            "rotation_clf_tube_radius": self.rotation_clf_tube_radius,
            "rotation_clf_contraction_rate": (
                self.rotation_clf_contraction_rate
            ),
            "recovery_progress_patience": (
                self.recovery_progress_patience
            ),
            "recovery_progress_min_delta": (
                self.recovery_progress_min_delta
            ),
            "field_speed_gain": self.field_speed_gain,
            "support_confidence_outer_score": (
                self.support_confidence_outer_score
            ),
            "support_confidence_inner_score": (
                self.support_confidence_inner_score
            ),
            "recovery_trigger_mode": self.recovery_trigger_mode,
            "require_state_innovation": self.require_state_innovation,
            "disturbance_response_mode": self.disturbance_response_mode,
            "disturbance_position_innovation_threshold": (
                self.disturbance_position_innovation_threshold
            ),
            "disturbance_rotation_innovation_threshold": (
                self.disturbance_rotation_innovation_threshold
            ),
            "disturbance_expected_motion_multiplier": (
                self.disturbance_expected_motion_multiplier
            ),
            "nominal_fusion_mode": self.nominal_fusion_mode,
            "transverse_fusion_inner_ratio": (
                self.transverse_fusion_inner_ratio
            ),
            "transverse_fusion_outer_ratio": (
                self.transverse_fusion_outer_ratio
            ),
            "transverse_fusion_max_weight": (
                self.transverse_fusion_max_weight
            ),
            "phase_stride": self.phase_stride,
        }


class LiberoPairedPoseBSplineRawActionHybridDSAdapter(
    LiberoPairedBSplineHybridDSAdapter
):
    """Use the canonical hybrid field with exact native action steps."""

    def __init__(self, base_policy, **kwargs) -> None:
        super().__init__(base_policy, **kwargs)
        self.representation = "paired_pose_bspline_raw_action_hybrid_ds"

    def get_policy_name(self):
        return (
            self.base_policy.get_policy_name()
            + "_pose_bspline_raw_action_hybrid_ds"
        )

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_pose_bspline_raw_action_hybrid_ds",
                "representation": self.representation,
                "action_representation": "exact_native_steps",
            }
        )
        return metrics

    def _decode_prediction(self, prediction):
        decoded = decode_libero_paired_pose_bspline_raw_action(
            prediction.detach().cpu().numpy(),
            output_steps=self.source_horizon,
            source_horizon=self.source_horizon,
            degree=self.degree,
            rotation_action_scale=self.field_config.rotation_action_scale,
            support_subdivisions=self.support_subdivisions,
        )
        return prediction.new_tensor(decoded)


class LiberoPairedPoseActionHybridDSAdapter(
    LiberoPairedBSplineHybridDSAdapter
):
    """Run last-safe directly on aligned raw pose and raw action steps."""

    def __init__(self, base_policy, **kwargs) -> None:
        subdivisions = int(kwargs.pop("support_subdivisions", 1))
        if subdivisions != 1:
            raise ValueError(
                "Raw pose/action sequences require support_subdivisions=1"
            )
        super().__init__(
            base_policy,
            support_subdivisions=1,
            **kwargs,
        )
        self.representation = "paired_pose_action_hybrid_ds"

    def get_policy_name(self):
        return self.base_policy.get_policy_name() + "_raw_pose_action_hybrid_ds"

    def get_execution_metrics(self) -> dict[str, object]:
        metrics = super().get_execution_metrics()
        metrics.update(
            {
                "executor": "paired_pose_action_hybrid_ds",
                "representation": self.representation,
                "pose_representation": (
                    "measured_native_timestep_increments"
                ),
                "action_representation": "exact_native_steps",
            }
        )
        return metrics

    def _decode_prediction(self, prediction):
        return prediction
