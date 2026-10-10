from __future__ import annotations

import argparse
import functools
import inspect
import importlib.util
import json
import os
import pathlib
import random
import sys
import types
from typing import Any

import numpy as np
import torch

# Spawn workers import Robosuite before main() can install this optional module.
sys.modules.setdefault(
    "robosuite.macros_private",
    types.ModuleType("robosuite.macros_private"),
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBERO_ROOT = pathlib.Path(
    os.environ.get("LIBERO_ROOT", ROOT / "thirdparty" / "LIBERO")
).resolve()
LIBERO_BENCHMARK_ROOT = LIBERO_ROOT / "libero" / "libero"
sys.path.insert(0, str(ROOT / "src"))

from ns_gaussian_ds_policy.libero_paired_hybrid_profiles import (
    METHOD_PROFILE_NAMES,
    PERTURBATION_PROFILE_NAMES,
    apply_evaluation_profiles,
)


def _prepare_imports() -> None:
    sys.path.insert(0, str(ROOT))
    expected_runner = os.environ.get("LIBERO_EXPECTED_RUNNER_PATH")
    if expected_runner:
        spec = importlib.util.find_spec("dsp_diffusion.env_runner.libero_runner")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot resolve the pilot LiberoRunner")
        actual_runner = pathlib.Path(spec.origin).resolve()
        if actual_runner != pathlib.Path(expected_runner).resolve():
            raise RuntimeError(f"Wrong pilot LiberoRunner: {actual_runner}")
        if "/success_{seed}" not in actual_runner.read_text():
            raise RuntimeError("Pilot LiberoRunner lacks explicit episode outcomes")
        print(f"[pilot-runtime] verified runner={actual_runner}", flush=True)
    os.chdir(ROOT)
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_libero_config()


def _ensure_libero_config() -> None:
    config_root = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", pathlib.Path.home() / ".libero"))
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / "config.yaml"
    if config_path.exists():
        return
    benchmark_root = LIBERO_BENCHMARK_ROOT.resolve()
    config_path.write_text(
        "\n".join(
            [
                f"benchmark_root: {benchmark_root.as_posix()}",
                f"bddl_files: {(benchmark_root / 'bddl_files').as_posix()}",
                f"init_states: {(benchmark_root / 'init_files').as_posix()}",
                f"datasets: {(benchmark_root.parent / 'datasets').as_posix()}",
                f"assets: {(benchmark_root / 'assets').as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _install_per_episode_rng_reset(
    policy: Any,
    *,
    eval_seed: int | None,
    n_parallel_envs: int,
) -> list[int]:
    """Reset sampling RNG after each serial environment reset."""
    if eval_seed is None:
        raise ValueError("per_episode RNG mode requires --eval-seed")
    if int(n_parallel_envs) != 1:
        raise ValueError("per_episode RNG mode requires --n-parallel-envs 1")

    original_reset = policy.reset
    episode_sampling_seeds: list[int] = []

    def seeded_reset() -> Any:
        result = original_reset()
        seed = int(eval_seed) + len(episode_sampling_seeds)
        _set_eval_seed(seed)
        episode_sampling_seeds.append(seed)
        print(
            f"[eval] episode_rng index={len(episode_sampling_seeds) - 1} "
            f"seed={seed}",
            flush=True,
        )
        return result

    policy.reset = seeded_reset
    return episode_sampling_seeds






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate matched raw, paired, or Gaussian LIBERO-10 policies."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--tokenizer-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--method-profile",
        choices=METHOD_PROFILE_NAMES,
        default="custom",
        help=(
            "Apply a versioned, immutable controller configuration. "
            "Profile-owned options override individual CLI values."
        ),
    )
    parser.add_argument(
        "--perturbation-profile",
        choices=PERTURBATION_PROFILE_NAMES,
        default="custom",
        help=(
            "Apply a versioned environment perturbation protocol. "
            "Profile-owned options override individual CLI values."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "raw_action",
            "command_center_field",
            "command_center_direct",
            "paired_pose_action_field",
            "paired_pose_action_direct",
            "paired_bspline_field",
            "paired_bspline_pose_field",
            "paired_bspline_hybrid_ds",
            "paired_bspline_integrated_field",
            "paired_bspline_direct",
            "paired_bspline_geometry_direct",
            "paired_pose_action_hybrid_ds",
            "action_bspline_direct",
        ],
        default="raw_action",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--eval-rng-mode",
        choices=("per_experiment", "per_episode", "continuous"),
        default="per_experiment",
        help=(
            "Reset to eval_seed + experiment index for each repeat, reset "
            "to eval_seed + serial episode index for matched common-random-"
            "number comparisons, or seed once and continue one RNG stream "
            "across all repeats."
        ),
    )
    parser.add_argument("--num-exp", type=int, default=1)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-test-vis", type=int, default=0)
    parser.add_argument(
        "--require-episode-outcomes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail evaluation unless every rollout has a retained task, "
            "environment seed, policy-sampling seed, and success outcome."
        ),
    )
    parser.add_argument(
        "--controller-debug-videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render task video beside synchronized ACTION/FIELD diagnostics "
            "for paired_bspline_hybrid_ds rollouts."
        ),
    )
    parser.add_argument("--n-parallel-envs", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=550)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--test-start-seed", type=int, default=None)
    parser.add_argument("--test-seeds", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--use-k-tokens", type=int, default=None)
    parser.add_argument("--max-xyz-action", type=float, default=1.0)
    parser.add_argument("--max-rot-action", type=float, default=0.5)
    parser.add_argument("--no-clip-gripper", action="store_true")
    parser.add_argument("--center-field-step-scale", type=float, default=1.0)
    parser.add_argument("--center-field-steps", type=int, default=16)
    parser.add_argument("--center-field-lookahead", type=int, default=3)
    parser.add_argument("--center-field-temporal-decay", type=float, default=0.7)
    parser.add_argument("--center-field-replan-interval", type=int, default=16)
    parser.add_argument("--center-field-active-weight-floor", type=float, default=0.35)
    parser.add_argument("--center-field-min-sigma", type=float, default=0.003)
    parser.add_argument("--center-field-max-sigma", type=float, default=0.10)
    parser.add_argument("--center-field-min-gain-scale", type=float, default=1.0)
    parser.add_argument("--center-field-max-gain-scale", type=float, default=220.0)
    parser.add_argument("--center-field-input-xyz-scale", type=float, default=1.0)
    parser.add_argument("--center-field-policy-index-stride", type=int, default=1)
    parser.add_argument("--success-hold-steps", type=int, default=None)
    parser.add_argument("--paired-plan-steps", type=int, default=16)
    parser.add_argument("--paired-bspline-source-horizon", type=int, default=32)
    parser.add_argument("--paired-bspline-degree", type=int, default=3)
    parser.add_argument("--geometry-direct-gain", type=float, default=1.0)
    parser.add_argument(
        "--paired-bspline-support-subdivisions",
        type=int,
        default=1,
    )
    parser.add_argument("--paired-pose-ensemble-samples", type=int, default=1)
    parser.add_argument(
        "--paired-pose-ensemble-cluster-cutoff",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--paired-pose-ensemble-min-cluster-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--paired-pose-ensemble-selection-mode",
        choices=("dominant_cluster", "all_samples", "manifold_kde"),
        default="dominant_cluster",
    )
    parser.add_argument(
        "--paired-pose-ensemble-distance-scale-mode",
        choices=("nearest_neighbor", "pairwise_median"),
        default="nearest_neighbor",
    )
    parser.add_argument(
        "--paired-pose-ensemble-support-members-mode",
        choices=(
            "selected_members",
            "all_samples",
            "kde_responsibility",
        ),
        default="selected_members",
    )
    parser.add_argument(
        "--paired-pose-ensemble-kde-position-bandwidth",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--paired-pose-ensemble-kde-rotation-bandwidth",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--paired-pose-ensemble-kde-gripper-bandwidth",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--paired-pose-ensemble-kde-support-mass",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--paired-pose-ensemble-adaptive-support",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--paired-pose-ensemble-adaptive-tube",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--paired-pose-ensemble-support-confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--paired-pose-ensemble-reentry-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--paired-pose-ensemble-dense-core-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--paired-pose-ensemble-dense-core-confidence",
        type=float,
        default=0.9,
    )
    parser.add_argument("--position-action-scale", type=float, default=0.05)
    parser.add_argument("--rotation-action-scale", type=float, default=0.5)
    parser.add_argument("--position-tolerance", type=float, default=0.002)
    parser.add_argument("--rotation-tolerance", type=float, default=0.05)
    parser.add_argument("--max-plan-steps", type=int, default=200)
    parser.add_argument(
        "--gripper-only-position-threshold",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--gripper-only-rotation-threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--gripper-close-command-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--gating-precision-scale", type=float, default=25.0)
    parser.add_argument("--position-recovery-gain", type=float, default=1.0)
    parser.add_argument("--rotation-recovery-gain", type=float, default=1.0)
    parser.add_argument(
        "--support-interpolation-rcond",
        type=float,
        default=1.0e-2,
    )
    parser.add_argument(
        "--tube-projection-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tube-radius", type=float, default=0.01)
    parser.add_argument(
        "--phase-conditioned-execution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--phase-position-tube-radius",
        type=float,
        default=0.075,
    )
    parser.add_argument(
        "--phase-rotation-tube-radius",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--phase-position-reentry-radius",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--phase-rotation-reentry-radius",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--phase-projection-lookahead",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--recovery-phase-advance-budget",
        type=float,
        default=None,
        help=(
            "Maximum cumulative support-phase advance per recovery; "
            "zero disables the cumulative limit."
        ),
    )
    parser.add_argument(
        "--phase-max-recovery-steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--hybrid-recovery-field-source",
        choices=("action", "geometry", "projected_action"),
        default="action",
    )
    parser.add_argument(
        "--hybrid-recovery-target-mode",
        choices=(
            "projected",
            "support_confidence",
            "support_confidence_prefix",
            "next_action_phase",
            "same_phase_attractor",
        ),
        default="projected",
    )
    parser.add_argument(
        "--hybrid-recovery-anchor-mode",
        choices=("predicted_pose", "observed_perturbation_start"),
        default="predicted_pose",
    )
    parser.add_argument(
        "--hybrid-observed-anchor-reentry-radius",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--hybrid-observed-anchor-stable-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--hybrid-freeze-recovery-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-autonomous-chunk-reentry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-replan-on-recovery-reentry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-rotation-clf-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-rotation-clf-radius",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--hybrid-rotation-clf-contraction-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hybrid-recovery-progress-patience",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hybrid-recovery-progress-min-delta",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--hybrid-field-speed-gain", type=float, default=1.0)
    parser.add_argument(
        "--hybrid-support-confidence-outer-score",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--hybrid-support-confidence-inner-score",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--hybrid-recovery-trigger-mode",
        choices=("support_only", "support_and_state_innovation"),
        default="support_only",
    )
    parser.add_argument(
        "--hybrid-nominal-fusion-mode",
        choices=(
            "native_action",
            "transverse_minimal",
            "transverse_outward_only",
        ),
        default="native_action",
    )
    parser.add_argument(
        "--hybrid-require-state-innovation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-transverse-fusion-inner-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hybrid-transverse-fusion-outer-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hybrid-transverse-fusion-max-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hybrid-disturbance-position-innovation-threshold",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--hybrid-disturbance-rotation-innovation-threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--hybrid-disturbance-expected-motion-multiplier",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--hybrid-disturbance-response-mode",
        choices=("recover_then_replan", "immediate_replan"),
        default="recover_then_replan",
    )
    parser.add_argument(
        "--direct-stepwise-execution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cache direct-policy action chunks and issue them one control "
            "step at a time without increasing policy query frequency."
        ),
    )
    parser.add_argument(
        "--direct-perturbation-radius-cm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--direct-perturbation-trigger-phase",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--direct-perturbation-direction-sign",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--direct-perturbation-max-steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--direct-perturbation-reach-fraction",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--direct-perturbation-gripper-event-guard",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--direct-perturbation-require-future-gripper-event",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-perturbation-radius-cm",
        type=float,
        default=0.0,
        help=(
            "Inject one transverse OSC displacement without advancing the "
            "cached hybrid executor; zero disables the diagnostic."
        ),
    )
    parser.add_argument(
        "--hybrid-perturbation-trigger-phase",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--hybrid-perturbation-direction-sign",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hybrid-perturbation-max-steps",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--hybrid-perturbation-reach-fraction",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--hybrid-perturbation-gripper-event-guard",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--hybrid-perturbation-require-future-gripper-event",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--concurrent-perturbation-distance-cm",
        type=float,
        default=0.0,
        help=(
            "Apply an environment-side EEF wrench while the policy keeps "
            "running; zero disables the protocol."
        ),
    )
    parser.add_argument(
        "--concurrent-perturbation-start-step",
        type=int,
        default=2,
        help="Zero-based controller step at which the external wrench starts.",
    )
    parser.add_argument(
        "--concurrent-perturbation-direction-x",
        type=float,
        default=0.3882361,
    )
    parser.add_argument(
        "--concurrent-perturbation-direction-y",
        type=float,
        default=-0.9215599,
    )
    parser.add_argument(
        "--concurrent-perturbation-direction-z",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--concurrent-perturbation-ramp-steps",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--concurrent-perturbation-hold-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--concurrent-perturbation-max-steps",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--concurrent-perturbation-stiffness",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--concurrent-perturbation-integral-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--concurrent-perturbation-integral-error-limit",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--concurrent-perturbation-damping",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--concurrent-perturbation-max-force",
        type=float,
        default=100.0,
    )
    parser.add_argument("--tube-contraction-rate", type=float, default=0.1)
    parser.add_argument(
        "--preserve-field-direction-on-saturation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    from ns_gaussian_ds_policy.recovery_baselines import add_recovery_arguments
    add_recovery_arguments(parser)
    parser.add_argument("--recovery-controller", choices=("none", "pd_return", "distance_field"), default="none")
    return parser.parse_args()


def _resolve_path(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_policy_from_checkpoint(
    checkpoint: pathlib.Path,
    tokenizer_checkpoint: pathlib.Path | None = None,
):
    if tokenizer_checkpoint is not None:
        raise ValueError('Tokenizer checkpoints are not part of the DSP diffusion release')
    from dsp_diffusion.checkpoint import load_policy
    return load_policy(checkpoint)


def _jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    path = getattr(value, "_path", None) or getattr(value, "path", None)
    if path is not None:
        return str(path)
    return str(value)


def _serializable_runner_log(log: dict) -> dict:
    return {str(key): _jsonable(value) for key, value in log.items()}


def _require_episode_outcome_capable_runner(env_runner: Any) -> None:
    run_method = getattr(type(env_runner), "run", None)
    try:
        source = inspect.getsource(run_method)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot verify that LiberoRunner retains per-episode outcomes"
        ) from exc
    if "/success_{seed}" not in source:
        raise RuntimeError(
            "LiberoRunner does not retain per-episode outcomes. Run "
            "scripts/patch_oat_runtime.py before formal evaluation."
        )


def _build_episode_outcome_records(
    run_log: dict[str, Any],
    *,
    task_names: list[str],
    environment_seeds: list[int],
    sampling_seeds: list[int] | None,
    experiment_index: int,
) -> list[dict[str, Any]]:
    if len(task_names) != len(environment_seeds):
        raise ValueError(
            "Runner task and environment-seed schedules have different "
            f"lengths: {len(task_names)} != {len(environment_seeds)}"
        )
    if sampling_seeds is not None and len(sampling_seeds) != len(task_names):
        raise ValueError(
            "Episode sampling-seed schedule has the wrong length: "
            f"{len(sampling_seeds)} != {len(task_names)}"
        )

    outcome_keys = [
        f"{task_name}/success_{environment_seed}"
        for task_name, environment_seed in zip(
            task_names,
            environment_seeds,
            strict=True,
        )
    ]
    present = [key in run_log for key in outcome_keys]
    if not any(present):
        return []
    if not all(present):
        missing = [key for key, exists in zip(outcome_keys, present) if not exists]
        raise ValueError(
            "Runner retained only a partial episode-outcome schedule; "
            f"first missing key: {missing[0]}"
        )

    records: list[dict[str, Any]] = []
    for episode_index, (task_name, environment_seed, outcome_key) in enumerate(
        zip(task_names, environment_seeds, outcome_keys, strict=True)
    ):
        raw_success = run_log[outcome_key]
        if isinstance(raw_success, bool):
            success = raw_success
        elif isinstance(raw_success, (int, float, np.number)) and float(
            raw_success
        ) in (0.0, 1.0):
            success = bool(int(raw_success))
        else:
            raise ValueError(
                f"Runner outcome {outcome_key} is not binary: {raw_success!r}"
            )

        success_step_key = f"{task_name}/success_step_{environment_seed}"
        success_step = run_log.get(success_step_key)
        if success_step is not None:
            success_step = int(success_step)
        records.append(
            {
                "experiment_index": int(experiment_index),
                "episode_index": int(episode_index),
                "task_name": str(task_name),
                "environment_seed": int(environment_seed),
                "sampling_seed": (
                    int(sampling_seeds[episode_index])
                    if sampling_seeds is not None
                    else None
                ),
                "success": bool(success),
                "success_step": success_step,
            }
        )
    return records


def main() -> None:
    args = apply_evaluation_profiles(parse_args())
    if (
        args.require_episode_outcomes
        and args.eval_rng_mode != "per_episode"
    ):
        raise ValueError(
            "--require-episode-outcomes requires "
            "--eval-rng-mode per_episode"
        )
    print(
        "[eval] profiles "
        f"method={args.method_profile} "
        f"perturbation={args.perturbation_profile}",
        flush=True,
    )
    args.checkpoint = _resolve_path(args.checkpoint)
    if args.tokenizer_checkpoint is not None:
        args.tokenizer_checkpoint = _resolve_path(args.tokenizer_checkpoint)
    args.output_dir = _resolve_path(args.output_dir)
    if args.success_hold_steps is not None:
        os.environ["LIBERO_SUCCESS_HOLD_STEPS"] = str(max(1, int(args.success_hold_steps)))

    _prepare_imports()

    import hydra
    from dsp_diffusion.env_runner.base_runner import BaseRunner

    if args.eval_seed is not None:
        _set_eval_seed(int(args.eval_seed))
        print(
            "[eval] sampling RNG "
            f"mode={args.eval_rng_mode} initial_seed={int(args.eval_seed)}",
            flush=True,
        )

    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists; remove it first or choose a new output dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy, cfg = _load_policy_from_checkpoint(args.checkpoint, args.tokenizer_checkpoint)
    common_adapter_info = {
        "mode": args.mode,
        "success_hold_steps": os.environ.get("LIBERO_SUCCESS_HOLD_STEPS", "1"),
    }
    if args.mode == "raw_action":
        adapter_info = {
            **common_adapter_info,
            "executor": "native_receding_horizon",
        }
    elif args.mode == "command_center_direct":
        from ns_gaussian_ds_policy.libero_command_center import (
            LiberoCommandCenterDirectAdapter,
        )

        adapter_info = {
            **common_adapter_info,
            "representation": "action_integrated_command_center",
            "position_action_scale": args.position_action_scale,
            "rotation_action_scale": args.rotation_action_scale,
        }
        policy = LiberoCommandCenterDirectAdapter(
            policy,
            position_action_scale=float(args.position_action_scale),
            rotation_action_scale=float(args.rotation_action_scale),
        )
    elif args.mode in {
        "paired_pose_action_direct",
        "paired_bspline_direct",
        "paired_bspline_geometry_direct",
        "action_bspline_direct",
    }:
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoActionBSplineDirectAdapter,
            LiberoPairedBSplineDirectAdapter,
            LiberoPairedBSplineGeometryDirectAdapter,
            LiberoPairedBSplinePoseMedoidPolicy,
            LiberoPairedPoseActionMedoidPolicy,
            LiberoPairedPoseActionDirectAdapter,
        )

        if (
            args.mode == "paired_pose_action_direct"
            and int(args.paired_pose_ensemble_samples) > 1
        ):
            policy = LiberoPairedPoseActionMedoidPolicy(
                policy,
                samples=int(args.paired_pose_ensemble_samples),
                source_horizon=int(args.paired_bspline_source_horizon),
                selection_steps=int(args.paired_plan_steps),
                support_subdivisions=1,
                position_action_scale=float(args.position_action_scale),
                rotation_action_scale=float(args.rotation_action_scale),
                cluster_cutoff=float(
                    args.paired_pose_ensemble_cluster_cutoff
                ),
                minimum_cluster_fraction=float(
                    args.paired_pose_ensemble_min_cluster_fraction
                ),
                selection_mode=str(
                    args.paired_pose_ensemble_selection_mode
                ),
                distance_scale_mode=str(
                    args.paired_pose_ensemble_distance_scale_mode
                ),
                support_members_mode=str(
                    args.paired_pose_ensemble_support_members_mode
                ),
                kde_position_bandwidth=float(
                    args.paired_pose_ensemble_kde_position_bandwidth
                ),
                kde_rotation_bandwidth=float(
                    args.paired_pose_ensemble_kde_rotation_bandwidth
                ),
                kde_gripper_bandwidth=float(
                    args.paired_pose_ensemble_kde_gripper_bandwidth
                ),
                kde_support_mass=float(
                    args.paired_pose_ensemble_kde_support_mass
                ),
                dense_core_fraction=(
                    None
                    if args.paired_pose_ensemble_dense_core_fraction is None
                    else float(
                        args.paired_pose_ensemble_dense_core_fraction
                    )
                ),
            )

        if args.mode == "action_bspline_direct":
            adapter_info = {
                **common_adapter_info,
                "representation": "action_bspline",
                "source_horizon": args.paired_bspline_source_horizon,
                "degree": args.paired_bspline_degree,
                "execution_steps": args.paired_plan_steps,
                "geometry_command_gain": (
                    args.geometry_direct_gain
                    if args.mode == "paired_bspline_geometry_direct"
                    else None
                ),
            }
            policy = LiberoActionBSplineDirectAdapter(
                policy,
                source_horizon=int(args.paired_bspline_source_horizon),
                degree=int(args.paired_bspline_degree),
                execution_steps=int(args.paired_plan_steps),
            )
        elif args.mode in {
            "paired_bspline_direct",
            "paired_bspline_geometry_direct",
        }:
            if int(args.paired_pose_ensemble_samples) > 1:
                policy = LiberoPairedBSplinePoseMedoidPolicy(
                    policy,
                    samples=int(args.paired_pose_ensemble_samples),
                    source_horizon=int(
                        args.paired_bspline_source_horizon
                    ),
                    selection_steps=int(args.paired_plan_steps),
                    degree=int(args.paired_bspline_degree),
                    support_subdivisions=int(
                        args.paired_bspline_support_subdivisions
                    ),
                    position_action_scale=float(
                        args.position_action_scale
                    ),
                    rotation_action_scale=float(
                        args.rotation_action_scale
                    ),
                    cluster_cutoff=float(
                        args.paired_pose_ensemble_cluster_cutoff
                    ),
                    minimum_cluster_fraction=float(
                        args.paired_pose_ensemble_min_cluster_fraction
                    ),
                    selection_mode=str(
                        args.paired_pose_ensemble_selection_mode
                    ),
                    distance_scale_mode=str(
                        args.paired_pose_ensemble_distance_scale_mode
                    ),
                    support_members_mode=str(
                        args.paired_pose_ensemble_support_members_mode
                    ),
                    kde_position_bandwidth=float(
                        args.paired_pose_ensemble_kde_position_bandwidth
                    ),
                    kde_rotation_bandwidth=float(
                        args.paired_pose_ensemble_kde_rotation_bandwidth
                    ),
                    kde_gripper_bandwidth=float(
                        args.paired_pose_ensemble_kde_gripper_bandwidth
                    ),
                    kde_support_mass=float(
                        args.paired_pose_ensemble_kde_support_mass
                    ),
                    dense_core_fraction=(
                        None
                        if args.paired_pose_ensemble_dense_core_fraction is None
                        else float(
                            args.paired_pose_ensemble_dense_core_fraction
                        )
                    ),
                )
            adapter_info = {
                **common_adapter_info,
                "representation": (
                    "paired_bspline_geometry_direct"
                    if args.mode == "paired_bspline_geometry_direct"
                    else "paired_bspline"
                ),
                "source_horizon": args.paired_bspline_source_horizon,
                "degree": args.paired_bspline_degree,
                "execution_steps": args.paired_plan_steps,
            }
            direct_adapter = (
                LiberoPairedBSplineGeometryDirectAdapter
                if args.mode == "paired_bspline_geometry_direct"
                else LiberoPairedBSplineDirectAdapter
            )
            direct_kwargs = dict(
                base_policy=policy,
                source_horizon=int(args.paired_bspline_source_horizon),
                degree=int(args.paired_bspline_degree),
                execution_steps=int(args.paired_plan_steps),
            )
            if args.mode == "paired_bspline_geometry_direct":
                direct_kwargs["geometry_command_gain"] = float(
                    args.geometry_direct_gain
                )
            policy = direct_adapter(**direct_kwargs)
        else:
            adapter_info = {
                **common_adapter_info,
                "representation": "paired_sampled_curve",
                "pose_representation": (
                    "measured_native_timestep_increments"
                ),
                "action_representation": "exact_native_steps",
                "execution_steps": args.paired_plan_steps,
            }
            policy = LiberoPairedPoseActionDirectAdapter(
                policy,
                execution_steps=int(args.paired_plan_steps),
            )
    elif args.mode in {
        "command_center_field",
        "paired_pose_action_field",
        "paired_bspline_field",
        "paired_bspline_pose_field",
        "paired_bspline_hybrid_ds",
        "paired_pose_action_hybrid_ds",
        "paired_bspline_integrated_field",
    }:
        from ns_gaussian_ds_policy.libero_command_center import (
            LiberoCommandCenterFieldAdapter,
        )
        from ns_gaussian_ds_policy.libero_paired_curve import (
            LiberoPairedBSplineFieldAdapter,
            LiberoPairedBSplineIntegratedFieldAdapter,
            LiberoPairedBSplinePoseMedoidPolicy,
            LiberoPairedBSplinePoseFieldAdapter,
            LiberoPairedPoseActionMedoidPolicy,
            LiberoPairedPoseActionFieldAdapter,
        )
        from ns_gaussian_ds_policy.libero_hybrid_ds import (
            LiberoPairedBSplineHybridDSAdapter,
            LiberoPairedPoseActionHybridDSAdapter,
        )

        is_bspline = args.mode in {
            "paired_bspline_field",
            "paired_bspline_pose_field",
            "paired_bspline_hybrid_ds",
            "paired_bspline_integrated_field",
        }
        is_hybrid_bspline = args.mode == "paired_bspline_hybrid_ds"
        is_raw_pose_action_hybrid = (
            args.mode == "paired_pose_action_hybrid_ds"
        )
        is_hybrid = is_hybrid_bspline or is_raw_pose_action_hybrid
        is_pose_only_bspline = args.mode == "paired_bspline_pose_field"
        is_integrated_bspline = (
            args.mode == "paired_bspline_integrated_field"
        )
        is_command_center = args.mode == "command_center_field"
        if is_command_center:
            representation = "action_integrated_command_center"
        elif is_raw_pose_action_hybrid:
            representation = "paired_pose_action_hybrid_ds"
        elif is_integrated_bspline:
            representation = "paired_bspline_action_integrated"
        elif is_hybrid_bspline:
            representation = "paired_bspline_hybrid_ds"
        elif is_bspline:
            representation = "paired_bspline"
        else:
            representation = "paired_sampled_curve"
        adapter_info = {
            **common_adapter_info,
            "representation": representation,
            "is_bspline": is_bspline,
            "plan_steps": args.paired_plan_steps,
            "position_action_scale": args.position_action_scale,
            "rotation_action_scale": args.rotation_action_scale,
            "position_tolerance": args.position_tolerance,
            "rotation_tolerance": args.rotation_tolerance,
            "max_plan_steps": args.max_plan_steps,
            "gripper_only_position_threshold": (
                args.gripper_only_position_threshold
            ),
            "gripper_only_rotation_threshold": (
                args.gripper_only_rotation_threshold
            ),
            "gating_precision_scale": args.gating_precision_scale,
            "position_recovery_gain": args.position_recovery_gain,
            "rotation_recovery_gain": args.rotation_recovery_gain,
            "support_interpolation_rcond": args.support_interpolation_rcond,
            "tube_projection_enabled": args.tube_projection_enabled,
            "tube_radius": args.tube_radius,
            "phase_conditioned_execution": (
                args.phase_conditioned_execution
            ),
            "phase_position_tube_radius": (
                args.phase_position_tube_radius
            ),
            "phase_rotation_tube_radius": (
                args.phase_rotation_tube_radius
            ),
            "phase_position_reentry_radius": (
                args.phase_position_reentry_radius
            ),
            "phase_rotation_reentry_radius": (
                args.phase_rotation_reentry_radius
            ),
            "phase_projection_lookahead": (
                args.phase_projection_lookahead
            ),
            "recovery_phase_advance_budget": (
                args.recovery_phase_advance_budget
            ),
            "phase_max_recovery_steps": (
                args.phase_max_recovery_steps
            ),
            "hybrid_recovery_field_source": (
                args.hybrid_recovery_field_source
            ),
            "hybrid_recovery_target_mode": (
                args.hybrid_recovery_target_mode
            ),
            "hybrid_support_confidence_outer_score": (
                args.hybrid_support_confidence_outer_score
            ),
            "hybrid_support_confidence_inner_score": (
                args.hybrid_support_confidence_inner_score
            ),
            "hybrid_recovery_trigger_mode": (
                args.hybrid_recovery_trigger_mode
            ),
            "hybrid_require_state_innovation": args.hybrid_require_state_innovation,
            "hybrid_nominal_fusion_mode": (
                args.hybrid_nominal_fusion_mode
            ),
            "hybrid_transverse_fusion_inner_ratio": (
                args.hybrid_transverse_fusion_inner_ratio
            ),
            "hybrid_transverse_fusion_outer_ratio": (
                args.hybrid_transverse_fusion_outer_ratio
            ),
            "hybrid_transverse_fusion_max_weight": (
                args.hybrid_transverse_fusion_max_weight
            ),
            "hybrid_disturbance_position_innovation_threshold": (
                args.hybrid_disturbance_position_innovation_threshold
            ),
            "hybrid_disturbance_rotation_innovation_threshold": (
                args.hybrid_disturbance_rotation_innovation_threshold
            ),
            "hybrid_disturbance_expected_motion_multiplier": (
                args.hybrid_disturbance_expected_motion_multiplier
            ),
            "hybrid_freeze_recovery_phase": (
                args.hybrid_freeze_recovery_phase
            ),
            "hybrid_autonomous_chunk_reentry": (
                args.hybrid_autonomous_chunk_reentry
            ),
            "hybrid_replan_on_recovery_reentry": (
                args.hybrid_replan_on_recovery_reentry
            ),
            "hybrid_rotation_clf_enabled": (
                args.hybrid_rotation_clf_enabled
            ),
            "hybrid_rotation_clf_radius": (
                args.hybrid_rotation_clf_radius
            ),
            "hybrid_rotation_clf_contraction_rate": (
                args.hybrid_rotation_clf_contraction_rate
            ),
            "hybrid_recovery_progress_patience": (
                args.hybrid_recovery_progress_patience
            ),
            "hybrid_recovery_progress_min_delta": (
                args.hybrid_recovery_progress_min_delta
            ),
            "hybrid_field_speed_gain": args.hybrid_field_speed_gain,
            "tube_contraction_rate": args.tube_contraction_rate,
            "preserve_field_direction_on_saturation": (
                args.preserve_field_direction_on_saturation
            ),
            "paired_pose_ensemble_samples": (
                args.paired_pose_ensemble_samples
            ),
            "paired_pose_ensemble_adaptive_support": (
                args.paired_pose_ensemble_adaptive_support
            ),
            "paired_pose_ensemble_adaptive_tube": (
                args.paired_pose_ensemble_adaptive_tube
            ),
        }
        if is_pose_only_bspline:
            adapter_info.update(
                {
                    "representation": "paired_bspline_pose_only_field",
                    "phase_conditioned_execution": False,
                    "arm_action_bspline_used": False,
                    "shared_gripper_schedule_used": True,
                }
            )
        if is_hybrid:
            if (
                bool(args.paired_pose_ensemble_adaptive_support)
                or bool(args.paired_pose_ensemble_adaptive_tube)
            ) and int(args.paired_pose_ensemble_samples) <= 1:
                raise ValueError(
                    "Adaptive pose support requires "
                    "--paired-pose-ensemble-samples > 1"
                )
            if int(args.paired_pose_ensemble_samples) > 1:
                medoid_policy = (
                    LiberoPairedPoseActionMedoidPolicy
                    if is_raw_pose_action_hybrid
                    else LiberoPairedBSplinePoseMedoidPolicy
                )
                policy = medoid_policy(
                    policy,
                    samples=int(args.paired_pose_ensemble_samples),
                    source_horizon=int(
                        args.paired_bspline_source_horizon
                    ),
                    selection_steps=int(args.paired_plan_steps),
                    degree=int(args.paired_bspline_degree),
                    support_subdivisions=(
                        1
                        if is_raw_pose_action_hybrid
                        else int(args.paired_bspline_support_subdivisions)
                    ),
                    position_action_scale=float(
                        args.position_action_scale
                    ),
                    rotation_action_scale=float(
                        args.rotation_action_scale
                    ),
                    cluster_cutoff=float(
                        args.paired_pose_ensemble_cluster_cutoff
                    ),
                    minimum_cluster_fraction=float(
                        args.paired_pose_ensemble_min_cluster_fraction
                    ),
                    selection_mode=str(
                        args.paired_pose_ensemble_selection_mode
                    ),
                    distance_scale_mode=str(
                        args.paired_pose_ensemble_distance_scale_mode
                    ),
                    support_members_mode=str(
                        args.paired_pose_ensemble_support_members_mode
                    ),
                    kde_position_bandwidth=float(
                        args.paired_pose_ensemble_kde_position_bandwidth
                    ),
                    kde_rotation_bandwidth=float(
                        args.paired_pose_ensemble_kde_rotation_bandwidth
                    ),
                    kde_gripper_bandwidth=float(
                        args.paired_pose_ensemble_kde_gripper_bandwidth
                    ),
                    kde_support_mass=float(
                        args.paired_pose_ensemble_kde_support_mass
                    ),
                    dense_core_fraction=(
                        None
                        if args.paired_pose_ensemble_dense_core_fraction is None
                        else float(
                            args.paired_pose_ensemble_dense_core_fraction
                        )
                    ),
                )
            adapter_class = (
                LiberoPairedPoseActionHybridDSAdapter
                if is_raw_pose_action_hybrid
                else LiberoPairedBSplineHybridDSAdapter
            )
        elif is_pose_only_bspline:
            adapter_class = LiberoPairedBSplinePoseFieldAdapter
        elif is_integrated_bspline:
            adapter_class = LiberoPairedBSplineIntegratedFieldAdapter
        elif is_bspline:
            adapter_class = LiberoPairedBSplineFieldAdapter
        elif is_command_center:
            adapter_class = LiberoCommandCenterFieldAdapter
        else:
            adapter_class = LiberoPairedPoseActionFieldAdapter
        adapter_kwargs = dict(
            base_policy=policy,
            plan_steps=int(args.paired_plan_steps),
            position_action_scale=float(args.position_action_scale),
            rotation_action_scale=float(args.rotation_action_scale),
            position_tolerance=float(args.position_tolerance),
            rotation_tolerance=float(args.rotation_tolerance),
            max_position_action=float(args.max_xyz_action),
            max_rotation_action=float(args.max_rot_action),
            max_plan_steps=int(args.max_plan_steps),
            gripper_only_position_threshold=float(
                args.gripper_only_position_threshold
            ),
            gripper_only_rotation_threshold=float(
                args.gripper_only_rotation_threshold
            ),
            gripper_close_command_threshold=float(
                args.gripper_close_command_threshold
            ),
            gating_precision_scale=float(args.gating_precision_scale),
            position_recovery_gain=float(args.position_recovery_gain),
            rotation_recovery_gain=float(args.rotation_recovery_gain),
            support_interpolation_rcond=float(
                args.support_interpolation_rcond
            ),
            min_sigma=float(args.center_field_min_sigma),
            tube_projection_enabled=bool(args.tube_projection_enabled),
            tube_radius=float(args.tube_radius),
            phase_conditioned_execution=bool(
                args.phase_conditioned_execution
            ),
            phase_position_tube_radius=float(
                args.phase_position_tube_radius
            ),
            phase_rotation_tube_radius=float(
                args.phase_rotation_tube_radius
            ),
            tube_contraction_rate=float(args.tube_contraction_rate),
            preserve_field_direction_on_saturation=bool(
                args.preserve_field_direction_on_saturation
            ),
        )
        if is_pose_only_bspline:
            adapter_kwargs["phase_conditioned_execution"] = False
        if is_bspline:
            adapter_info.update(
                {
                    "source_horizon": args.paired_bspline_source_horizon,
                    "degree": args.paired_bspline_degree,
                    "support_subdivisions": (
                        args.paired_bspline_support_subdivisions
                    ),
                }
            )
            adapter_kwargs.update(
                source_horizon=int(args.paired_bspline_source_horizon),
                degree=int(args.paired_bspline_degree),
                support_subdivisions=int(
                    args.paired_bspline_support_subdivisions
                ),
            )
        elif is_raw_pose_action_hybrid:
            adapter_info.update(
                {
                    "source_horizon": args.paired_bspline_source_horizon,
                    "support_subdivisions": 1,
                    "pose_representation": (
                        "measured_native_timestep_increments"
                    ),
                    "action_representation": "exact_native_steps",
                }
            )
            adapter_kwargs.update(
                source_horizon=int(args.paired_bspline_source_horizon),
                support_subdivisions=1,
            )
        if is_hybrid:
            adapter_kwargs.update(
                phase_position_reentry_radius=float(
                    args.phase_position_reentry_radius
                ),
                phase_rotation_reentry_radius=float(
                    args.phase_rotation_reentry_radius
                ),
                phase_projection_lookahead=(
                    None
                    if args.phase_projection_lookahead is None
                    else float(args.phase_projection_lookahead)
                ),
                recovery_phase_advance_budget=(
                    None
                    if args.recovery_phase_advance_budget is None
                    else float(args.recovery_phase_advance_budget)
                ),
                phase_max_recovery_steps=int(
                    args.phase_max_recovery_steps
                ),
                recovery_field_source=(
                    args.hybrid_recovery_field_source
                ),
                recovery_target_mode=(
                    args.hybrid_recovery_target_mode
                ),
                recovery_anchor_mode=(
                    args.hybrid_recovery_anchor_mode
                ),
                observed_anchor_reentry_radius=float(
                    args.hybrid_observed_anchor_reentry_radius
                ),
                observed_anchor_stable_steps=int(
                    args.hybrid_observed_anchor_stable_steps
                ),
                freeze_recovery_phase=bool(
                    args.hybrid_freeze_recovery_phase
                ),
                autonomous_chunk_reentry=bool(
                    args.hybrid_autonomous_chunk_reentry
                ),
                replan_on_recovery_reentry=bool(
                    args.hybrid_replan_on_recovery_reentry
                ),
                rotation_clf_enabled=bool(
                    args.hybrid_rotation_clf_enabled
                ),
                rotation_clf_tube_radius=float(
                    args.hybrid_rotation_clf_radius
                ),
                rotation_clf_contraction_rate=(
                    None
                    if args.hybrid_rotation_clf_contraction_rate is None
                    else float(
                        args.hybrid_rotation_clf_contraction_rate
                    )
                ),
                recovery_progress_patience=int(
                    args.hybrid_recovery_progress_patience
                ),
                recovery_progress_min_delta=float(
                    args.hybrid_recovery_progress_min_delta
                ),
                field_speed_gain=float(args.hybrid_field_speed_gain),
                support_confidence_outer_score=float(
                    args.hybrid_support_confidence_outer_score
                ),
                support_confidence_inner_score=float(
                    args.hybrid_support_confidence_inner_score
                ),
                recovery_trigger_mode=(
                    args.hybrid_recovery_trigger_mode
                ),
                require_state_innovation=args.hybrid_require_state_innovation,
                nominal_fusion_mode=(
                    args.hybrid_nominal_fusion_mode
                ),
                transverse_fusion_inner_ratio=float(
                    args.hybrid_transverse_fusion_inner_ratio
                ),
                transverse_fusion_outer_ratio=float(
                    args.hybrid_transverse_fusion_outer_ratio
                ),
                transverse_fusion_max_weight=float(
                    args.hybrid_transverse_fusion_max_weight
                ),
                disturbance_position_innovation_threshold=float(
                    args.hybrid_disturbance_position_innovation_threshold
                ),
                disturbance_rotation_innovation_threshold=float(
                    args.hybrid_disturbance_rotation_innovation_threshold
                ),
                disturbance_expected_motion_multiplier=float(
                    args.hybrid_disturbance_expected_motion_multiplier
                ),
                disturbance_response_mode=(
                    args.hybrid_disturbance_response_mode
                ),
                adaptive_pose_support_enabled=bool(
                    args.paired_pose_ensemble_adaptive_support
                ),
                adaptive_pose_tube_enabled=bool(
                    args.paired_pose_ensemble_adaptive_tube
                ),
                pose_support_confidence=float(
                    args.paired_pose_ensemble_support_confidence
                ),
                pose_reentry_fraction=float(
                    args.paired_pose_ensemble_reentry_fraction
                ),
                pose_dense_core_confidence=(
                    None
                    if args.paired_pose_ensemble_dense_core_fraction is None
                    else float(
                        args.paired_pose_ensemble_dense_core_confidence
                    )
                ),
            )
        policy = adapter_class(**adapter_kwargs)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if args.recovery_controller != "none":
        if args.mode not in {"paired_bspline_direct", "paired_pose_action_direct"}:
            raise ValueError("Geometric recovery requires a paired direct decoder")
        from ns_gaussian_ds_policy.recovery_baselines import (
            PairedGeometricRecoveryPolicy, RecoveryConfig, recovery_kwargs,
        )
        policy = PairedGeometricRecoveryPolicy(
            policy, args.recovery_controller, RecoveryConfig(**recovery_kwargs(args))
        )
        adapter_info["recovery_controller"] = args.recovery_controller
        adapter_info["recovery_config"] = recovery_kwargs(args)
    use_direct_stepper = bool(args.direct_stepwise_execution) or (
        float(args.direct_perturbation_radius_cm) > 0.0
    )
    if use_direct_stepper:
        if args.mode not in {"raw_action", "action_bspline_direct"}:
            raise ValueError(
                "Stepwise direct execution only supports raw_action and "
                "action_bspline_direct"
            )
        if int(args.n_parallel_envs) != 1:
            raise ValueError(
                "Stepwise direct execution requires --n-parallel-envs 1"
            )
        from ns_gaussian_ds_policy.libero_direct_perturbation import (
            LiberoDirectChunkPerturbationAdapter,
        )

        policy = LiberoDirectChunkPerturbationAdapter(
            policy,
            radius_m=float(args.direct_perturbation_radius_cm) / 100.0,
            trigger_phase=int(args.direct_perturbation_trigger_phase),
            direction_sign=float(
                args.direct_perturbation_direction_sign
            ),
            position_action_scale=float(args.position_action_scale),
            max_perturbation_steps=int(
                args.direct_perturbation_max_steps
            ),
            reach_fraction=float(
                args.direct_perturbation_reach_fraction
            ),
            gripper_event_guard=int(
                args.direct_perturbation_gripper_event_guard
            ),
            require_future_gripper_event=bool(
                args.direct_perturbation_require_future_gripper_event
            ),
        )
        adapter_info["direct_stepwise_execution"] = True
        adapter_info["controlled_perturbation"] = {
            "radius_cm": float(args.direct_perturbation_radius_cm),
            "trigger_phase": int(args.direct_perturbation_trigger_phase),
            "direction_sign": float(
                args.direct_perturbation_direction_sign
            ),
        }

    if float(args.hybrid_perturbation_radius_cm) > 0.0:
        if args.mode not in {
            "paired_bspline_hybrid_ds",
            "paired_pose_action_hybrid_ds",
        }:
            raise ValueError(
                "Controlled perturbations require a paired hybrid DS mode"
            )
        if int(args.n_parallel_envs) != 1:
            raise ValueError(
                "Controlled perturbations require --n-parallel-envs 1"
            )
        from ns_gaussian_ds_policy.libero_hybrid_perturbation import (
            LiberoHybridPerturbationAdapter,
        )

        policy = LiberoHybridPerturbationAdapter(
            policy,
            radius_m=float(args.hybrid_perturbation_radius_cm) / 100.0,
            trigger_phase=float(args.hybrid_perturbation_trigger_phase),
            direction_sign=float(args.hybrid_perturbation_direction_sign),
            position_action_scale=float(args.position_action_scale),
            max_perturbation_steps=int(
                args.hybrid_perturbation_max_steps
            ),
            reach_fraction=float(
                args.hybrid_perturbation_reach_fraction
            ),
            gripper_event_guard=float(
                args.hybrid_perturbation_gripper_event_guard
            ),
            require_future_gripper_event=bool(
                args.hybrid_perturbation_require_future_gripper_event
            ),
        )
        adapter_info["controlled_perturbation"] = {
            "radius_cm": float(args.hybrid_perturbation_radius_cm),
            "trigger_phase": float(
                args.hybrid_perturbation_trigger_phase
            ),
            "direction_sign": float(
                args.hybrid_perturbation_direction_sign
            ),
        }

    print(
        f"Rollout adapter {args.mode}: "
        + json.dumps(adapter_info, indent=2),
        flush=True,
    )

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    if args.controller_debug_videos:
        enable_diagnostics = getattr(
            policy,
            "enable_controller_diagnostics",
            None,
        )
        if not callable(enable_diagnostics):
            raise ValueError(
                "--controller-debug-videos requires "
                "--mode paired_bspline_hybrid_ds"
            )
        enable_diagnostics()

    kwargs = {}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.topk is not None:
        kwargs["topk"] = args.topk
    if args.use_k_tokens is not None:
        kwargs["use_k_tokens"] = args.use_k_tokens

    test_seeds = None
    if args.test_seeds is not None:
        test_seeds = [int(part.strip()) for part in str(args.test_seeds).split(",") if part.strip()]
        if len(test_seeds) != int(args.n_test):
            raise ValueError(f"--test-seeds has {len(test_seeds)} seeds but --n-test is {args.n_test}")

    env_runner: BaseRunner = hydra.utils.instantiate(
        cfg.task.policy.env_runner,
        output_dir=str(args.output_dir),
        n_test=int(args.n_test),
        n_test_vis=int(args.n_test_vis),
        n_parallel_envs=int(args.n_parallel_envs),
        max_episode_steps=int(args.max_episode_steps),
        **({} if args.task_name is None else {"task_name": str(args.task_name)}),
        **({} if args.test_start_seed is None else {"test_start_seed": int(args.test_start_seed)}),
        **({} if test_seeds is None else {"test_seeds": test_seeds}),
    )
    if args.require_episode_outcomes:
        _require_episode_outcome_capable_runner(env_runner)

    concurrent_perturbation_enabled = bool(
        float(args.concurrent_perturbation_distance_cm) > 0.0
    )
    concurrent_perturbation_runs: list[dict[str, Any]] = []
    concurrent_perturbation_collector = None
    if concurrent_perturbation_enabled:
        if args.mode not in {
            "raw_action",
            "action_bspline_direct",
            "paired_bspline_hybrid_ds",
            "paired_bspline_direct",
            "paired_pose_action_hybrid_ds",
            "paired_pose_action_direct",
        }:
            raise ValueError(
                "Concurrent perturbation evaluation requires "
                "--mode paired_bspline_hybrid_ds or "
                "--mode paired_bspline_direct, or the corresponding "
                "paired_pose_action mode"
            )
        if int(args.num_exp) != 1 or int(args.n_parallel_envs) != 1:
            raise ValueError(
                "Concurrent perturbation evaluation currently requires "
                "--num-exp 1 --n-parallel-envs 1"
            )
        if (
            float(args.hybrid_perturbation_radius_cm) > 0.0
            or float(args.direct_perturbation_radius_cm) > 0.0
        ):
            raise ValueError(
                "Do not combine policy-replacing and concurrent "
                "perturbation protocols"
            )

        import dill

        from ns_gaussian_ds_policy.libero_concurrent_perturbation import (
            ConcurrentPerturbationConfig,
            collect_concurrent_perturbation,
            install_concurrent_perturbation,
        )

        concurrent_config = ConcurrentPerturbationConfig(
            start_step=int(args.concurrent_perturbation_start_step),
            distance=float(args.concurrent_perturbation_distance_cm) / 100.0,
            direction=(
                float(args.concurrent_perturbation_direction_x),
                float(args.concurrent_perturbation_direction_y),
                float(args.concurrent_perturbation_direction_z),
            ),
            ramp_steps=int(args.concurrent_perturbation_ramp_steps),
            hold_steps=int(args.concurrent_perturbation_hold_steps),
            max_push_steps=int(args.concurrent_perturbation_max_steps),
            stiffness=float(args.concurrent_perturbation_stiffness),
            integral_gain=float(args.concurrent_perturbation_integral_gain),
            integral_error_limit=float(
                args.concurrent_perturbation_integral_error_limit
            ),
            damping=float(args.concurrent_perturbation_damping),
            max_force=float(args.concurrent_perturbation_max_force),
        )
        installer = functools.partial(
            install_concurrent_perturbation,
            config=concurrent_config,
        )
        install_results = env_runner.env.call_each(
            "run_dill_function",
            args_list=[(dill.dumps(installer),)],
        )
        concurrent_perturbation_collector = dill.dumps(
            collect_concurrent_perturbation
        )
        adapter_info["concurrent_environment_perturbation"] = _jsonable(
            install_results[0]
        )
        print(
            "[eval] installed concurrent environment perturbation "
            f"step={concurrent_config.start_step} "
            f"distance_cm={100.0 * concurrent_config.distance:.1f} "
            f"direction={concurrent_config.direction}",
            flush=True,
        )

    episode_sampling_seeds: list[int] = []
    if args.eval_rng_mode == "per_episode":
        episode_sampling_seeds = _install_per_episode_rng_reset(
            policy,
            eval_seed=args.eval_seed,
            n_parallel_envs=int(args.n_parallel_envs),
        )

    runner_log = None
    runs = []
    episode_outcomes: list[dict[str, Any]] = []
    controller_debug_videos = []
    try:
        for exp_idx in range(int(args.num_exp)):
            if (
                args.eval_seed is not None
                and args.eval_rng_mode == "per_experiment"
            ):
                _set_eval_seed(int(args.eval_seed) + exp_idx)
            sampling_seed_start = len(episode_sampling_seeds)
            this_log = env_runner.run(policy, **kwargs)
            print(f"Exp {exp_idx + 1}: success rate = {this_log['mean_success_rate']}", flush=True)
            if concurrent_perturbation_collector is not None:
                collected = env_runner.env.call_each(
                    "run_dill_function",
                    args_list=[(concurrent_perturbation_collector,)],
                )
                concurrent_perturbation_runs.append(_jsonable(collected[0]))
                records = collected[0].get("episode_records", [])
                if records:
                    record = records[-1]
                    reached_count = sum(
                        bool(item.get("target_reached", False))
                        for item in records
                    )
                    achieved = record["release_signed_displacement_m"]
                    if achieved is None:
                        achieved = record[
                            "max_signed_displacement_during_push_m"
                        ]
                    print(
                        "[eval] concurrent perturbation "
                        f"episodes={len(records)} "
                        f"targets_reached={reached_count} "
                        f"target_reached={record['target_reached']} "
                        "release_displacement_cm="
                        f"{100.0 * achieved:.2f} "
                        f"force_steps={record['force_steps']} "
                        "policy_actions_replaced="
                        f"{record['policy_actions_replaced']}",
                        flush=True,
                    )
            if args.controller_debug_videos:
                consume_traces = getattr(
                    policy,
                    "consume_controller_diagnostic_traces",
                    None,
                )
                if not callable(consume_traces):
                    raise ValueError(
                        "--controller-debug-videos requires "
                        "--mode paired_bspline_hybrid_ds"
                    )
                from ns_gaussian_ds_policy.controller_debug_video import (
                    render_libero_controller_debug_videos,
                )

                rendered = render_libero_controller_debug_videos(
                    output_dir=args.output_dir,
                    task_names=list(env_runner.env_task_names),
                    seeds=list(env_runner.env_seeds),
                    trace_chunks=consume_traces(),
                    n_test_vis=int(args.n_test_vis),
                    experiment_index=exp_idx,
                )
                controller_debug_videos.extend(rendered)
                print(
                    f"[controller-debug] rendered {len(rendered)} videos",
                    flush=True,
                )
            serial = _serializable_runner_log(this_log)
            runs.append(serial)
            run_sampling_seeds = (
                episode_sampling_seeds[sampling_seed_start:]
                if args.eval_rng_mode == "per_episode"
                else None
            )
            records = _build_episode_outcome_records(
                serial,
                task_names=[str(name) for name in env_runner.env_task_names],
                environment_seeds=[int(seed) for seed in env_runner.env_seeds],
                sampling_seeds=run_sampling_seeds,
                experiment_index=exp_idx,
            )
            if args.require_episode_outcomes and len(records) != int(args.n_test):
                raise RuntimeError(
                    "Formal evaluation requires one retained outcome per "
                    f"rollout; found {len(records)}/{int(args.n_test)}. "
                    "Run scripts/patch_oat_runtime.py before eval."
                )
            episode_outcomes.extend(records)
            if runner_log is None:
                runner_log = dict(serial)
            else:
                for key, value in serial.items():
                    if isinstance(value, (int, float, np.number)):
                        runner_log[key] = float(runner_log.get(key, 0.0)) + float(value)
    finally:
        env_runner.close()

    assert runner_log is not None
    if int(args.num_exp) > 1:
        for key, value in list(runner_log.items()):
            if isinstance(value, (int, float, np.number)):
                runner_log[key] = float(value) / float(args.num_exp)

    out = {
        "checkpoint": str(args.checkpoint),
        "tokenizer_checkpoint": str(args.tokenizer_checkpoint) if args.tokenizer_checkpoint is not None else None,
        "mode": args.mode,
        "num_exp": int(args.num_exp),
        "n_test": int(args.n_test),
        "n_test_vis": int(args.n_test_vis),
        "n_parallel_envs": int(args.n_parallel_envs),
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
        "command_line": " ".join(sys.argv),
        "runs": runs,
        "episode_outcome_schema": "libero_episode_outcomes_v1",
        "episode_outcomes": episode_outcomes,
        "adapter": adapter_info,
        "controller_debug_videos": controller_debug_videos,
        **runner_log,
    }
    if args.eval_rng_mode == "per_episode":
        out["episode_sampling_seeds"] = episode_sampling_seeds
    if concurrent_perturbation_runs:
        out["concurrent_environment_perturbation"] = (
            concurrent_perturbation_runs[0]
            if len(concurrent_perturbation_runs) == 1
            else concurrent_perturbation_runs
        )
    if len(runs) > 1:
        success_rate_keys = [
            key
            for key, value in runs[0].items()
            if isinstance(value, (int, float, np.number))
            and (
                key == "mean_success_rate"
                or key.endswith("/mean_success_rate")
            )
        ]
        for key in success_rate_keys:
            values = np.asarray(
                [float(run[key]) for run in runs],
                dtype=np.float64,
            )
            sample_std = float(np.std(values, ddof=1))
            out[f"{key}_std"] = sample_std
            out[f"{key}_stderr"] = sample_std / float(np.sqrt(len(values)))
    get_execution_metrics = getattr(policy, "get_execution_metrics", None)
    if callable(get_execution_metrics):
        out["executor"] = _jsonable(get_execution_metrics())
    (args.output_dir / "eval_log.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
