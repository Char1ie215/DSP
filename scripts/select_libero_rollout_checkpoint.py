from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


CHECKPOINT_PATTERN = re.compile(
    r"^ep-(?P<epoch>\d+)_vl-(?P<loss>[0-9.eE+-]+)\.ckpt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a LIBERO paired-curve checkpoint on held-out rollouts."
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", required=True)
    parser.add_argument("--expected-method-profile", default=None)
    parser.add_argument("--expected-eval-rng-mode", default=None)
    parser.add_argument("--expected-eval-seed", type=int, default=None)
    parser.add_argument("--expected-n-parallel-envs", type=int, default=None)
    parser.add_argument("--expected-start-seed", type=int, default=200000)
    parser.add_argument("--expected-rollouts", type=int, default=500)
    parser.add_argument("--expected-candidates", type=int, default=6)
    return parser.parse_args()


def _checkpoint_metadata(checkpoint: Path) -> tuple[int, float]:
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError(f"Cannot recover epoch and val_loss from {checkpoint}")
    return int(match.group("epoch")), float(match.group("loss"))


def _successes(summary: dict[str, object], expected_rollouts: int) -> int:
    outcomes = summary.get("episode_outcomes")
    if isinstance(outcomes, list) and len(outcomes) == expected_rollouts:
        if all(isinstance(item, dict) and "success" in item for item in outcomes):
            return sum(bool(item["success"]) for item in outcomes)
    runs = summary.get("runs")
    if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict):
        values = [
            value
            for key, value in runs[0].items()
            if "/success_" in str(key) and isinstance(value, (int, float))
        ]
        if len(values) == expected_rollouts:
            return int(round(sum(float(value) for value in values)))
    return int(round(float(summary["mean_success_rate"]) * expected_rollouts))


def select_checkpoint(
    evaluation_root: Path,
    run_dir: Path,
    *,
    expected_mode: str,
    expected_method_profile: str | None = None,
    expected_eval_rng_mode: str | None = None,
    expected_eval_seed: int | None = None,
    expected_n_parallel_envs: int | None = None,
    expected_start_seed: int,
    expected_rollouts: int,
    expected_candidates: int,
) -> dict[str, object]:
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    candidates = []
    seen_checkpoints: set[Path] = set()
    for result_path in sorted(evaluation_root.rglob("eval_log.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("mode") != expected_mode:
            continue
        if int(result["n_test"]) != expected_rollouts:
            raise ValueError(
                f"Unexpected rollout count in {result_path}: {result['n_test']}"
            )
        if int(result["num_exp"]) != 1:
            raise ValueError(f"Selection requires one run per candidate: {result_path}")
        result_args = result.get("args", {})
        if not isinstance(result_args, dict):
            raise ValueError(f"Missing evaluator args in {result_path}")
        if (
            expected_method_profile is not None
            and result_args.get("method_profile") != expected_method_profile
        ):
            raise ValueError(
                f"Unexpected method profile in {result_path}: "
                f"{result_args.get('method_profile')}"
            )
        if (
            expected_eval_rng_mode is not None
            and result_args.get("eval_rng_mode") != expected_eval_rng_mode
        ):
            raise ValueError(
                f"Unexpected RNG mode in {result_path}: "
                f"{result_args.get('eval_rng_mode')}"
            )
        if (
            expected_eval_seed is not None
            and int(result_args.get("eval_seed", -1)) != expected_eval_seed
        ):
            raise ValueError(
                f"Unexpected evaluation seed in {result_path}: "
                f"{result_args.get('eval_seed')}"
            )
        if (
            expected_n_parallel_envs is not None
            and int(result.get("n_parallel_envs", -1))
            != expected_n_parallel_envs
        ):
            raise ValueError(
                f"Unexpected parallel environment count in {result_path}: "
                f"{result.get('n_parallel_envs')}"
            )
        if int(result_args.get("test_start_seed", -1)) != expected_start_seed:
            raise ValueError(
                f"Unexpected validation start seed in {result_path}: "
                f"{result_args.get('test_start_seed')}"
            )
        if expected_eval_rng_mode == "per_episode":
            if expected_eval_seed is None:
                raise ValueError(
                    "Per-episode checkpoint selection requires an expected eval seed"
                )
            outcomes = result.get("episode_outcomes")
            if not isinstance(outcomes, list) or len(outcomes) != expected_rollouts:
                raise ValueError(
                    f"Expected {expected_rollouts} episode outcomes in {result_path}"
                )
            expected_sampling_seeds = list(
                range(
                    expected_eval_seed,
                    expected_eval_seed + expected_rollouts,
                )
            )
            if result.get("episode_sampling_seeds") != expected_sampling_seeds:
                raise ValueError(
                    f"Unexpected per-episode sampling seeds in {result_path}"
                )

        checkpoint = Path(result["checkpoint"]).resolve()
        if checkpoint.parent != checkpoint_dir:
            raise ValueError(f"Checkpoint is outside {checkpoint_dir}: {checkpoint}")
        if checkpoint in seen_checkpoints:
            raise ValueError(f"Duplicate checkpoint evaluation: {checkpoint}")
        seen_checkpoints.add(checkpoint)
        epoch, val_loss = _checkpoint_metadata(checkpoint)
        successes = _successes(result, expected_rollouts)
        candidates.append(
            {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "val_loss": val_loss,
                "successes": successes,
                "rollouts": expected_rollouts,
                "success_rate": successes / expected_rollouts,
                "result": str(result_path.resolve()),
            }
        )

    if len(candidates) != expected_candidates:
        raise ValueError(
            f"Expected {expected_candidates} candidate results, found "
            f"{len(candidates)} under {evaluation_root}"
        )
    selected = min(
        candidates,
        key=lambda item: (
            -int(item["successes"]),
            float(item["val_loss"]),
            int(item["epoch"]),
        ),
    )
    return {
        "selection_protocol": {
            "primary": "held_out_rollout_successes_max",
            "secondary": "offline_val_loss_min",
            "tertiary": "epoch_min",
            "mode": expected_mode,
            "method_profile": expected_method_profile,
            "eval_rng_mode": expected_eval_rng_mode,
            "eval_seed": expected_eval_seed,
            "n_parallel_envs": expected_n_parallel_envs,
            "start_seed": expected_start_seed,
            "rollouts_per_checkpoint": expected_rollouts,
            "candidate_count": expected_candidates,
            "final_test_seeds_used": False,
        },
        "selected": selected,
        "candidates": sorted(candidates, key=lambda item: int(item["epoch"])),
    }


def main() -> None:
    args = parse_args()
    evaluation_root = args.evaluation_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    report = select_checkpoint(
        evaluation_root,
        run_dir,
        expected_mode=args.expected_mode,
        expected_method_profile=args.expected_method_profile,
        expected_eval_rng_mode=args.expected_eval_rng_mode,
        expected_eval_seed=args.expected_eval_seed,
        expected_n_parallel_envs=args.expected_n_parallel_envs,
        expected_start_seed=args.expected_start_seed,
        expected_rollouts=args.expected_rollouts,
        expected_candidates=args.expected_candidates,
    )

    checkpoint_dir = run_dir / "checkpoints"
    destination = checkpoint_dir / "best_rollout.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    link_path = checkpoint_dir / "best_rollout.ckpt"
    link_path.unlink(missing_ok=True)
    selected = Path(str(report["selected"]["checkpoint"]))
    link_path.symlink_to(selected.name)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[libero-select] wrote {destination}", flush=True)
    print(f"[libero-select] linked {link_path} -> {selected.name}", flush=True)


if __name__ == "__main__":
    main()
