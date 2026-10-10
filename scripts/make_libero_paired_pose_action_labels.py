from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ns_gaussian_ds_policy.libero_paired_curve import (  # noqa: E402
    LIBERO_ACTION_DIM,
    PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    PAIRED_BSPLINE_DEFAULT_DEGREE,
    PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    PAIRED_POSE_ACTION_DIM,
    build_libero_action_bspline_labels,
    build_libero_paired_bspline_labels,
    build_libero_paired_pose_action_labels,
)

from make_libero_gaussian_ds_labels import (  # noqa: E402
    _data_group,
    _episode_ends,
)


DEFAULT_ACTION_KEY = "paired_pose_action13_s1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add aligned geometry6 + native_command6 + gripper1 labels to a "
            "converted LIBERO zarr dataset."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=(
            "paired_sampled_curve",
            "paired_bspline",
            "action_bspline",
        ),
        default="paired_sampled_curve",
    )
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--position-action-scale", type=float, default=0.05)
    parser.add_argument("--rotation-action-scale", type=float, default=0.5)
    parser.add_argument(
        "--source-horizon",
        type=int,
        default=PAIRED_BSPLINE_DEFAULT_SOURCE_HORIZON,
    )
    parser.add_argument(
        "--control-points",
        type=int,
        default=PAIRED_BSPLINE_DEFAULT_CONTROL_POINTS,
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=PAIRED_BSPLINE_DEFAULT_DEGREE,
    )
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=1.0e-4)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.array(
        name,
        data=values,
        shape=values.shape,
        chunks=(min(4096, len(values)),) + values.shape[1:],
        dtype=str(values.dtype),
        overwrite=True,
    )


def main() -> None:
    args = parse_args()
    if args.action_key is None:
        if args.representation == "paired_bspline":
            args.action_key = (
                "paired_bspline13_"
                f"cp{int(args.control_points)}_h{int(args.source_horizon)}"
            )
        elif args.representation == "action_bspline":
            args.action_key = (
                "action_bspline7_"
                f"cp{int(args.control_points)}_h{int(args.source_horizon)}"
            )
        else:
            args.action_key = DEFAULT_ACTION_KEY
    root = zarr.open(str(args.input), mode="r" if args.dry_run else "a")
    data = _data_group(root)
    episode_ends = _episode_ends(root)
    required = (
        ("action",)
        if args.representation == "action_bspline"
        else (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "action",
        )
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"LIBERO zarr is missing required arrays: {missing}")

    num_steps = int(data["action"].shape[0])
    if args.representation == "paired_bspline":
        expected_shape = (
            num_steps,
            int(args.control_points),
            PAIRED_POSE_ACTION_DIM,
        )
    elif args.representation == "action_bspline":
        expected_shape = (
            num_steps,
            int(args.control_points),
            LIBERO_ACTION_DIM,
        )
    else:
        expected_shape = (num_steps, PAIRED_POSE_ACTION_DIM)
    target_index_key = f"{args.action_key}_target_idx"
    if args.action_key in data and not args.overwrite and not args.dry_run:
        stored = data[args.action_key]
        if tuple(stored.shape) != expected_shape:
            raise RuntimeError(
                f"Existing {args.action_key} has shape {stored.shape}; "
                f"expected {expected_shape}. Pass --overwrite to replace it."
            )
        sampled_indices_ready = (
            args.representation in {"paired_bspline", "action_bspline"}
            or (
                target_index_key in data
                and tuple(data[target_index_key].shape) == (num_steps,)
            )
        )
        if sampled_indices_ready:
            summary_key = f"{args.action_key}_summary"
            stored_summary = data.attrs.get(summary_key)
            summary = (
                json.loads(stored_summary)
                if isinstance(stored_summary, str)
                else {
                    "dataset": str(args.input),
                    "action_key": str(args.action_key),
                    "shape": list(expected_shape),
                    "representation": str(args.representation),
                }
            )
            print(
                f"[paired-labels] {args.action_key} already exists with "
                f"shape={stored.shape}; skipping label construction",
                flush=True,
            )
            print(json.dumps(summary, indent=2), flush=True)
            return

    actions = data["action"][:]
    if args.representation == "paired_bspline":
        positions = data["robot0_eef_pos"][:]
        quaternions = data["robot0_eef_quat"][:]
        labels, stats = build_libero_paired_bspline_labels(
            positions,
            quaternions,
            actions,
            episode_ends,
            source_horizon=int(args.source_horizon),
            num_control_points=int(args.control_points),
            degree=int(args.degree),
            position_action_scale=float(args.position_action_scale),
            rotation_action_scale=float(args.rotation_action_scale),
            velocity_weight=float(args.velocity_weight),
            smoothness_weight=float(args.smoothness_weight),
        )
        target_indices = None
        summary = {
            "dataset": str(args.input),
            "action_key": str(args.action_key),
            "shape": list(labels.shape),
            "representation": "paired_bspline",
            "is_bspline": True,
            **stats,
        }
        if labels.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected paired B-spline label shape {labels.shape}"
            )
    elif args.representation == "action_bspline":
        labels, stats = build_libero_action_bspline_labels(
            actions,
            episode_ends,
            source_horizon=int(args.source_horizon),
            num_control_points=int(args.control_points),
            degree=int(args.degree),
            velocity_weight=float(args.velocity_weight),
            smoothness_weight=float(args.smoothness_weight),
        )
        target_indices = None
        summary = {
            "dataset": str(args.input),
            "action_key": str(args.action_key),
            "shape": list(labels.shape),
            "representation": "action_bspline",
            "is_bspline": True,
            **stats,
        }
        if labels.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected action B-spline label shape {labels.shape}"
            )
    else:
        positions = data["robot0_eef_pos"][:]
        quaternions = data["robot0_eef_quat"][:]
        labels, target_indices, stats = build_libero_paired_pose_action_labels(
            positions,
            quaternions,
            actions,
            episode_ends,
            position_action_scale=float(args.position_action_scale),
            rotation_action_scale=float(args.rotation_action_scale),
            clip=not bool(args.no_clip),
        )
        summary = {
            "dataset": str(args.input),
            "action_key": str(args.action_key),
            "shape": list(labels.shape),
            "layout": "geometry_delta6 + native_command6 + gripper1",
            "representation": "paired_sampled_curve",
            "is_bspline": False,
            **stats,
        }
        if labels.shape[1] != PAIRED_POSE_ACTION_DIM:
            raise RuntimeError(f"Unexpected paired label shape {labels.shape}")
    if args.dry_run:
        print(json.dumps(summary, indent=2), flush=True)
        return
    if args.action_key in data and not args.overwrite:
        stored = data[args.action_key]
        if tuple(stored.shape) != tuple(labels.shape):
            raise RuntimeError(
                f"Existing {args.action_key} has shape {stored.shape}; "
                f"expected {labels.shape}. Pass --overwrite to replace it."
            )
        print(
            f"[paired-labels] {args.action_key} already exists with "
            f"shape={stored.shape}; keeping it",
            flush=True,
        )
        if target_indices is not None and target_index_key not in data:
            _write_array(data, target_index_key, target_indices)
            data.attrs[f"{args.action_key}_summary"] = json.dumps(
                summary,
                sort_keys=True,
            )
            print(
                f"[paired-labels] repaired missing {target_index_key}",
                flush=True,
            )
    else:
        _write_array(data, str(args.action_key), labels)
        if target_indices is not None:
            _write_array(data, target_index_key, target_indices)
        data.attrs[f"{args.action_key}_summary"] = json.dumps(
            summary,
            sort_keys=True,
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
