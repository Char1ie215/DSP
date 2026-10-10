from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr


QUERY_DIM = 16
STRUCTURED_ACTION_KEY = "gaussian_ds_action"
STRUCTURED_ACTION_DIM = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a LIBERO zarr copy with a task-agnostic Gaussian-DS supervised action. "
            "The added action is [query16, subgoal_delta3, speed1, rot_tail3, gripper1]."
        )
    )
    parser.add_argument(
        "--action-key",
        default=STRUCTURED_ACTION_KEY,
        help="Zarr data key to write the 24D structured action into.",
    )
    parser.add_argument("--input", type=Path, default=Path("data/libero/libero10_N500.zarr"))
    parser.add_argument("--output", type=Path, default=Path("data/libero/libero10_N500_gaussian_ds.zarr"))
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument(
        "--waypoint-mode",
        choices=["fixed_lookahead", "segment_dense", "segment_query_fixed_waypoint"],
        default="fixed_lookahead",
        help=(
            "fixed_lookahead uses t+H EEF as target. segment_dense first extracts "
            "trajectory keypoints and uses min(t+H, next_keypoint) as a dense local attractor. "
            "segment_query_fixed_waypoint uses segment-level query labels but keeps the dense "
            "fixed-lookahead waypoint target."
        ),
    )
    parser.add_argument("--max-segment-len", type=int, default=32)
    parser.add_argument("--min-segment-len", type=int, default=4)
    parser.add_argument("--curvature-threshold", type=float, default=0.45)
    parser.add_argument("--gripper-change-threshold", type=float, default=0.5)
    parser.add_argument("--speed-min-quantile", type=float, default=0.20)
    parser.add_argument(
        "--tail-mode",
        choices=["quat_delta", "raw_action"],
        default="quat_delta",
        help=(
            "quat_delta stores eef_quat[target,1:4] - eef_quat[t,1:4] in dims 20:23. "
            "raw_action stores the original LIBERO rotation action action[:,3:6] there, "
            "leaving rotation/gripper semantics aligned with OAT raw actions."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help=(
            "If output already exists, do not delete or recopy the source zarr; "
            "just add/replace data/gaussian_ds_action in that existing output."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Add/replace data/gaussian_ds_action directly inside --input and ignore --output.",
    )
    return parser.parse_args()


def _episode_ends(root: zarr.Group) -> np.ndarray:
    if "meta" in root and "episode_ends" in root["meta"]:
        return np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if "episode_ends" in root:
        return np.asarray(root["episode_ends"][:], dtype=np.int64)
    raise KeyError("Could not find episode_ends in zarr root/meta")


def _data_group(root: zarr.Group) -> zarr.Group:
    return root["data"] if "data" in root else root


def _safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.maximum(np.linalg.norm(x, axis=axis, keepdims=keepdims), 1e-8)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / _safe_norm(a, axis=-1, keepdims=True)
    b_norm = b / _safe_norm(b, axis=-1, keepdims=True)
    cos = np.sum(a_norm * b_norm, axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0)).astype(np.float32)


def _add_keypoint(keypoints: set[int], idx: int, n: int) -> None:
    if 0 <= int(idx) < int(n):
        keypoints.add(int(idx))


def _filter_keypoints(candidates: set[int], n: int, min_segment_len: int) -> list[int]:
    ordered = sorted(int(v) for v in candidates if 0 <= int(v) < int(n))
    if not ordered or ordered[0] != 0:
        ordered.insert(0, 0)
    if ordered[-1] != n - 1:
        ordered.append(n - 1)

    filtered = [ordered[0]]
    for idx in ordered[1:-1]:
        if idx - filtered[-1] >= int(min_segment_len):
            filtered.append(idx)
    if filtered[-1] != n - 1:
        if n - 1 - filtered[-1] < int(min_segment_len) and len(filtered) > 1:
            filtered[-1] = n - 1
        else:
            filtered.append(n - 1)
    return sorted(set(filtered))


def _episode_keypoints(
    pos: np.ndarray,
    action: np.ndarray,
    max_segment_len: int,
    min_segment_len: int,
    curvature_threshold: float,
    gripper_change_threshold: float,
    speed_min_quantile: float,
) -> list[int]:
    n = int(pos.shape[0])
    keypoints: set[int] = {0, max(n - 1, 0)}
    if n <= 2:
        return sorted(keypoints)

    raw_speed = np.linalg.norm(action[:, :3], axis=-1).astype(np.float32)
    grip = action[:, -1].astype(np.float32)

    # Contact / terminal-action events are hard segment boundaries.
    grip_change = np.abs(np.diff(grip))
    for idx in np.where(grip_change >= float(gripper_change_threshold))[0]:
        _add_keypoint(keypoints, int(idx), n)
        _add_keypoint(keypoints, int(idx) + 1, n)

    # Geometry events: local direction changes in the EEF path.
    d_prev = pos[1:-1] - pos[:-2]
    d_next = pos[2:] - pos[1:-1]
    curvature = _angle_between(d_prev, d_next)
    if curvature.size:
        peak = np.zeros_like(curvature, dtype=bool)
        if curvature.size == 1:
            peak[0] = curvature[0] >= float(curvature_threshold)
        else:
            peak[1:-1] = (
                (curvature[1:-1] >= curvature[:-2])
                & (curvature[1:-1] >= curvature[2:])
                & (curvature[1:-1] >= float(curvature_threshold))
            )
            peak[0] = curvature[0] >= float(curvature_threshold)
            peak[-1] = curvature[-1] >= float(curvature_threshold)
        for idx in np.where(peak)[0] + 1:
            _add_keypoint(keypoints, int(idx), n)

    # Slow-down points often correspond to pre-grasp / placement / insertion bottlenecks.
    moving = raw_speed[raw_speed > 1e-6]
    if moving.size:
        speed_threshold = float(np.quantile(moving, np.clip(float(speed_min_quantile), 0.0, 1.0)))
        for idx in range(1, n - 1):
            if raw_speed[idx] <= speed_threshold and raw_speed[idx] <= raw_speed[idx - 1] and raw_speed[idx] <= raw_speed[idx + 1]:
                _add_keypoint(keypoints, idx, n)

    keypoint_list = _filter_keypoints(keypoints, n, int(min_segment_len))

    # Cap long segments so the policy never has to regress very distant attractors.
    capped: list[int] = [keypoint_list[0]]
    for right in keypoint_list[1:]:
        while right - capped[-1] > int(max_segment_len):
            capped.append(capped[-1] + int(max_segment_len))
        if right != capped[-1]:
            capped.append(right)
    return _filter_keypoints(set(capped), n, int(min_segment_len))


def _segment_query(
    pos: np.ndarray,
    action: np.ndarray,
    gripper: np.ndarray,
    task_uid: np.ndarray,
    left: int,
    right: int,
    speed_ref: float,
) -> np.ndarray:
    seg = slice(int(left), int(right) + 1)
    displacement = (pos[int(right)] - pos[int(left)]).astype(np.float32)
    direction = displacement / _safe_norm(displacement, keepdims=False)
    log_dist = np.asarray([np.log1p(float(np.linalg.norm(displacement)) / 0.05)], dtype=np.float32)

    mean_raw_xyz = action[seg, :3].mean(axis=0).astype(np.float32)
    segment_center = pos[seg].mean(axis=0).astype(np.float32)
    speed = _safe_norm(action[seg, :3], axis=-1, keepdims=True)
    speed_scale_median = np.asarray(
        [np.clip(float(np.median(speed)) / max(float(speed_ref), 1e-6), 0.0, 3.0)],
        dtype=np.float32,
    )

    if right > left + 1:
        curvature = _angle_between(pos[left + 1 : right] - pos[left : right - 1], pos[left + 2 : right + 1] - pos[left + 1 : right])
        curvature_mean = np.asarray([float(np.mean(curvature)) if curvature.size else 0.0], dtype=np.float32)
    else:
        curvature_mean = np.zeros((1,), dtype=np.float32)

    vertical_ratio = np.asarray(
        [float(np.mean(np.abs(action[seg, 2:3]) / _safe_norm(action[seg, :3], axis=-1, keepdims=True)))],
        dtype=np.float32,
    )
    gripper_start = np.asarray([float(action[int(left), -1])], dtype=np.float32)
    gripper_end = np.asarray([float(action[int(right), -1])], dtype=np.float32)
    task_norm = np.asarray([float(task_uid[int(left), 0]) / 99.0], dtype=np.float32)

    query = np.concatenate(
        [
            direction.astype(np.float32),        # 3 segment geometry
            log_dist.astype(np.float32),         # 1 segment geometry
            mean_raw_xyz,                        # 3 segment motion style
            segment_center,                      # 3 segment context
            speed_scale_median,                  # 1 style
            curvature_mean,                      # 1 geometry/style
            vertical_ratio,                      # 1 style
            gripper_start,                       # 1 contact/terminal context
            gripper_end,                         # 1 contact/terminal context
            task_norm,                           # 1 task context
        ],
        axis=-1,
    ).astype(np.float32)
    if query.shape[-1] != QUERY_DIM:
        raise RuntimeError(f"segment query dim mismatch: {query.shape}")
    return query


def _fixed_lookahead_targets(n: int, episode_ends: np.ndarray, lookahead: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_idx = np.arange(n, dtype=np.int64)
    segment_id = np.zeros((n,), dtype=np.int64)
    next_keypoint_idx = np.arange(n, dtype=np.int64)
    start = 0
    sid = 0
    for end in episode_ends:
        end = int(end)
        idx = np.arange(start, end, dtype=np.int64)
        target_idx[start:end] = np.minimum(idx + max(int(lookahead), 0), end - 1)
        next_keypoint_idx[start:end] = target_idx[start:end]
        segment_id[start:end] = sid
        sid += 1
        start = end
    return target_idx, segment_id, next_keypoint_idx


def _segment_dense_targets(
    eef_pos: np.ndarray,
    action: np.ndarray,
    gripper: np.ndarray,
    task_uid: np.ndarray,
    episode_ends: np.ndarray,
    lookahead: int,
    max_segment_len: int,
    min_segment_len: int,
    curvature_threshold: float,
    gripper_change_threshold: float,
    speed_min_quantile: float,
    speed_ref: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = int(eef_pos.shape[0])
    target_idx = np.arange(n, dtype=np.int64)
    segment_id = np.full((n,), -1, dtype=np.int64)
    next_keypoint_idx = np.arange(n, dtype=np.int64)
    query = np.zeros((n, QUERY_DIM), dtype=np.float32)
    counts = []

    start = 0
    sid = 0
    for end in episode_ends:
        end = int(end)
        local_pos = eef_pos[start:end]
        local_action = action[start:end]
        keypoints = _episode_keypoints(
            local_pos,
            local_action,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
            curvature_threshold=curvature_threshold,
            gripper_change_threshold=gripper_change_threshold,
            speed_min_quantile=speed_min_quantile,
        )
        counts.append(len(keypoints))
        for left, right in zip(keypoints[:-1], keypoints[1:]):
            abs_left = start + int(left)
            abs_right = start + int(right)
            seg_query = _segment_query(eef_pos, action, gripper, task_uid, abs_left, abs_right, speed_ref)
            frame_idx = np.arange(abs_left, abs_right, dtype=np.int64)
            if frame_idx.size == 0:
                continue
            target_idx[frame_idx] = np.minimum(frame_idx + max(int(lookahead), 0), abs_right)
            next_keypoint_idx[frame_idx] = abs_right
            segment_id[frame_idx] = sid
            query[frame_idx] = seg_query[None, :]
            sid += 1
        # Final frame belongs to the previous segment and targets itself.
        if end > start:
            final = end - 1
            segment_id[final] = max(sid - 1, 0)
            target_idx[final] = final
            next_keypoint_idx[final] = final
            if final > start:
                query[final] = query[final - 1]
            else:
                query[final] = _segment_query(eef_pos, action, gripper, task_uid, final, final, speed_ref)
        start = end

    if np.any(segment_id < 0):
        bad = int(np.where(segment_id < 0)[0][0])
        raise RuntimeError(f"Failed to assign segment id for frame {bad}")
    stats = {
        "num_segments": int(sid),
        "keypoints_per_episode_min": int(np.min(counts)) if counts else 0,
        "keypoints_per_episode_mean": float(np.mean(counts)) if counts else 0.0,
        "keypoints_per_episode_max": int(np.max(counts)) if counts else 0,
    }
    return target_idx, segment_id, next_keypoint_idx, query, stats


def _build_labels(
    data: zarr.Group,
    episode_ends: np.ndarray,
    lookahead: int,
    waypoint_mode: str,
    max_segment_len: int,
    min_segment_len: int,
    curvature_threshold: float,
    gripper_change_threshold: float,
    speed_min_quantile: float,
    tail_mode: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    action = np.asarray(data["action"][:], dtype=np.float32)
    eef_pos = np.asarray(data["robot0_eef_pos"][:], dtype=np.float32)
    eef_quat = np.asarray(data["robot0_eef_quat"][:], dtype=np.float32)
    gripper = np.asarray(data["robot0_gripper_qpos"][:], dtype=np.float32)
    task_uid = np.asarray(data["task_uid"][:], dtype=np.float32).reshape(len(action), -1)

    speed = _safe_norm(action[:, :3], axis=-1, keepdims=True)
    speed_ref = float(np.median(speed[speed[:, 0] > 1e-6])) if np.any(speed[:, 0] > 1e-6) else 1.0
    speed_scale = np.clip(speed / max(speed_ref, 1e-6), 0.0, 3.0).astype(np.float32)
    grip_scalar = action[:, -1:].astype(np.float32)

    n = int(action.shape[0])
    metadata: dict[str, np.ndarray] = {}
    stats: dict = {"waypoint_mode": waypoint_mode, "tail_mode": tail_mode}
    if waypoint_mode == "fixed_lookahead":
        target_idx, segment_id, next_keypoint_idx = _fixed_lookahead_targets(n, episode_ends, lookahead)
        subgoal_delta = eef_pos[target_idx] - eef_pos
        direction = subgoal_delta / _safe_norm(subgoal_delta, axis=-1, keepdims=True)
        log_dist = np.log1p(_safe_norm(subgoal_delta, axis=-1, keepdims=True) / 0.05).astype(np.float32)
        raw_dir = action[:, :3] / _safe_norm(action[:, :3], axis=-1, keepdims=True)
        lateral = np.clip(1.0 - np.sum(raw_dir * direction, axis=-1, keepdims=True), 0.0, 2.0).astype(np.float32)
        vertical_ratio = np.abs(action[:, 2:3]) / _safe_norm(action[:, :3], axis=-1, keepdims=True)
        gripper_mean = np.mean(gripper, axis=-1, keepdims=True).astype(np.float32)
        task_norm = (task_uid[:, :1] / 99.0).astype(np.float32)
        query = np.concatenate(
            [
                direction,                 # 3 geometry
                log_dist,                  # 1 geometry
                action[:, :3],             # 3 local raw motion cue
                eef_pos,                   # 3 current pose context
                speed_scale,               # 1 style
                lateral,                   # 1 style/risk proxy
                vertical_ratio,            # 1 style/risk proxy
                grip_scalar,               # 1 terminal/contact
                gripper_mean,              # 1 robot context
                task_norm,                 # 1 task context
            ],
            axis=-1,
        ).astype(np.float32)
        stats.update({"num_segments": int(len(episode_ends))})
    elif waypoint_mode in {"segment_dense", "segment_query_fixed_waypoint"}:
        segment_target_idx, segment_id, next_keypoint_idx, query, seg_stats = _segment_dense_targets(
            eef_pos=eef_pos,
            action=action,
            gripper=gripper,
            task_uid=task_uid,
            episode_ends=episode_ends,
            lookahead=lookahead,
            max_segment_len=max_segment_len,
            min_segment_len=min_segment_len,
            curvature_threshold=curvature_threshold,
            gripper_change_threshold=gripper_change_threshold,
            speed_min_quantile=speed_min_quantile,
            speed_ref=speed_ref,
        )
        if waypoint_mode == "segment_dense":
            target_idx = segment_target_idx
        else:
            # Hybrid supervision: retrieve/query is segment-level, but the
            # waypoint remains a dense t+H target so closed-loop execution is
            # less brittle around keypoint boundaries.
            target_idx, _, _ = _fixed_lookahead_targets(n, episode_ends, lookahead)
        subgoal_delta = eef_pos[target_idx] - eef_pos
        stats.update(seg_stats)
    else:
        raise ValueError(f"Unknown waypoint_mode: {waypoint_mode}")

    if tail_mode == "quat_delta":
        rot_tail = eef_quat[target_idx, 1:4] - eef_quat[:, 1:4]
        tail_layout = "quat_delta3"
    elif tail_mode == "raw_action":
        rot_tail = action[:, 3:6]
        tail_layout = "raw_rot_delta3"
    else:
        raise ValueError(f"Unknown tail_mode: {tail_mode}")
    if query.shape[-1] != QUERY_DIM:
        raise RuntimeError(f"query dim mismatch: {query.shape}")

    metadata["segment_id"] = segment_id.astype(np.int64)
    metadata["target_idx"] = target_idx.astype(np.int64)
    metadata["next_keypoint_idx"] = next_keypoint_idx.astype(np.int64)
    labels = np.concatenate([query, subgoal_delta, speed_scale, rot_tail.astype(np.float32), grip_scalar], axis=-1).astype(np.float32)
    stats["tail_layout"] = tail_layout
    return labels, metadata, stats


def main() -> None:
    args = parse_args()
    if args.in_place:
        args.output = args.input
        print(f"[make_labels] in-place mode: {args.output}", flush=True)
    elif args.output.exists():
        if args.reuse_output:
            print(f"[make_labels] reusing existing output zarr: {args.output}", flush=True)
        elif args.overwrite:
            print(f"[make_labels] removing existing output zarr: {args.output}", flush=True)
            shutil.rmtree(args.output)
            print(f"[make_labels] copying source zarr: {args.input} -> {args.output}", flush=True)
            shutil.copytree(args.input, args.output)
        else:
            raise FileExistsError(f"{args.output} exists; pass --overwrite or --reuse-output")
    else:
        print(f"[make_labels] copying source zarr: {args.input} -> {args.output}", flush=True)
        shutil.copytree(args.input, args.output)

    print(f"[make_labels] opening output zarr: {args.output}", flush=True)
    root = zarr.open(str(args.output), mode="a")
    data = _data_group(root)
    ends = _episode_ends(root)
    print(f"[make_labels] building {args.action_key} labels", flush=True)
    labels, metadata, build_stats = _build_labels(
        data,
        ends,
        int(args.lookahead),
        waypoint_mode=str(args.waypoint_mode),
        max_segment_len=int(args.max_segment_len),
        min_segment_len=int(args.min_segment_len),
        curvature_threshold=float(args.curvature_threshold),
        gripper_change_threshold=float(args.gripper_change_threshold),
        speed_min_quantile=float(args.speed_min_quantile),
        tail_mode=str(args.tail_mode),
    )
    if args.action_key in data:
        print(f"[make_labels] replacing existing {args.action_key}", flush=True)
        del data[args.action_key]
    print(f"[make_labels] writing {args.action_key} with shape {labels.shape}", flush=True)
    data.array(args.action_key, labels, chunks=(min(4096, len(labels)), labels.shape[-1]), dtype="float32", overwrite=True)

    for suffix, values in metadata.items():
        key = f"{args.action_key}_{suffix}"
        if key in data:
            del data[key]
        data.array(key, values, chunks=(min(4096, len(values)),), dtype=str(values.dtype), overwrite=True)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "lookahead": int(args.lookahead),
        "waypoint_mode": str(args.waypoint_mode),
        "max_segment_len": int(args.max_segment_len),
        "min_segment_len": int(args.min_segment_len),
        "curvature_threshold": float(args.curvature_threshold),
        "gripper_change_threshold": float(args.gripper_change_threshold),
        "speed_min_quantile": float(args.speed_min_quantile),
        "tail_mode": str(args.tail_mode),
        "key": args.action_key,
        "shape": list(labels.shape),
        "layout": f"[query16, subgoal_delta3, speed1, {build_stats['tail_layout']}, gripper1]",
        **build_stats,
    }
    summary_name = f"{args.action_key}_summary.json"
    (args.output / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
