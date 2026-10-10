"""Render synchronized task and controller-mode diagnostic videos."""

from __future__ import annotations

import json
import math
import pathlib
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_BACKGROUND = (25, 29, 35)
_PANEL = (34, 39, 47)
_MUTED = (158, 168, 181)
_TEXT = (244, 246, 248)
_ACTION = (35, 174, 139)
_FIELD = (239, 139, 47)
_PERTURB = (205, 87, 108)
_QUERY = (112, 123, 246)


@lru_cache(maxsize=None)
def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _mode_color(mode: str) -> tuple[int, int, int]:
    if mode == "field":
        return _FIELD
    if mode == "perturb":
        return _PERTURB
    return _ACTION


def _timeline_pixels(
    trace: list[dict[str, object]],
    width: int,
    height: int = 18,
) -> np.ndarray:
    timeline = np.zeros((height, width, 3), dtype=np.uint8)
    if not trace:
        timeline[:] = _MUTED
        return timeline
    indices = np.minimum(
        (np.arange(width, dtype=np.float64) * len(trace) / width).astype(int),
        len(trace) - 1,
    )
    colors = np.asarray(
        [_mode_color(str(trace[index].get("mode", "action"))) for index in indices],
        dtype=np.uint8,
    )
    timeline[:] = colors[None, :, :]
    return timeline


def _draw_meter(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    value: float,
    reference: float,
    color: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + 12),
        radius=3,
        fill=(57, 64, 75),
    )
    ratio = min(max(value / max(reference, 1.0e-9), 0.0), 2.0) / 2.0
    draw.rounded_rectangle(
        (x, y, x + int(width * ratio), y + 12),
        radius=3,
        fill=color,
    )
    midpoint = x + width // 2
    draw.line((midpoint, y - 3, midpoint, y + 15), fill=_TEXT, width=2)


def compose_controller_debug_frame(
    task_frame: np.ndarray,
    record: dict[str, object],
    *,
    frame_index: int,
    trace_length: int,
    timeline: np.ndarray | None = None,
    panel_width: int = 400,
) -> np.ndarray:
    """Put a controller dashboard to the right of one RGB task frame."""

    frame = np.asarray(task_frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"task_frame must be HxWx3 RGB, got {frame.shape}")
    height, width = frame.shape[:2]
    panel_width = max(int(panel_width), 320)
    canvas = Image.new("RGB", (width + panel_width, height), _BACKGROUND)
    canvas.paste(Image.fromarray(frame, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    panel_x = width
    draw.rectangle((panel_x, 0, width + panel_width, height), fill=_PANEL)

    mode = str(record.get("mode", "action"))
    color = _mode_color(mode)
    if mode == "field":
        status = "GAUSSIAN FIELD"
        subtitle = "Recovery execution"
    elif mode == "perturb":
        status = "PERTURBATION"
        subtitle = "External displacement; policy phase paused"
    else:
        status = "ACTION SPLINE"
        subtitle = "Nominal execution"
    left = panel_x + 32
    usable = panel_width - 64

    draw.text((left, 25), "CONTROLLER MODE", font=_font(17, bold=True), fill=_MUTED)
    draw.rounded_rectangle(
        (left, 58, left + usable, 137),
        radius=7,
        fill=tuple(max(channel - 95, 0) for channel in color),
        outline=color,
        width=3,
    )
    draw.ellipse((left + 18, 78, left + 38, 98), fill=color)
    draw.text((left + 52, 69), status, font=_font(25, bold=True), fill=_TEXT)
    draw.text((left + 52, 105), subtitle, font=_font(14), fill=_MUTED)

    query_text = "VISUAL QUERY" if bool(record.get("policy_query")) else "CACHED PLAN"
    query_color = _QUERY if bool(record.get("policy_query")) else (91, 101, 115)
    draw.rounded_rectangle(
        (left, 153, left + 140, 180),
        radius=5,
        fill=query_color,
    )
    draw.text((left + 12, 158), query_text, font=_font(13, bold=True), fill=_TEXT)

    phase = float(record.get("phase", 0.0))
    phase_count = max(int(record.get("phase_count", 1)), 1)
    phase_ratio = min(max(phase / max(phase_count - 1, 1), 0.0), 1.0)
    draw.text((left, 204), "PLAN PHASE", font=_font(14, bold=True), fill=_MUTED)
    draw.text(
        (left + usable, 201),
        f"{phase:4.1f} / {phase_count - 1}",
        anchor="ra",
        font=_font(16, bold=True),
        fill=_TEXT,
    )
    draw.rounded_rectangle(
        (left, 230, left + usable, 244),
        radius=4,
        fill=(57, 64, 75),
    )
    draw.rounded_rectangle(
        (left, 230, left + int(usable * phase_ratio), 244),
        radius=4,
        fill=color,
    )

    position_error = float(record.get("position_error", 0.0))
    outer_radius = float(record.get("position_tube_radius", 0.02))
    reentry_radius = float(record.get("position_reentry_radius", 0.01))
    rotation_error = float(record.get("rotation_error", 0.0))
    draw.text((left, 271), "POSITION ERROR", font=_font(14, bold=True), fill=_MUTED)
    draw.text(
        (left + usable, 267),
        f"{100.0 * position_error:4.1f} cm",
        anchor="ra",
        font=_font(18, bold=True),
        fill=color if position_error > outer_radius else _TEXT,
    )
    _draw_meter(
        draw,
        x=left,
        y=299,
        width=usable,
        value=position_error,
        reference=outer_radius,
        color=color,
    )
    draw.text(
        (left, 320),
        f"re-enter {100.0 * reentry_radius:.1f} cm",
        font=_font(12),
        fill=_MUTED,
    )
    draw.text(
        (left + usable, 320),
        f"switch {100.0 * outer_radius:.1f} cm",
        anchor="ra",
        font=_font(12),
        fill=_MUTED,
    )

    draw.text((left, 351), "ROTATION ERROR", font=_font(14, bold=True), fill=_MUTED)
    draw.text(
        (left + usable, 347),
        f"{math.degrees(rotation_error):4.1f} deg",
        anchor="ra",
        font=_font(18, bold=True),
        fill=_TEXT,
    )
    recovery_steps = int(record.get("recovery_steps", 0))
    draw.text(
        (left, 384),
        f"Recovery steps: {recovery_steps}",
        font=_font(13),
        fill=_MUTED,
    )
    gripper_command = record.get("gripper_command")
    gripper_aperture = record.get("gripper_aperture")
    if gripper_command is not None or gripper_aperture is not None:
        command_text = (
            "n/a" if gripper_command is None else f"{float(gripper_command):+.1f}"
        )
        aperture_text = (
            "n/a"
            if gripper_aperture is None
            else f"{1000.0 * float(gripper_aperture):.1f} mm"
        )
        draw.text(
            (left, 409),
            f"Gripper cmd {command_text}   measured qpos {aperture_text}",
            font=_font(12),
            fill=_TEXT if mode == "perturb" else _MUTED,
        )

    timeline_y = height - 59
    draw.text((left, timeline_y - 24), "FULL ROLLOUT", font=_font(13, bold=True), fill=_MUTED)
    if timeline is None:
        timeline = np.full((18, usable, 3), _MUTED, dtype=np.uint8)
    timeline_image = Image.fromarray(timeline, mode="RGB")
    canvas.paste(timeline_image, (left, timeline_y))
    marker_ratio = frame_index / max(trace_length - 1, 1)
    marker_x = left + int(marker_ratio * (usable - 1))
    draw.line((marker_x, timeline_y - 4, marker_x, timeline_y + 22), fill=_TEXT, width=3)
    draw.text((left, timeline_y + 25), "ACTION", font=_font(11, bold=True), fill=_ACTION)
    draw.text(
        (left + usable, timeline_y + 25),
        "FIELD",
        anchor="ra",
        font=_font(11, bold=True),
        fill=_FIELD,
    )

    draw.rectangle((0, height - 7, width, height), fill=color)
    return np.asarray(canvas, dtype=np.uint8)


def render_controller_debug_video(
    task_video: pathlib.Path,
    output_video: pathlib.Path,
    trace: Iterable[dict[str, object]],
    *,
    panel_width: int = 400,
) -> dict[str, object]:
    """Render one side-by-side MP4 and return synchronization metadata."""

    import av

    task_video = pathlib.Path(task_video)
    output_video = pathlib.Path(output_video)
    records = [dict(record) for record in trace]
    if not records:
        raise ValueError("controller trace is empty")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_video.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    input_container = av.open(str(task_video))
    input_stream = input_container.streams.video[0]
    raw_trace_steps = len(records)
    encoded_frame_count = int(input_stream.frames or 0)
    if encoded_frame_count:
        records = records[:encoded_frame_count]
    timeline = _timeline_pixels(records, max(panel_width - 64, 256))
    rate = input_stream.average_rate or input_stream.base_rate or 20
    output_container = av.open(str(output_video), mode="w")
    output_stream = output_container.add_stream("h264", rate=rate)
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "20", "profile": "high"}
    output_stream.thread_type = "FRAME"
    output_stream.thread_count = 2

    frame_count = 0
    try:
        for frame_count, frame in enumerate(
            input_container.decode(input_stream),
            start=1,
        ):
            trace_index = min(frame_count - 1, len(records) - 1)
            combined = compose_controller_debug_frame(
                frame.to_ndarray(format="rgb24"),
                records[trace_index],
                frame_index=trace_index,
                trace_length=len(records),
                timeline=timeline,
                panel_width=panel_width,
            )
            if frame_count == 1:
                output_stream.width = int(combined.shape[1])
                output_stream.height = int(combined.shape[0])
            output_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()

    mode_counts = {
        mode: sum(record.get("mode") == mode for record in records)
        for mode in ("action", "field", "perturb")
    }
    transitions = sum(
        records[index].get("mode") != records[index - 1].get("mode")
        for index in range(1, len(records))
    )
    return {
        "task_video": str(task_video),
        "output_video": str(output_video),
        "trace": str(trace_path),
        "video_frames": int(frame_count),
        "trace_steps": len(records),
        "raw_trace_steps": raw_trace_steps,
        "frame_trace_delta": int(frame_count - len(records)),
        "mode_counts": mode_counts,
        "mode_transitions": int(transitions),
    }


def render_libero_controller_debug_videos(
    *,
    output_dir: pathlib.Path,
    task_names: list[str],
    seeds: list[int],
    trace_chunks: list[list[list[dict[str, object]]]],
    n_test_vis: int,
    experiment_index: int = 0,
) -> list[dict[str, object]]:
    """Match LiberoRunner videos with executor traces and render dashboards."""

    if not trace_chunks or n_test_vis <= 0:
        return []
    output_dir = pathlib.Path(output_dir)
    chunk_width = len(trace_chunks[0])
    summaries = []
    for global_index in range(min(n_test_vis, len(task_names))):
        chunk_index, local_index = divmod(global_index, chunk_width)
        if chunk_index >= len(trace_chunks):
            break
        trace = trace_chunks[chunk_index][local_index]
        task_name = task_names[global_index]
        seed = int(seeds[global_index])
        task_video = output_dir / "media" / task_name / f"seed_{seed}.mp4"
        output_video = (
            output_dir
            / "controller_debug"
            / f"exp_{experiment_index:03d}"
            / task_name
            / f"seed_{seed}_action_field.mp4"
        )
        summaries.append(
            render_controller_debug_video(task_video, output_video, trace)
        )

    summary_path = (
        output_dir
        / "controller_debug"
        / f"exp_{experiment_index:03d}"
        / "render_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
