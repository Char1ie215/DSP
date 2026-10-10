from __future__ import annotations

from typing import Any

import numpy as np


def _closest_segment_points(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    first_norm_sq = float(np.dot(first, first))
    second_norm_sq = float(np.dot(second, second))
    first_offset = float(np.dot(first, offset))
    second_offset = float(np.dot(second, offset))
    epsilon = 1.0e-12

    if first_norm_sq <= epsilon and second_norm_sq <= epsilon:
        first_ratio = 0.0
        second_ratio = 0.0
    elif first_norm_sq <= epsilon:
        first_ratio = 0.0
        second_ratio = float(
            np.clip(second_offset / second_norm_sq, 0.0, 1.0)
        )
    elif second_norm_sq <= epsilon:
        second_ratio = 0.0
        first_ratio = float(
            np.clip(-first_offset / first_norm_sq, 0.0, 1.0)
        )
    else:
        cross = float(np.dot(first, second))
        determinant = first_norm_sq * second_norm_sq - cross * cross
        if determinant > epsilon:
            first_ratio = float(
                np.clip(
                    (cross * second_offset - first_offset * second_norm_sq)
                    / determinant,
                    0.0,
                    1.0,
                )
            )
        else:
            first_ratio = 0.0
        second_ratio = (cross * first_ratio + second_offset) / second_norm_sq
        if second_ratio < 0.0:
            second_ratio = 0.0
            first_ratio = float(
                np.clip(-first_offset / first_norm_sq, 0.0, 1.0)
            )
        elif second_ratio > 1.0:
            second_ratio = 1.0
            first_ratio = float(
                np.clip(
                    (cross - first_offset) / first_norm_sq,
                    0.0,
                    1.0,
                )
            )

    return (
        first_start + first_ratio * first,
        second_start + second_ratio * second,
        first_ratio,
        second_ratio,
    )


def find_state_chain_ambiguities(
    centers: Any,
    *,
    intersection_margin: float = 0.0,
    tangent_alignment_cosine: float = 0.95,
) -> tuple[tuple[int, int], ...]:
    """Find nonlocal revisits that require conflicting local velocities."""

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError(f"Expected centers [count, dim], got {centers.shape}.")
    if intersection_margin < 0.0:
        raise ValueError("intersection_margin must be non-negative.")
    if not -1.0 <= tangent_alignment_cosine <= 1.0:
        raise ValueError("tangent_alignment_cosine must be in [-1, 1].")

    conflicts: list[tuple[int, int]] = []
    vectors = np.diff(centers, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    for first_index in range(len(vectors)):
        if lengths[first_index] <= 1.0e-10:
            continue
        for second_index in range(first_index + 2, len(vectors)):
            if lengths[second_index] <= 1.0e-10:
                continue
            cosine = float(
                np.dot(vectors[first_index], vectors[second_index])
                / (lengths[first_index] * lengths[second_index])
            )
            if cosine >= tangent_alignment_cosine:
                continue
            first_point, second_point, first_ratio, second_ratio = (
                _closest_segment_points(
                    centers[first_index],
                    centers[first_index + 1],
                    centers[second_index],
                    centers[second_index + 1],
                )
            )
            arc_separation = float(
                (1.0 - first_ratio) * lengths[first_index]
                + np.sum(lengths[first_index + 1 : second_index])
                + second_ratio * lengths[second_index]
            )
            if arc_separation <= float(intersection_margin) + 1.0e-9:
                continue
            distance = float(np.linalg.norm(first_point - second_point))
            if distance <= float(intersection_margin) + 1.0e-9:
                conflicts.append((first_index, second_index))
    return tuple(conflicts)


def _select_nonintersection_boundary(
    centers: np.ndarray,
    first_segment: int,
    second_segment: int,
) -> int:
    first_point, second_point, first_ratio, second_ratio = (
        _closest_segment_points(
            centers[first_segment],
            centers[first_segment + 1],
            centers[second_segment],
            centers[second_segment + 1],
        )
    )
    conflict_point = 0.5 * (first_point + second_point)
    first_visit = first_segment + first_ratio
    second_visit = second_segment + second_ratio
    candidates = [
        index
        for index in range(first_segment + 1, second_segment + 1)
        if index > first_visit + 1.0e-6
        and index < second_visit - 1.0e-6
    ]
    if not candidates:
        candidates = list(range(first_segment + 1, second_segment + 1))

    distances = np.asarray(
        [np.linalg.norm(centers[index] - conflict_point) for index in candidates],
        dtype=np.float64,
    )
    maximum_distance = float(np.max(distances))
    if maximum_distance > 1.0e-10:
        safe = [
            index
            for index, distance in zip(candidates, distances)
            if distance >= 0.75 * maximum_distance
        ]
    else:
        safe = candidates

    def turn_cost(index: int) -> tuple[float, float]:
        incoming = centers[index] - centers[index - 1]
        outgoing = centers[index + 1] - centers[index]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= 1.0e-10 or outgoing_norm <= 1.0e-10:
            curvature = float("inf")
        else:
            cosine = float(
                np.clip(
                    np.dot(incoming, outgoing)
                    / (incoming_norm * outgoing_norm),
                    -1.0,
                    1.0,
                )
            )
            curvature = 1.0 - cosine
        distance = float(np.linalg.norm(centers[index] - conflict_point))
        return curvature, -distance

    return min(safe, key=turn_cost)


def partition_state_chain(
    centers: Any,
    *,
    intersection_margin: float = 0.0,
    tangent_alignment_cosine: float = 0.95,
) -> tuple[tuple[int, int], ...]:
    """Split a center chain only at conflicting nonlocal state revisits."""

    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or len(centers) < 2:
        raise ValueError(
            f"Expected at least two centers [count, dim], got {centers.shape}."
        )

    chunks: list[tuple[int, int]] = []
    start = 0
    final = len(centers) - 1
    while start < final:
        conflicts = find_state_chain_ambiguities(
            centers[start:],
            intersection_margin=intersection_margin,
            tangent_alignment_cosine=tangent_alignment_cosine,
        )
        if not conflicts:
            chunks.append((start, final))
            break
        local_first, local_second = min(
            conflicts,
            key=lambda pair: (pair[1], pair[0]),
        )
        first = start + local_first
        second = start + local_second
        split = _select_nonintersection_boundary(centers, first, second)
        chunks.append((start, split))
        start = split
    return tuple(chunks)
