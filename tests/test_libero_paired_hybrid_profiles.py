from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from ns_gaussian_ds_policy.libero_paired_hybrid_profiles import (
    CONCURRENT_WRENCH_20CM_V1,
    CONCURRENT_WRENCH_20CM_V1_VALUES,
    LAST_SAFE_V1,
    LAST_SAFE_V1_VALUES,
    apply_evaluation_profiles,
)


def _namespace(**overrides) -> Namespace:
    values = {
        **{name: "unset" for name in LAST_SAFE_V1_VALUES},
        **{
            name: "unset"
            for name in CONCURRENT_WRENCH_20CM_V1_VALUES
        },
        "mode": "paired_bspline_hybrid_ds",
        "method_profile": LAST_SAFE_V1,
        "perturbation_profile": CONCURRENT_WRENCH_20CM_V1,
    }
    values.update(overrides)
    return Namespace(**values)


def test_lastsafe_profile_resolves_complete_formal_method() -> None:
    args = apply_evaluation_profiles(_namespace())

    for name, expected in LAST_SAFE_V1_VALUES.items():
        assert getattr(args, name) == expected
    for name, expected in CONCURRENT_WRENCH_20CM_V1_VALUES.items():
        assert getattr(args, name) == expected

    assert args.paired_pose_ensemble_samples == 16
    assert args.hybrid_recovery_field_source == "geometry"
    assert args.hybrid_recovery_target_mode == "support_confidence_prefix"
    assert args.hybrid_recovery_anchor_mode == "observed_perturbation_start"
    assert args.hybrid_freeze_recovery_phase
    assert args.hybrid_replan_on_recovery_reentry


def test_lastsafe_profile_rejects_unrelated_executor() -> None:
    with pytest.raises(ValueError, match="does not support"):
        apply_evaluation_profiles(_namespace(mode="raw_action"))


@pytest.mark.parametrize(
    "mode",
    ("paired_pose_action_direct", "paired_pose_action_hybrid_ds"),
)
def test_lastsafe_profile_accepts_raw_paired_executor(mode: str) -> None:
    args = apply_evaluation_profiles(_namespace(mode=mode))

    assert args.paired_pose_ensemble_samples == 16


def test_custom_profiles_leave_values_unchanged() -> None:
    args = _namespace(
        method_profile="custom",
        perturbation_profile="custom",
        paired_pose_ensemble_samples=3,
        concurrent_perturbation_distance_cm=7.0,
    )

    resolved = apply_evaluation_profiles(args)

    assert resolved.paired_pose_ensemble_samples == 3
    assert resolved.concurrent_perturbation_distance_cm == 7.0


def test_profile_fields_exist_in_the_evaluator_cli() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "eval_libero_policy.py"
    ).read_text(encoding="utf-8")

    for name in {
        *LAST_SAFE_V1_VALUES,
        *CONCURRENT_WRENCH_20CM_V1_VALUES,
    }:
        assert f'"--{name.replace("_", "-")}"' in source
