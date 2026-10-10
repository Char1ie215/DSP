from pathlib import Path

import pytest

import evaluate
from ns_gaussian_ds_policy.libero_paired_hybrid_profiles import apply_evaluation_profiles


def test_default_runtime_does_not_reuse_global_libero_config(monkeypatch):
    import os
    monkeypatch.delenv('LIBERO_CONFIG_PATH', raising=False)
    monkeypatch.delenv('LIBERO_ROOT', raising=False)
    evaluate.configure_runtime()
    assert Path(os.environ['LIBERO_CONFIG_PATH']) == evaluate.ROOT/'.runtime/libero'
    assert Path(os.environ['LIBERO_ROOT']) == evaluate.ROOT/'thirdparty/LIBERO'


def options(*extra):
    return evaluate.parser().parse_args([
        '--representation', 'paired_raw', '--checkpoint', 'weights.ckpt',
        '--output', 'outputs/test', '--task', 'example_task', *extra,
    ])


def test_recovery_uses_final_method_a_contract():
    args = evaluate.resolve(options('--perturbed'))
    assert args.mode == 'paired_pose_action_hybrid_ds'
    assert args.hybrid_nominal_fusion_mode == 'native_action'
    assert args.hybrid_require_state_innovation is True
    assert args.hybrid_freeze_recovery_phase is True
    assert args.hybrid_replan_on_recovery_reentry is True
    assert args.hybrid_recovery_field_source == 'geometry'
    assert args.hybrid_recovery_anchor_mode == 'observed_perturbation_start'
    assert args.hybrid_disturbance_response_mode == 'recover_then_replan'
    assert args.concurrent_perturbation_distance_cm == 20.0
    assert args.max_episode_steps == 575
    assert args.max_plan_steps == 1000
    assert args.paired_plan_steps == 16
    assert args.paired_bspline_source_horizon == 32
    assert args.eval_rng_mode == 'per_episode'
    assert args.require_episode_outcomes is True
    assert args.n_parallel_envs == args.num_exp == 1
    assert Path(args.checkpoint).is_absolute()
    assert Path(args.output_dir).is_absolute()


@pytest.mark.parametrize('representation,mode', [
    ('paired_raw', 'paired_pose_action_direct'),
    ('paired_bsp', 'paired_bspline_direct'),
])
def test_k1_survives_backend_profile_application(representation, mode):
    args = evaluate.resolve(options('--representation', representation,
                                    '--strategy', 'direct', '--candidates', '1'))
    args = apply_evaluation_profiles(args)
    assert args.paired_pose_ensemble_samples == 1
    assert args.mode == mode
    assert args.concurrent_perturbation_distance_cm == 0


def test_immediate_replan_is_explicit():
    args = evaluate.resolve(options('--strategy', 'replan', '--perturbed'))
    assert args.hybrid_disturbance_response_mode == 'immediate_replan'
    assert args.hybrid_require_state_innovation is True


@pytest.mark.parametrize('representation,mode', [
    ('raw_action', 'raw_action'), ('action_bsp', 'action_bspline_direct'),
])
def test_action_baselines_use_stepwise_execution(representation, mode):
    args = evaluate.resolve(options('--representation', representation,
                                    '--strategy', 'direct', '--candidates', '1'))
    assert args.mode == mode
    assert args.direct_stepwise_execution is True


@pytest.mark.parametrize('extra', [
    ('--representation', 'raw_action'), ('--candidates', '1'),
    ('--episodes', '0'),
])
def test_reject_incompatible_settings(extra):
    with pytest.raises(ValueError):
        evaluate.resolve(options(*extra))
