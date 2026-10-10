"""LIBERO single-task evaluation with explicit DSP and ablation choices."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
REPRESENTATIONS = ('paired_raw', 'paired_bsp', 'raw_action', 'action_bsp')


def configure_runtime():
    os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT/'.runtime/libero'))
    os.environ.setdefault('LIBERO_ROOT', str(ROOT/'thirdparty/LIBERO'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--representation', choices=REPRESENTATIONS, required=True)
    p.add_argument('--strategy', choices=('recovery', 'direct', 'replan'), default='recovery')
    p.add_argument('--candidates', type=int, choices=(1,16), default=16)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', required=True, help='LIBERO task name, not a suite name')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--environment-seed', type=int, default=900000)
    p.add_argument('--sampling-seed', type=int, default=0)
    p.add_argument('--perturbed', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    return p


def resolve(options):
    import eval_libero_policy as backend
    from ns_gaussian_ds_policy.libero_paired_hybrid_profiles import (
        apply_evaluation_profiles,
    )
    paired = options.representation.startswith('paired_')
    if not paired and (options.strategy != 'direct' or options.candidates != 1):
        raise ValueError('Action-only baselines require --strategy direct --candidates 1')
    if options.candidates == 1 and options.strategy != 'direct':
        raise ValueError('The K=1 ablation uses direct execution only')
    if options.episodes < 1:
        raise ValueError('Episodes must be positive')
    modes = {'paired_raw':'paired_pose_action', 'paired_bsp':'paired_bspline',
             'raw_action':'raw_action', 'action_bsp':'action_bspline_direct'}
    mode = modes[options.representation]
    if paired:
        mode += '_direct' if options.strategy == 'direct' else '_hybrid_ds'
    argv = [str(ROOT/'scripts/eval_libero_policy.py'), '--checkpoint',str(options.checkpoint.resolve()),
            '--output-dir',str(options.output.resolve()), '--mode',mode,
            '--method-profile','custom', '--perturbation-profile','custom',
            '--task-name',options.task, '--n-test',str(options.episodes),
            '--n-parallel-envs','1', '--num-exp','1', '--max-episode-steps','575',
            '--max-plan-steps','1000', '--paired-plan-steps','16',
            '--paired-bspline-source-horizon','32', '--n-test-vis','0',
            '--eval-rng-mode','per_episode', '--require-episode-outcomes',
            '--eval-seed',str(options.sampling_seed), '--test-start-seed',str(options.environment_seed),
            '--device',options.device]
    old = sys.argv
    try:
        sys.argv = argv
        args = backend.parse_args()
    finally:
        sys.argv = old
    args.method_profile = 'last_safe_no_fusion_a' if paired else 'custom'
    args.perturbation_profile = 'concurrent_wrench_20cm_v1' if options.perturbed else 'custom'
    args = apply_evaluation_profiles(args)
    if paired:
        args.paired_pose_ensemble_samples = options.candidates
    else:
        args.direct_stepwise_execution = True
    args.hybrid_disturbance_response_mode = (
        'immediate_replan' if options.strategy == 'replan' else 'recover_then_replan'
    )
    # The backend applies profiles again. Keep the resolved K=1 override intact.
    args.method_profile = args.perturbation_profile = 'custom'
    return args


def main():
    configure_runtime()
    p = parser()
    options = p.parse_args()
    options.checkpoint = options.checkpoint.resolve()
    options.output = options.output.resolve()
    try:
        args = resolve(options)
    except ValueError as error:
        p.error(str(error))
    if options.dry_run:
        print(json.dumps(vars(args),default=str,indent=2))
        return
    if not options.checkpoint.is_file():
        p.error('Checkpoint missing; no weights are bundled with this source release')
    if options.output.exists():
        p.error('Output exists; choose a new directory')
    digest = hashlib.sha256()
    with options.checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    import eval_libero_policy as backend
    original = backend.parse_args
    try:
        backend.parse_args = lambda: args
        backend.main()
    finally:
        backend.parse_args = original
    record = {'representation':options.representation,'strategy':options.strategy,
              'candidates':options.candidates,'checkpoint_sha256':digest.hexdigest(),
              'base_perturbation_profile':'concurrent_wrench_20cm_v1' if options.perturbed else 'none',
              'base_profile':'last_safe_no_fusion_a' if options.representation.startswith('paired_') else 'native',
              'resolved_arguments':vars(args)}
    (options.output.resolve()/'release_run.json').write_text(json.dumps(record,default=str,indent=2)+'\n')


if __name__ == '__main__':
    main()
