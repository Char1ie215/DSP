"""Run the four released models, one nominal and one perturbed episode each."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = 'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket'


def summarize(data, perturbed):
    episodes = data.get('episode_outcomes', [])
    pushes = data.get('concurrent_environment_perturbation', {}).get('episode_records', [])
    executor = data.get('executor', {})
    return {
        'execution_passed': len(episodes) == 1 and (
            not perturbed or any(p.get('force_steps', 0) > 0 for p in pushes)),
        'success_rate': data.get('mean_success_rate'),
        'episode_count': len(episodes),
        'push_applied': any(p.get('force_steps', 0) > 0 for p in pushes),
        'push_target_reached': any(p.get('target_reached', False) for p in pushes),
        'field_recovery_steps': executor.get('field_recovery_steps', 0),
        'recovery_reentry_replans': executor.get('recovery_reentry_replans', 0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default=TASK)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output exists; choose a new directory')
    models = ('paired_raw', 'paired_bsp', 'raw_action', 'action_bsp')
    catalog = json.loads((ROOT/'models/libero10.json').read_text())
    hashes = {model['id']: model['expected_sha256'] for model in catalog['models']}
    for model in models:
        checkpoint = ROOT/'checkpoints/libero10'/f'{model}.ckpt'
        if not checkpoint.is_file():
            parser.error(f'Missing checkpoint: {model}')
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != hashes[model]:
            parser.error(f'Checkpoint checksum mismatch: {model}')
    output.mkdir(parents=True)
    results = []
    for model in models:
        paired = model.startswith('paired_')
        for perturbed in (False, True):
            name = model + ('_perturbed' if perturbed else '_nominal')
            command = [sys.executable, str(ROOT/'scripts/evaluate.py'), '--representation', model,
                       '--strategy', 'recovery' if paired else 'direct', '--candidates', '16' if paired else '1',
                       '--checkpoint', str(ROOT/'checkpoints/libero10'/f'{model}.ckpt'),
                       '--output', str(output/name), '--task', args.task, '--episodes', '1',
                       '--environment-seed', '910000', '--sampling-seed', '17000', '--device', args.device]
            if perturbed:
                command.append('--perturbed')
            print(f'Running {name}', flush=True)
            with (output/f'{name}.log').open('w') as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         env=os.environ.copy())
            log = output/name/'eval_log.json'
            record = {'case': name, 'exit_code': process.returncode, 'execution_passed': False}
            if process.returncode == 0 and log.is_file():
                data = json.loads(log.read_text())
                record.update(summarize(data, perturbed))
            results.append(record)
            (output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(record), flush=True)
    if not all(item['execution_passed'] for item in results):
        raise SystemExit('One or more execution checks failed; inspect the logs')
    print('All eight execution checks passed. This does not estimate task success rates.')


if __name__ == '__main__':
    main()
