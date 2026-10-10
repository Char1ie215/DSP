"""Portable entry point to the retained diffusion training script (Linux/Bash)."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'configs/libero_train.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', choices=tuple(config['representations']), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings = {**config['common'], **config['representations'][args.representation],
                'DATA_PATH':str(args.data.resolve()), 'OUT_DIR':str(args.output.resolve())}
    print(json.dumps(settings, indent=2))
    if args.dry_run:
        return
    if args.output.exists():
        parser.error('Output exists; choose a new training directory')
    if not args.data.is_dir():
        parser.error('Dataset directory does not exist; prepare labels first')
    if not (ROOT/'thirdparty/LIBERO/libero').is_dir():
        parser.error('Install the runtime with bash scripts/setup_runtime.sh first')
    from evaluate import configure_runtime
    configure_runtime()
    from eval_libero_policy import _ensure_libero_config
    _ensure_libero_config()
    subprocess.run(['bash', str(ROOT/'scripts/train_diffusion.sh')],
                   cwd=ROOT, env={**os.environ, **settings}, check=True)


if __name__ == '__main__':
    main()
