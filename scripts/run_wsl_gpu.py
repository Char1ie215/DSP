"""Run local WSL evaluation with a verified NVIDIA D3D12 renderer.

Usage: python run_wsl_gpu.py [--render-report PATH] -- EVALUATOR.py [ARGS ...]
This changes child-process settings only, not the system or release configuration.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PROBE = '''
import json
import mujoco
from OpenGL import GL
context = mujoco.GLContext(128, 128)
try:
    context.make_current()
    print(json.dumps({name: GL.glGetString(value).decode() for name, value in (
        ('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER),
        ('version', GL.GL_VERSION))}))
finally:
    context.free()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('Provide an evaluator script and its arguments after --')
    if sys.platform != 'linux' or 'microsoft' not in platform.release().lower():
        parser.error('This launcher is for local WSL only; do not use it on the cluster.')
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', GALLIUM_DRIVER='d3d12',
               MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    probe = subprocess.run([sys.executable, '-c', PROBE], env=env,
                           capture_output=True, text=True, timeout=60)
    if probe.returncode:
        sys.stderr.write(probe.stderr)
        raise SystemExit('GPU rendering preflight failed; evaluation was not started.')
    report = json.loads(probe.stdout)
    renderer = report['renderer']
    if 'D3D12' not in renderer or 'NVIDIA' not in renderer:
        raise SystemExit(f'Expected NVIDIA D3D12 rendering, got {renderer!r}; evaluation was not started.')
    report['python'] = sys.executable
    report['environment'] = {key: env[key] for key in (
        'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'GALLIUM_DRIVER', 'MESA_D3D12_DEFAULT_ADAPTER_NAME')}
    report['command'] = command
    if args.render_report:
        args.render_report.parent.mkdir(parents=True, exist_ok=True)
        with args.render_report.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print(f'[render-preflight] {renderer}', flush=True)
    raise SystemExit(subprocess.call([sys.executable, *command], env=env))


if __name__ == '__main__':
    main()
