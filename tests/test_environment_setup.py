import json
from pathlib import Path
import subprocess
import sys

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import run_wsl_gpu


ROOT = Path(__file__).resolve().parents[1]


def test_cluster_requirements_are_portable_and_pinned():
    pins = {}
    for line in (ROOT / 'requirements-sim.txt').read_text().splitlines():
        if not line or line.startswith('#') or line.startswith('--extra-index-url '):
            continue
        requirement = Requirement(line)
        assert requirement.url is None
        specs = list(requirement.specifier)
        assert len(specs) == 1 and specs[0].operator == '=='
        name = canonicalize_name(requirement.name)
        assert name not in pins
        pins[name] = specs[0].version
    assert pins['torch'] == '2.10.0+cu128'
    assert pins['torchvision'] == '0.25.0+cu128'
    assert pins['diffusers'] == '0.38.0'
    assert pins['mujoco'] == '2.3.7'
    assert not {'oat', 'transformers', 'vector-quantize-pytorch', 'gsutil'} & pins.keys()


@pytest.mark.parametrize('renderer,returncode,accepted', [
    ('D3D12 (NVIDIA GeForce RTX)', 0, True),
    ('llvmpipe (LLVM)', 0, False),
    ('D3D12 (AMD)', 0, False),
    ('', 1, False),
])
def test_wsl_render_guard(monkeypatch, tmp_path, renderer, returncode, accepted):
    report = tmp_path / 'render.json'
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(run_wsl_gpu.platform, 'release', lambda: 'test-microsoft-WSL2')
    monkeypatch.setattr(sys, 'argv', ['run_wsl_gpu.py', '--render-report', str(report),
                                     '--', 'evaluate.py', '--episodes', '2'])
    monkeypatch.setattr(run_wsl_gpu.subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess([], returncode,
                                                    json.dumps({'renderer': renderer}), ''))
    calls = []

    def launch(command, env):
        calls.append(command)
        assert env['GALLIUM_DRIVER'] == 'd3d12'
        assert env['MESA_D3D12_DEFAULT_ADAPTER_NAME'] == 'NVIDIA'
        return 7

    monkeypatch.setattr(run_wsl_gpu.subprocess, 'call', launch)
    with pytest.raises(SystemExit) as error:
        run_wsl_gpu.main()
    assert bool(calls) is accepted
    assert report.exists() is accepted
    if accepted:
        assert error.value.code == 7
        assert calls[0][1:] == ['evaluate.py', '--episodes', '2']
    else:
        assert 'not started' in str(error.value)
