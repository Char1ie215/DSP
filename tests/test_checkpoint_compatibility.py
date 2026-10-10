import ast
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from dsp_diffusion.checkpoint import migrate_config, load_policy


def config(ema=True):
    return OmegaConf.create({
        '_target_': 'oat.workspace.train_policy.TrainPolicyWorkspace',
        'horizon': 32,
        'policy': {'_target_': 'oat.policy.diffpolicy.DiffusionTransformerPolicy',
                   'horizon': '${horizon}'},
        'training': {'use_ema': ema},
    })


def test_migration_keeps_interpolations_and_does_not_mutate_input():
    original = config()
    migrated = migrate_config(original)
    assert original.policy._target_.startswith('oat.')
    assert migrated.policy._target_.startswith('dsp_diffusion.')
    assert migrated.policy.horizon == 32
    assert '${horizon}' in OmegaConf.to_yaml(migrated, resolve=False)


def test_unreleased_legacy_models_are_rejected():
    original = config()
    original.policy._target_ = 'oat.policy.oatpolicy.OATPolicy'
    with pytest.raises(ValueError, match='Unsupported legacy'):
        migrate_config(original)


@pytest.mark.parametrize('ema', [True, False])
def test_loader_selects_correct_state_without_workspace_or_optimizer(tmp_path, monkeypatch, ema):
    captured = {}

    class Policy:
        def load_state_dict(self, state, strict):
            captured.update(state=state, strict=strict)

        def eval(self):
            captured['eval'] = True

    monkeypatch.setattr('dsp_diffusion.checkpoint.hydra.utils.instantiate', lambda cfg: Policy())
    path = tmp_path/'weights.ckpt'
    torch.save({'cfg': config(ema), 'state_dicts': {'model': {'weight': torch.tensor(1)},
                                                  'ema_model': {'weight': torch.tensor(2)}}}, path)
    load_policy(path)
    assert captured['strict'] and captured['eval']
    assert captured['state']['weight'].item() == (2 if ema else 1)


def test_runtime_has_no_oat_imports_or_tokenizer_package():
    root = Path(__file__).resolve().parents[1]
    for folder in ('src', 'scripts'):
        for path in (root/folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                assert not any(name == 'oat' or name.startswith('oat.') for name in names), str(path)
    assert not (root/'src/dsp_diffusion/model/common/misc.py').exists()
    assert not (root/'src/dsp_diffusion/tokenizer').exists()
