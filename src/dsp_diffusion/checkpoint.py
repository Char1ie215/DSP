"""Load legacy diffusion checkpoints without importing the original OAT package."""
from pathlib import Path

import dill
import hydra
from omegaconf import OmegaConf
import torch

from dsp_diffusion.common.hydra_util import register_new_resolvers


LEGACY_TARGETS = {
    'policy.diffpolicy.DiffusionTransformerPolicy',
    'perception.fused_obs_encoder.FusedObservationEncoder',
    'perception.robomimic_vision_encoder.RobomimicRgbEncoder',
    'perception.state_encoder.ProjectionStateEncoder',
    'env_runner.libero_runner.LiberoRunner',
    'workspace.train_policy.TrainPolicyWorkspace',
    'dataset.zarr_dataset.ZarrDataset',
    'dataset.window_zarr_dataset.WindowZarrDataset',
    'model.diffusion.ema_model.EMAModel',
}


def migrate_config(cfg):
    register_new_resolvers()
    values = OmegaConf.to_container(cfg, resolve=False)

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == '_target_' and isinstance(item, str) and item.startswith('oat.'):
                    target = item.removeprefix('oat.')
                    if target not in LEGACY_TARGETS:
                        raise ValueError(f'Unsupported legacy checkpoint target: {item}')
                    value[key] = 'dsp_diffusion.'+target
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(values)
    return OmegaConf.create(values)


def load_policy(checkpoint: str | Path):
    # Legacy artifacts contain OmegaConf metadata and must come from a trusted source.
    with Path(checkpoint).open('rb') as stream:
        payload = torch.load(stream, map_location='cpu', pickle_module=dill, weights_only=False)
    cfg = migrate_config(payload['cfg'])
    expected = 'dsp_diffusion.policy.diffpolicy.DiffusionTransformerPolicy'
    if cfg.policy._target_ != expected:
        raise ValueError(f'Only the released DiffusionTransformerPolicy is supported: {cfg.policy._target_}')
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = 'ema_model' if cfg.training.get('use_ema', False) else 'model'
    policy.load_state_dict(payload['state_dicts'][state_key], strict=True)
    policy.eval()
    return policy, cfg
