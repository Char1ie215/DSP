"""Run DSP's diffusion training workspace from the packaged configuration."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import hydra
from omegaconf import OmegaConf
from dsp_diffusion.common.hydra_util import register_new_resolvers
from dsp_diffusion.workspace.train_policy import TrainPolicyWorkspace

register_new_resolvers()


@hydra.main(version_base=None, config_path=str(ROOT/'src/dsp_diffusion/config'),
            config_name='train_diffpolicy')
def main(cfg):
    OmegaConf.resolve(cfg)
    TrainPolicyWorkspace(cfg).run()


if __name__ == '__main__':
    main()
