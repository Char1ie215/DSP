"""Episode-local policy RNG streams for serial matched evaluations."""
import random

import numpy as np
import torch


def install_episode_rng(policy, seed_base, n_envs):
    if seed_base is None:
        return None
    if n_envs != 1:
        raise ValueError("Episode-local RNG requires n_envs=1")
    original_reset = policy.reset
    seeds = []

    def reset():
        result = original_reset()
        seed = int(seed_base) + len(seeds)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeds.append(seed)
        return result

    policy.reset = reset
    return seeds


def annotate_episode_rng(episodes, seeds):
    if seeds is None:
        return
    if len(episodes) != len(seeds):
        raise RuntimeError(f"Policy reset count {len(seeds)} != episodes {len(episodes)}")
    for episode, seed in zip(episodes, seeds):
        episode["sampling_seed"] = seed
