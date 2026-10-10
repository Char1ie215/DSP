# Dynamic System Policy

**Dynamic System Policy: Robust Visuomotor Policies via Gaussian Dynamic Fields**

Joint motion/command prediction, candidate selection, geometric support, and
phase-locked recovery for LIBERO-10. Includes paired RAW/BSP models, action-only
baselines, and direct/replan/recovery ablations. Real-robot deployment and other
simulation suites are outside this release.

## Install

Linux x86-64 (or WSL2), Conda, Git, and an NVIDIA GPU are required for the
tested GPU setup. The installation locks use the cluster's Python 3.10.20,
PyTorch 2.10.0+cu128, MuJoCo 2.3.7, and diffusers 0.38.0. The Conda lock
provides the base environment; the pip overlay installs the CUDA 12.8 runtime.
Use a new environment:

```bash
conda create -n dsp --file environment-linux64.lock
conda activate dsp
bash scripts/setup_runtime.sh
python -m pytest -q
```

The setup downloads pinned LIBERO source and dependencies; allow several GB
of disk space. It does not download training data or model weights. Dependency
licenses remain separate; see [third-party notices](THIRD_PARTY_NOTICES.md).

The cluster uses NVIDIA A100 GPUs with native NVIDIA EGL rendering. GPU
drivers and OpenGL/EGL libraries are provided by the host, not the Python lock.
For a headless Linux GPU session, set `MUJOCO_GL=egl`. On a shared GPU server,
ensure EGL selects the allocated GPU as well as CUDA.

For WSL2 on an NVIDIA GPU, run evaluation through the checked launcher:

```bash
python scripts/run_wsl_gpu.py -- scripts/evaluate.py --help
```

Use the same prefix for the evaluation commands below. It selects the NVIDIA
D3D12 renderer and stops if hardware rendering is unavailable. Setting only
`MUJOCO_GL=egl` can select CPU software rendering even when CUDA is working.
Dependency provenance is recorded in [the environment manifest](docs/cluster-environment.json).

Only the required diffusion implementation and training/evaluation utilities are
vendored under `src/dsp_diffusion`. No OAT installation, action tokenizer, or
autoregressive policy is required. Legacy checkpoint configuration paths are
translated at load time; the uploaded weights do not need conversion.

## Models

Download [DSP_LIBERO10_weights.zip (2.29 GiB)](https://drive.google.com/file/d/1wsS6QJ2tbp92CggpW69OgXAtoXvnRHtj/view?usp=sharing) and extract it
**inside this source directory**. It supplies `checkpoints/libero10/*.ckpt`.
For this standalone version, the destination is `DSP_STANDALONE_RELEASE`,
regardless of the older source-folder name in the weight archive's README.
Do not use historical unsanitized archives.
[Model hashes](models/libero10.json) identify the four files.

| Representation | Model | Strategy | Candidates |
| --- | --- | --- | --- |
| `paired_raw` | Ours (RAW) | `recovery`, `replan`, `direct` | 16; direct also supports 1 |
| `paired_bsp` | Ours (BSP) | `recovery`, `replan`, `direct` | 16; direct also supports 1 |
| `raw_action` | Raw Action | `direct` | 1 |
| `action_bsp` | Action BSP | `direct` | 1 |

Release checkpoints replace only the private dataset path. Model, EMA,
normalization, and optimizer tensors are unchanged; original and release hashes
are recorded together. Load only trusted checkpoints (PyTorch pickle format).

## Evaluate

```bash
python scripts/evaluate.py --representation paired_raw --strategy recovery --checkpoint checkpoints/libero10/paired_raw.ckpt --output outputs/raw_perturbed --task LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket --episodes 1 --perturbed
```

Omit `--perturbed` for nominal evaluation; add `--dry-run` to inspect settings.
Outputs must use a new directory. Run all four models in both conditions with:

```bash
python scripts/smoke_test.py --output outputs/smoke
```

This is an eight-episode execution smoke test, not a paper success-rate estimate.
It verifies checkpoint hashes before loading and reports task success, applied
pushes, recovery steps, and recovery-triggered replanning separately.
See [evaluation protocol](docs/evaluation.md) for the full seed/budget protocol
and the known disturbance-onset recovery reference.

## Train

```bash
bash scripts/download_oat_libero10_n500.sh
python scripts/make_libero_paired_pose_action_labels.py --input data/libero/libero10_N500.zarr --representation paired_sampled_curve
python scripts/train.py --representation paired_raw --data data/libero/libero10_N500.zarr --output outputs/train_raw --dry-run
```

Prepare labels on a working dataset copy: the builder writes new Zarr arrays.
For paired BSP use label representation `paired_bspline`; for Action BSP use
`action_bspline`. Remove `--dry-run` to train. Configurations in
`configs/libero_train.json` are starting presets, not certified reconstructions
of the original paper training runs.

## Scope

DSP controllers are in `src/ns_gaussian_ds_policy`; the extracted diffusion
runtime is in `src/dsp_diffusion`. Supported entry points are `scripts/train.py`
and `scripts/evaluate.py`.
See [validation status](VALIDATION.md). Source licensing is awaiting author
confirmation; no project-wide license is granted by third-party notices alone.
