# Validation

## Tested Environment and Replay

The installation locks were updated on 2026-10-10 from the cluster export:
Python 3.10.20, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.2.6,
MuJoCo 2.3.7, robosuite 1.4.0, and diffusers 0.38.0. The cluster GPU was an
NVIDIA A100-SXM4-80GB with native NVIDIA EGL rendering. OAT, its tokenizer
dependencies, and cloud-transfer tools are excluded from the standalone setup.
LIBERO is installed at the pinned commit; egl-probe and pip CMake are retained
as release installation helpers rather than copied cluster runtime entries.
Conda creation preflight and clean pip dependency resolution passed for these
locks. All 142 pip packages resolved to their pinned versions. The updated
source passed 244 tests in the existing local validation environment.

On 2026-10-09, the local paired-RAW check used NVIDIA D3D12 hardware rendering
under WSL2 on an RTX 5060 Laptop GPU. For two sequential nominal episodes,
seed labels 900000/900001 and sampling seeds 0/1, both the frozen cluster source
and standalone source produced outcomes `[false, true]`, matching the cluster's
two outcomes. The two local source versions had identical initial observations
and all 981 output actions. The successful episode took 406 actions locally
and 274 on the cluster. These local runs used PyTorch CUDA 12.9 and diffusers
0.37.1; they predate the updated cluster-derived installation locks.

## Independent Runtime Check

Validated on 2026-10-02 in a newly installed WSL2 Ubuntu environment, with the
standalone source, pinned LIBERO, an isolated LIBERO configuration, and an NVIDIA
RTX 5060 Laptop GPU. Imports resolved to this checkout and the new environment,
not the research workspace. Python 3.10.20, PyTorch 2.10.0 (CUDA 12.9), NumPy
2.2.6, and MuJoCo 2.3.7. This earlier check used a local CUDA 12.9 environment;
the current installation locks instead track the cluster export described above.
Official cached installation packages were reused; installed research packages
and editable research checkouts were not copied. OAT, Transformers, and
vector-quantize-pytorch were not installed in this environment.

- 239 tests passed, with no skips.
- Standalone installation completed; `pip check` found no broken requirements.
- The Python package wheel built successfully, including both training YAML
  configurations and third-party license notices.
- Three label-generation CLIs ran on synthetic data.
- All four representations completed one real optimizer step, one validation
  batch, reconstruction, EMA/checkpoint saving, and standalone checkpoint reload.
  These used four synthetic 64-step episodes and batch size 2, not demonstration
  data or a complete training run. All training and validation losses were finite.
- Four released checkpoints each completed one nominal and one perturbed
  evaluation, with one serial environment and a 575-step episode budget.

Task: `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`.
Environment seed: 910000. Policy-sampling seed: 17000. Paired models used K=16;
action-only models used K=1. These are smoke-test settings, not a benchmark sample.

| Model | Nominal process | Perturbed process | Perturbed recovery steps | Recovery reentry replans | Task successes |
| --- | --- | --- | ---: | ---: | ---: |
| Paired RAW | Passed | Passed | 49 | 1 | 0/2 |
| Paired BSP | Passed | Passed | 64 | 1 | 0/2 |
| Raw Action | Passed | Passed | 0 | 0 | 0/2 |
| Action BSP | Passed | Passed | 0 | 0 | 0/2 |

All four perturbations were applied and reached their target. Both paired
policies entered recovery, kept the recovery phase fixed, and replanned after
reentry. Every run's checkpoint hash matched the public catalog. For all eight
runs, the executor metrics, episode outcomes, and perturbation records exactly
matched the previous OAT-backed release on the same hardware and seeds.

Later inspection found software camera rendering in the local EGL setup;
the hardware-rendered replay above corrects this setup. All eight earlier task outcomes were
failures. Process completion and exercising
recovery do not establish benchmark performance. This check verifies execution
paths, not the paper's success rates or statistical reproducibility.

## Checkpoint Integrity

All four released checkpoints were loaded independently in the previous and new
runtimes. With synthetic RGB/state inputs, RNG seed 731, and batch sizes 1 and
16, CUDA `action` and `action_pred` tensors were finite and bitwise identical.
The new loader translates legacy configuration targets in memory and loads the
model/EMA state directly, without constructing a training workspace or optimizer.
No checkpoint files or uploaded weight archives were modified by this extraction.

Four released checkpoints were verified against their original paper hashes.
During the earlier sanitization, only the private dataset path was changed.
Every model/EMA/normalization/optimizer
tensor remained byte-identical. Same-seed synthetic-input GPU inference was
identical before and after sanitization in the earlier research environment.

## Not Validated

Full training on demonstration data, the complete multi-task/multi-seed benchmark,
other GPUs/platforms, other simulation suites, and real-robot deployment were not
rerun. Training presets are not certified reproductions of the original runs.

DSP's original code is released under the MIT License in `LICENSE`.
Retained upstream terms and excluded components are documented in
`THIRD_PARTY_NOTICES.md`. The checkpoint download link is recorded in the README
and model catalog; the source-code license does not assign license terms to the
separately distributed weights. Technical validation does not establish
licensing clearance for third-party materials.
