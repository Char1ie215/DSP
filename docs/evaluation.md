# LIBERO Evaluation Contract

This documents the retained final experiment protocol, not newly reproduced
results. Exact numerical reproduction additionally requires the frozen cluster
source, checkpoint identities, and matching dependency state.

## Method and Ablations

`scripts/evaluate.py` resolves `last_safe_no_fusion_a` for paired policies:
normal execution uses native predicted commands, the intervention requires
motion innovation, recovery freezes the reference phase, and recovery reentry
causes replanning. The field source is predicted geometry. The older
`last_safe_v1` includes nominal transverse fusion and is not equivalent.

The wrapper resolves profiles first and passes explicit settings to the backend
as `custom`, preventing a second profile application from overwriting K=1.
`release_run.json` records the base profile and the resolved arguments.

- Recovery ablation: paired RAW checkpoint, K=16, `direct`, `replan`, and
  `recovery`, all under perturbations.
- Candidate ablation: paired BSP checkpoint, nominal `direct`, K=1 versus K=16.
- Action-only baselines: `raw_action` and `action_bsp`, direct stepwise execution.

The final perturbation profile is `concurrent_wrench_20cm_v1`; 20 cm is the target
displacement, not a promise that each trial achieves exactly 20 cm. Preserve
all attempted episodes in the success denominator, including invalid-target
trials, and report achieved displacement separately.

## Running Ablations

These Bash commands use the same entry point as the README. On WSL2, use the
checked launcher prefix instead of `python scripts/evaluate.py`.
Run from the repository root with weights extracted and the environment active.

Recovery comparison: paired RAW, injected perturbations, K=16 for all strategies:

```bash
export MUJOCO_GL=egl
TASK=LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
for strategy in direct replan recovery; do
  python scripts/evaluate.py \
    --representation paired_raw --checkpoint checkpoints/libero10/paired_raw.ckpt \
    --strategy "$strategy" --perturbed --candidates 16 \
    --task "$TASK" --episodes 10 --environment-seed 900000 --sampling-seed 0 \
    --output "outputs/raw_perturbed_${strategy}"
done
```

Candidate comparison: paired BSP, no injected perturbation, direct execution:

```bash
export MUJOCO_GL=egl
TASK=LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
for k in 1 16; do
  python scripts/evaluate.py \
    --representation paired_bsp --checkpoint checkpoints/libero10/paired_bsp.ckpt \
    --strategy direct --candidates "$k" \
    --task "$TASK" --episodes 10 --environment-seed 900000 --sampling-seed 0 \
    --output "outputs/bsp_nominal_k${k}"
done
```

These are small single-task comparisons, not the full paper matrix. Add
`--dry-run` to inspect each configuration before simulation. Choose fresh output
paths on subsequent runs. Recovery remains conditional: enabling it does not
mean every episode will trigger a field.

## Inspecting Outputs

Each completed evaluation writes:

- `release_run.json`: checkpoint SHA256, representation, strategy, candidate
  count, base profiles, and all resolved backend arguments.
- `eval_log.json`: `mean_success_rate`, `episode_outcomes`, and controller
  diagnostics. Inspect `executor.field_recovery_steps` for field execution and
  `executor.recovery_reentry_replans` for replanning after recovery reentry.
- For perturbed runs, `eval_log.json` also contains
  `concurrent_environment_perturbation.episode_records`; `force_steps`
  indicates applied pushes and `target_reached` indicates target displacement.

Check the diagnostics together: a completed process, an applied push, field
execution, and task success are different outcomes. The standard wrapper
disables video recording (`n_test_vis=0`); its rollout count does not specify a
number of saved videos.

## Code Locations and Advanced Settings

| File | Responsibility |
| --- | --- |
| [scripts/evaluate.py](../scripts/evaluate.py) | Public CLI, model/strategy selection, episode count, seeds, and final-profile resolution. |
| [libero_paired_hybrid_profiles.py](../src/ns_gaussian_ds_policy/libero_paired_hybrid_profiles.py) | Named method and perturbation presets, including support, innovation, recovery, and wrench settings. |
| [libero_hybrid_ds.py](../src/ns_gaussian_ds_policy/libero_hybrid_ds.py) | `ProjectedPhaseHybridExecutor` and LIBERO adapters: intervention, reference/phase locking, field recovery, and replanning. |
| [autonomous_se3_field.py](../src/ns_gaussian_ds_policy/autonomous_se3_field.py) | Local field fitting and SE(3) field execution primitives used by the hybrid controller. |
| [libero_concurrent_perturbation.py](../src/ns_gaussian_ds_policy/libero_concurrent_perturbation.py) | Simulator-side external-wrench injection and perturbation records. |
| [scripts/eval_libero_policy.py](../scripts/eval_libero_policy.py) | Lower-level evaluator and full advanced argument list (`--help`). |

The public wrapper deliberately keeps the paper controller settings fixed.
It does not expose arbitrary candidate counts, push strengths, recovery gains,
parallel environment counts, or video options. For custom experiments, inspect
the resolved namespace with `--dry-run` and the backend's `--help`. Named
profiles overwrite their covered parameters; do not expect an extra backend
flag to override a named preset. Use `custom` with explicitly preserved
resolved settings when changing those parameters, and record the changes.
Backend parser defaults alone are not the paper preset.

## Known Onset Versus Autonomous Reference

The retained final simulation profile sets
`hybrid_recovery_anchor_mode=observed_perturbation_start`. The evaluator can use
the measured pose at the known injected disturbance onset as a recovery
reference. Do not describe this protocol as an entirely onset-blind autonomous
evaluation, or silently replace it with the last accepted-state reference.
The underlying controller also contains autonomous recovery behavior; a changed
reference protocol must be reported as a separate experiment.

## Budget and Seeds

- 575 environment steps per episode; 1,000 maximum planning steps.
- 16 command steps per query; BSP source horizon 32.
- One parallel environment, one experiment per invocation, explicit per-episode
  outcomes, and `per_episode` policy RNG mode.
- Each reported group: 10 tasks x 50 episodes x 5 policy-sampling replicates =
  2,500 attempts. The aggregate contains 12 groups, or 30,000 attempts.
- For task index `t` in the frozen task order (0 through 9) and replicate `r`
  (0 through 4), use `--environment-seed 900000+50*t` and
  `--sampling-seed 500*r+50*t`, evaluating 50 episodes. Supply computed integers,
  not arithmetic strings, to the CLI.

The five sampling replicates reuse the same environment seed pairs. They are
not five independently trained models. Keep checkpoint-selection seeds separate
from the reported evaluation seeds. The four checkpoint hashes have been
confirmed against the final-run manifest; see `models/libero10.json`. The frozen
task order and runtime still need reconciliation before executing the full matrix.

`results/libero_paper_aggregate.json` preserves the available aggregate, including
replicate counts. It is not a substitute for per-episode logs or checkpoint
provenance, and was not generated by testing this draft.
