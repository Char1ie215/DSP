#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

CONFIG_NAME="${CONFIG_NAME:-train_diffpolicy}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/libero/libero10_N500.zarr}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/oat/libero10_paired_diffpolicy}"
TASK_POLICY_CONFIG="${TASK_POLICY_CONFIG:-libero/libero10}"
ACTION_KEY="${ACTION_KEY:?ACTION_KEY is required}"
ACTION_DIM="${ACTION_DIM:-13}"
DATASET_TARGET="${DATASET_TARGET:-dsp_diffusion.dataset.zarr_dataset.ZarrDataset}"
SEED="${SEED:-42}"
NUM_DEMO="${NUM_DEMO:-500}"
NUM_EPOCHS="${NUM_EPOCHS:-1001}"
HORIZON="${HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
GRADIENT_ACCUMULATE_EVERY="${GRADIENT_ACCUMULATE_EVERY:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-null}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-null}"
MAX_RECONST_STEPS="${MAX_RECONST_STEPS:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
LATEST_CHECKPOINT_EVERY="${LATEST_CHECKPOINT_EVERY:-${CHECKPOINT_EVERY}}"
VAL_EVERY="${VAL_EVERY:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
ROLLOUT_EVERY="${ROLLOUT_EVERY:-200}"
LOGGING_MODE="${LOGGING_MODE:-disabled}"
TRAINING_RESUME="${TRAINING_RESUME:-true}"
ALLOW_BF16="${ALLOW_BF16:-true}"
USE_EMA="${USE_EMA:-true}"
POLICY_LR="${POLICY_LR:-5.0e-5}"
OBS_ENC_LR="${OBS_ENC_LR:-1.0e-5}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
NOISE_TRAIN_TIMESTEPS="${NOISE_TRAIN_TIMESTEPS:-100}"
CHECKPOINT_MONITOR_KEY="${CHECKPOINT_MONITOR_KEY:-mean_success_rate}"
CHECKPOINT_MONITOR_MODE="${CHECKPOINT_MONITOR_MODE:-max}"
CHECKPOINT_TOPK="${CHECKPOINT_TOPK:-3}"
CHECKPOINT_FORMAT_STR="${CHECKPOINT_FORMAT_STR:-}"
if [[ -z "${CHECKPOINT_FORMAT_STR}" ]]; then
  CHECKPOINT_FORMAT_STR='ep-{epoch:04d}_sr-{mean_success_rate:.3f}_vl-{val_loss:.6f}.ckpt'
fi
LAZY_EVAL="${LAZY_EVAL:-false}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-1}"
ACCELERATE_NUM_MACHINES="${ACCELERATE_NUM_MACHINES:-1}"
ENV_RUNNER_N_TEST="${ENV_RUNNER_N_TEST:-500}"
ENV_RUNNER_N_TEST_VIS="${ENV_RUNNER_N_TEST_VIS:-0}"
ENV_RUNNER_N_PARALLEL_ENVS="${ENV_RUNNER_N_PARALLEL_ENVS:-20}"
ENV_RUNNER_TEST_START_SEED="${ENV_RUNNER_TEST_START_SEED:-200000}"
ROLLOUT_ADAPTER="${ROLLOUT_ADAPTER:-paired_pose_action_field}"
ROLLOUT_PLAN_STEPS="${ROLLOUT_PLAN_STEPS:-16}"
ROLLOUT_MAX_XYZ_ACTION="${ROLLOUT_MAX_XYZ_ACTION:-1.0}"
ROLLOUT_MAX_ROT_ACTION="${ROLLOUT_MAX_ROT_ACTION:-0.5}"
ROLLOUT_POSITION_ACTION_SCALE="${ROLLOUT_POSITION_ACTION_SCALE:-0.05}"
ROLLOUT_ROTATION_ACTION_SCALE="${ROLLOUT_ROTATION_ACTION_SCALE:-0.5}"
ROLLOUT_POSITION_TOLERANCE="${ROLLOUT_POSITION_TOLERANCE:-0.002}"
ROLLOUT_ROTATION_TOLERANCE="${ROLLOUT_ROTATION_TOLERANCE:-0.05}"
ROLLOUT_MAX_PLAN_STEPS="${ROLLOUT_MAX_PLAN_STEPS:-200}"
ROLLOUT_GRIPPER_ONLY_POSITION_THRESHOLD="${ROLLOUT_GRIPPER_ONLY_POSITION_THRESHOLD:-0.002}"
ROLLOUT_GRIPPER_ONLY_ROTATION_THRESHOLD="${ROLLOUT_GRIPPER_ONLY_ROTATION_THRESHOLD:-0.05}"
ROLLOUT_GRIPPER_CLOSE_COMMAND_THRESHOLD="${ROLLOUT_GRIPPER_CLOSE_COMMAND_THRESHOLD:-0.5}"
ROLLOUT_GATING_PRECISION_SCALE="${ROLLOUT_GATING_PRECISION_SCALE:-25.0}"
ROLLOUT_POSITION_RECOVERY_GAIN="${ROLLOUT_POSITION_RECOVERY_GAIN:-1.0}"
ROLLOUT_ROTATION_RECOVERY_GAIN="${ROLLOUT_ROTATION_RECOVERY_GAIN:-1.0}"
ROLLOUT_TERMINAL_STALL_REPLAN="${ROLLOUT_TERMINAL_STALL_REPLAN:-true}"
ROLLOUT_BSPLINE_SOURCE_HORIZON="${ROLLOUT_BSPLINE_SOURCE_HORIZON:-32}"
ROLLOUT_BSPLINE_DEGREE="${ROLLOUT_BSPLINE_DEGREE:-3}"
ROLLOUT_BSPLINE_SUPPORT_SUBDIVISIONS="${ROLLOUT_BSPLINE_SUPPORT_SUBDIVISIONS:-1}"
ROLLOUT_TUBE_PROJECTION_ENABLED="${ROLLOUT_TUBE_PROJECTION_ENABLED:-true}"
ROLLOUT_SUPPORT_INTERPOLATION_RCOND="${ROLLOUT_SUPPORT_INTERPOLATION_RCOND:-0.01}"
ROLLOUT_TUBE_RADIUS="${ROLLOUT_TUBE_RADIUS:-0.01}"
ROLLOUT_PHASE_CONDITIONED_EXECUTION="${ROLLOUT_PHASE_CONDITIONED_EXECUTION:-true}"
ROLLOUT_PHASE_POSITION_TUBE_RADIUS="${ROLLOUT_PHASE_POSITION_TUBE_RADIUS:-0.075}"
ROLLOUT_PHASE_ROTATION_TUBE_RADIUS="${ROLLOUT_PHASE_ROTATION_TUBE_RADIUS:-0.2}"
ROLLOUT_PHASE_POSITION_REENTRY_RADIUS="${ROLLOUT_PHASE_POSITION_REENTRY_RADIUS:-0.02}"
ROLLOUT_PHASE_ROTATION_REENTRY_RADIUS="${ROLLOUT_PHASE_ROTATION_REENTRY_RADIUS:-0.1}"
ROLLOUT_PHASE_PROJECTION_LOOKAHEAD="${ROLLOUT_PHASE_PROJECTION_LOOKAHEAD:-null}"
ROLLOUT_RECOVERY_PHASE_ADVANCE_BUDGET="${ROLLOUT_RECOVERY_PHASE_ADVANCE_BUDGET:-null}"
ROLLOUT_PHASE_MAX_RECOVERY_STEPS="${ROLLOUT_PHASE_MAX_RECOVERY_STEPS:-16}"
ROLLOUT_RECOVERY_FIELD_SOURCE="${ROLLOUT_RECOVERY_FIELD_SOURCE:-action}"
ROLLOUT_FREEZE_RECOVERY_PHASE="${ROLLOUT_FREEZE_RECOVERY_PHASE:-false}"
ROLLOUT_AUTONOMOUS_CHUNK_REENTRY="${ROLLOUT_AUTONOMOUS_CHUNK_REENTRY:-false}"
ROLLOUT_ROTATION_CLF_ENABLED="${ROLLOUT_ROTATION_CLF_ENABLED:-false}"
ROLLOUT_ROTATION_CLF_TUBE_RADIUS="${ROLLOUT_ROTATION_CLF_TUBE_RADIUS:-0.05}"
ROLLOUT_ROTATION_CLF_CONTRACTION_RATE="${ROLLOUT_ROTATION_CLF_CONTRACTION_RATE:-null}"
ROLLOUT_RECOVERY_PROGRESS_PATIENCE="${ROLLOUT_RECOVERY_PROGRESS_PATIENCE:-0}"
ROLLOUT_RECOVERY_PROGRESS_MIN_DELTA="${ROLLOUT_RECOVERY_PROGRESS_MIN_DELTA:-0.001}"
ROLLOUT_TUBE_CONTRACTION_RATE="${ROLLOUT_TUBE_CONTRACTION_RATE:-0.1}"
ROLLOUT_DIRECTION_PRESERVING_SATURATION="${ROLLOUT_DIRECTION_PRESERVING_SATURATION:-true}"

if [[ "${CHECKPOINT_MONITOR_KEY}" == "mean_success_rate" ]]; then
  if [[ "${LAZY_EVAL}" != "false" ]]; then
    echo "[paired-diffusion] mean_success_rate selection requires online rollout" >&2
    exit 2
  fi
  if (( CHECKPOINT_EVERY % ROLLOUT_EVERY != 0 )); then
    echo "[paired-diffusion] checkpoint_every must be divisible by rollout_every" >&2
    exit 2
  fi
fi
if (( CHECKPOINT_EVERY % VAL_EVERY != 0 )); then
  echo "[paired-diffusion] checkpoint_every must be divisible by val_every" >&2
  exit 2
fi

if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="${ROOT_DIR}/${DATA_PATH}"
fi
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${ROOT_DIR}/${OUT_DIR}"
fi

RUN_CMD=(
  scripts/run_workspace.py
  --config-name="${CONFIG_NAME}"
  task/policy="${TASK_POLICY_CONFIG}"
  task.policy.dataset.zarr_path="${DATA_PATH}"
  task.policy.dataset._target_="${DATASET_TARGET}"
  task.policy.dataset.action_key="${ACTION_KEY}"
  task.policy.shape_meta.action.shape="[${ACTION_DIM}]"
  task.policy.lazy_eval="${LAZY_EVAL}"
  seed="${SEED}"
  horizon="${HORIZON}"
  n_action_steps="${N_ACTION_STEPS}"
  n_obs_steps="${N_OBS_STEPS}"
  policy.num_inference_steps="${NUM_INFERENCE_STEPS}"
  policy.noise_scheduler.num_train_timesteps="${NOISE_TRAIN_TIMESTEPS}"
  training.max_train_steps="${MAX_TRAIN_STEPS}"
  training.max_val_steps="${MAX_VAL_STEPS}"
  training.max_reconst_steps="${MAX_RECONST_STEPS}"
  training.num_epochs="${NUM_EPOCHS}"
  training.resume="${TRAINING_RESUME}"
  training.rollout_every="${ROLLOUT_EVERY}"
  training.val_every="${VAL_EVERY}"
  training.sample_every="${SAMPLE_EVERY}"
  training.checkpoint_every="${CHECKPOINT_EVERY}"
  ++training.latest_checkpoint_every="${LATEST_CHECKPOINT_EVERY}"
  training.num_demo="${NUM_DEMO}"
  training.allow_bf16="${ALLOW_BF16}"
  training.use_ema="${USE_EMA}"
  training.gradient_accumulate_every="${GRADIENT_ACCUMULATE_EVERY}"
  optimizer.policy_lr="${POLICY_LR}"
  optimizer.obs_enc_lr="${OBS_ENC_LR}"
  dataloader.batch_size="${BATCH_SIZE}"
  dataloader.num_workers="${NUM_WORKERS}"
  dataloader.persistent_workers="${PERSISTENT_WORKERS}"
  val_dataloader.batch_size="${BATCH_SIZE}"
  val_dataloader.num_workers="${NUM_WORKERS}"
  val_dataloader.persistent_workers="${PERSISTENT_WORKERS}"
  logging.mode="${LOGGING_MODE}"
  checkpoint.topk.monitor_key="${CHECKPOINT_MONITOR_KEY}"
  checkpoint.topk.mode="${CHECKPOINT_MONITOR_MODE}"
  checkpoint.topk.k="${CHECKPOINT_TOPK}"
  checkpoint.topk.format_str="'${CHECKPOINT_FORMAT_STR}'"
  task.policy.env_runner.n_test="${ENV_RUNNER_N_TEST}"
  task.policy.env_runner.n_test_vis="${ENV_RUNNER_N_TEST_VIS}"
  task.policy.env_runner.n_parallel_envs="${ENV_RUNNER_N_PARALLEL_ENVS}"
  ++task.policy.env_runner.test_start_seed="${ENV_RUNNER_TEST_START_SEED}"
  hydra.run.dir="${OUT_DIR}"
)

if [[ -n "${ROLLOUT_ADAPTER}" && "${ROLLOUT_ADAPTER}" != "none" ]]; then
  RUN_CMD+=(
    +task.policy.rollout_adapter.type="${ROLLOUT_ADAPTER}"
    +task.policy.rollout_adapter.plan_steps="${ROLLOUT_PLAN_STEPS}"
    +task.policy.rollout_adapter.max_xyz_action="${ROLLOUT_MAX_XYZ_ACTION}"
    +task.policy.rollout_adapter.max_rot_action="${ROLLOUT_MAX_ROT_ACTION}"
    +task.policy.rollout_adapter.position_action_scale="${ROLLOUT_POSITION_ACTION_SCALE}"
    +task.policy.rollout_adapter.rotation_action_scale="${ROLLOUT_ROTATION_ACTION_SCALE}"
    +task.policy.rollout_adapter.position_tolerance="${ROLLOUT_POSITION_TOLERANCE}"
    +task.policy.rollout_adapter.rotation_tolerance="${ROLLOUT_ROTATION_TOLERANCE}"
    +task.policy.rollout_adapter.max_plan_steps="${ROLLOUT_MAX_PLAN_STEPS}"
    +task.policy.rollout_adapter.gripper_only_position_threshold="${ROLLOUT_GRIPPER_ONLY_POSITION_THRESHOLD}"
    +task.policy.rollout_adapter.gripper_only_rotation_threshold="${ROLLOUT_GRIPPER_ONLY_ROTATION_THRESHOLD}"
    +task.policy.rollout_adapter.gripper_close_command_threshold="${ROLLOUT_GRIPPER_CLOSE_COMMAND_THRESHOLD}"
    +task.policy.rollout_adapter.gating_precision_scale="${ROLLOUT_GATING_PRECISION_SCALE}"
    +task.policy.rollout_adapter.position_recovery_gain="${ROLLOUT_POSITION_RECOVERY_GAIN}"
    +task.policy.rollout_adapter.rotation_recovery_gain="${ROLLOUT_ROTATION_RECOVERY_GAIN}"
    +task.policy.rollout_adapter.terminal_stall_replan="${ROLLOUT_TERMINAL_STALL_REPLAN}"
    +task.policy.rollout_adapter.bspline_source_horizon="${ROLLOUT_BSPLINE_SOURCE_HORIZON}"
    +task.policy.rollout_adapter.bspline_degree="${ROLLOUT_BSPLINE_DEGREE}"
    +task.policy.rollout_adapter.bspline_support_subdivisions="${ROLLOUT_BSPLINE_SUPPORT_SUBDIVISIONS}"
    +task.policy.rollout_adapter.tube_projection_enabled="${ROLLOUT_TUBE_PROJECTION_ENABLED}"
    +task.policy.rollout_adapter.support_interpolation_rcond="${ROLLOUT_SUPPORT_INTERPOLATION_RCOND}"
    +task.policy.rollout_adapter.tube_radius="${ROLLOUT_TUBE_RADIUS}"
    +task.policy.rollout_adapter.phase_conditioned_execution="${ROLLOUT_PHASE_CONDITIONED_EXECUTION}"
    +task.policy.rollout_adapter.phase_position_tube_radius="${ROLLOUT_PHASE_POSITION_TUBE_RADIUS}"
    +task.policy.rollout_adapter.phase_rotation_tube_radius="${ROLLOUT_PHASE_ROTATION_TUBE_RADIUS}"
    +task.policy.rollout_adapter.phase_position_reentry_radius="${ROLLOUT_PHASE_POSITION_REENTRY_RADIUS}"
    +task.policy.rollout_adapter.phase_rotation_reentry_radius="${ROLLOUT_PHASE_ROTATION_REENTRY_RADIUS}"
    +task.policy.rollout_adapter.phase_projection_lookahead="${ROLLOUT_PHASE_PROJECTION_LOOKAHEAD}"
    +task.policy.rollout_adapter.recovery_phase_advance_budget="${ROLLOUT_RECOVERY_PHASE_ADVANCE_BUDGET}"
    +task.policy.rollout_adapter.phase_max_recovery_steps="${ROLLOUT_PHASE_MAX_RECOVERY_STEPS}"
    +task.policy.rollout_adapter.recovery_field_source="${ROLLOUT_RECOVERY_FIELD_SOURCE}"
    +task.policy.rollout_adapter.freeze_recovery_phase="${ROLLOUT_FREEZE_RECOVERY_PHASE}"
    +task.policy.rollout_adapter.autonomous_chunk_reentry="${ROLLOUT_AUTONOMOUS_CHUNK_REENTRY}"
    +task.policy.rollout_adapter.rotation_clf_enabled="${ROLLOUT_ROTATION_CLF_ENABLED}"
    +task.policy.rollout_adapter.rotation_clf_tube_radius="${ROLLOUT_ROTATION_CLF_TUBE_RADIUS}"
    +task.policy.rollout_adapter.rotation_clf_contraction_rate="${ROLLOUT_ROTATION_CLF_CONTRACTION_RATE}"
    +task.policy.rollout_adapter.recovery_progress_patience="${ROLLOUT_RECOVERY_PROGRESS_PATIENCE}"
    +task.policy.rollout_adapter.recovery_progress_min_delta="${ROLLOUT_RECOVERY_PROGRESS_MIN_DELTA}"
    +task.policy.rollout_adapter.tube_contraction_rate="${ROLLOUT_TUBE_CONTRACTION_RATE}"
    +task.policy.rollout_adapter.direction_preserving_saturation="${ROLLOUT_DIRECTION_PRESERVING_SATURATION}"
  )
fi

if (( ACCELERATE_NUM_PROCESSES > 1 )); then
  accelerate launch \
    --num_machines "${ACCELERATE_NUM_MACHINES}" \
    --multi_gpu \
    --num_processes "${ACCELERATE_NUM_PROCESSES}" \
    "${RUN_CMD[@]}"
else
  python "${RUN_CMD[@]}"
fi

OUT_DIR="${OUT_DIR}" CHECKPOINT_MONITOR_KEY="${CHECKPOINT_MONITOR_KEY}" python - <<'PY'
import os
import re
from pathlib import Path

ckpt_dir = Path(os.environ["OUT_DIR"]) / "checkpoints"
monitor_key = os.environ["CHECKPOINT_MONITOR_KEY"]
if monitor_key == "mean_success_rate":
    pattern = re.compile(
        r"^ep-(?P<epoch>\d+)_sr-(?P<success>[0-9.]+)_vl-(?P<val>[0-9.]+)\.ckpt$"
    )
    candidates = []
    for path in ckpt_dir.glob("ep-*_sr-*_vl-*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append(
                (
                    -float(match.group("success")),
                    float(match.group("val")),
                    int(match.group("epoch")),
                    path,
                )
            )
    destination = ckpt_dir / "best.ckpt"
    label = "rollout-ranked"
else:
    pattern = re.compile(r"^ep-(?P<epoch>\d+)_vl-(?P<val>[0-9.]+)\.ckpt$")
    candidates = []
    for path in ckpt_dir.glob("ep-*_vl-*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append(
                (
                    float(match.group("val")),
                    int(match.group("epoch")),
                    path,
                )
            )
    destination = ckpt_dir / "best_val.ckpt"
    label = "validation-ranked"

if not candidates:
    raise RuntimeError(f"No {label} checkpoint found in {ckpt_dir}")
best = min(candidates)[-1]
destination.unlink(missing_ok=True)
destination.symlink_to(best.name)
print(f"[paired-diffusion] linked {destination.name} -> {best.name}", flush=True)
PY
