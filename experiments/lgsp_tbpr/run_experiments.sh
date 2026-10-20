#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-frequency}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  baseline) ID=E0 ;;
  local) ID=E1 ;;
  frequency) ID=E2 ;;
  both) ID=E3 ;;
  *) echo "Usage: $0 {baseline|local|frequency|both}" >&2; exit 2 ;;
esac
: "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to the existing IRRA best.pth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG_ARGS=()
if [ -n "${BASE_CONFIG:-}" ]; then CONFIG_ARGS=(--config_file "$BASE_CONFIG"); fi
python train.py "${CONFIG_ARGS[@]}" --name "${ID}_${MODE}" --lgsp_mode "$MODE" \
  --init_checkpoint "$INIT_CHECKPOINT" --img_aug --batch_size "${BATCH_SIZE:-64}" \
  --MLM --loss_names sdm+mlm+id --pretrain_choice ViT-B/16 \
  --dataset_name ICFG-PEDES --root_dir "${DATA_ROOT:-/kaggle/working/data}" \
  --num_epoch "${EPOCHS:-5}" --warmup_epochs 1 --lr 1e-5 --lgsp_lr 1e-3 \
  --seed "${SEED:-42}" --deterministic \
  --val_dataset "${VAL_DATASET:-holdout}" --holdout_seed 42 --holdout_fraction 0.1 \
  --test_batch_size "${TEST_BATCH_SIZE:-128}" \
  --output_dir "${OUTPUT_ROOT:-/kaggle/working/IRRA_LGSP_runs}/${ID}_${MODE}" "$@"
