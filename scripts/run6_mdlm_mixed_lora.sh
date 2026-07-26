#!/usr/bin/env bash
# Run 6 — MDLM twin model, MIXED domains (gsm8k+math+code round-robin), LoRA, 300 steps.
# Paper-identical recipe (train_mdlm.py): AR-order rollout + original per-token loss.
# Compared against Run 5 (MDLM GSM8K single) for the single-vs-mixed question on MDLM.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train_mdlm.py \
  --model_path /home/ubuntu/data/models/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --dataset mixed \
  --code_data_path datasets/acecode_hard.jsonl \
  --grad_accum 8 \
  --lora \
  --total_steps 300 \
  --run_dir "$RUNS_DIR/run6-mdlm-mixed-lora" \
  2>&1 | tee -a "$RUNS_DIR/run6-mdlm-mixed-lora-train.log"
