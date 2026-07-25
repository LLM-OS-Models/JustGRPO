#!/usr/bin/env bash
# Run 5 — MDLM twin model (Qwen3-0.6B-diffusion-mdlm-v0.1), GSM8K, LoRA, 200 steps.
# PAPER-IDENTICAL recipe: AR-order rollout (block=1, original generate()) +
# original per-token bidirectional loss (grpo_mdlm.py / train_mdlm.py).
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train_mdlm.py \
  --model_path /home/ubuntu/data/models/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --dataset gsm8k \
  --grad_accum 8 \
  --lora \
  --total_steps 200 \
  --run_dir "$RUNS_DIR/run5-mdlm-gsm8k-lora" \
  2>&1 | tee -a "$RUNS_DIR/run5-mdlm-gsm8k-lora-train.log"
