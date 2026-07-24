#!/usr/bin/env bash
# Run 1 — single-domain GSM8K (math), LoRA (r=128, lr 5e-5), 200 steps.
# Math baseline: per-task JustGRPO recipe on Qwen3-0.6B BD3LM.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train.py \
  --model_path "$MODEL_PATH" \
  --dataset gsm8k \
  --grad_accum 8 \
  --lora \
  --total_steps 200 \
  --run_dir "$RUNS_DIR/run1-gsm8k-lora" \
  2>&1 | tee -a "$RUNS_DIR/run1-gsm8k-lora-train.log"
