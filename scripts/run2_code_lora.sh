#!/usr/bin/env bash
# Run 2 — single-domain code (AceCode-Hard 21K), LoRA, 200 steps.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train.py \
  --model_path "$MODEL_PATH" \
  --dataset code \
  --code_data_path datasets/acecode_hard.jsonl \
  --grad_accum 8 \
  --lora \
  --total_steps 200 \
  --run_dir "$RUNS_DIR/run2-code-lora" \
  2>&1 | tee -a "$RUNS_DIR/run2-code-lora-train.log"
