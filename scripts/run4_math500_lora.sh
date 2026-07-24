#!/usr/bin/env bash
# Run 4 (optional) — single-domain MATH-500, LoRA, 200 steps. Completes the matrix.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train.py \
  --model_path "$MODEL_PATH" \
  --dataset math \
  --grad_accum 8 \
  --lora \
  --total_steps 200 \
  --run_dir "$RUNS_DIR/run4-math500-lora" \
  2>&1 | tee -a "$RUNS_DIR/run4-math500-lora-train.log"
