#!/usr/bin/env bash
# Run 3 — mixed-domain (gsm8k + math + code round-robin), LoRA, 300 steps.
# 300 (not 200) so each domain gets comparable exposure to the single-domain runs.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train.py \
  --model_path "$MODEL_PATH" \
  --dataset mixed \
  --code_data_path datasets/acecode_hard.jsonl \
  --grad_accum 8 \
  --lora \
  --total_steps 300 \
  --run_dir "$RUNS_DIR/run3-mixed-lora" \
  2>&1 | tee -a "$RUNS_DIR/run3-mixed-lora-train.log"
