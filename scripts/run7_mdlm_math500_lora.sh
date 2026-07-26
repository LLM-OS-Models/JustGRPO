#!/usr/bin/env bash
# Run 7 — MDLM twin model, MATH(수학응용, ankner/math-500 train) 단독, LoRA, 200 steps.
# Paper-identical recipe (train_mdlm.py). Launch after Run 5 frees its slot.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train_mdlm.py \
  --model_path /home/ubuntu/data/models/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --dataset math \
  --grad_accum 8 \
  --lora \
  --total_steps 200 \
  --run_dir "$RUNS_DIR/run7-mdlm-math500-lora" \
  2>&1 | tee -a "$RUNS_DIR/run7-mdlm-math500-lora-train.log"
