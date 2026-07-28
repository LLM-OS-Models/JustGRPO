#!/usr/bin/env bash
# Run 8 — MDLM twin model, 코드(AceCode-Hard 21K) 단독, LoRA, 200 steps.
# Paper-identical recipe (train_mdlm.py). Launch after Run 6 frees its slot.
set -e
source "$(dirname "$0")/common.sh"

accelerate launch --num_processes 1 train_mdlm.py \
  --model_path /home/ubuntu/data/models/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --dataset code \
  --code_data_path datasets/acecode_hard.jsonl \
  --grad_accum 8 \
  --lora \
  --total_steps 150 \
  --save_every 50 \
  --num_generations 16 \
  --repeat_times 1 \
  --loss_chunk 2 \
  --run_dir "$RUNS_DIR/run8-mdlm-code-lora" \
  2>&1 | tee -a "$RUNS_DIR/run8-mdlm-code-lora-train.log"
