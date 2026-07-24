#!/usr/bin/env bash
# Shared environment for all runs (1x H100 box)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source /home/ubuntu/justgrpo-venv/bin/activate
export HF_HOME=/home/ubuntu/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH=/home/ubuntu/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1
RUNS_DIR=/home/ubuntu/data/runs
mkdir -p "$RUNS_DIR"
