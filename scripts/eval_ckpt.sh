#!/usr/bin/env bash
# Evaluate a (merged) model dir on the 4 JustGRPO tasks with the dllm framework.
# Runs the 4 tasks in parallel on one GPU (batch 16 per task — fits H100 80GB).
#
# Usage: bash scripts/eval_ckpt.sh <model_dir> <result_name>
#   e.g. bash scripts/eval_ckpt.sh ~/data/models/run1-gsm8k-lora-merged run1-gsm8k-lora
set -e
MODEL=${1:?model dir}
NAME=${2:?result name}

DLLM_DIR=/home/ubuntu/dllm
OUT=/home/ubuntu/data/bench/$NAME
mkdir -p "$OUT"
cd "$DLLM_DIR"
source .venv/bin/activate

export PYTHONPATH=$DLLM_DIR:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export HF_HOME=/home/ubuntu/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_MODEL_ARGS="pretrained=${MODEL},max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0"

run_task () {
  local task=$1 port=$2; shift 2
  accelerate launch --num_processes 1 --main_process_port "$port" \
    dllm/pipelines/a2d/eval.py \
    --tasks "$task" --model a2d_bd3lm --apply_chat_template \
    --num_fewshot 0 --batch_size 16 \
    --model_args "$COMMON_MODEL_ARGS" \
    --output_path "$OUT" \
    "$@" > "$OUT/${task}.log" 2>&1
  echo "TASK ${task} EXITED WITH CODE $?"
}

run_task gsm8k_cot          29531 &
run_task humaneval_instruct 29532 --confirm_run_unsafe_code &
run_task mbpp_instruct      29533 --confirm_run_unsafe_code &
run_task minerva_math       29534 &
wait
echo "ALL 4 TASKS DONE -> $OUT"
