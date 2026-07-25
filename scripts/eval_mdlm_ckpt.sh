#!/usr/bin/env bash
# Evaluate an MDLM-family (merged) model on the 4 JustGRPO tasks with the dllm
# framework. Same as eval_ckpt.sh but --model a2d_mdlm and block_size=256
# (MDLM = full-sequence diffusion), matching dLLM's official mdlm eval settings.
#
# Usage: bash scripts/eval_mdlm_ckpt.sh <model_dir> <result_name>
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

COMMON_MODEL_ARGS="pretrained=${MODEL},max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0"

run_task () {
  local task=$1 port=$2; shift 2
  accelerate launch --num_processes 1 --main_process_port "$port" \
    dllm/pipelines/a2d/eval.py \
    --tasks "$task" --model a2d_mdlm --apply_chat_template \
    --num_fewshot 0 --batch_size 8 \
    --model_args "$COMMON_MODEL_ARGS" \
    --output_path "$OUT" \
    "$@" > "$OUT/${task}.log" 2>&1
  echo "TASK ${task} EXITED WITH CODE $?"
}

run_task gsm8k_cot          29561 &
run_task humaneval_instruct 29562 --confirm_run_unsafe_code &
wait
echo "STAGE1 DONE"
run_task mbpp_instruct 29563 --confirm_run_unsafe_code &
run_task minerva_math  29564 &
wait
echo "ALL MDLM EVAL DONE -> $OUT"
