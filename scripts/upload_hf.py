"""Upload a finished run (LoRA adapter + merged model) to the LLM-OS-Models2 HF org,
with a model card covering provenance, method, usage, and results.

Usage:
  export HF_TOKEN=...   # or keep it in the repo .env
  python scripts/upload_hf.py --run run1-gsm8k-lora --dataset gsm8k \
      --adapter ~/data/runs/run1-gsm8k-lora/ckpt-000200 \
      --merged ~/data/models/run1-gsm8k-lora-merged \
      [--results "GSM8K 45.72->XX.X, ..."]
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ORG = "LLM-OS-Models2"
BASE = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
REPO_CODE = "https://github.com/LLM-OS-Models/JustGRPO"

CARD = """---
license: apache-2.0
base_model: {base}
tags:
  - diffusion-language-model
  - block-diffusion
  - bd3lm
  - grpo
  - rlvr
  - lora
  - qwen3
language:
  - en
---

# {repo_name}

**JustGRPO-style RL (LoRA)** applied to the tiny block-diffusion LM
[`{base}`](https://huggingface.co/{base}).
Trained on **{dataset_desc}** with verifiable rewards.

- Method & code: [{repo_code}]({repo_code})
- Papers this work builds on:
  [JustGRPO (arXiv:2601.15165)](https://arxiv.org/abs/2601.15165) ·
  [dLLM (arXiv:2602.22661)](https://arxiv.org/abs/2602.22661) ·
  [BD3LM (arXiv:2503.09573)](https://arxiv.org/abs/2503.09573)

## How it was trained

- Rollouts: native block-diffusion sampling (block_size 32, temperature 1.0,
  256 denoising steps, gen length 256) — AR-order rollout collapses on this base
  model, see the repo's ADAPTATION.md for the full analysis.
- Loss: exact autoregressive log-likelihood of each sampled token, computed in a
  single forward pass via the BD3LM `[x0 || xt]` concat-attention trick
  (mathematically identical to the per-token loop; verified to 2e-5 in fp32),
  weighted by GRPO group-normalized advantages with PPO-style clipping.
- LoRA r=128, alpha=64, dropout 0.05 on q/k/v/o/up/down/gate projections,
  lr 5e-5, {steps} steps, 8 prompts x 16 rollouts per step, 1x H100.
- Rewards: {reward_desc}

## Results

{results}

## Usage

This repo contains {content_desc}. Generation uses block diffusion (NOT vanilla
`model.generate`); the easiest path is the [dllm](https://github.com/ZHZisZZ/dllm)
sampler:

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig

tok = AutoTokenizer.from_pretrained("{org}/{repo_name}")
model = AutoModelForMaskedLM.from_pretrained(
    "{org}/{repo_name}", trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda().eval()

sampler = BD3LMSampler(model=model, tokenizer=tok)
prompt = tok.apply_chat_template(
    [[{{"role": "user", "content": "Natalia sold clips to 48 friends in April, "
       "then half as many in May. How many altogether?"}}]],
    add_generation_prompt=True, tokenize=True)
seqs = sampler.sample(prompt, config=BD3LMSamplerConfig(
    max_new_tokens=256, steps=256, block_size=32, temperature=0.0))
print(tok.decode(seqs[0], skip_special_tokens=True))
```

To evaluate: use the dllm eval harness
(`--model a2d_bd3lm`, `max_new_tokens=256,steps=256,block_size=32`) as described in
[{repo_code}]({repo_code}).
"""

DATASET_DESC = {
    "gsm8k": "GSM8K train split (7.4k grade-school math problems)",
    "math": "MATH training problems (ankner/math-500 train split)",
    "code": "AceCode-Hard 21K (verifiable unit-test coding problems)",
    "mixed": "a round-robin mix of GSM8K + MATH + AceCode-Hard 21K (multi-domain RLVR)",
}
REWARD_DESC = {
    "gsm8k": "+1 correct / -1 incorrect (final-answer match)",
    "math": "+1 correct / -1 incorrect (math_equal on boxed answers)",
    "code": "format reward (0-1) + unit-test pass rate (0-1)",
    "mixed": "per-domain: math +-1 exact match; code format + unit-test pass rate",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. run1-gsm8k-lora")
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DESC))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", default=None)
    ap.add_argument("--steps", default="200")
    ap.add_argument("--results", default="Evaluation in progress — see the GitHub repo's BENCHMARKS.md for the live table.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    assert token, "set HF_TOKEN"
    api = HfApi(token=token)

    def push(repo_name, folder, content_desc):
        repo_id = f"{ORG}/{repo_name}"
        api.create_repo(repo_id, private=False, exist_ok=True)
        card = CARD.format(
            base=BASE, repo_name=repo_name, org=ORG, repo_code=REPO_CODE,
            dataset_desc=DATASET_DESC[args.dataset], reward_desc=REWARD_DESC[args.dataset],
            steps=args.steps, results=args.results, content_desc=content_desc,
        )
        card_path = Path(folder) / "README.md"
        card_path.write_text(card)
        api.upload_folder(repo_id=repo_id, folder_path=folder)
        print(f"uploaded -> https://huggingface.co/{repo_id}")

    base_name = f"Qwen3-0.6B-diffusion-bd3lm-justgrpo-{args.run}"
    push(f"{base_name}-adapter", args.adapter, "the LoRA adapter only (load onto the base model with PEFT)")
    if args.merged:
        push(base_name, args.merged, "the merged full model (base + LoRA)")


if __name__ == "__main__":
    main()
