"""Merge a trained LoRA adapter into the BD3LM base model for evaluation.

Usage:
  python scripts/merge_lora.py --adapter ~/data/runs/run1-gsm8k-lora/ckpt-000200 \
      --out ~/data/models/run1-gsm8k-lora-merged
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/ubuntu/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForMaskedLM.from_pretrained(
        args.base, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.out)

    # save_pretrained does not carry the remote-code module or chat template along
    for fname in ["modeling_qwen3.py", "chat_template.jinja"]:
        src = Path(args.base) / fname
        if src.exists():
            shutil.copy(src, Path(args.out) / fname)

    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
