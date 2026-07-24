"""Verification for the BD3LM adaptation (run on a GPU box):

1. one-pass AR logprobs == per-token loop logprobs (BD3LM concat-mask trick correctness)
2. AR rollout produces coherent text via the chat template
3. LoRA backward pass populates adapter gradients

Usage: /home/ubuntu/dllm/.venv/bin/python tests/test_adaptation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils.ar_diffusion import causal_mask_4d, generate_ar, ar_logprobs_onepass

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/ubuntu/data/models/Qwen3-0.6B-diffusion-bd3lm-v0.1")


def loop_logprobs(model, prompt_ids, completion_ids, mask_id):
    """Reference: per-token forward loop (original JustGRPO style, causal mask)."""
    B, P = prompt_ids.shape
    G = completion_ids.shape[1]
    device = prompt_ids.device
    out = torch.zeros(B, G, device=device)
    for t in range(G):
        x = torch.cat([prompt_ids, completion_ids[:, :t],
                       torch.full((B, 1), mask_id, dtype=torch.long, device=device)], dim=1)
        L = x.shape[1]
        logits = model(input_ids=x,
                       attention_mask=causal_mask_4d(L, device),
                       position_ids=torch.arange(L, device=device).view(1, L).expand(B, L),
                       logits_to_keep=1).logits[:, -1, :]
        lp = F.log_softmax(logits.float(), dim=-1)
        out[:, t] = lp.gather(-1, completion_ids[:, t:t + 1]).squeeze(-1)
    return out


def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    print(f"mask_id={mask_id} eos_id={eos_id}")
    assert mask_id is not None

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she "
          "sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"}],
        add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # --- Test 2 first (also produces a real completion for test 1) ---
    print("\n[test 2] AR rollout (greedy, 96 tokens)...")
    with torch.no_grad():
        gen = generate_ar(model, prompt_ids, gen_length=96, mask_id=mask_id, temperature=0.0)
    text = tokenizer.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"  output: {text!r}")
    assert len(text.strip()) > 10, "rollout produced (near-)empty text"

    # --- Test 1: one-pass vs loop equivalence on the sampled completion ---
    # Run in fp32: if the mask logic is right the two paths are mathematically identical,
    # so any fp32 difference beyond ~1e-3 indicates a real bug (bf16 shows ~0.1 noise).
    print("\n[test 1] one-pass vs per-token-loop logprob equivalence (G=32, fp32)...")
    completion_ids = gen[:, prompt_ids.shape[1]:prompt_ids.shape[1] + 32]
    model.float()
    with torch.no_grad():
        lp_fast = ar_logprobs_onepass(model, prompt_ids, completion_ids, mask_id=mask_id)
        lp_slow = loop_logprobs(model, prompt_ids, completion_ids, mask_id=mask_id)
    diff = (lp_fast - lp_slow).abs().max().item()
    print(f"  max |diff| = {diff:.6f}  (fast[0,:5]={lp_fast[0, :5].tolist()})")
    assert diff < 1e-3, f"one-pass logprobs diverge from loop reference: {diff}"
    model.bfloat16()

    # --- Test 3: LoRA backward ---
    print("\n[test 3] LoRA backward...")
    from peft import LoraConfig, get_peft_model
    lora_model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM"))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lp = ar_logprobs_onepass(lora_model, prompt_ids, completion_ids, mask_id=mask_id)
        loss = -(lp.mean())
    loss.backward()
    grads = [p.grad for n, p in lora_model.named_parameters() if "lora_" in n and p.grad is not None]
    total = sum(g.abs().sum().item() for g in grads)
    print(f"  lora params with grad: {len(grads)}, total |grad| = {total:.4f}")
    assert len(grads) > 0 and total > 0, "no gradients reached LoRA adapters"

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
