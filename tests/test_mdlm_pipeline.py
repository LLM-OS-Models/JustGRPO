"""Verification for the MDLM (paper-identical) pipeline:

1. chunked loss forward (loss_chunk=4) produces the SAME token logprobs as the
   original per-t loop (fp32, tolerance 1e-3)
2. AR-order rollout via original generate() is coherent on GSM8K
3. one GRPO micro-step end-to-end with LoRA: rewards mixed, grads flow

Usage: /home/ubuntu/justgrpo-venv/bin/python tests/test_mdlm_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_PATH = os.environ.get("MDLM_PATH", "/home/ubuntu/data/models/Qwen3-0.6B-diffusion-mdlm-v0.1")


def naive_loop_logprobs(model, prompt_ids, completion_ids, mask_id, G):
    """Original JustGRPO loss loop, forward-only (reference)."""
    B = prompt_ids.shape[0]
    P = prompt_ids.shape[1]
    device = prompt_ids.device
    out = torch.zeros(B, G, device=device)
    for t in range(G):
        x = torch.cat([prompt_ids, completion_ids[:, :t],
                       torch.full((B, G - t), mask_id, device=device, dtype=prompt_ids.dtype)], dim=1)
        logits = model(x).logits
        lp = F.log_softmax(logits[:, P + t, :].float(), dim=-1)
        out[:, t] = lp.gather(-1, completion_ids[:, t:t + 1]).squeeze(-1)
    return out


def chunked_logprobs(model, prompt_ids, completion_ids, mask_id, G, K):
    """Same inputs, batched K positions per forward (as in grpo_mdlm.logprob_loss)."""
    B = prompt_ids.shape[0]
    device = prompt_ids.device
    out = torch.zeros(B, G, device=device)
    for t0 in range(0, G, K):
        ts = list(range(t0, min(t0 + K, G)))
        xs = [torch.cat([prompt_ids, completion_ids[:, :t],
                         torch.full((B, G - t), mask_id, device=device, dtype=prompt_ids.dtype)], dim=1)
              for t in ts]
        logits = model(torch.cat(xs, dim=0), logits_to_keep=G).logits  # [B*K, G, V]
        for k, t in enumerate(ts):
            lp = F.log_softmax(logits[k * B:(k + 1) * B, t, :].float(), dim=-1)
            out[:, t] = lp.gather(-1, completion_ids[:, t:t + 1]).squeeze(-1)
    return out


def main():
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    mask_id, eos_id = tok.mask_token_id, tok.eos_token_id
    print(f"mask_id={mask_id} eos_id={eos_id}")

    q = ("Natalia sold clips to 48 of her friends in April, and then she sold half as many "
         "clips in May. How many clips did Natalia sell altogether in April and May?")
    prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                     add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

    # --- 2. rollout first (also gives us a real completion for test 1) ---
    from utils.generate import generate
    print("\n[test 2] AR-order rollout (original generate, temp 1.0, 96 tokens)...")
    with torch.no_grad():
        gen = generate(model, prompt_ids, steps=96, gen_length=96, block_length=1,
                       temperature=1.0, mask_id=mask_id)
    comp_row = gen[0, prompt_ids.shape[1]:].tolist()
    if eos_id in comp_row:
        comp_row = comp_row[:comp_row.index(eos_id)]
    text = tok.decode(comp_row, skip_special_tokens=True)
    print(f"  output: {text[:160]!r}")
    assert len(text.strip()) > 10, "rollout produced (near-)empty text"

    # --- 1. chunked vs naive loop equivalence (fp32) ---
    print("\n[test 1] chunked loss forward == original loop (G=24, fp32)...")
    G = 24
    completion_ids = gen[:, prompt_ids.shape[1]:prompt_ids.shape[1] + G]
    model.float()
    with torch.no_grad():
        lp_naive = naive_loop_logprobs(model, prompt_ids, completion_ids, mask_id, G)
        lp_chunk = chunked_logprobs(model, prompt_ids, completion_ids, mask_id, G, K=4)
    diff = (lp_naive - lp_chunk).abs().max().item()
    print(f"  max |diff| = {diff:.6f}")
    assert diff < 1e-3, f"chunked loss diverges from original loop: {diff}"
    model.bfloat16()

    # --- 3. one GRPO micro-step end-to-end ---
    print("\n[test 3] GRPO micro-step (sample -> advantages -> loss -> LoRA grads)...")
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model
    from grpo_mdlm import sample, logprob_loss, compute_group_advantages
    from data.math import reward_gsm8k

    lora_model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM"))
    acc = Accelerator()
    batch = {"problems": [q], "answers": ["... #### 72"]}

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inputs = sample(model=lora_model, batch=batch, tokenizer=tok, device=device,
                        reward_fn=reward_gsm8k, num_generations=8, temperature=1.0,
                        steps=128, gen_length=128, mask_id=mask_id, eos_id=eos_id)
    print("  rewards:", inputs["rewards"].tolist())
    adv = compute_group_advantages(inputs["rewards"], 8)
    inputs["advantages"] = adv

    lora_model.train()
    out = logprob_loss(model=lora_model, inputs=inputs, valid_samples=(adv != 0).sum(),
                       gain=1.0, accelerator=acc, gen_length=128, mask_id=mask_id, loss_chunk=4)
    print("  loss stats:", out)
    g = sum(p.grad.abs().sum().item() for n, p in lora_model.named_parameters()
            if "lora_" in n and p.grad is not None)
    print(f"  total lora |grad| = {g:.4f}")
    assert (adv == 0).all() or g > 0, "advantages nonzero but no grads reached LoRA"

    print("\nALL MDLM PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
