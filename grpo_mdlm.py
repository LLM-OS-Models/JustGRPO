"""Paper-identical JustGRPO for MDLM-style diffusion models
(e.g. dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1).

This is the ORIGINAL LLaDA recipe (see `git show 1a2fddb:grpo.py`) with only:
  - mask_id / eos_id parameterized (LLaDA hardcoded 126336),
  - responses truncated at the first EOS before reward computation,
  - the per-token loss loop optionally batched over `loss_chunk` positions per
    forward — mathematically identical inputs, just concatenated along the batch
    dim for GPU efficiency (verified against the naive loop in
    tests/test_mdlm_pipeline.py).

Rollout is AR-order (block_length=1) with the original `utils/generate.py`
(full bidirectional attention, exactly the paper's setup). Do NOT use this file
for BD3LM checkpoints — that is what grpo.py handles.
"""

import torch
import torch.nn.functional as F

from utils.generate import generate  # original reference (kept for tests/ablation)
from utils.mdlm_fast import generate_ar_mdlm_fast


@torch.no_grad()
def sample(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1.,
           steps=256, gen_length=256, mask_id=None, eos_id=None):
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Rollout with AR order (block_length=1). generate_ar_mdlm_fast is
    # compute-identical to the original generate(..., block_length=1) — greedy
    # outputs verified token-identical — it just skips the vocab projection at
    # positions whose logits are discarded (~1.5-2x wall-clock).
    generated_ids = generate_ar_mdlm_fast(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                                          gen_length=gen_length, temperature=temperature,
                                          mask_id=mask_id)

    # Truncate at EOS before decoding so post-EOS continuation can't confuse the reward
    responses = []
    for row in generated_ids[:, prompt_ids.shape[1]:].tolist():
        if eos_id is not None and eos_id in row:
            row = row[:row.index(eos_id)]
        responses.append(tokenizer.decode(row, skip_special_tokens=True))

    return {
        'generated_ids': generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, num_generations, device).float(),
    }


def logprob_loss(model, inputs, valid_samples, eps=0.2, gain=1.0, temperature=1., accelerator=None,
                 gen_length=256, mask_id=None, loss_chunk=4):
    advantages, generated_ids, prompt_len = inputs['advantages'], inputs['generated_ids'], inputs['prompt_len']
    batch_size, device = advantages.shape[0], generated_ids.device
    prompt_ids, completion_ids = generated_ids[:, :prompt_len], generated_ids[:, prompt_len:]

    valid_samples = accelerator.gather(valid_samples).float().mean().item()
    scale = gain / gen_length / (valid_samples + 1e-5)

    for t0 in range(0, gen_length, loss_chunk):
        ts = list(range(t0, min(t0 + loss_chunk, gen_length)))

        # Original per-t input: [prompt, comp[:t], MASK x (G - t)] — full bidirectional
        # attention (no attn mask), exactly the paper. Variants for the ts in this
        # chunk are concatenated along the batch dim.
        xs = [torch.cat([prompt_ids, completion_ids[:, :t],
                         torch.full((batch_size, gen_length - t), mask_id,
                                    device=device, dtype=generated_ids.dtype)], dim=1)
              for t in ts]
        x_cat = torch.cat(xs, dim=0)  # [B * K, P + G]

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # logits_to_keep=G keeps only the completion region (saves the 152k-vocab
            # projection over the prompt); position P + t maps to index t within it.
            logits = model(x_cat, logits_to_keep=gen_length).logits / temperature  # [B*K, G, V]

        sel = torch.cat([logits[k * batch_size:(k + 1) * batch_size, t, :]
                         for k, t in enumerate(ts)], dim=0)          # [B*K, V]
        log_prob = F.log_softmax(sel.float(), dim=-1)
        tok = torch.cat([completion_ids[:, t:t + 1] for t in ts], dim=0)  # [B*K, 1]
        token_log_prob = log_prob.gather(-1, tok).squeeze(-1)

        adv = advantages.repeat(len(ts))
        ratio = (token_log_prob - token_log_prob.detach()).exp()
        clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
        loss = -torch.min(ratio * adv, clipped_ratio * adv)

        accelerator.backward(loss.mul(scale).sum())

    return {
        "reward": accelerator.gather(inputs['rewards'].detach()).mean().item(),
        "valid_samples": valid_samples,
    }


def compute_group_advantages(rewards, group_size):
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    return (rewards - mean) / (std + 1e-4)
