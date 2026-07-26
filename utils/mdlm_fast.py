"""Compute-identical fast AR-order rollout for MDLM models.

The original utils/generate.py (LLaDA reference) computes full-sequence x
152k-vocab logits at EVERY denoising step, but with block_length=1 only the
current position's logits are ever used — on a 0.6B model that vocab projection
is ~80% of the FLOPs. This variant runs the same full-sequence transformer body
(required for MDLM's conditioning on trailing mask tokens) and applies lm_head
only at the current position.

Greedy (temperature=0) outputs are verified token-identical to the original
generate() in tests/test_mdlm_fast.py. At temperature>0 the sampling
distribution is identical (per-position Gumbel-max); only the RNG stream
differs, exactly like changing a seed.
"""

import torch
import torch.nn.functional as F


def _split_body_head(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


@torch.no_grad()
def generate_ar_mdlm_fast(model, prompt, gen_length, mask_id, temperature=1.0):
    """AR-order (block_length=1) MDLM rollout, same semantics as
    utils.generate.generate(..., block_length=1, remasking='low_confidence').

    prompt: [B, P]; returns [B, P + gen_length].
    """
    body, head = _split_body_head(model)
    B, P = prompt.shape
    device = prompt.device

    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt

    for t in range(gen_length):
        pos = P + t
        h = body(x).last_hidden_state[:, pos, :]
        logits = head(h)  # [B, V]

        if temperature == 0:
            next_tok = logits.argmax(dim=-1)
        else:
            # Same Gumbel-max scheme as utils.generate.add_gumbel_noise, applied
            # only at the position whose sample is actually consumed.
            logits64 = logits.to(torch.float64)
            noise = torch.rand_like(logits64)
            gumbel = (-torch.log(noise)) ** temperature
            next_tok = (logits64.exp() / gumbel).argmax(dim=-1)

        x[:, pos] = next_tok

    return x
