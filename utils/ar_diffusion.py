"""AR-order rollout and one-pass logprob utilities for BD3LM-style diffusion models
(e.g. dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1).

Unlike LLaDA (full bidirectional attention, mask id 126336), BD3LM models are trained
with block-causal attention and need an explicit 4D attention mask + position_ids.
With block size 1, block-causal attention reduces to standard causal attention, which
is exactly JustGRPO's AR training mode.
"""

import torch
import torch.nn.functional as F


def causal_mask_4d(seq_len, device):
    """Bool [1, 1, T, T] lower-triangular mask (True = attend). Shared across batch."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril_().view(1, 1, seq_len, seq_len)


@torch.no_grad()
def generate_ar(model, prompt, gen_length, mask_id, temperature=1.0,
                eos_id=None, min_new_tokens=0):
    """AR-order (block size 1) rollout for a BD3LM model.

    At step t the sequence is [prompt, sampled tokens < t, <mask>]; causal attention
    makes the mask position attend to the clean prefix and itself, so its logits give
    p(token_t | prefix) — the exact AR conditional.

    min_new_tokens: suppress eos_id for the first N sampled tokens. Needed for this
    model family: BD3LM SFT pads the final block with EOS, which inflates the EOS
    probability (~49% at the very first AR position) and otherwise yields empty rollouts.

    Args:
        prompt: [B, P] tensor (rows may repeat for group sampling).
    Returns:
        [B, P + gen_length] generated ids.
    """
    B, P = prompt.shape
    T = P + gen_length
    device = prompt.device

    x = torch.full((B, T), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt
    full_mask = causal_mask_4d(T, device)
    full_pos = torch.arange(T, device=device).view(1, T).expand(B, T)

    for t in range(P, T):
        # logits_to_keep=1: only the lm_head projection for the last (mask) position —
        # significant with this model's 152k vocab
        logits = model(
            input_ids=x[:, : t + 1],
            attention_mask=full_mask[:, :, : t + 1, : t + 1],
            position_ids=full_pos[:, : t + 1],
            logits_to_keep=1,
        ).logits[:, -1, :]

        if eos_id is not None and (t - P) < min_new_tokens:
            logits[:, eos_id] = -float("inf")

        if temperature > 0:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tok = logits.argmax(dim=-1)
        x[:, t] = next_tok

    return x


def build_onepass_inputs(prompt_ids, completion_ids, mask_id):
    """Build the concatenated sequence + mask that yields every AR conditional
    p(y_t | prompt, y_<t) in a single forward pass.

    Layout: [x0 | xt] where x0 = prompt + clean completion (length T = P + G) and
    xt = G mask tokens standing in for the completion. This is the BD3LM training
    trick (arXiv:2503.09573) specialized to block size 1 with all completion tokens
    masked; the xt copy of the prompt is dropped since nothing attends to it.

    Attention rule (True = attend):
      - x0 query i  -> x0 keys with pos <= i (causal)
      - xt query at logical pos p -> x0 keys with pos < p, plus itself
      - xt keys are never attended by other positions

    Returns:
        input_ids [B, T+G], attn_mask [1, 1, T+G, T+G], position_ids [B, T+G]
    """
    B, P = prompt_ids.shape
    G = completion_ids.shape[1]
    T = P + G
    S = T + G
    device = prompt_ids.device

    input_ids = torch.cat(
        [prompt_ids, completion_ids,
         torch.full((B, G), mask_id, dtype=torch.long, device=device)], dim=1)

    # logical positions: x0 half 0..T-1, xt half P..T-1
    lp = torch.cat([torch.arange(T, device=device), torch.arange(P, T, device=device)])
    is_xt = torch.zeros(S, dtype=torch.bool, device=device)
    is_xt[T:] = True

    allow = (lp.view(1, S) < lp.view(S, 1)) & ~is_xt.view(1, S)
    allow |= torch.eye(S, dtype=torch.bool, device=device)
    attn_mask = allow.view(1, 1, S, S)

    position_ids = lp.view(1, S).expand(B, S)
    return input_ids, attn_mask, position_ids


def ar_logprobs_onepass(model, prompt_ids, completion_ids, mask_id, temperature=1.0):
    """Log-probs of every completion token under the AR factorization, one forward.

    Returns [B, G] token log-probs (fp32). Gradients flow through the model call.
    """
    B, P = prompt_ids.shape
    G = completion_ids.shape[1]
    input_ids, attn_mask, position_ids = build_onepass_inputs(prompt_ids, completion_ids, mask_id)

    # The xt half occupies the last G positions, so logits_to_keep=G projects exactly
    # the positions we need (saves a [B, P+G, 152k] logits allocation).
    logits = model(
        input_ids=input_ids, attention_mask=attn_mask, position_ids=position_ids,
        logits_to_keep=G,
    ).logits / temperature

    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
