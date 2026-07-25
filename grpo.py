import torch
import torch.nn.functional as F

from utils.ar_diffusion import generate_ar, ar_logprobs_onepass


@torch.no_grad()
def sample(model, batch, tokenizer, device, reward_fn=None, num_generations=1, temperature=1.,
           steps=256, gen_length=256, mask_id=None, eos_id=None,
           sampler=None, rollout_block_size=32):
    """Rollout + reward.

    Rollouts use the model's native block-diffusion sampler (dllm BD3LMSampler):
    AR-order (block=1) generation from this 0.6B BD3LM checkpoint degenerates
    (~49% EOS mass at the first position, digit loops), while block-diffusion sampling
    solves GSM8K problems reliably. The GRPO update below still optimizes the exact AR
    factorization, so training remains "AR order" as in JustGRPO.
    """
    prompts = tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in batch['problems']],
                                            add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompts, return_tensors='pt', padding=True)['input_ids'].to(device)

    if sampler is not None:
        # Native block-diffusion rollout (recommended for BD3LM checkpoints)
        from dllm.core.samplers import BD3LMSamplerConfig
        prompt_list = tokenizer(prompts)['input_ids'] * num_generations  # tiled like repeat()
        seqs = sampler.sample(prompt_list, config=BD3LMSamplerConfig(
            max_new_tokens=gen_length, steps=steps, block_size=rollout_block_size,
            temperature=temperature, remasking="low_confidence"))
        # The sampler left-pads the prompt to a multiple of block_size and stops early
        # once every sequence hit EOS, so the generation starts at padded_P and may be
        # shorter than gen_length — pad the tail with EOS (the model's SFT convention).
        padded_P = -(-max(len(p) for p in prompt_list) // rollout_block_size) * rollout_block_size
        completion_ids = seqs[:, padded_P:].to(device)
        if completion_ids.shape[1] < gen_length:
            completion_ids = F.pad(completion_ids, (0, gen_length - completion_ids.shape[1]), value=eos_id)
        generated_ids = torch.cat([prompt_ids.repeat(num_generations, 1), completion_ids], dim=1)
    else:
        # AR-order rollout (JustGRPO original mode; weak for this base model)
        generated_ids = generate_ar(model=model, prompt=prompt_ids.repeat(num_generations, 1),
                                    gen_length=gen_length, mask_id=mask_id, temperature=temperature,
                                    eos_id=eos_id, min_new_tokens=16)

    # Truncate at EOS before decoding so post-EOS continuation can't confuse the reward
    responses = []
    for row in generated_ids[:, prompt_ids.shape[1]:].tolist():
        if eos_id in row:
            row = row[:row.index(eos_id)]
        responses.append(tokenizer.decode(row, skip_special_tokens=True))

    return {
        'generated_ids': generated_ids,
        'prompt_len': prompt_ids.shape[1],
        'rewards': reward_fn(batch, responses, num_generations, device).float(),
    }


def logprob_loss(model, inputs, valid_samples, eps=0.2, gain=1.0, temperature=1., accelerator=None,
                 gen_length=256, mask_id=None, eos_id=None):
    advantages, generated_ids, prompt_len = inputs['advantages'], inputs['generated_ids'], inputs['prompt_len']
    prompt_ids, completion_ids = generated_ids[:, :prompt_len], generated_ids[:, prompt_len:]

    valid_samples = accelerator.gather(valid_samples).float().mean().item()
    scale = gain / gen_length / (valid_samples + 1e-5)

    # All AR conditionals p(y_t | y_<t) in ONE forward via the BD3LM [x0 | xt] trick
    # (block size 1, completion fully masked) — replaces the per-token forward loop
    # the original LLaDA implementation needed.
    token_log_prob = ar_logprobs_onepass(model, prompt_ids, completion_ids,
                                         mask_id=mask_id, temperature=temperature)  # [B, G]

    ratio = (token_log_prob - token_log_prob.detach()).exp()
    clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
    adv = advantages.unsqueeze(1)
    loss = -torch.min(ratio * adv, clipped_ratio * adv)

    # Train only up to (and including) the first EOS: the block-diffusion sampler
    # stops early and sample() pads the tail with EOS tokens the policy never
    # generated. Reinforcing that artificial padding inflates the marginal EOS
    # probability and collapses generation into empty responses (observed on the
    # first mixed run: HumanEval 46.9 -> 18.3 with '<think></think>' + instant EOS).
    if eos_id is not None:
        is_eos = completion_ids == eos_id
        after_first_eos = (is_eos.cumsum(dim=1) - is_eos.int()) > 0
        loss = loss * (~after_first_eos).float()

    accelerator.backward(loss.mul(scale).sum())

    return {
        "reward": accelerator.gather(inputs['rewards'].detach()).mean().item(),
        "valid_samples": valid_samples,
    }


def compute_group_advantages(rewards, group_size):
    mean = rewards.view(group_size, -1).mean(dim=0).repeat(group_size)
    std = rewards.view(group_size, -1).std(dim=0).repeat(group_size)
    return (rewards - mean) / (std + 1e-4)
