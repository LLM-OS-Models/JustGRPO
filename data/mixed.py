"""Mixed-domain dataloader for multi-domain GRPO (gsm8k + math + code).

Round-robin at the batch level: each optimizer micro-step draws its prompt batch from
the next domain in the cycle, and the reward is routed to that domain's reward_fn via
the batch's 'domain' tag. GRPO's per-group advantage normalization absorbs the
different reward scales (math ±1 vs code 0..2), the same property that lets
production multi-environment RLVR pipelines (e.g. Nemotron 3 Super's 21-env stage)
mix heterogeneous rewards in one run.
"""

from data.math import load_gsm8k_dataset_and_reward, load_math_dataset_and_reward
from data.code import load_code_dataset_and_reward

DOMAINS = ("gsm8k", "math", "code")


class MixedRoundRobinLoader:
    """Infinite iterator cycling gsm8k -> math -> code batches (each loader is
    already infinite via InfiniteSampler). Tags every batch with its domain."""

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        iters = {d: iter(self.loaders[d]) for d in DOMAINS}
        i = 0
        while True:
            domain = DOMAINS[i % len(DOMAINS)]
            i += 1
            batch = next(iters[domain])
            batch["domain"] = domain
            yield batch


def load_mixed_dataset_and_reward(code_data_path, batch_size=1, num_workers=2, seed=112):
    loaders, rewards = {}, {}
    loaders["gsm8k"], rewards["gsm8k"] = load_gsm8k_dataset_and_reward(
        local_path="openai/gsm8k", batch_size=batch_size, num_workers=num_workers, seed=seed)
    loaders["math"], rewards["math"] = load_math_dataset_and_reward(
        local_path="ankner/math-500", batch_size=batch_size, num_workers=num_workers, seed=seed)
    loaders["code"], rewards["code"] = load_code_dataset_and_reward(
        local_path=code_data_path, batch_size=batch_size, num_workers=num_workers, seed=seed)

    def mixed_reward(batch, responses, num_generations, device):
        return rewards[batch["domain"]](batch, responses, num_generations, device)

    return MixedRoundRobinLoader(loaders), mixed_reward
