from easy_rl.data import make_dataloader
from easy_rl.models import HFPolicy
from easy_rl.rewards import reward

import torch

# CONFIG
# dataclass or dict?

def train_reinforce(config: dict):
    policy = HFPolicy(config["model_id"])
    dataloader = make_dataloader(config["batch_size"])
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        for batch in dataloader:
            prompts = batch["prompts"]
            texts, logprobs = policy.generate_with_logprobs(prompts)
            rewards = reward(texts)
            # REINFORCE objective: maximise E[r*sum log pi]  -> minimise negative
            # without baseline
            loss = -(logprobs * rewards).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config["grad_clip"])
            optimizer.step()
