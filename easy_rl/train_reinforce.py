import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from dataclasses import dataclass

from easy_rl.data import make_dataloader, TEST_TOPIC
from easy_rl.hf_policy import HFPolicy
from easy_rl.rewards import reward

@dataclass
class TrainConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    grad_clip: float = 1.0


def train_reinforce(policy: HFPolicy, dataloader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig) -> None:
    policy.model.train()
    for epoch in range(config.num_epochs):
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            optimizer.zero_grad()
            prompts = batch["prompts"]
            texts, logprobs = policy.generate_with_logprobs(prompts)
            rewards = reward(texts)
            # REINFORCE objective: maximise E[r*sum log pi]  -> minimise negative
            # without baseline
            loss = -(logprobs * rewards).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
            optimizer.step()
        print(f"Epoch {epoch+1}/{config.num_epochs} loss: {loss.item()}")
        torch.save(policy.model.state_dict(), f"model_{epoch+1}.pth")
        print(f"Model saved to model_{epoch+1}.pth")
    policy.model.eval()

def main():
    config = TrainConfig()
    policy = HFPolicy(config.model_id)
    test_text = policy.generate([TEST_TOPIC])
    dataloader = make_dataloader(batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config.learning_rate)
    train_reinforce(policy, dataloader, optimizer, config)
    test_text_after = policy.generate([TEST_TOPIC])
    print("Test text before training:")
    print(test_text)
    print("Test text after training:")
    print(test_text_after)

if __name__ == "__main__":
    main()
