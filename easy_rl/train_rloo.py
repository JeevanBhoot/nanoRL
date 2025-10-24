"""
Train a policy using the REINFORCE Leave-One-Out (RLOO) algorithm.
https://arxiv.org/pdf/2402.14740

1. Generate multiple generations for each prompt.
2. Compute the reward for each generation.
3. Baseline = mean reward of the other generations for the same prompt.
"""

import csv
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from easy_rl.data import TOPICS, TEST_TOPICS, make_dataloader
from easy_rl.hf_policy import HFPolicy
from easy_rl.rewards import reward
from easy_rl.plot import plot_training_metrics

@dataclass
class TrainConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 6
    grad_clip: float = 1.0
    output_dir: Path = Path("results")
    save_every: int = None
    group_size: int = 4


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    return parser.parse_args()


def train_rloo(policy: HFPolicy, dataloader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig) -> List[Tuple[int, str, str]]:
    """Train the policy using the RLOO algorithm."""
    policy.model.train()
    epoch_losses = []
    epoch_rewards = []
    output_dir = config.output_dir
    train_samples: List[Tuple[int, str, str]] = []

    # Track generations of the first training prompt
    tracking_prompt = TOPICS[0]
    # Generate the initial generation before training
    initial_sample = policy.generate([tracking_prompt])[0]
    train_samples.append((0, tracking_prompt, initial_sample))

    for epoch in range(config.num_epochs):
        batch_losses = []
        batch_rewards = []
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            optimizer.zero_grad()
            prompts = batch["prompts"]
            texts_per_prompt, logprobs = policy.generate_k_with_logprobs(
                prompts,
                k=config.group_size,
            )

            # Rewards -> [B, k]
            flat_texts = [text for texts in texts_per_prompt for text in texts]
            rewards_flat = reward(flat_texts).to(device=logprobs.device, dtype=logprobs.dtype)
            rewards_tensor = rewards_flat.view(len(prompts), config.group_size)

            if config.group_size < 2:
                raise ValueError("RLOO requires group_size >= 2 for leave-one-out baseline.")
            
            # Leave-One-Out baseline and advantages
            sum_rewards = rewards_tensor.sum(dim=1, keepdim=True)                     # [B, 1]
            loo_baseline = (sum_rewards - rewards_tensor) / (config.group_size - 1)   # [B, k]
            advantages = (rewards_tensor - loo_baseline).detach()                     # [B, k]

            # REINFORCE objective: maximise E[adv*sum log pi]  ->   minimise negative
            loss = -(logprobs * advantages).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
            optimizer.step()

            batch_losses.append(loss.detach().item())
            batch_rewards.append(rewards_tensor.detach().mean().item())

        # Log mean loss and reward
        mean_loss = sum(batch_losses) / max(len(batch_losses), 1)
        mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
        epoch_losses.append(mean_loss)
        epoch_rewards.append(mean_reward)
        print(f"Epoch {epoch+1}/{config.num_epochs} loss: {mean_loss:.4f} reward: {mean_reward:.4f}")

        # Save the model every `save_every` epochs
        if config.save_every and (epoch + 1) % config.save_every == 0:
            ckpt_path = output_dir / f"model_{epoch+1}.pth"
            torch.save(policy.model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

        # Generate sample after each epoch
        sample_text = policy.generate([tracking_prompt])[0]
        train_samples.append((epoch + 1, tracking_prompt, sample_text))

    # Save the final model
    ckpt_path = output_dir / f"model_final.pth"
    torch.save(policy.model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")
    policy.model.eval()

    plot_training_metrics(epoch_losses, epoch_rewards, output_dir)

    # Dump loss and reward to CSV
    metrics_path = output_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "reward"])
        for epoch_idx, (loss_val, reward_val) in enumerate(zip(epoch_losses, epoch_rewards), start=1):
            writer.writerow([epoch_idx, loss_val, reward_val])
    return train_samples


def main():
    args = parse_args()
    # Initialise output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("model_*.pth")):
        print(f"Existing checkpoints found in {output_dir}, exiting without training.")
        return
    if args.group_size < 2:
        raise ValueError("RLOO requires --group-size >= 2.")

    # Initialise training configuration
    config = TrainConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        grad_clip=args.grad_clip,
        output_dir=output_dir,
        save_every=args.save_every,
        group_size=args.group_size,
    )

    # Initialise policy (LLM), dataloader, and optimizer
    policy = HFPolicy(config.model_id)
    dataloader = make_dataloader(batch_size=config.batch_size)
    test_dataloader = make_dataloader(topics=TEST_TOPICS, batch_size=1, shuffle=False)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=config.learning_rate)

    # Generate completions on test prompts before training
    test_texts_before = []
    for batch in test_dataloader:
        prompts = batch["prompts"]
        texts = policy.generate(prompts)[0]
        test_texts_before.append(texts)

    # Train the policy
    train_samples = train_rloo(policy, dataloader, optimizer, config)

    # Generate completions on test prompts after training
    test_texts_after = []
    for batch in test_dataloader:
        prompts = batch["prompts"]
        texts = policy.generate(prompts)[0]
        test_texts_after.append(texts)

    print("Test text before training:")
    print(test_texts_before[0])
    print("Test text after training:")
    print(test_texts_after[0])

    # Save test generations
    test_log_path = output_dir / "test_generations.txt"
    with test_log_path.open("w", encoding="utf-8") as f:
        f.write("Test topics:\n")
        for topic in TEST_TOPICS:
            f.write(topic.strip() + "\n\n")
        f.write("Test texts before training:\n")
        for text in test_texts_before:
            f.write(text.strip() + "\n\n")
        f.write("Test texts after training:\n")
        for text in test_texts_after:
            f.write(text.strip() + "\n\n")

    # Save train generations (one sample per epoch)
    if train_samples:
        samples_path = output_dir / "train_generations.txt"
        with samples_path.open("w", encoding="utf-8") as f:
            for epoch, prompt, generation in train_samples:
                label = "Initial" if epoch == 0 else f"Epoch {epoch}"
                f.write(f"{label}\n")
                f.write("Prompt:\n")
                f.write(prompt.strip() + "\n\n")
                f.write("Generation:\n")
                f.write(generation.strip() + "\n\n")


if __name__ == "__main__":
    main()
