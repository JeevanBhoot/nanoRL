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
from typing import List, Optional, Tuple

from easy_rl.data import TOPICS, TEST_TOPICS, make_dataloader
from easy_rl.hf_policy import HFPolicy
from easy_rl.rewards import reward
from easy_rl.plot import plot_training_metrics
from easy_rl.train_reinforce import train_reinforce


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
    alpha: float = 0.5


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
    parser.add_argument("--alpha", type=float, default=0.5, help="Scale for brevity penalty.")
    return parser.parse_args()


def rloo(policy: HFPolicy, prompts: List[str], config: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the log-probs, rewards, and advantages for a batch of prompts using the RLOO algorithm."""
    # Generate k completions for each prompt
    texts_per_prompt, logprobs = policy.generate_with_logprobs(prompts, k=config.group_size)

    # Compute rewards for all completions
    flat_texts = [text for texts in texts_per_prompt for text in texts]
    rewards_flat = reward(flat_texts, alpha=config.alpha).to(device=logprobs.device, dtype=logprobs.dtype)
    rewards_tensor = rewards_flat.view(len(prompts), config.group_size)

    # Leave-One-Out baseline and advantages
    sum_rewards = rewards_tensor.sum(dim=1, keepdim=True)                     # [B, 1]
    loo_baseline = (sum_rewards - rewards_tensor) / (config.group_size - 1)   # [B, k]
    advantages = (rewards_tensor - loo_baseline).detach()                     # [B, k]

    return logprobs, rewards_tensor, advantages


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
        alpha=args.alpha,
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
    train_samples = train_reinforce(policy, dataloader, optimizer, config, rloo)

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
