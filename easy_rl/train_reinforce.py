import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from easy_rl.data import make_dataloader, TEST_TOPIC
from easy_rl.hf_policy import HFPolicy
from easy_rl.rewards import reward

@dataclass
class TrainConfig:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 25
    grad_clip: float = 1.0
    output_dir: Path = Path("results")
    save_every: int = None


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    return parser.parse_args()


def plot_training_metrics(epoch_losses: List[float], epoch_rewards: List[float], output_dir: Path) -> None:
    epochs = range(1, len(epoch_losses) + 1)
    fig, ax_loss = plt.subplots(figsize=(8, 5))
    color_loss = "#1f77b4"
    color_reward = "#ff7f0e"

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color=color_loss)
    ax_loss.plot(epochs, epoch_losses, color=color_loss, label="Loss")
    ax_loss.tick_params(axis="y", labelcolor=color_loss)

    ax_reward = ax_loss.twinx()
    ax_reward.set_ylabel("Reward", color=color_reward)
    ax_reward.plot(epochs, epoch_rewards, color=color_reward, label="Reward")
    ax_reward.tick_params(axis="y", labelcolor=color_reward)

    fig.suptitle("Training Loss and Reward")
    fig.tight_layout()
    output_path = output_dir / "training_metrics.png"
    fig.savefig(output_path)
    plt.close(fig)


def train_reinforce(policy: HFPolicy, dataloader: DataLoader, optimizer: torch.optim.Optimizer, config: TrainConfig) -> None:
    policy.model.train()
    epoch_losses = []
    epoch_rewards = []
    output_dir = config.output_dir
    for epoch in range(config.num_epochs):
        batch_losses = []
        batch_rewards = []
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            optimizer.zero_grad()
            prompts = batch["prompts"]
            texts, logprobs = policy.generate_with_logprobs(prompts)
            rewards = reward(texts).to(device=logprobs.device, dtype=logprobs.dtype)
            # REINFORCE objective: maximise E[r*sum log pi]  -> minimise negative
            # without baseline
            loss = -(logprobs * rewards).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), config.grad_clip)
            optimizer.step()
            batch_losses.append(loss.detach().item())
            batch_rewards.append(rewards.detach().mean().item())
        mean_loss = sum(batch_losses) / max(len(batch_losses), 1)
        mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
        epoch_losses.append(mean_loss)
        epoch_rewards.append(mean_reward)
        print(f"Epoch {epoch+1}/{config.num_epochs} loss: {mean_loss:.4f} reward: {mean_reward:.4f}")
        if config.save_every and (epoch + 1) % config.save_every == 0:
            ckpt_path = output_dir / f"model_{epoch+1}.pth"
            torch.save(policy.model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")
    ckpt_path = output_dir / f"model_final.pth"
    torch.save(policy.model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")
    policy.model.eval()
    plot_training_metrics(epoch_losses, epoch_rewards, output_dir)


def main():
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("model_*.pth")):
        print(f"Existing checkpoints found in {output_dir}, exiting without training.")
        return
    config = TrainConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        grad_clip=args.grad_clip,
        output_dir=output_dir,
    )
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
