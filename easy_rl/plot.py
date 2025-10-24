import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

def plot_training_metrics(epoch_losses: List[float], epoch_rewards: List[float], output_dir: Path) -> None:
    """Plot training metrics (loss and reward) over epochs."""
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