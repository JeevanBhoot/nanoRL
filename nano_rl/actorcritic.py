"""
Train a policy using the Actor-Critic algorithm.
"""

import torch
import torch.nn as nn
import tqdm
import wandb
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from nano_rl.data import TOPICS, TEST_TOPICS, make_dataloader
from nano_rl.model import load_model, generate, evaluate
from nano_rl.logging import save_test_generations, save_training_samples
from nano_rl.rewards import reward, ttr


class CausalLMWithValueHead(nn.Module):
    """Wraps a CausalLM with a value head for Actor-Critic."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # Value head: maps hidden states to scalar value
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        self.value_head = self.value_head.to(device=base_model.device, dtype=base_model.dtype)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, **kwargs
        )
        hidden_states = outputs.hidden_states[-1]                   # [B, T, H]
        values = self.value_head(hidden_states).squeeze(-1)         # [B, T]
        return outputs.logits, values
    
    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)
    
    @property
    def device(self):
        return self.base_model.device


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--value-coef", type=float, default=0.5, help="Coefficient for critic loss")
    parser.add_argument("--brevity-penalty-scale", type=float, default=0.0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir or f"results/{datetime.now():%Y%m%d-%H%M%S}")
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return args


def generate_with_logprobs_and_values(model, tokenizer, prompts):
    """Generate completions and compute per-token log-probs and values."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # 1) Sample actions from policy (no gradients during rollout)
    with torch.no_grad():
        output_ids = model.generate(                                                                # [B, T]
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )

    # 2) Forward pass with gradients to get log π(a_t | s_{<t}) and V(s_t)
    attention_mask = (output_ids != tokenizer.pad_token_id).long()                                  # [B, T]
    logits, values = model(
        input_ids=output_ids,
        attention_mask=attention_mask,
    )                                                                                               # [B, T, V], [B, T]

    # 3) Compute log-probs: logits[:, t, :] predicts token at position t+1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)                                        # [B, T-1, V]
    token_log_probs = log_probs.gather(-1, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)             # [B, T-1]

    # 4) Align values: values[:, t] = V(s_t) = value before generating token t+1
    values = values[:, :-1]                                                                         # [B, T-1]

    # 5) Mask out prompt and padding tokens
    # Generated tokens start at position prompt_len in output_ids, which is index prompt_len-1 in shifted view
    seq_len = token_log_probs.shape[1]
    is_generated = torch.arange(seq_len, device=model.device) >= (prompt_len - 1)                   # [T-1]
    is_valid = attention_mask[:, 1:].bool()                                                         # [B, T-1]
    mask = is_generated & is_valid                                                                  # [B, T-1]

    texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
    return texts, token_log_probs, values, mask


def actorcritic_step(model, tokenizer, prompts, args):
    """Compute actor and critic losses for a batch of prompts."""
    texts, token_log_probs, values, mask = generate_with_logprobs_and_values(
        model, tokenizer, prompts
    )
    rewards = reward(texts, alpha=args.brevity_penalty_scale).to(model.device)                      # [B]

    # Monte Carlo return: same final reward R for all token positions
    rewards_expanded = rewards.unsqueeze(1).expand_as(values)                                       # [B, T-1]

    # Per-token advantage: A_t = R - V(s_t), 
    # detached so actor loss doesn't backprop through critic
    advantages = (rewards_expanded - values).detach()                                               # [B, T-1]

    # Actor loss: -E[A_t * log π(y_t | s_t)]
    # sum over tokens per sequence, then average over batch
    per_seq_actor = (token_log_probs * advantages * mask).sum(dim=1)                                # [B]
    actor_loss = -per_seq_actor.mean()

    # Critic loss: E[(V(s_t) - R)²]
    critic_loss = ((values - rewards_expanded) ** 2 * mask).sum() / mask.sum()

    loss = actor_loss + args.value_coef * critic_loss
    return texts, rewards, actor_loss, critic_loss, loss


def train(model, tokenizer, dataloader, test_dataloader, optimizer, args):
    samples = []
    track_prompt = TOPICS[0]
    samples.append((0, track_prompt, generate(model, tokenizer, [track_prompt])[0]))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_actor, epoch_critic = [], [], []
        epoch_reward, epoch_ttr, epoch_gen_len = [], [], []
        
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()
            texts, rewards, actor_loss, critic_loss, loss = actorcritic_step(
                model, tokenizer, batch["prompts"], args
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_actor.append(actor_loss.item())
            epoch_critic.append(critic_loss.item())
            epoch_reward.append(rewards.mean().item())
            epoch_ttr.append(sum(ttr(t) for t in texts) / len(texts))
            epoch_gen_len.append(sum(len(t.split()) for t in texts) / len(texts))

        mean_loss = sum(epoch_loss) / len(epoch_loss)
        mean_actor = sum(epoch_actor) / len(epoch_actor)
        mean_critic = sum(epoch_critic) / len(epoch_critic)
        mean_reward = sum(epoch_reward) / len(epoch_reward)
        mean_ttr = sum(epoch_ttr) / len(epoch_ttr)
        mean_gen_len = sum(epoch_gen_len) / len(epoch_gen_len)

        _, test_reward, test_ttr, test_gen_len = evaluate(model, tokenizer, test_dataloader, args)

        wandb.log({
            "train/loss": mean_loss, "train/actor_loss": mean_actor, "train/critic_loss": mean_critic,
            "train/reward": mean_reward, "train/ttr": mean_ttr, "train/gen_length": mean_gen_len,
            "test/reward": test_reward, "test/ttr": test_ttr, "test/gen_length": test_gen_len, "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1}: loss={mean_loss:.4f}, actor={mean_actor:.4f}, critic={mean_critic:.4f}, "
              f"reward={mean_reward:.4f}, test_reward={test_reward:.4f}, test_ttr={test_ttr:.4f}")

        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), args.output_dir / f"model_{epoch+1}.pth")

        samples.append((epoch + 1, track_prompt, generate(model, tokenizer, [track_prompt])[0]))

    torch.save(model.state_dict(), args.output_dir / "model_final.pth")
    return samples


def main():
    args = parse_args()
    if any(args.output_dir.glob("model_*.pth")):
        raise ValueError(f"Checkpoints exist in {args.output_dir}, exiting.")

    wandb.init(project="nano-rl", name=args.output_dir.name, config=vars(args))
    tokenizer, base_model = load_model(args.model_id)
    model = CausalLMWithValueHead(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_dataloader = make_dataloader(batch_size=args.batch_size)
    test_dataloader = make_dataloader(topics=TEST_TOPICS, batch_size=len(TEST_TOPICS), shuffle=False)

    before_texts, before_reward, before_ttr, before_gen_len = evaluate(model, tokenizer, test_dataloader, args)
    wandb.log({"test/reward": before_reward, "test/ttr": before_ttr, "test/gen_length": before_gen_len, "epoch": 0})

    samples = train(model, tokenizer, train_dataloader, test_dataloader, optimizer, args)

    after_texts, _, _, _ = evaluate(model, tokenizer, test_dataloader, args)

    save_test_generations(args.output_dir / "test_generations.txt", before_texts, after_texts, TEST_TOPICS)
    save_training_samples(args.output_dir / "train_generations.txt", samples)
    wandb.finish()


if __name__ == "__main__":
    main()
