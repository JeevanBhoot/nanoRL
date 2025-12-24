"""
Train a policy using the REINFORCE Leave-One-Out (RLOO) algorithm.
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


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4, help="Number of completions per prompt (k)")
    parser.add_argument("--brevity-penalty-scale", type=float, default=0.0)
    args = parser.parse_args()
    
    if args.group_size < 2:
        raise ValueError("RLOO requires --group-size >= 2")
    output_dir = Path(args.output_dir or f"results/{datetime.now():%Y%m%d-%H%M%S}")
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return args


def generate_with_logprobs(model, tokenizer, prompts, k):
    """Generate k completions per prompt and compute log-probabilities of the generated tokens."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    B = len(prompts)

    # 1) Sample k actions per prompt (no gradients during rollout)
    with torch.no_grad():
        output_ids = model.generate(                                                                # [B*k, T]
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=k
        )

    # 2) Forward pass with gradients to get log π(a_t | s_{<t})
    attention_mask = (output_ids != tokenizer.pad_token_id).long()                                  # [B*k, T]
    logits = model(input_ids=output_ids, attention_mask=attention_mask, use_cache=False).logits     # [B*k, T, V]
    
    # log_probs[b, t, v] = log π(v | tokens_{≤t})
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)                                        # [B*k, T-1, V] log-probs over entire vocab
    token_log_probs = log_probs.gather(-1, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)             # [B*k, T-1] log-probs of generated tokens only
    
    # Mask out prompt and padding tokens
    # Generated tokens are at indices [prompt_len-1, ...] in shifted view
    seq_len = token_log_probs.shape[1]
    is_generated = torch.arange(seq_len, device=model.device) >= (prompt_len - 1)                   # [T-1]
    is_valid = attention_mask[:, 1:].bool()                                                         # [B*k, T-1]
    mask = is_generated & is_valid                                                                  # [B*k, T-1]
    
    sum_log_probs = (token_log_probs * mask).sum(dim=1)                                             # [B*k]
    
    # Decode and reshape: k completions per prompt
    texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
    texts = [texts[i*k:(i+1)*k] for i in range(B)]                                                  # List[List[str]] of shape [B][k]
    sum_log_probs = sum_log_probs.view(B, k)                                                        # [B, k]
    
    return texts, sum_log_probs


def rloo_step(model, tokenizer, prompts, args):
    """
    Compute log-probs, rewards, and advantages using RLOO.
    For each of k completions, baseline = mean reward of the other k-1 completions.
    """
    texts_per_prompt, logprobs = generate_with_logprobs(model, tokenizer, prompts, k=args.group_size)
    
    # Compute rewards for all B*k completions
    flat_texts = [text for texts in texts_per_prompt for text in texts]
    rewards = reward(flat_texts, alpha=args.brevity_penalty_scale).to(logprobs.device)
    rewards = rewards.view(len(prompts), args.group_size)                                           # [B, k]
    
    # Leave-One-Out baseline: for completion i, baseline = mean of other k-1 rewards
    sum_rewards = rewards.sum(dim=1, keepdim=True)                                                  # [B, 1]
    loo_baseline = (sum_rewards - rewards) / (args.group_size - 1)                                  # [B, k]
    advantages = (rewards - loo_baseline).detach()                                                  # [B, k]
    
    return flat_texts, logprobs, rewards, advantages


def train(model, tokenizer, dataloader, test_dataloader, optimizer, args):
    samples = []
    track_prompt = TOPICS[0]
    samples.append((0, track_prompt, generate(model, tokenizer, [track_prompt])[0]))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_reward, epoch_ttr, epoch_gen_len = [], [], [], []
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()
            texts, logprobs, rewards, advantages = rloo_step(model, tokenizer, batch["prompts"], args)
            # RLOO objective: same as REINFORCE, minimise -E[adv*sum log pi], but over all B*k completions
            loss = -(logprobs * advantages).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            epoch_loss.append(loss.detach().item())
            epoch_reward.append(rewards.detach().mean().item())
            epoch_ttr.append(sum(ttr(t) for t in texts) / len(texts))
            epoch_gen_len.append(sum(len(t.split()) for t in texts) / len(texts))
        
        mean_loss = sum(epoch_loss) / len(epoch_loss)
        mean_reward = sum(epoch_reward) / len(epoch_reward)
        mean_ttr = sum(epoch_ttr) / len(epoch_ttr)
        mean_gen_len = sum(epoch_gen_len) / len(epoch_gen_len)
        
        _, test_reward, test_ttr, test_gen_len = evaluate(model, tokenizer, test_dataloader, args)
        
        wandb.log({
            "train/loss": mean_loss, "train/reward": mean_reward, "train/ttr": mean_ttr, "train/gen_length": mean_gen_len,
            "test/reward": test_reward, "test/ttr": test_ttr, "test/gen_length": test_gen_len, "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1}: loss={mean_loss:.4f}, reward={mean_reward:.4f}, ttr={mean_ttr:.4f}, test_reward={test_reward:.4f}, test_ttr={test_ttr:.4f}")
        
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
    tokenizer, model = load_model(args.model_id)
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
