"""
Train a policy using the REINFORCE algorithm.
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--baseline", choices=["mean", "ema", "scst"], default=None)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--brevity-penalty-scale", type=float, default=0.0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir or f"results/{datetime.now():%Y%m%d-%H%M%S}")
    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return args


def generate_with_logprobs(model, tokenizer, prompts):
    """Generate completions for a batch of prompts and compute the log-probabilities of the generated tokens."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs.input_ids.shape[1] 

    # 1) Sample actions from policy (no gradients during rollout)
    with torch.no_grad():
        output_ids = model.generate(                                                                # [B, T]
            **inputs, max_new_tokens=256, do_sample=True, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )

    # 2) Forward pass with gradients to get log π(a_t | s_{<t})
    attention_mask = (output_ids != tokenizer.pad_token_id).long()                                  # [B, T]
    logits = model(input_ids=output_ids, attention_mask=attention_mask, use_cache=False).logits     # [B, T, V]
    
    # log_probs[b, t, v] = log π(v | tokens_{≤t})
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)                                        # [B, T-1, V] log-probs over entire vocab
    token_log_probs = log_probs.gather(-1, output_ids[:, 1:].unsqueeze(-1)).squeeze(-1)             # [B, T-1] log-probs of generated tokens only
    
    # Mask out prompt and padding tokens
    # Generated tokens are at indices [prompt_len-1, ...] in shifted view
    is_generated = torch.arange(token_log_probs.shape[1], device=model.device) >= (prompt_len - 1)  # [T-1]
    is_valid = attention_mask[:, 1:].bool()                                                         # [B, T-1]
    mask = is_generated & is_valid                                                                  # [B, T-1]
    
    sum_log_probs = (token_log_probs * mask).sum(dim=1)                                             # [B]
    texts = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)            # Decode generated tokens
    
    return texts, sum_log_probs


def reinforce_step(model, tokenizer, prompts, args, ema_baseline):
    """Compute the log-probs, rewards, and advantages for a batch of prompts."""
    texts, logprobs = generate_with_logprobs(model, tokenizer, prompts)
    rewards = reward(texts, alpha=args.brevity_penalty_scale).to(logprobs.device)
    
    baseline, new_ema = None, ema_baseline
    if args.baseline == "mean":
        baseline = rewards.mean()
    elif args.baseline == "ema":
        # Exponential moving average of the mean batch reward
        current_mean = rewards.mean()
        new_ema = current_mean.item() if ema_baseline is None else args.ema_beta * ema_baseline + (1 - args.ema_beta) * current_mean.item()
        baseline = torch.tensor(new_ema, device=rewards.device)
    elif args.baseline == "scst":
        # Self-critical sequence training (SCST) https://arxiv.org/pdf/1612.00563
        # Baseline = reward of the model's own greedy decode for the same prompt.
        baseline = reward(generate(model, tokenizer, prompts, do_sample=False), alpha=args.brevity_penalty_scale).to(rewards.device)
    # Advantage = reward - baseline -> Baseline reduces variance in policy gradient estimates, while keeping it unbiased.
    advantages = (rewards - baseline).detach() if baseline is not None else rewards.detach()
    return texts, logprobs, rewards, advantages, new_ema


def train(model, tokenizer, dataloader, test_dataloader, optimizer, args):
    samples = []
    ema = None
    track_prompt = TOPICS[0]
    samples.append((0, track_prompt, generate(model, tokenizer, [track_prompt])[0]))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_reward, epoch_ttr, epoch_gen_len = [], [], [], []
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()
            texts, logprobs, rewards, advantages, ema = reinforce_step(model, tokenizer, batch["prompts"], args, ema)
            # REINFORCE objective: maximise E[adv*sum log pi]  ->   minimise negative
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