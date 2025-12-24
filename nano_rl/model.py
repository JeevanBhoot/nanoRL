import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nano_rl.rewards import reward


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tokenizer, model


def generate(model, tokenizer, prompts, do_sample=True):
    """Generate completions for a batch of prompts."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=do_sample, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def evaluate(model, tokenizer, test_dataloader, args):
    """Evaluate on test set, return (texts, mean_reward, mean_gen_length)."""
    model.eval()
    all_texts = []
    for batch in test_dataloader:
        texts = generate(model, tokenizer, batch["prompts"])
        all_texts.extend(texts)
    rewards = reward(all_texts, alpha=args.brevity_penalty_scale)
    mean_gen_len = sum(len(t.split()) for t in all_texts) / len(all_texts)
    return all_texts, rewards.mean().item(), mean_gen_len