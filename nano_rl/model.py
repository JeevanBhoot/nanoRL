import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tokenizer, model