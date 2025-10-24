from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenConfig:
    """Configuration container for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 150
    do_sample: bool = True
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None

    def to_generate_kwargs(self, default_pad: Optional[int], default_eos: Optional[Union[int, List[int]]]) -> dict:
        """Return a dict of kwargs suitable for `model.generate`."""
        cfg = asdict(self)
        if cfg["pad_token_id"] is None:
            cfg["pad_token_id"] = default_pad
        if cfg["eos_token_id"] is None:
            cfg["eos_token_id"] = default_eos
        return {k: v for k, v in cfg.items() if v is not None}


class HFPolicy:
    """Wrapper around a Hugging Face LM for RL training."""

    def __init__(
        self,
        model_id: str,
        dtype: Union[str, torch.dtype] = "bfloat16",
        device_map: Optional[Union[str, dict]] = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _tokenize(self, prompts: Sequence[str]):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_lens = inputs.attention_mask.sum(-1)
        return inputs, prompt_lens

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], gen_cfg: Optional[GenConfig] = None) -> List[str]:
        """Generate completions for a batch of prompts."""
        gen_cfg = gen_cfg or GenConfig()
        gen_cfg.do_sample = False
        inputs, lens = self._tokenize(prompts)
        out = self.model.generate(
            **inputs,
            **gen_cfg.to_generate_kwargs(
                default_pad=self.tokenizer.pad_token_id,
                default_eos=self.tokenizer.eos_token_id,
            ),
        )
        texts = []
        for i in range(len(prompts)):
            gen_tokens = out[i, lens[i].item():]
            texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
        return texts

    def generate_with_logprobs(self, prompts: Sequence[str], gen_cfg: Optional[GenConfig] = None):
        """Sample continuations, then compute differentiable sum(log p) over the generated tokens."""
        gen_cfg = gen_cfg or GenConfig()
        # 1) sample actions (no gradients needed for sampling)
        inputs, lens = self._tokenize(prompts)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_cfg.to_generate_kwargs(
                    default_pad=self.tokenizer.pad_token_id,
                    default_eos=self.tokenizer.eos_token_id,
                ),
                return_dict_in_generate=True,
            )
        seqs = out.sequences  # [B, T_max] (prompt + gen, padded)

        # 2) differentiable forward on the sampled sequences
        attn = (seqs != self.tokenizer.pad_token_id).long()
        outputs = self.model(input_ids=seqs, attention_mask=attn)
        logits = outputs.logits  # [B, T_max, V]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)    # predict next token
        tgt = seqs[:, 1:]                                      # target = next token ids

        # build a mask for the generated region only (exclude prompt and padding)
        B, Tm1 = tgt.shape
        idx = torch.arange(Tm1, device=self.device).unsqueeze(0)          # [1, T-1]
        prompt_starts = lens.unsqueeze(1).to(self.device)                  # [B, 1]
        seq_lens = attn.sum(dim=1).unsqueeze(1)                            # [B, 1] true lengths
        gen_mask = (idx >= prompt_starts) & (idx < seq_lens - 1)           # [B, T-1]

        # gather token log-probs and sum over generated positions
        tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # [B, T-1]
        tok_logp = tok_logp * gen_mask
        logprob_sums = tok_logp.sum(dim=1)                                  # [B]

        # decode continuations (for logging/evaluation)
        texts = []
        for i in range(seqs.size(0)):
            gen_tokens = seqs[i, lens[i].item():seq_lens[i].item()]
            texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))

        return texts, logprob_sums

    def generate_k_with_logprobs(
        self,
        prompts: Sequence[str],
        k: int,
        gen_cfg: Optional[GenConfig] = None,
    ):
        """Sample `k` continuations per prompt and return log-probs per sample."""
        if k < 1:
            raise ValueError("k must be >= 1 for generate_k_with_logprobs.")
        gen_cfg = gen_cfg or GenConfig()

        # 1) sample actions (no gradients needed for sampling)
        inputs, lens = self._tokenize(prompts)
        batch_size = len(prompts)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                **gen_cfg.to_generate_kwargs(
                    default_pad=self.tokenizer.pad_token_id,
                    default_eos=self.tokenizer.eos_token_id,
                ),
                num_return_sequences=k,
                return_dict_in_generate=True,
            )

        seqs = out.sequences  # [B*k, T_max]
        if seqs.size(0) != batch_size * k:
            raise RuntimeError("Unexpected number of generated sequences returned.")

        # 2) differentiable forward on the sampled sequences
        attn = (seqs != self.tokenizer.pad_token_id).long()
        logits = self.model(input_ids=seqs, attention_mask=attn, use_cache=False).logits  # [B*k, T_max, V]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)              # predict next token
        tgt = seqs[:, 1:]                                                # [B*k, T_max-1]

        _, Tm1 = tgt.shape
        idx = torch.arange(Tm1, device=self.device).unsqueeze(0)         # [1, T-1]
        prompt_lens = lens.repeat_interleave(k).unsqueeze(1)             # [B*k, 1]
        seq_lens = attn.sum(dim=1).unsqueeze(1)                          # [B*k, 1] true lengths

        start_idx = (prompt_lens - 1).clamp_min(0)
        gen_mask = (idx >= start_idx) & (idx < seq_lens - 1)             # [B*k, T-1]

        tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)        # [B*k, T-1]
        tok_logp = tok_logp * gen_mask                                   
        logprob_sums = tok_logp.sum(dim=1).view(batch_size, k)           # [B, k]

        prompt_lens_flat = prompt_lens.squeeze(1)
        seq_lens_flat = seq_lens.squeeze(1)
        texts: List[List[str]] = []
        for i in range(batch_size):
            prompt_texts = []
            for j in range(k):
                idx_flat = i * k + j
                start = int(prompt_lens_flat[idx_flat].item())
                end = int(seq_lens_flat[idx_flat].item())
                gen_tokens = seqs[idx_flat, start:end]
                prompt_texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
            texts.append(prompt_texts)
        return texts, logprob_sums