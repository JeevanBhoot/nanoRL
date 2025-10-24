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
        forward_chunk_size: Optional[int] = 4,
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
        self.forward_chunk_size = max(1, forward_chunk_size) if forward_chunk_size else None

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

        # 2) differentiable forward on the sampled sequences, optionally in chunks
        attn = (seqs != self.tokenizer.pad_token_id).long()
        seq_lens = attn.sum(dim=1, keepdim=True)
        start_idx = lens.unsqueeze(1)
        logprob_sums = self._compute_logprob_sums(
            seqs=seqs,
            attn=attn,
            start_idx=start_idx,
            seq_lens=seq_lens,
        )

        # decode continuations (for logging/evaluation)
        seq_lens_flat = seq_lens.squeeze(1)
        texts = []
        for i in range(seqs.size(0)):
            gen_tokens = seqs[i, lens[i].item():seq_lens_flat[i].item()]
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
        seq_lens = attn.sum(dim=1, keepdim=True)
        prompt_lens = lens.repeat_interleave(k).unsqueeze(1)
        start_idx = (prompt_lens - 1).clamp_min(0)
        flat_logprob_sums = self._compute_logprob_sums(
            seqs=seqs,
            attn=attn,
            start_idx=start_idx,
            seq_lens=seq_lens,
        )
        logprob_sums = flat_logprob_sums.view(batch_size, k)

        # decode generated text only
        prompt_lens_flat = prompt_lens.squeeze(1)
        seq_lens_flat = seq_lens.squeeze(1)

        texts: List[List[str]] = []
        for i in range(batch_size):
            group = []
            for j in range(k):
                idx_flat = i * k + j
                start = int(prompt_lens_flat[idx_flat].item())
                end = int(seq_lens_flat[idx_flat].item())
                gen_tokens = seqs[idx_flat, start:end]
                group.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
            texts.append(group)

        return texts, logprob_sums

    def _compute_logprob_sums(
        self,
        seqs: torch.Tensor,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-prob sums for sequences, processing in chunks."""
        total = seqs.size(0)
        chunk = self.forward_chunk_size or total
        logprob_sums = []
        for offset in range(0, total, chunk):
            end = min(offset + chunk, total)
            seqs_chunk = seqs[offset:end]
            attn_chunk = attn[offset:end]
            outputs = self.model(input_ids=seqs_chunk, attention_mask=attn_chunk, use_cache=False)
            logits = outputs.logits  # [chunk, T_max, V]
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = seqs_chunk[:, 1:]

            valid = (torch.arange(tgt.size(1), device=self.device)[None, :] < attn_chunk.sum(dim=1, keepdim=True) - 1)
            logprob_sums.append((logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * valid).sum(dim=1))
            
        return torch.cat(logprob_sums, dim=0)
