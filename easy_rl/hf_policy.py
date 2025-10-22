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
        torch_dtype: Union[str, torch.dtype] = "bfloat16",
        device_map: Optional[Union[str, dict]] = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
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
    def generate(self, prompts: Sequence[str], gen_cfg: Optional[GenConfig] = GenConfig()) -> List[str]:
        """Generate completions for a batch of prompts."""
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
            gen_tokens = out[i, lens[i] :]
            texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
        return texts

    @torch.no_grad()
    def generate_with_logprobs(self, prompts: Sequence[str], gen_cfg: Optional[GenConfig] = GenConfig()):
        """Generate completions and return token log-probabilities."""
        inputs, lens = self._tokenize(prompts)
        out = self.model.generate(
            **inputs,
            **gen_cfg.to_generate_kwargs(
                default_pad=self.tokenizer.pad_token_id,
                default_eos=self.tokenizer.eos_token_id,
            ),
            return_dict_in_generate=True,
            output_scores=True,
        )
        seqs, scores = out.sequences, out.scores  # scores: list of [B, V] per gen step

        # log-probabilities of the generated tokens at each step
        # shape: [B, gen_len]
        logprobs = self.model.compute_transition_scores(
            sequences=seqs, scores=scores, normalize_logits=True
        )

        texts, logprob_sums = [], []
        for i in range(len(prompts)):
            gen_tokens = seqs[i, lens[i].item():]
            texts.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
            logprob_sums.append(logprobs[i].sum().item())
        return texts, torch.tensor(logprob_sums, device=self.device)
