import re
from typing import List

import torch

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words(text):
    return [w.lower() for w in _word_re.findall(text)]


def ttr(text):
    """Type-token ratio: number of unique words / total number of words."""
    ws = words(text)
    return len(set(ws)) / max(len(ws), 1)


def brevity_penalty(text, min_length=80):
    """
    Penalty for short text.

    Hinge penalty: 0 when length >= min_length, otherwise linear.
    """
    L = len(words(text))
    return max(0.0, (min_length - L)/min_length)


def reward(texts: List[str], alpha: float = 0.5, min_length: int = 80):
    rewards = []
    for text in texts:
        rewards.append(ttr(text) - alpha*brevity_penalty(text, min_length))
    return torch.tensor(rewards)
