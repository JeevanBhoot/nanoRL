import csv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def save_test_generations(path: Path, before: List[str], after: List[str], topics: List[str]) -> None:
    """Save before/after training generations to a text file."""
    with path.open("w") as f:
        f.write("Before Training:\n" + "\n".join([f"Q: {q}\nA: {a}\n" for q, a in zip(topics, before)]))
        f.write("\nAfter Training:\n" + "\n".join([f"Q: {q}\nA: {a}\n" for q, a in zip(topics, after)]))


def save_training_samples(path: Path, samples: List[Tuple[int, str, str]]) -> None:
    """Save tracked generations over training epochs."""
    with path.open("w") as f:
        f.write("\n\n".join([f"Epoch {e}\nPrompt: {p}\nGen: {g}" for e, p, g in samples]))
