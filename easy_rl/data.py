from typing import Iterable, List, Optional
from torch.utils.data import Dataset, DataLoader

TOPICS = [
    "Explain why bicycles stay upright while moving.",
    "Describe a hidden corner of your city worth visiting.",
    "What makes a great cup of tea?",
    "Tell a short scene about two strangers on a train.",
    "Describe the sound of rain without using the word 'rain'.",
    "Tell a short story about a cat with a human voice.",
    "Why is the sky blue?",
    "Tell me about the capital of India.",
    "Tell me the history of the internet.",
    "Who invented the light bulb? What else did they invent?",
    "Explain attention mechanism in transformers.",
    "Explain the plot of Harry Potter and the Philosopher's Stone.",
    "Give instructions to a robot to clean a room.",
    "Tell me about COVID-19 and how it spread.",
    "Write a cinematic story about cherry blossoms.",
    "How can I clean white trainers?",
    "Tell me about types of silicon wafers and their applications.",
    "Write a list of exceptional foods to eat when bulking.",
    "Who are the 10 greatest athletes of all time?",
    "Describe a bustling market without naming the city.",
    "Explain how a seed becomes a tree.",
    "Write a vivid scene about a storm ending.",
    "What makes a memorable cup of tea?",
    "Describe the texture of autumn light.",
]

def batch_prompts(batch_size=8):
    import random
    return [f"Write ~100 words: {random.choice(TOPICS)}\n\n" for _ in range(batch_size)]


class TopicDataset(Dataset):
    """Formats topics into prompts with a simple template."""

    def __init__(self, topics: Iterable[str], template: str = "Write ~100 words: {topic}\n\n"):
        topics = list(topics)
        self.topics = topics
        self.template = template

    def __len__(self) -> int:
        return len(self.topics)

    def __getitem__(self, idx: int) -> str:
        return self.template.format(topic=self.topics[idx])


def _collate_prompts(batch: List[str]) -> List[str]:
    """Keep batches as plain lists of strings."""
    return batch


def make_dataloader(
    topics: Optional[Iterable[str]] = TOPICS,
    template: str = "Write ~100 words: {topic}\n\n",
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a simple DataLoader that yields batches of prompt strings.
    """
    ds = TopicDataset(topics, template=template)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate_prompts,
    )