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

TEST_TOPICS = [
    "Why would you rather live in the forest than in the city?",
    "Write a short story about a fish with dreams of becoming a pilot.",
    "Give me a recipe for a meal with the following ingredients: grass, rose, water, salmon, and a secret ingredient.",
]

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


def _collate_prompts(batch: List[str]) -> dict:
    """Return the batch in the format expected by the trainer."""
    return {"prompts": batch}


def make_dataloader(
    topics: Optional[Iterable[str]] = None,
    template: str = "Write ~100 words: {topic}\n\n",
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a simple DataLoader that yields batches of prompt strings.
    """
    topics = topics or TOPICS
    ds = TopicDataset(topics, template=template)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate_prompts,
    )
