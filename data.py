"""IMDB reviews -> sequences of GloVe word embeddings."""

import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

MAX_LENGTH = 100  # reviews are truncated / zero-padded to this many tokens
TRAIN_SIZE = 30_000  # first 30k reviews for training, remaining 20k for testing


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)  # links
    text = re.sub(r"[^A-Za-z]+", " ", text)  # non-alphabetic characters (also strips <br /> tags)
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)  # single characters
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return clean(text).lower().split()[:MAX_LENGTH]


class GloVe:
    """Minimal GloVe lookup: unknown words map to the zero vector."""

    def __init__(self, path: str):
        self.vectors: dict[str, np.ndarray] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, *values = line.rstrip().split(" ")
                self.vectors[word] = np.asarray(values, dtype=np.float32)
        self.dim = len(next(iter(self.vectors.values())))

    def embed(self, tokens: list[str]) -> torch.Tensor:
        """Return a (MAX_LENGTH, dim) tensor, zero-padded at the end."""
        out = np.zeros((MAX_LENGTH, self.dim), dtype=np.float32)
        for i, token in enumerate(tokens):
            vec = self.vectors.get(token)
            if vec is not None:
                out[i] = vec
        return torch.from_numpy(out)


class ReviewDataset(Dataset):
    def __init__(self, reviews: list[str], labels: list[str]):
        self.reviews = list(reviews)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int):
        return self.reviews[idx], self.labels[idx]


def make_collate(glove: GloVe):
    """Batch -> (one-hot labels [B, 2] with index 0 = positive, embeddings [B, T, D], token lists)."""

    def collate(batch):
        tokens = [tokenize(review) for review, _ in batch]
        labels = torch.tensor([[1.0, 0.0] if label.lower().startswith("pos") else [0.0, 1.0] for _, label in batch])
        embeddings = torch.stack([glove.embed(t) for t in tokens])
        return labels, embeddings, tokens

    return collate


def get_loaders(csv_path: str, glove: GloVe, batch_size: int = 32):
    data = pd.read_csv(csv_path)
    train = ReviewDataset(data["review"][:TRAIN_SIZE], data["sentiment"][:TRAIN_SIZE])
    test = ReviewDataset(data["review"][TRAIN_SIZE:], data["sentiment"][TRAIN_SIZE:])
    collate = make_collate(glove)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate),
        DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate),
    )
