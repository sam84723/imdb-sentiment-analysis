"""Four sentiment classifiers over GloVe embeddings, implemented from scratch.

Recurrent models read the review word by word and classify from the last output.
Token-wise models give every word its own (positive, negative) sub-score and
average them, which makes their decisions directly interpretable per word.
"""

import math

import torch
import torch.nn as nn
from torch.nn.functional import pad


class MatMul(nn.Module):
    """Linear layer acting on the last dimension of tensors of any rank."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        return x + self.bias if self.bias is not None else x


class RNN(nn.Module):
    """Vanilla (Elman) recurrent cell."""

    recurrent = True

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def step(self, x, h):
        h = torch.tanh(self.in2hidden(torch.cat((x, h), dim=1)))
        return self.hidden2out(h), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            out, h = self.step(x[:, t], h)
        return out


class GRU(RNN):
    """Gated recurrent unit: update gate z, reset gate r, candidate state."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__(input_size, hidden_size, output_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def step(self, x, h):
        xh = torch.cat((x, h), dim=1)
        z = torch.sigmoid(self.update_gate(xh))
        r = torch.sigmoid(self.reset_gate(xh))
        h_tilde = torch.tanh(self.candidate(torch.cat((x, r * h), dim=1)))
        h = (1 - z) * h + z * h_tilde
        return self.hidden2out(h), h


class TokenMLP(nn.Module):
    """Each word is scored independently; the review score is the mean of word scores."""

    recurrent = False

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2):
        super().__init__()
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)

    def word_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x)))  # (B, T, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.word_scores(x).mean(dim=1)


class AttentionMLP(TokenMLP):
    """Token-wise MLP where each word first attends to its neighbours within +/- `window` positions.

    Restricted self-attention with learned relative positional embeddings, so a word
    like "good" can be re-scored when it follows "not".
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 2, window: int = 5):
        super().__init__(input_size, hidden_size, output_size)
        self.window = window
        self.scale = math.sqrt(hidden_size)
        self.query = MatMul(hidden_size, hidden_size, bias=False)
        self.key = MatMul(hidden_size, hidden_size, bias=False)
        self.value = MatMul(hidden_size, hidden_size, bias=False)
        self.relative_position = nn.Embedding(2 * window + 1, hidden_size)

    def word_scores(self, x: torch.Tensor, return_attention: bool = False):
        w = self.window
        h = torch.relu(self.layer1(x))  # (B, T, H)

        # Stack the 2w+1 neighbours of every position: (B, T, 2w+1, H)
        padded = pad(h, (0, 0, w, w))
        neighbours = torch.stack([torch.roll(padded, k, 1)[:, w:-w] for k in range(-w, w + 1)], dim=2)

        q = self.query(h).unsqueeze(2)  # (B, T, 1, H)
        k = self.key(neighbours) + self.relative_position.weight  # (B, T, 2w+1, H)
        v = self.value(neighbours)
        attention = torch.softmax((q @ k.transpose(-1, -2)).squeeze(2) / self.scale, dim=-1)  # (B, T, 2w+1)
        context = (v * attention.unsqueeze(-1)).sum(dim=2)  # (B, T, H)

        scores = self.layer2(context)
        return (scores, attention) if return_attention else scores


MODELS = {"rnn": RNN, "gru": GRU, "mlp": TokenMLP, "attention": AttentionMLP}
