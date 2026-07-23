"""Temporal classifiers over (B, T=30, F=55) skeleton-feature windows.

Both models are deliberately tiny (<1 M params) — the pose backbone has
already done the heavy perceptual lifting; the temporal net only needs
to recognize a ~0.5 s kinematic signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """2-layer GRU, hidden 64 -> 1 logit. ~<0.1 M params."""

    def __init__(self, input_dim: int = 55, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F) -> (B,)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)   # last hidden state


class CNN1DClassifier(nn.Module):
    """1D-CNN baseline: Conv-BN-ReLU stack + global max-pool.

    Rationale for the comparison: a temporal CNN sees fixed-size local
    motifs (the impact spike) but has no gating/memory; if the GRU only
    matches it, the recurrence isn't buying anything.
    """

    def __init__(self, input_dim: int = 55, channels=(64, 128, 128),
                 kernel: int = 5, dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], input_dim
        for c_out in channels:
            layers += [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                       nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F) -> (B,)
        z = self.body(x.transpose(1, 2))       # (B,C,T)
        z = z.max(dim=2).values                # global max pool over time
        return self.head(z).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) on logits.

    FALL windows are ~5-10 % of URFD windows; focal loss down-weights
    the easy NORMAL majority (gamma) and re-balances classes (alpha)
    without throwing away negatives via undersampling.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)                            # prob of the true class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["type"] == "gru":
        return GRUClassifier(m["input_dim"], m["hidden"], m["layers"], m["dropout"])
    if m["type"] == "cnn":
        return CNN1DClassifier(m["input_dim"], tuple(m["cnn_channels"]),
                               m["cnn_kernel"], m["dropout"])
    raise ValueError(f"unknown model type {m['type']!r}")
