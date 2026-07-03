import torch
import torch.nn as nn


class GlobalMeanReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim_multiplier = 3

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        g1 = x1.mean(dim=(1, 2))
        g2 = x2.mean(dim=(1, 2))
        g3 = x3.mean(dim=(1, 2))
        return torch.cat([g1, g2, g3], dim=-1)


class WeightedLevelReadout(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_score1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_score2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_score3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.level_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.out_dim_multiplier = 3

    @staticmethod
    def _weighted_pool(x: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        batch_size, height, width, channels = x.shape
        x_flat = x.view(batch_size, height * width, channels)
        alpha = torch.softmax(scorer(x_flat), dim=1)
        return (alpha * x_flat).sum(dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        g1 = self._weighted_pool(x1, self.node_score1)
        g2 = self._weighted_pool(x2, self.node_score2)
        g3 = self._weighted_pool(x3, self.node_score3)

        g_cat = torch.cat([g1, g2, g3], dim=-1)
        level_alpha = torch.softmax(self.level_gate(g_cat), dim=-1)

        g1 = g1 * level_alpha[:, 0:1]
        g2 = g2 * level_alpha[:, 1:2]
        g3 = g3 * level_alpha[:, 2:3]
        return torch.cat([g1, g2, g3], dim=-1)


def build_readout(hidden_dim: int, readout_type: str = "mean") -> nn.Module:
    readout_type = readout_type.lower()
    if readout_type == "mean":
        return GlobalMeanReadout()
    if readout_type == "weighted":
        return WeightedLevelReadout(hidden_dim=hidden_dim)
    raise ValueError(f"Unsupported readout_type: {readout_type}")


def build_classifier_head(
    hidden_dim: int,
    num_classes: int,
    in_dim_multiplier: int = 3,
) -> nn.Sequential:
    in_dim = hidden_dim * in_dim_multiplier
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, num_classes),
    )
