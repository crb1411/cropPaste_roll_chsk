import torch
import torch.nn as nn
import torch.nn.functional as F

class InversePatchEmbeddingMLP(nn.Module):
    """
    Input:  x [B, 196, D]
    Output: g [B, D]
    """
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=14, padding=0),
        )

    def forward(self, x):
        B, N, D = x.shape
        assert N == 196, "N must be 196 (14x14)"

        # [B, 196, D] -> [B, D, 14, 14]
        x = x.view(B, 14, 14, D).permute(0, 3, 1, 2)

        # [B, D, 1, 1]
        g = self.net(x)

        # [B, D]
        g = g.squeeze(-1).squeeze(-1)
        return g
