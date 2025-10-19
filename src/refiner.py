# src/models/refiner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreRefiner(nn.Module):
    """
    Input:  A  (D,) audio emb (L2-normalized)
            Vt (T,D) video embs (L2-normalized)
    Output: logits over T (unnormalized scores per timestep)
    """
    def __init__(self, dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        # simple feature fusion: [V, A, V*A, |V-A|] -> 4D
        in_dim = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, a: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        a: (D,) or (B,D)
        V: (T,D) or (B,T,D)
        returns: logits (T,) or (B,T)
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)     # (1,D)
        if V.dim() == 2:
            V = V.unsqueeze(0)     # (1,T,D)

        B, T, D = V.shape
        A = a[:, None, :].expand(B, T, D)         # (B,T,D)

        feats = torch.cat([V, A, V * A, torch.abs(V - A)], dim=-1)   # (B,T,4D)
        logits = self.mlp(feats).squeeze(-1)      # (B,T)
        return logits