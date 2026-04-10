"""Invertible bottleneck head: pair representation ↔ contact probability.

Forward:  z[B,N,N,128] → W_enc → h[B,N,N,k] → dot(v) → sigmoid → C[B,N,N]
Inverse:  C[N,N] → logit → broadcast × v^T → [N,N,k] → W_dec → pseudo_z[N,N,128]
"""

import torch
import torch.nn as nn


class ContactProjectionHead(nn.Module):
    """Encoder-decoder bottleneck for pair → contact with inverse path.

    Parameters (~8 k for k=32):
        W_enc: Linear(c_z, bottleneck_dim)   — 128×32 = 4096
        v:     Parameter(bottleneck_dim)      — 32
        W_dec: Linear(bottleneck_dim, c_z)    — 32×128 = 4096
    """

    def __init__(
        self,
        c_z: int = 128,
        bottleneck_dim: int = 32,
    ) -> None:
        super().__init__()
        self.c_z = c_z
        self.bottleneck_dim = bottleneck_dim

        # Encoder: pair space → bottleneck
        self.w_enc = nn.Linear(c_z, bottleneck_dim, bias=False)

        # Contact vector: bottleneck → scalar
        self.v = nn.Parameter(torch.randn(bottleneck_dim))

        # Decoder: bottleneck → pair space (inverse path)
        self.w_dec = nn.Linear(bottleneck_dim, c_z, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode pair repr to contact probability.

        Args:
            z: [B, N, N, c_z] pair representation.

        Returns:
            c_pred: [B, N, N] symmetric, diagonal=0, values in [0,1].
        """
        # Symmetrise input
        z_sym = 0.5 * (z + z.transpose(1, 2))

        # Encode to bottleneck
        h = self.w_enc(z_sym)  # [B, N, N, k]

        # Dot with contact vector → scalar logit
        logits = (h * self.v).sum(dim=-1)  # [B, N, N]

        # Symmetrise logits
        logits = 0.5 * (logits + logits.transpose(-1, -2))

        # Zero diagonal
        n = logits.shape[-1]
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=logits.device)
        logits = logits * diag_mask

        # Sigmoid
        c_pred = torch.sigmoid(logits)
        c_pred = c_pred * diag_mask  # re-zero (sigmoid(0)=0.5)

        return c_pred

    def encode_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        """Return bottleneck representation h for analysis.

        Args:
            z: [B, N, N, c_z]

        Returns:
            h: [B, N, N, bottleneck_dim]
        """
        z_sym = 0.5 * (z + z.transpose(1, 2))
        return self.w_enc(z_sym)

    @torch.no_grad()
    def inverse(self, c: torch.Tensor) -> torch.Tensor:
        """Reconstruct pseudo pair representation from contact map.

        Args:
            c: [N, N] contact probabilities in (0, 1).

        Returns:
            pseudo_z: [N, N, c_z] approximate pair representation.
        """
        # Sigmoid inverse (logit)
        c_clamped = c.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(c_clamped / (1.0 - c_clamped))  # [N, N]

        # Broadcast through contact vector → bottleneck
        v_norm = self.v / (self.v.norm() + 1e-8)
        h_approx = logit.unsqueeze(-1) * v_norm  # [N, N, k]

        # Decode to pair space
        pseudo_z = self.w_dec(h_approx)  # [N, N, c_z]

        return pseudo_z
