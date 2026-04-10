"""GNMContactLearner: frozen OpenFold3 trunk + trainable contact head."""

from typing import Any, Dict

import torch
import torch.nn as nn

from .contact_head import ContactProjectionHead


class GNMContactLearner(nn.Module):
    """Full model: OpenFold3 (frozen) + ContactProjectionHead (trainable).

    Forward returns a dict:
        C_pred:          [B, N, N]    contact probabilities
        pair_repr:       [B, N, N, c_z] raw pair repr (detached)
        pair_repr_recon: [B, N, N, c_z] encoder→decoder reconstruction
    """

    def __init__(
        self,
        openfold_model: nn.Module,
        c_z: int = 128,
        bottleneck_dim: int = 32,
    ) -> None:
        super().__init__()

        # Freeze the OpenFold3 backbone
        self.openfold = openfold_model
        for param in self.openfold.parameters():
            param.requires_grad_(False)
        self.openfold.eval()

        # Trainable head
        self.contact_head = ContactProjectionHead(
            c_z=c_z, bottleneck_dim=bottleneck_dim
        )

    def forward(
        self, batch: dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict consumed by ``openfold.run_trunk()``.

        Returns:
            dict with keys: C_pred, pair_repr, pair_repr_recon
        """
        with torch.no_grad():
            _s_input, _s, z = self.openfold.run_trunk(batch)

        # Contact prediction (forward path)
        c_pred = self.contact_head(z)

        # Reconstruction path (for L_recon): encode → decode
        z_sym = 0.5 * (z + z.transpose(1, 2))
        h = self.contact_head.w_enc(z_sym)       # [B, N, N, k]
        z_recon = self.contact_head.w_dec(h)      # [B, N, N, c_z]

        return {
            "C_pred": c_pred,
            "pair_repr": z.detach(),
            "pair_repr_recon": z_recon,
        }

    def train(self, mode: bool = True) -> "GNMContactLearner":
        """Keep openfold frozen even when model.train() is called."""
        super().train(mode)
        self.openfold.eval()
        return self
