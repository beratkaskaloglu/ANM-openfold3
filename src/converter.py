"""Bidirectional converter: pair representation <-> contact map.

Loads trained ContactProjectionHead weights and provides:
  - z_to_contact(z)   : [N, N, 128] → [N, N] connectivity
  - contact_to_z(C)   : [N, N]      → [N, N, 128] pseudo pair repr
  - analyze(C or z)   : GNM decomposition (eigenvalues, eigenvectors, B-factors)
"""

from pathlib import Path
from typing import Tuple

import torch

from .contact_head import ContactProjectionHead
from .kirchhoff import gnm_decompose, soft_kirchhoff


class PairContactConverter:
    """Bidirectional pair_repr <-> contact converter with GNM analysis.

    Args:
        checkpoint: path to best_model.pt (or dict with model_state_dict).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        checkpoint: str | Path | dict | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Load checkpoint metadata
        if checkpoint is not None:
            if isinstance(checkpoint, dict):
                ckpt = checkpoint
            else:
                ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            c_z = ckpt.get("c_z", 128)
            bottleneck_dim = ckpt.get("bottleneck_dim", 64)
        else:
            ckpt = None
            c_z = 128
            bottleneck_dim = 64

        self.head = ContactProjectionHead(c_z=c_z, bottleneck_dim=bottleneck_dim)

        if ckpt is not None:
            self.head.load_state_dict(ckpt["model_state_dict"])

        self.head.to(self.device)
        self.head.eval()

    @torch.no_grad()
    def z_to_contact(self, z: torch.Tensor) -> torch.Tensor:
        """Convert pair representation to contact map.

        Args:
            z: [N, N, 128] or [B, N, N, 128] pair representation.

        Returns:
            C: [N, N] or [B, N, N] contact probabilities in [0, 1].
        """
        squeeze = z.dim() == 3
        if squeeze:
            z = z.unsqueeze(0)

        z = z.to(self.device)
        c = self.head(z)

        if squeeze:
            c = c.squeeze(0)
        return c

    @torch.no_grad()
    def contact_to_z(self, c: torch.Tensor) -> torch.Tensor:
        """Reconstruct pseudo pair representation from contact map.

        Args:
            c: [N, N] contact probabilities in (0, 1).

        Returns:
            pseudo_z: [N, N, 128] approximate pair representation.
        """
        c = c.to(self.device)
        return self.head.inverse(c)

    @torch.no_grad()
    def analyze(
        self,
        x: torch.Tensor,
        n_modes: int = 20,
        is_contact: bool | None = None,
    ) -> dict:
        """Run GNM analysis on a contact map or pair representation.

        Args:
            x: [N, N] contact map or [N, N, 128] pair representation.
            n_modes: number of non-trivial modes.
            is_contact: if None, inferred from shape. True = contact, False = pair repr.

        Returns:
            dict with keys:
                contact:      [N, N]        contact map
                kirchhoff:    [N, N]        Kirchhoff matrix
                eigenvalues:  [n_modes]     non-trivial eigenvalues
                eigenvectors: [N, n_modes]  eigenvectors
                b_factors:    [N]           per-residue B-factors
        """
        if is_contact is None:
            is_contact = x.dim() == 2

        if is_contact:
            contact = x.to(self.device)
        else:
            contact = self.z_to_contact(x)

        gamma = soft_kirchhoff(contact)
        eigenvalues, eigenvectors, b_factors = gnm_decompose(gamma, n_modes)

        return {
            "contact": contact,
            "kirchhoff": gamma,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "b_factors": b_factors,
        }

    @torch.no_grad()
    def roundtrip(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """z -> contact -> z_recon, returns (contact, z_recon, mse).

        Useful for checking reconstruction quality.
        """
        contact = self.z_to_contact(z)
        z_recon = self.contact_to_z(contact)

        # Truncate to match shapes if needed
        n = min(z.shape[0], z_recon.shape[0])
        mse = ((z[:n, :n] - z_recon[:n, :n]) ** 2).mean().item()

        return contact, z_recon, mse
