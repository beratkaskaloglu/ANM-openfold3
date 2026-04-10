"""Inverse path: coordinates → pseudo pair representation."""

from pathlib import Path

import torch

from .contact_head import ContactProjectionHead
from .data import extract_ca_coords
from .ground_truth import compute_gt_probability_matrix


class PairReprFromCoords:
    """Generate pseudo pair representation from protein coordinates.

    Uses the trained inverse path of ContactProjectionHead:
        coords → distance → sigmoid contact → logit → ×v^T → W_dec → pseudo_z

    The resulting tensor lives in OpenFold3's pair representation space
    and can be used for downstream ANM/TE analyses.
    """

    def __init__(self, trained_contact_head: ContactProjectionHead) -> None:
        self.head = trained_contact_head
        self.head.eval()

    @torch.no_grad()
    def __call__(
        self,
        coords_ca: torch.Tensor,
        r_cut: float = 10.0,
        tau: float = 1.5,
    ) -> torch.Tensor:
        """Generate pseudo pair repr from Cα coordinates.

        Args:
            coords_ca: [N, 3] Cα positions in Ångströms.
            r_cut: sigmoid cutoff centre.
            tau: sigmoid temperature.

        Returns:
            pseudo_pair: [N, N, c_z] approximate pair representation.
        """
        c = compute_gt_probability_matrix(coords_ca, r_cut=r_cut, tau=tau)
        return self.head.inverse(c)

    @torch.no_grad()
    def from_pdb(
        self,
        pdb_path: str | Path,
        r_cut: float = 10.0,
        tau: float = 1.5,
    ) -> torch.Tensor:
        """Generate pseudo pair repr directly from a PDB file.

        Args:
            pdb_path: path to PDB or mmCIF file.
            r_cut: sigmoid cutoff centre.
            tau: sigmoid temperature.

        Returns:
            pseudo_pair: [N, N, c_z]
        """
        coords_ca = extract_ca_coords(pdb_path)
        return self(coords_ca, r_cut=r_cut, tau=tau)
