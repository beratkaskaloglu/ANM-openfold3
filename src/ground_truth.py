"""Ground truth soft contact map from PDB Cα coordinates."""

import torch


def compute_gt_probability_matrix(
    coords_ca: torch.Tensor,
    r_cut: float = 10.0,
    tau: float = 1.5,
) -> torch.Tensor:
    """Compute sigmoid-based soft contact map from Cα coordinates.

    Args:
        coords_ca: [N, 3] Cα atom positions in Ångströms.
        r_cut: Distance cutoff centre (Å). Pairs at this distance get p ≈ 0.5.
        tau: Sigmoid temperature — smaller → sharper transition.

    Returns:
        C_gt: [N, N] symmetric matrix in [0, 1], diagonal = 0.
    """
    # Pairwise Euclidean distances  [N, N]
    dist = torch.cdist(
        coords_ca.unsqueeze(0), coords_ca.unsqueeze(0)
    ).squeeze(0)

    # Sigmoid soft cutoff: σ(-(d_ij − r_cut) / τ)
    c_gt = torch.sigmoid(-(dist - r_cut) / tau)

    # Self-contact is undefined → zero
    c_gt.fill_diagonal_(0.0)

    return c_gt
