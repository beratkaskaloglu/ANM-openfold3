"""Convert arbitrary 3D coordinates to a soft contact map.

Same sigmoid formula as ground_truth.py but for displaced/generated
coordinates in the mode-drive pipeline.
"""

import torch


def coords_to_contact(
    coords: torch.Tensor,
    r_cut: float = 10.0,
    tau: float = 1.5,
) -> torch.Tensor:
    """Compute soft contact map from coordinates.

    C_ij = sigmoid(-(d_ij - r_cut) / tau), C_ii = 0.

    Args:
        coords: [N, 3] atom positions.
        r_cut:  Distance cutoff centre (Angstrom).
        tau:    Sigmoid temperature.

    Returns:
        C: [N, N] symmetric contact probabilities in [0, 1].
    """
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    c = torch.sigmoid(-(dist - r_cut) / tau)
    c.fill_diagonal_(0.0)
    return c
