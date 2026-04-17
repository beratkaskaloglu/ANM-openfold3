"""Structural utility functions for the ANM Mode-Drive pipeline.

Kabsch superimposition, RMSD, TM-score, contact-to-distance inversion,
classical MDS, and pseudo-diffusion construction.

Note: iw_enm/analysis.py has numpy equivalents of kabsch/RMSD/TM-score
for the IW-ENM MD trajectory analysis path.  These torch versions are
used by the Mode-Drive pipeline which operates on torch tensors.
"""

from __future__ import annotations

from typing import Callable

import torch

from .converter import PairContactConverter
from .coords_to_contact import coords_to_contact


def kabsch_superimpose(
    ref: torch.Tensor,
    mobile: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kabsch superimposition: align mobile onto ref.

    Args:
        ref: [N, 3] reference coordinates
        mobile: [N, 3] mobile coordinates to align

    Returns:
        aligned: [N, 3] mobile after optimal rotation+translation
        rmsd: scalar RMSD after alignment
    """
    # 1. Center both
    ref_center = ref.mean(dim=0)
    mob_center = mobile.mean(dim=0)
    ref_centered = ref - ref_center
    mob_centered = mobile - mob_center

    # 2. Covariance matrix
    H = mob_centered.T @ ref_centered  # [3, 3]

    # 3. SVD
    U, S, Vt = torch.linalg.svd(H)

    # 4. Correct reflection
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.diag(
        torch.tensor([1.0, 1.0, torch.sign(d)], device=ref.device, dtype=ref.dtype),
    )

    # 5. Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # 6. Apply
    aligned = (mob_centered @ R.T) + ref_center

    # 7. RMSD
    rmsd_val = ((ref - aligned) ** 2).sum(dim=-1).mean().sqrt()

    return aligned, rmsd_val


def compute_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """RMSD between two coordinate sets after Kabsch superimposition [N, 3]."""
    _, rmsd_val = kabsch_superimpose(a, b)
    return rmsd_val.item()


def tm_score(
    coords_model: torch.Tensor,
    coords_ref: torch.Tensor,
) -> float:
    """Approximate TM-score between two CA coordinate sets.

    TM-score = (1/N) * sum_i 1 / (1 + (d_i / d0)^2)

    where d_i is per-residue distance after Kabsch superimposition
    and d0 = 1.24 * (N - 15)^(1/3) - 1.8

    Args:
        coords_model: [N, 3] model coordinates
        coords_ref:   [N, 3] reference coordinates

    Returns:
        TM-score in [0, 1]
    """
    aligned, _ = kabsch_superimpose(coords_ref, coords_model)
    N = coords_ref.shape[0]
    d0 = 1.24 * (max(N, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    di = ((coords_ref - aligned) ** 2).sum(dim=-1).sqrt()
    scores = 1.0 / (1.0 + (di / d0) ** 2)
    return scores.mean().item()


def contact_to_distance(contact: torch.Tensor, r_cut: float, tau: float) -> torch.Tensor:
    """Invert sigmoid soft contact to approximate distances.

    d_ij = r_cut - tau * ln(C / (1 - C))

    Args:
        contact: [N, N] contact probabilities in (0, 1).
        r_cut:   Cutoff centre used in coords_to_contact.
        tau:     Sigmoid temperature.

    Returns:
        dist: [N, N] approximate pairwise distances.
    """
    c = contact.clamp(1e-6, 1.0 - 1e-6)
    logit = torch.log(c / (1.0 - c))  # sigmoid inverse
    dist = r_cut - tau * logit
    dist = dist.clamp(min=0.0)
    dist.fill_diagonal_(0.0)
    # Symmetrize
    dist = 0.5 * (dist + dist.T)
    return dist


def classical_mds(dist_matrix: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """Classical multidimensional scaling: distance matrix -> 3D coordinates.

    Args:
        dist_matrix: [N, N] symmetric distance matrix.
        dim: Embedding dimension (3 for 3D coords).

    Returns:
        coords: [N, dim] embedded coordinates.
    """
    N = dist_matrix.shape[0]
    D2 = dist_matrix ** 2

    # Centering matrix: H = I - (1/N) * 11^T
    H = torch.eye(N, device=dist_matrix.device, dtype=dist_matrix.dtype) - 1.0 / N

    # Double-centered matrix: B = -0.5 * H * D^2 * H
    B = -0.5 * H @ D2 @ H

    # Eigendecompose (float64 for stability)
    B64 = B.to(dtype=torch.float64, device="cpu")
    vals, vecs = torch.linalg.eigh(B64)
    vals = vals.to(dtype=dist_matrix.dtype, device=dist_matrix.device)
    vecs = vecs.to(dtype=dist_matrix.dtype, device=dist_matrix.device)

    # Take top `dim` eigenvalues (largest, at the end)
    top_vals = vals[-dim:].flip(0).clamp(min=0.0)
    top_vecs = vecs[:, -dim:].flip(1)

    # Coordinates: X = V * sqrt(Lambda)
    coords = top_vecs * top_vals.sqrt().unsqueeze(0)

    return coords


def make_pseudo_diffusion(
    converter: PairContactConverter,
    r_cut: float = 10.0,
    tau: float = 1.5,
    reference_coords: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a pseudo-diffusion function for testing without OF3.

    Converts blended z_mod back to 3D coordinates via:
        z_mod -> contact (forward head) -> distances (invert sigmoid) -> MDS -> coords

    If reference_coords is provided, the MDS output is Kabsch-aligned
    to the reference to maintain consistent orientation.

    Args:
        converter:        Trained PairContactConverter.
        r_cut:            Contact cutoff used in coords_to_contact.
        tau:              Sigmoid temperature.
        reference_coords: [N, 3] coords for alignment (typically initial structure).

    Returns:
        diffusion_fn: Callable([N, N, 128]) -> [N, 3]
    """
    def _pseudo_diffuse(z_mod: torch.Tensor) -> torch.Tensor:
        # z_mod [N, N, 128] -> contact [N, N]
        contact = converter.z_to_contact(z_mod)

        # contact -> approximate distance matrix
        dist = contact_to_distance(contact, r_cut, tau)

        # distance -> 3D coordinates via classical MDS
        coords = classical_mds(dist, dim=3)

        # Align to reference if available
        if reference_coords is not None:
            coords, _ = kabsch_superimpose(reference_coords, coords)

        return coords

    return _pseudo_diffuse
