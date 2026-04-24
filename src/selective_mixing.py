"""Selective Z-Mixing: spatially-adaptive pair representation blending.

Instead of applying a uniform alpha to all (i,j) pairs, this module
computes a per-pair change score based on connectivity and distance
changes, then maps it to a per-pair alpha mask for z blending.

Regions that moved significantly get higher alpha (more z_pseudo signal),
while static regions keep their original z_trunk values.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .coords_to_contact import coords_to_contact
from .mode_drive_utils import kabsch_superimpose


def compute_change_score(
    coords_before: Tensor,
    coords_after: Tensor,
    initial_coords: Tensor,
    r_cut: float,
    tau: float,
    w_c: float = 0.5,
    w_d: float = 0.5,
    distance_mode: str = "max",
) -> Tensor:
    """Compute per-pair change score from connectivity and displacement.

    Combines two signals:
      1. delta_C: absolute change in soft contact map (topological change)
      2. D_ij:    pairwise displacement magnitude (conformational change)

    Both are normalized to [0, 1] and combined via weighted geometric mean.

    Args:
        coords_before:  [N, 3] current coordinates (before displacement).
        coords_after:   [N, 3] displaced coordinates (after displacement).
        initial_coords: [N, 3] initial structure (used as Kabsch reference).
        r_cut:          Contact distance cutoff (Angstrom).
        tau:            Contact sigmoid temperature.
        w_c:            Weight for connectivity change (default 0.5).
        w_d:            Weight for distance change (default 0.5).
        distance_mode:  "max" or "mean" for pairwise displacement (default "max").

    Returns:
        S: [N, N] symmetric change score in [0, 1].
    """
    eps = 1e-8

    # 1. Connectivity change: |C_after - C_before|
    c_before = coords_to_contact(coords_before, r_cut, tau)
    c_after = coords_to_contact(coords_after, r_cut, tau)
    delta_c = (c_after - c_before).abs()  # [N, N]

    # 2. Distance change: per-residue displacement after Kabsch alignment
    coords_after_aligned, _ = kabsch_superimpose(initial_coords, coords_after)
    d_per_residue = (coords_after_aligned - coords_before).pow(2).sum(dim=-1).sqrt()  # [N]

    # Pairwise displacement: max(d_i, d_j) or mean(d_i, d_j)
    d_i = d_per_residue.unsqueeze(1)  # [N, 1]
    d_j = d_per_residue.unsqueeze(0)  # [1, N]

    if distance_mode == "mean":
        d_ij = 0.5 * (d_i + d_j)  # [N, N]
    else:  # "max"
        d_ij = torch.max(d_i.expand_as(delta_c), d_j.expand_as(delta_c))  # [N, N]

    # 3. Normalize both to [0, 1]
    dc_max = delta_c.max()
    delta_c_norm = delta_c / (dc_max + eps) if dc_max > eps else delta_c

    dij_max = d_ij.max()
    d_norm = d_ij / (dij_max + eps) if dij_max > eps else d_ij

    # 4. Combined score: weighted geometric mean
    s = delta_c_norm.pow(w_c) * d_norm.pow(w_d)  # [N, N]

    # Ensure symmetry (should already be symmetric but enforce it)
    s = 0.5 * (s + s.T)

    # Zero diagonal
    s.fill_diagonal_(0.0)

    return s


def compute_alpha_mask(
    change_score: Tensor,
    change_cutoff: float = 0.1,
    alpha_base: float = 0.0,
    alpha_max: float = 1.0,
    mapping: str = "linear",
) -> Tensor:
    """Map change score to per-pair alpha mask.

    Args:
        change_score:  [N, N] change score from compute_change_score.
        change_cutoff: Scores below this threshold get alpha_base (default 0.1).
        alpha_base:    Alpha for unchanged pairs (default 0.0).
        alpha_max:     Alpha for maximally changed pairs (default 1.0).
        mapping:       "linear", "sigmoid", or "step" (default "linear").

    Returns:
        alpha_mask: [N, N] per-pair alpha values in [alpha_base, alpha_max].
    """
    eps = 1e-8

    # Apply cutoff: zero out sub-threshold scores
    s = change_score.clone()
    below_cutoff = s < change_cutoff
    s[below_cutoff] = 0.0

    if mapping == "step":
        # Binary: alpha_base or alpha_max
        alpha = torch.where(below_cutoff, alpha_base, alpha_max)

    elif mapping == "sigmoid":
        # Smooth sigmoid transition around midpoint
        s_max = s.max()
        if s_max > eps:
            s_normalized = s / s_max  # [0, 1]
        else:
            s_normalized = s
        # Sigmoid centered at 0.5, temperature=10 for reasonable sharpness
        sigmoid_input = (s_normalized - 0.5) * 10.0
        t = torch.sigmoid(sigmoid_input)
        # Re-apply cutoff mask
        t[below_cutoff] = 0.0
        alpha = alpha_base + (alpha_max - alpha_base) * t

    else:  # "linear"
        # Linear interpolation from alpha_base to alpha_max
        s_max = s.max()
        if s_max > eps:
            s_normalized = s / s_max  # [0, 1]
        else:
            s_normalized = s
        alpha = alpha_base + (alpha_max - alpha_base) * s_normalized

    # Ensure result is a tensor with proper type
    if not isinstance(alpha, Tensor):
        alpha = torch.full_like(change_score, alpha)

    # Ensure symmetry
    alpha = 0.5 * (alpha + alpha.T)

    # Zero diagonal
    alpha.fill_diagonal_(0.0)

    return alpha


def selective_blend_z(
    z_pseudo: Tensor,
    z_trunk: Tensor,
    alpha_mask: Tensor,
    normalize: bool = True,
    direction: str = "plus",
) -> Tensor:
    """Blend z_pseudo into z_trunk with per-pair alpha mask.

    Args:
        z_pseudo:   [N, N, C] pseudo pair representation from contact map.
        z_trunk:    [N, N, C] trunk pair representation from OF3.
        alpha_mask: [N, N] per-pair alpha values.
        normalize:  If True, normalize z_pseudo stats to match z_trunk.
        direction:  "plus" or "minus" — add or subtract delta_z.

    Returns:
        z_blended: [N, N, C] blended pair representation.
    """
    if normalize:
        z_pseudo = (z_pseudo - z_pseudo.mean()) / (z_pseudo.std() + 1e-8)
        z_pseudo = z_pseudo * z_trunk.std() + z_trunk.mean()

    delta_z = z_pseudo - z_trunk

    # Expand alpha_mask to broadcast over channel dimension: [N, N, 1]
    alpha_expanded = alpha_mask.unsqueeze(-1)

    if direction == "minus":
        return z_trunk - alpha_expanded * delta_z
    else:  # "plus"
        return z_trunk + alpha_expanded * delta_z
