"""Anisotropic Network Model (ANM): Hessian, eigendecomposition, displacement.

ANM extends GNM from scalar (N x N) to vectorial (3N x 3N), providing
per-residue 3D displacement directions for each normal mode.

References:
    - Atilgan et al. (2001) Biophys J 80:505-515
    - Eyal et al. (2006) Bioinformatics 22:2619-2627
"""

from typing import Tuple

import torch


def build_hessian(
    coords: torch.Tensor,
    cutoff: float = 15.0,
    gamma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """Build the 3N x 3N ANM Hessian from CA coordinates.

    Off-diagonal 3x3 block: H_ij = -gamma * w_ij * (e_ij outer e_ij)
    Diagonal 3x3 block:     H_ii = -sum_{j!=i} H_ij

    Args:
        coords: [N, 3] CA atom positions.
        cutoff: Distance cutoff centre (Angstrom).
        gamma:  Uniform spring constant.
        tau:    Sigmoid temperature for soft cutoff.

    Returns:
        H: [3N, 3N] symmetric Hessian matrix.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    # Pairwise difference vectors: [N, N, 3]
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # r_j - r_i

    # Pairwise distances: [N, N]
    dist = diff.norm(dim=-1)

    # Soft contact weights: [N, N]
    w = torch.sigmoid(-(dist - cutoff) / tau)

    # Zero self-contacts
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    w = w * mask.float()

    # Unit vectors: [N, N, 3]
    e = diff / (dist.unsqueeze(-1) + 1e-8)

    # Outer product: [N, N, 3, 3]
    outer = e.unsqueeze(-1) * e.unsqueeze(-2)

    # Off-diagonal super-elements: [N, N, 3, 3]
    H_blocks = -gamma * w.unsqueeze(-1).unsqueeze(-1) * outer

    # Diagonal super-elements: H_ii = -sum_{j!=i} H_ij
    diag_blocks = -H_blocks.sum(dim=1)  # [N, 3, 3]
    idx = torch.arange(N, device=device)
    H_blocks[idx, idx] = diag_blocks

    # Reshape [N, N, 3, 3] -> [3N, 3N]
    H = H_blocks.permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

    return H


def anm_modes(
    hessian: torch.Tensor,
    n_modes: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eigendecompose ANM Hessian, skipping 6 trivial rigid-body modes.

    Args:
        hessian: [3N, 3N] symmetric Hessian.
        n_modes: Number of non-trivial modes to return.

    Returns:
        eigenvalues:  [n_modes] ascending non-trivial eigenvalues.
        eigenvectors: [N, n_modes, 3] per-residue 3D displacement directions.
    """
    device = hessian.device
    orig_dtype = hessian.dtype
    dim = hessian.shape[0]
    N = dim // 3

    # float64 for numerical stability, CPU for eigh reliability
    H64 = hessian.to(dtype=torch.float64, device="cpu")
    vals, vecs = torch.linalg.eigh(H64)
    vals = vals.to(dtype=orig_dtype, device=device)
    vecs = vecs.to(dtype=orig_dtype, device=device)

    # Skip first 6 trivial modes (3 translation + 3 rotation)
    max_modes = dim - 6
    k = min(n_modes, max_modes)

    eig_vals = vals[6 : 6 + k]          # [k]
    eig_vecs = vecs[:, 6 : 6 + k]       # [3N, k]

    # Reshape to per-residue: [3N, k] -> [N, k, 3]
    eig_vecs = eig_vecs.reshape(N, 3, k).permute(0, 2, 1)  # [N, k, 3]

    return eig_vals, eig_vecs


def displace(
    coords: torch.Tensor,
    mode_vectors: torch.Tensor,
    dfs: torch.Tensor,
) -> torch.Tensor:
    """Apply mode-driven displacement to coordinates.

    new_coords = coords + sum_k(df_k * v_k)

    Args:
        coords:       [N, 3] original CA positions.
        mode_vectors: [N, n_selected, 3] selected mode displacement vectors.
        dfs:          [n_selected] displacement factors.

    Returns:
        new_coords: [N, 3] displaced positions.
    """
    # [N, n_sel, 3] * [n_sel] -> sum -> [N, 3]
    displacement = (mode_vectors * dfs[None, :, None]).sum(dim=1)
    return coords + displacement


def collectivity(eigenvectors: torch.Tensor) -> torch.Tensor:
    """Compute collectivity for each ANM/GNM mode.

    Collectivity measures how many residues participate in a mode.
    κ_k = (1/N) * exp(-Σ_i  u²_ki * ln(u²_ki))

    where u²_ki = ||v_k_i||² / Σ_j ||v_k_j||² is the normalized
    squared displacement of residue i in mode k.

    Values range from 1/N (localized) to 1.0 (maximally collective).

    References:
        Bruschweiler (1995) J Chem Phys 102:3396-3403

    Args:
        eigenvectors: [N, n_modes, 3] per-residue displacement vectors.

    Returns:
        kappa: [n_modes] collectivity values in (0, 1].
    """
    N, n_modes, _ = eigenvectors.shape

    # ||v_k_i||^2 for each residue and mode: [N, n_modes]
    sq_norms = (eigenvectors ** 2).sum(dim=-1)

    # Normalize per mode: u²_ki = sq_norm_ik / sum_j(sq_norm_jk)
    u2 = sq_norms / (sq_norms.sum(dim=0, keepdim=True) + 1e-30)  # [N, n_modes]

    # Shannon entropy: -Σ_i u²_ki * ln(u²_ki)
    # Clamp to avoid log(0)
    u2_safe = u2.clamp(min=1e-30)
    entropy = -(u2_safe * u2_safe.log()).sum(dim=0)  # [n_modes]

    # Collectivity: κ_k = (1/N) * exp(entropy)
    kappa = (1.0 / N) * entropy.exp()

    return kappa


def combo_collectivity(
    eigenvectors: torch.Tensor,
    mode_indices: tuple[int, ...],
) -> float:
    """Compute combined collectivity for a set of modes.

    For a multi-mode combination, sums the displacement vectors
    (unit-weighted) and computes the collectivity of the summed field.

    Args:
        eigenvectors: [N, n_modes, 3] per-residue displacement vectors.
        mode_indices: Indices of modes in the combination.

    Returns:
        Combined collectivity score (higher = more collective).
    """
    N = eigenvectors.shape[0]

    # Sum selected mode vectors: [N, 3]
    selected = eigenvectors[:, list(mode_indices), :]  # [N, k, 3]
    combined = selected.sum(dim=1)  # [N, 3]

    # Squared displacement per residue: [N]
    sq_norms = (combined ** 2).sum(dim=-1)

    # Normalize
    u2 = sq_norms / (sq_norms.sum() + 1e-30)

    # Shannon entropy
    u2_safe = u2.clamp(min=1e-30)
    entropy = -(u2_safe * u2_safe.log()).sum()

    return ((1.0 / N) * entropy.exp()).item()


def batch_combo_collectivity(
    eigenvectors: torch.Tensor,
    combo_indices_list: list[tuple[int, ...]],
) -> torch.Tensor:
    """Vectorized collectivity for many mode combos at once.

    Instead of looping Python-side over each combo, this builds a
    [n_combos, N, 3] tensor of summed displacement fields via a
    sparse mask and computes all collectivities in one shot.

    Args:
        eigenvectors:       [N, n_modes, 3] per-residue displacement vectors.
        combo_indices_list: List of tuples, each containing mode indices.

    Returns:
        kappa: [n_combos] collectivity scores.
    """
    N, n_modes, _ = eigenvectors.shape
    n_combos = len(combo_indices_list)
    device = eigenvectors.device
    dtype = eigenvectors.dtype

    # Build mask: [n_combos, n_modes] — 1.0 where mode is in combo
    mask = torch.zeros(n_combos, n_modes, device=device, dtype=dtype)
    for i, indices in enumerate(combo_indices_list):
        for j in indices:
            mask[i, j] = 1.0

    # mask:         [n_combos, n_modes]
    # eigenvectors: [N, n_modes, 3]
    # We want: combined[c, i, d] = sum_m mask[c, m] * eigvec[i, m, d]
    # einsum: cm, imd -> cid
    combined = torch.einsum("cm, imd -> cid", mask, eigenvectors)
    # result: [n_combos, N, 3]

    # Squared displacement per residue: [n_combos, N]
    sq_norms = (combined ** 2).sum(dim=-1)

    # Normalize per combo: [n_combos, N]
    u2 = sq_norms / (sq_norms.sum(dim=-1, keepdim=True) + 1e-30)

    # Shannon entropy: [n_combos]
    u2_safe = u2.clamp(min=1e-30)
    entropy = -(u2_safe * u2_safe.log()).sum(dim=-1)

    # Collectivity
    kappa = (1.0 / N) * entropy.exp()

    return kappa


def anm_bfactors(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
) -> torch.Tensor:
    """Compute per-residue B-factors from ANM modes.

    B_i = sum_k(||v_k_i||^2 / lambda_k)

    Args:
        eigenvalues:  [n_modes] non-trivial eigenvalues.
        eigenvectors: [N, n_modes, 3] per-residue displacement vectors.

    Returns:
        b_factors: [N] per-residue B-factors (unnormalized).
    """
    # ||v_k_i||^2 for each residue and mode: [N, n_modes]
    sq_norms = (eigenvectors ** 2).sum(dim=-1)

    # 1 / lambda_k: [n_modes]
    inv_vals = 1.0 / (eigenvalues + 1e-10)

    # B_i = sum_k(sq_norm_ik / lambda_k): [N]
    b_factors = sq_norms @ inv_vals

    return b_factors
