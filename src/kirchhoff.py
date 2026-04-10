"""Differentiable Kirchhoff matrix and GNM eigendecomposition."""

from typing import Tuple

import torch


def soft_kirchhoff(
    c: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a Kirchhoff (Laplacian) matrix from a soft contact matrix.

    Γ_ij = -C_ij  (i ≠ j)
    Γ_ii = Σ_j C_ij
    Then add εI for numerical stability during eigendecomposition.

    Args:
        c: [N, N] contact probabilities in [0, 1].
        eps: Regularisation added to diagonal.

    Returns:
        gamma: [N, N] real-symmetric positive-semi-definite matrix.
    """
    n = c.shape[-1]
    gamma = -c.clone()
    gamma.fill_diagonal_(0.0)

    # Diagonal ← coordination number
    diag_vals = c.sum(dim=-1)  # [N]
    gamma.diagonal().copy_(diag_vals)

    # Regularise to avoid degenerate eigenvalues → NaN gradients
    gamma = gamma + eps * torch.eye(n, device=c.device, dtype=c.dtype)

    return gamma


def gnm_decompose(
    gamma: torch.Tensor,
    n_modes: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable GNM eigendecomposition.

    Args:
        gamma: [N, N] Kirchhoff matrix (real symmetric).
        n_modes: Number of non-trivial modes to keep (skipping the
                 zero mode at index 0).

    Returns:
        eigenvalues:  [n_modes]    ascending non-trivial eigenvalues.
        eigenvectors: [N, n_modes] corresponding eigenvectors.
        b_factors:    [N]          B_i = Σ_k (V_ik² / λ_k).
    """
    vals, vecs = torch.linalg.eigh(gamma)  # ascending order

    # Clamp n_modes to available non-trivial modes
    max_modes = vals.shape[-1] - 1
    k = min(n_modes, max_modes)

    eig_vals = vals[1 : k + 1]        # [k]
    eig_vecs = vecs[:, 1 : k + 1]     # [N, k]

    # B-factors:  B_i = Σ_k  V_ik² / λ_k
    inv_vals = 1.0 / (eig_vals + 1e-10)
    b_factors = (eig_vecs ** 2) @ inv_vals  # [N]

    return eig_vals, eig_vecs, b_factors
