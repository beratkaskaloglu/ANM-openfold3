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

    # Zero out diagonal of c, then negate for off-diagonal Kirchhoff
    mask = ~torch.eye(n, dtype=torch.bool, device=c.device)
    c_offdiag = c * mask.float()          # zero diagonal, keep off-diag
    gamma = -c_offdiag                    # Γ_ij = -C_ij for i≠j

    # Diagonal ← coordination number: Γ_ii = Σ_j C_ij
    diag_vals = c_offdiag.sum(dim=-1)     # [N]
    gamma = gamma + torch.diag(diag_vals)

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
    # Run eigh on CPU to avoid CUSOLVER_STATUS_INTERNAL_ERROR on some GPUs
    # .float().cpu() ensures a clean contiguous copy for CUDA-sourced tensors
    device = gamma.device
    gamma_cpu = gamma.to(dtype=torch.float64, device='cpu')
    vals, vecs = torch.linalg.eigh(gamma_cpu)
    vals = vals.to(dtype=gamma.dtype, device=device)
    vecs = vecs.to(dtype=gamma.dtype, device=device)

    # Clamp n_modes to available non-trivial modes
    max_modes = vals.shape[-1] - 1
    k = min(n_modes, max_modes)

    eig_vals = vals[1 : k + 1]        # [k]
    eig_vecs = vecs[:, 1 : k + 1]     # [N, k]

    # B-factors:  B_i = Σ_k  V_ik² / λ_k
    inv_vals = 1.0 / (eig_vals + 1e-10)
    b_factors = (eig_vecs ** 2) @ inv_vals  # [N]

    return eig_vals, eig_vecs, b_factors
