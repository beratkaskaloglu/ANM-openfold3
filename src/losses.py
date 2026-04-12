"""Loss functions for GNM-Contact Learner."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .kirchhoff import gnm_decompose, soft_kirchhoff


def focal_loss(
    c_pred: torch.Tensor,
    c_gt: torch.Tensor,
    seq_sep_min: int = 6,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> torch.Tensor:
    """Focal loss with sequence-separation filter.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples and focuses on hard boundary cases.
    Better than BCE for imbalanced contact maps (~10% positive).

    Args:
        c_pred: [B, N, N] or [N, N] predicted probabilities.
        c_gt:   same shape, ground-truth soft contacts.
        seq_sep_min: minimum sequence separation.
        focal_gamma: focusing parameter (higher = more focus on hard examples).
        focal_alpha: balance weight for positive class.

    Returns:
        Scalar loss.
    """
    n = c_pred.shape[-1]
    idx = torch.arange(n, device=c_pred.device)
    sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= seq_sep_min

    pred_masked = c_pred[..., sep_mask].clamp(1e-7, 1.0 - 1e-7)
    gt_masked = c_gt[..., sep_mask]

    # Per-element BCE (no reduction)
    bce = F.binary_cross_entropy(pred_masked, gt_masked, reduction="none")

    # p_t = p if y=1, else 1-p
    p_t = pred_masked * gt_masked + (1.0 - pred_masked) * (1.0 - gt_masked)

    # Alpha weighting: alpha for positives, (1-alpha) for negatives
    alpha_t = focal_alpha * gt_masked + (1.0 - focal_alpha) * (1.0 - gt_masked)

    # Focal modulation
    focal_weight = alpha_t * (1.0 - p_t) ** focal_gamma

    loss = (focal_weight * bce).mean()
    return loss


def contact_loss(
    c_pred: torch.Tensor,
    c_gt: torch.Tensor,
    seq_sep_min: int = 6,
) -> torch.Tensor:
    """Weighted BCE loss with sequence-separation filter.

    Only residue pairs with |i − j| ≥ seq_sep_min contribute.

    Args:
        c_pred: [B, N, N] or [N, N] predicted probabilities.
        c_gt:   same shape, ground-truth soft contacts.
        seq_sep_min: minimum sequence separation.

    Returns:
        Scalar loss.
    """
    n = c_pred.shape[-1]
    idx = torch.arange(n, device=c_pred.device)
    sep_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= seq_sep_min

    pred_masked = c_pred[..., sep_mask]
    gt_masked = c_gt[..., sep_mask]

    # Clamp to avoid log(0)
    pred_clamped = pred_masked.clamp(1e-7, 1.0 - 1e-7)

    loss = F.binary_cross_entropy(pred_clamped, gt_masked, reduction="mean")
    return loss


def gnm_loss(
    c_pred: torch.Tensor,
    c_gt: torch.Tensor,
    n_modes: int = 20,
    w_eigenvalue: float = 1.0,
    w_bfactor: float = 1.0,
    w_eigvec: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Physics-informed GNM loss comparing Kirchhoff spectra.

    Both inputs go through: soft_kirchhoff → eigh → compare.

    Args:
        c_pred: [N, N] predicted contact matrix.
        c_gt:   [N, N] ground-truth contact matrix.
        n_modes: non-trivial modes to compare.
        w_eigenvalue, w_bfactor, w_eigvec: component weights.

    Returns:
        (scalar loss, dict of component values)
    """
    # Run entire GNM pipeline on CPU to avoid CUDA assert errors
    device = c_pred.device
    c_pred_cpu = c_pred.detach().cpu()
    c_gt_cpu = c_gt.detach().cpu()

    gamma_pred = soft_kirchhoff(c_pred_cpu)
    gamma_gt = soft_kirchhoff(c_gt_cpu)

    vals_p, vecs_p, bf_p = gnm_decompose(gamma_pred, n_modes)
    vals_g, vecs_g, bf_g = gnm_decompose(gamma_gt, n_modes)

    # --- L_eigenvalue: MSE on normalised inverse eigenvalues ---
    inv_p = 1.0 / (vals_p + 1e-10)
    inv_g = 1.0 / (vals_g + 1e-10)
    inv_p_norm = inv_p / (inv_p.sum() + 1e-10)
    inv_g_norm = inv_g / (inv_g.sum() + 1e-10)
    l_eig = F.mse_loss(inv_p_norm, inv_g_norm)

    # --- L_bfactor: MSE on normalised B-factor profiles ---
    bf_p_norm = bf_p / (bf_p.max() + 1e-10)
    bf_g_norm = bf_g / (bf_g.max() + 1e-10)
    l_bf = F.mse_loss(bf_p_norm, bf_g_norm)

    # --- L_eigvec: 1 − |cos(v_pred, v_gt)| (phase-invariant) ---
    cos_sim = torch.abs(
        F.cosine_similarity(vecs_p.T, vecs_g.T, dim=-1)
    )  # [n_modes]
    l_vec = (1.0 - cos_sim).mean()

    loss = w_eigenvalue * l_eig + w_bfactor * l_bf + w_eigvec * l_vec
    loss = loss.to(device)

    details = {
        "L_eigenvalue": l_eig.item(),
        "L_bfactor": l_bf.item(),
        "L_eigvec": l_vec.item(),
    }
    return loss, details


def reconstruction_loss(
    z_original: torch.Tensor,
    z_reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Autoencoder reconstruction loss (stabilises inverse path).

    Args:
        z_original:      [B, N, N, c_z] symmetrised pair repr.
        z_reconstructed: [B, N, N, c_z] encode → decode output.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(z_reconstructed, z_original)


def total_loss(
    c_pred: torch.Tensor,
    c_gt: torch.Tensor,
    z_original: torch.Tensor | None = None,
    z_reconstructed: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    n_modes: int = 20,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined loss: L = α·L_contact + β·L_gnm + γ·L_recon.

    Args:
        c_pred: [N, N] predicted contact matrix.
        c_gt:   [N, N] ground-truth contact matrix.
        z_original: [B, N, N, c_z] symmetrised pair repr (optional).
        z_reconstructed: [B, N, N, c_z] reconstruction (optional).
        alpha:  contact-loss weight.
        beta:   GNM-loss weight.
        gamma:  reconstruction-loss weight.
        n_modes: modes for GNM loss.
        use_focal: use focal loss instead of BCE.
        focal_gamma: focal loss focusing parameter.
        focal_alpha: focal loss balance weight.

    Returns:
        (scalar loss, dict of all component values)
    """
    if use_focal:
        l_contact = focal_loss(
            c_pred, c_gt, focal_gamma=focal_gamma, focal_alpha=focal_alpha
        )
    else:
        l_contact = contact_loss(c_pred, c_gt)
    l_gnm, gnm_details = gnm_loss(c_pred, c_gt, n_modes=n_modes)

    loss = alpha * l_contact + beta * l_gnm

    details: Dict[str, float] = {
        "L_contact": l_contact.item(),
        "L_gnm": l_gnm.item(),
        **gnm_details,
    }

    if z_original is not None and z_reconstructed is not None:
        l_recon = reconstruction_loss(z_original, z_reconstructed)
        loss = loss + gamma * l_recon
        details["L_recon"] = l_recon.item()

    return loss, details
