"""Training loop for GNM-Contact Learner."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .ground_truth import compute_gt_probability_matrix
from .losses import total_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for training."""

    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    n_modes: int = 20
    r_cut: float = 10.0
    tau: float = 1.5
    use_wandb: bool = False
    wandb_project: str = "gnm-contact-learner"
    device: str = "cpu"


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.norm() * y_c.norm()).clamp(min=1e-8)
    return (num / den).item()


def _adjacency_accuracy(
    c_pred: torch.Tensor, c_gt: torch.Tensor, threshold: float = 0.5
) -> float:
    """Fraction of entries where binarised pred matches binarised gt."""
    pred_bin = (c_pred > threshold).float()
    gt_bin = (c_gt > threshold).float()
    return (pred_bin == gt_bin).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Run one training epoch. Returns average metrics."""
    model.train()
    total_n = 0
    running: Dict[str, float] = {}

    for batch in loader:
        c_gt = batch["c_gt"].to(cfg.device)

        z_sym = None
        z_recon = None

        if "pair_repr" in batch:
            pair_repr = batch["pair_repr"].to(cfg.device)
            c_pred = model.contact_head(pair_repr)
            # Reconstruction path
            z_sym = 0.5 * (pair_repr + pair_repr.transpose(1, 2))
            h = model.contact_head.w_enc(z_sym)
            z_recon = model.contact_head.w_dec(h)
        else:
            out = model(batch)
            c_pred = out["C_pred"]
            z_sym = 0.5 * (out["pair_repr"] + out["pair_repr"].transpose(1, 2))
            z_recon = out["pair_repr_recon"]

        # Unbatched losses (per-sample in batch dim)
        loss, details = total_loss(
            c_pred.squeeze(0),
            c_gt.squeeze(0),
            z_original=z_sym,
            z_reconstructed=z_recon,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            n_modes=cfg.n_modes,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.contact_head.parameters(), max_norm=cfg.max_grad_norm
        )
        optimizer.step()

        # Accumulate
        total_n += 1
        for k, v in details.items():
            running[k] = running.get(k, 0.0) + v
        running["loss"] = running.get("loss", 0.0) + loss.item()

    return {k: v / max(total_n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Run validation. Returns average metrics + adj accuracy + B-factor r."""
    model.eval()
    total_n = 0
    running: Dict[str, float] = {}

    from .kirchhoff import gnm_decompose, soft_kirchhoff

    for batch in loader:
        c_gt = batch["c_gt"].to(cfg.device)

        if "pair_repr" in batch:
            pair_repr = batch["pair_repr"].to(cfg.device)
            c_pred = model.contact_head(pair_repr)
        else:
            out = model(batch)
            c_pred = out["C_pred"]

        c_p = c_pred.squeeze(0)
        c_g = c_gt.squeeze(0)

        loss, details = total_loss(
            c_p, c_g, alpha=cfg.alpha, beta=cfg.beta, n_modes=cfg.n_modes
        )

        adj_acc = _adjacency_accuracy(c_p, c_g)

        # B-factor Pearson correlation
        gamma_p = soft_kirchhoff(c_p)
        gamma_g = soft_kirchhoff(c_g)
        _, _, bf_p = gnm_decompose(gamma_p, cfg.n_modes)
        _, _, bf_g = gnm_decompose(gamma_g, cfg.n_modes)
        bf_corr = _pearson_corr(bf_p, bf_g)

        total_n += 1
        for k, v in details.items():
            running[k] = running.get(k, 0.0) + v
        running["loss"] = running.get("loss", 0.0) + loss.item()
        running["adj_acc"] = running.get("adj_acc", 0.0) + adj_acc
        running["bf_pearson"] = running.get("bf_pearson", 0.0) + bf_corr

    return {k: v / max(total_n, 1) for k, v in running.items()}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[TrainConfig] = None,
) -> None:
    """Full training loop with optional WandB logging."""
    if cfg is None:
        cfg = TrainConfig()

    wandb_run: Any = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, config=vars(cfg))

    optimizer = torch.optim.AdamW(
        model.contact_head.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, cfg)
        scheduler.step()

        log_msg = f"Epoch {epoch+1}/{cfg.epochs}"
        for k, v in train_metrics.items():
            log_msg += f"  {k}={v:.4f}"

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, cfg)
            log_msg += "  |  val:"
            for k, v in val_metrics.items():
                log_msg += f"  {k}={v:.4f}"

        logger.info(log_msg)

        if wandb_run is not None:
            wandb_run.log(
                {
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch + 1,
                }
            )

    if wandb_run is not None:
        wandb_run.finish()
