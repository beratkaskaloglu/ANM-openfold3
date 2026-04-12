#!/usr/bin/env python3
"""Phase 4: Large-scale training with focal loss, OneCycleLR, grad accumulation.

Usage:
    python scripts/train_large.py
    python scripts/train_large.py --epochs 500 --lr 3e-4 --bottleneck-dim 64
    python scripts/train_large.py --resume checkpoints/latest.pt
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contact_head import ContactProjectionHead
from src.data import ShardedPairReprDataset
from src.kirchhoff import gnm_decompose, soft_kirchhoff
from src.losses import total_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_large.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainLargeConfig:
    """Hyperparameters for large-scale training."""

    # Architecture
    c_z: int = 128
    bottleneck_dim: int = 64

    # Training
    epochs: int = 500
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    grad_accum_steps: int = 4

    # Loss weights
    alpha: float = 1.0       # L_contact (focal)
    beta: float = 0.3        # L_gnm
    gamma: float = 0.05      # L_recon

    # Focal loss
    use_focal: bool = True
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # GNM
    n_modes: int = 20
    r_cut: float = 8.0
    tau: float = 1.0

    # Data
    shard_dir: str = "data/shards"
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Scheduler
    warmup_fraction: float = 0.05

    # Early stopping
    patience: int = 50
    min_delta: float = 1e-5

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_every: int = 50
    log_every: int = 5

    # Device
    device: str = "cuda"

    # WandB
    use_wandb: bool = False
    wandb_project: str = "gnm-contact-learner-v2"


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = (xc.norm() * yc.norm()).clamp(min=1e-8)
    return (num / den).item()


def adjacency_accuracy(
    c_pred: torch.Tensor, c_gt: torch.Tensor, threshold: float = 0.5
) -> float:
    """Fraction of entries where binarised pred matches binarised gt."""
    pred_bin = (c_pred > threshold).float()
    gt_bin = (c_gt > threshold).float()
    return (pred_bin == gt_bin).float().mean().item()


def split_dataset(
    dataset: ShardedPairReprDataset,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple:
    """Split dataset indices into train/val/test."""
    n = len(dataset)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def train_one_epoch(
    head: ContactProjectionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cfg: TrainLargeConfig,
) -> Dict[str, float]:
    """Train one epoch with gradient accumulation."""
    head.train()
    running: Dict[str, float] = {}
    total_n = 0

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        z = batch["pair_repr"].to(cfg.device)
        c_gt = batch["c_gt"].to(cfg.device)

        # Forward
        c_pred = head(z)

        # Reconstruction path
        z_sym = 0.5 * (z + z.transpose(-3, -2))
        h = head.w_enc(z_sym)
        z_recon = head.w_dec(h)

        # Loss (squeeze batch dim for per-protein)
        loss, details = total_loss(
            c_pred.squeeze(0),
            c_gt.squeeze(0),
            z_original=z_sym,
            z_reconstructed=z_recon,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            n_modes=cfg.n_modes,
            use_focal=cfg.use_focal,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / cfg.grad_accum_steps
        scaled_loss.backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Accumulate metrics
        total_n += 1
        for k, v in details.items():
            running[k] = running.get(k, 0.0) + v
        running["loss"] = running.get("loss", 0.0) + loss.item()

    return {k: v / max(total_n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    head: ContactProjectionHead,
    loader: DataLoader,
    cfg: TrainLargeConfig,
) -> Dict[str, float]:
    """Validate and compute metrics."""
    head.eval()
    running: Dict[str, float] = {}
    total_n = 0

    for batch in loader:
        z = batch["pair_repr"].to(cfg.device)
        c_gt = batch["c_gt"].to(cfg.device)

        c_pred = head(z)
        c_p = c_pred.squeeze(0)
        c_g = c_gt.squeeze(0)

        loss, details = total_loss(
            c_p, c_g,
            alpha=cfg.alpha,
            beta=cfg.beta,
            n_modes=cfg.n_modes,
            use_focal=cfg.use_focal,
            focal_gamma=cfg.focal_gamma,
            focal_alpha=cfg.focal_alpha,
        )

        # Additional metrics
        acc = adjacency_accuracy(c_p, c_g)

        gamma_p = soft_kirchhoff(c_p)
        gamma_g = soft_kirchhoff(c_g)
        _, _, bf_p = gnm_decompose(gamma_p, cfg.n_modes)
        _, _, bf_g = gnm_decompose(gamma_g, cfg.n_modes)
        bf_corr = pearson_corr(bf_p, bf_g)

        total_n += 1
        for k, v in details.items():
            running[k] = running.get(k, 0.0) + v
        running["loss"] = running.get("loss", 0.0) + loss.item()
        running["adj_acc"] = running.get("adj_acc", 0.0) + acc
        running["bf_pearson"] = running.get("bf_pearson", 0.0) + bf_corr

    return {k: v / max(total_n, 1) for k, v in running.items()}


def save_checkpoint(
    head: ContactProjectionHead,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_loss: float,
    cfg: TrainLargeConfig,
    path: Path,
) -> None:
    """Save training checkpoint."""
    torch.save(
        {
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "c_z": cfg.c_z,
            "bottleneck_dim": cfg.bottleneck_dim,
            "config": asdict(cfg),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Large-scale GNM-Contact training")

    # Architecture
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--bottleneck-dim", type=int, default=64)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=4)

    # Loss
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--no-focal", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)

    # GNM
    parser.add_argument("--n-modes", type=int, default=20)
    parser.add_argument("--r-cut", type=float, default=8.0)
    parser.add_argument("--tau", type=float, default=1.0)

    # Data
    parser.add_argument("--shard-dir", type=str, default="data/shards")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)

    # Training control
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)

    # Checkpointing
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gnm-contact-learner-v2")

    args = parser.parse_args()

    # Build config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainLargeConfig(
        c_z=args.c_z,
        bottleneck_dim=args.bottleneck_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        use_focal=not args.no_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        n_modes=args.n_modes,
        r_cut=args.r_cut,
        tau=args.tau,
        shard_dir=args.shard_dir,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        patience=args.patience,
        save_every=args.save_every,
        log_every=args.log_every,
        ckpt_dir=args.ckpt_dir,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    # Create dirs
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = ckpt_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2))

    # Load shards
    shard_dir = Path(cfg.shard_dir)
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    if not shard_paths:
        logger.error("No shards found in %s", shard_dir)
        sys.exit(1)

    logger.info("Found %d shards in %s", len(shard_paths), shard_dir)

    # Create dataset
    dataset = ShardedPairReprDataset(
        shard_paths=shard_paths,
        r_cut=cfg.r_cut,
        tau=cfg.tau,
    )
    logger.info("Total proteins: %d", len(dataset))

    # Split
    train_ds, val_ds, test_ds = split_dataset(
        dataset, cfg.val_fraction, cfg.test_fraction
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    head = ContactProjectionHead(c_z=cfg.c_z, bottleneck_dim=cfg.bottleneck_dim)
    head = head.to(cfg.device)
    n_params = sum(p.numel() for p in head.parameters())
    logger.info("ContactProjectionHead: %d parameters", n_params)

    # Optimizer
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # OneCycleLR: warmup + cosine decay
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_fraction,
        anneal_strategy="cos",
        div_factor=10.0,      # initial_lr = max_lr / 10
        final_div_factor=100.0,  # final_lr = max_lr / 1000
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.device, weights_only=False)
        head.load_state_dict(ckpt["model_state_dict"])
        # Don't restore optimizer/scheduler — fresh training run on new data
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(
            "Resumed model weights (prev val_loss=%.4f), training %d fresh epochs",
            best_val_loss, cfg.epochs,
        )

    # WandB
    wandb_run = None
    if cfg.use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            config=asdict(cfg),
            resume="allow" if args.resume else None,
        )

    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting training: %d epochs, lr=%.1e, device=%s", cfg.epochs, cfg.lr, cfg.device)
    logger.info(
        "Loss: %s, alpha=%.2f, beta=%.2f, gamma=%.2f",
        "Focal" if cfg.use_focal else "BCE",
        cfg.alpha,
        cfg.beta,
        cfg.gamma,
    )
    logger.info("r_cut=%.1f, tau=%.1f, n_modes=%d", cfg.r_cut, cfg.tau, cfg.n_modes)
    logger.info("bottleneck_dim=%d, grad_accum=%d", cfg.bottleneck_dim, cfg.grad_accum_steps)
    logger.info("=" * 60 + "\n")

    history: List[Dict[str, Any]] = []
    epochs_no_improve = 0
    epoch = start_epoch  # default if loop doesn't run
    t0 = time.time()

    if start_epoch >= cfg.epochs:
        logger.info("Already trained %d epochs (max=%d), skipping.", start_epoch, cfg.epochs)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_t0 = time.time()

        # Train
        train_metrics = train_one_epoch(head, train_loader, optimizer, scheduler, cfg)

        # Validate
        val_metrics = validate(head, val_loader, cfg)

        epoch_time = time.time() - epoch_t0

        # Record
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_adj_acc": val_metrics.get("adj_acc", 0),
            "val_bf_pearson": val_metrics.get("bf_pearson", 0),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        # Logging
        if (epoch + 1) % cfg.log_every == 0 or epoch == start_epoch:
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "adj_acc=%.3f  bf_r=%.3f  lr=%.1e  (%.1fs)",
                epoch + 1,
                cfg.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics.get("adj_acc", 0),
                val_metrics.get("bf_pearson", 0),
                optimizer.param_groups[0]["lr"],
                epoch_time,
            )

        # WandB
        if wandb_run is not None:
            wandb_run.log(record)

        # NaN guard
        if torch.isnan(torch.tensor(val_metrics["loss"])):
            logger.warning("Epoch %d: val_loss is NaN, skipping checkpoint", epoch + 1)
            continue

        # Best model
        if val_metrics["loss"] < best_val_loss - cfg.min_delta:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            save_checkpoint(
                head, optimizer, scheduler, epoch,
                best_val_loss, cfg,
                ckpt_dir / "best_model.pt",
            )
            logger.info("  -> New best model! val_loss=%.4f", best_val_loss)
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                head, optimizer, scheduler, epoch,
                val_metrics["loss"], cfg,
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
            )

        # Always save latest (for resume)
        save_checkpoint(
            head, optimizer, scheduler, epoch,
            val_metrics["loss"], cfg,
            ckpt_dir / "latest.pt",
        )

        # Early stopping
        if epochs_no_improve >= cfg.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1,
                cfg.patience,
            )
            break

    total_time = time.time() - t0

    # Save history
    history_path = ckpt_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    # Final test evaluation
    logger.info("\n" + "=" * 60)

    best_model_path = ckpt_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info("Final evaluation on test set...")

        best_ckpt = torch.load(
            best_model_path, map_location=cfg.device, weights_only=False
        )
        head.load_state_dict(best_ckpt["model_state_dict"])

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        test_metrics = validate(head, test_loader, cfg)

        logger.info(
            "TEST RESULTS:\n"
            "  loss=%.4f\n"
            "  adj_acc=%.4f\n"
            "  bf_pearson=%.4f\n"
            "  L_contact=%.4f\n"
            "  L_gnm=%.4f",
            test_metrics["loss"],
            test_metrics.get("adj_acc", 0),
            test_metrics.get("bf_pearson", 0),
            test_metrics.get("L_contact", 0),
            test_metrics.get("L_gnm", 0),
        )

        test_results = {
            "test_metrics": test_metrics,
            "best_epoch": best_ckpt["epoch"] + 1,
            "best_val_loss": best_ckpt["val_loss"],
            "total_epochs": epoch + 1,
            "total_time_min": total_time / 60,
            "config": asdict(cfg),
        }
        (ckpt_dir / "test_results.json").write_text(json.dumps(test_results, indent=2))

        logger.info(
            "\n=== DONE ===\n"
            "  Best epoch: %d\n"
            "  Best val_loss: %.4f\n"
            "  Test adj_acc: %.4f\n"
            "  Test bf_pearson: %.4f\n"
            "  Total time: %.1f min\n"
            "  Checkpoints: %s",
            best_ckpt["epoch"] + 1,
            best_ckpt["val_loss"],
            test_metrics.get("adj_acc", 0),
            test_metrics.get("bf_pearson", 0),
            total_time / 60,
            ckpt_dir,
        )
    else:
        logger.warning(
            "best_model.pt not found (val_loss may have been NaN). "
            "Skipping test evaluation. latest.pt still saved."
        )
        logger.info(
            "\n=== DONE (no best model) ===\n"
            "  Total epochs: %d\n"
            "  Total time: %.1f min\n"
            "  Checkpoints: %s",
            epoch + 1,
            total_time / 60,
            ckpt_dir,
        )

    if wandb_run is not None:
        wandb_run.log({"test": test_metrics} if best_model_path.exists() else {})
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training crashed with exception:")
        sys.exit(1)
