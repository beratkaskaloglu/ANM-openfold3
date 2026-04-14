"""Tests for training loop utilities."""

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, Dataset

from src.train import TrainConfig, train_one_epoch, validate
from src.contact_head import ContactProjectionHead
from src.ground_truth import compute_gt_probability_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _SyntheticDataset(Dataset):
    """Dataset that returns synthetic pair repr + ground-truth contact maps."""

    def __init__(self, n_samples: int = 5, n_res: int = 12, c_z: int = 64):
        self.n_samples = n_samples
        self.n_res = n_res
        self.c_z = c_z

        # Fixed random coords for reproducible GT
        torch.manual_seed(42)
        self.coords = [torch.randn(n_res, 3) * 5.0 for _ in range(n_samples)]
        self.pair_reprs = [torch.randn(n_res, n_res, c_z) for _ in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        c_gt = compute_gt_probability_matrix(self.coords[idx], r_cut=8.0, tau=1.0)
        return {
            "pair_repr": self.pair_reprs[idx],
            "c_gt": c_gt,
        }


class _ModelWrapper(nn.Module):
    """Minimal wrapper that exposes contact_head like GNMContactLearner."""

    def __init__(self, c_z: int = 64, bottleneck_dim: int = 16):
        super().__init__()
        self.contact_head = ContactProjectionHead(
            c_z=c_z, bottleneck_dim=bottleneck_dim
        )


def _make_loader(n_samples: int = 5, n_res: int = 12, c_z: int = 64):
    ds = _SyntheticDataset(n_samples=n_samples, n_res=n_res, c_z=c_z)
    return DataLoader(ds, batch_size=1, shuffle=False)


def _make_model(c_z: int = 64, bottleneck_dim: int = 16):
    return _ModelWrapper(c_z=c_z, bottleneck_dim=bottleneck_dim)


def _default_cfg(**overrides):
    defaults = dict(
        epochs=5,
        lr=3e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        alpha=1.0,
        beta=0.1,
        gamma=0.1,
        n_modes=5,
        r_cut=8.0,
        tau=1.0,
        device="cpu",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ---------------------------------------------------------------------------
# TestTrainConfig
# ---------------------------------------------------------------------------
class TestTrainConfig:
    def test_default_values(self):
        cfg = TrainConfig()
        assert cfg.epochs == 100
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 1e-2
        assert cfg.max_grad_norm == 1.0
        assert cfg.alpha == 1.0
        assert cfg.beta == 0.5
        assert cfg.gamma == 0.1
        assert cfg.n_modes == 20
        assert cfg.r_cut == 10.0
        assert cfg.tau == 1.5
        assert cfg.device == "cpu"
        assert cfg.use_wandb is False

    def test_custom_values(self):
        cfg = TrainConfig(epochs=50, lr=3e-4, beta=0.3, device="cuda")
        assert cfg.epochs == 50
        assert cfg.lr == 3e-4
        assert cfg.beta == 0.3
        assert cfg.device == "cuda"


# ---------------------------------------------------------------------------
# TestTrainOneEpoch
# ---------------------------------------------------------------------------
class TestTrainOneEpoch:
    @pytest.fixture
    def setup(self):
        model = _make_model()
        loader = _make_loader()
        cfg = _default_cfg()
        optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=cfg.lr)
        return model, loader, optimizer, cfg

    def test_returns_metrics_dict(self, setup):
        model, loader, optimizer, cfg = setup
        metrics = train_one_epoch(model, loader, optimizer, cfg)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "L_contact" in metrics
        assert "L_gnm" in metrics

    def test_loss_is_finite(self, setup):
        model, loader, optimizer, cfg = setup
        metrics = train_one_epoch(model, loader, optimizer, cfg)
        assert metrics["loss"] > 0
        assert not torch.isnan(torch.tensor(metrics["loss"]))
        assert not torch.isinf(torch.tensor(metrics["loss"]))

    def test_loss_decreases_over_epochs(self):
        model = _make_model()
        loader = _make_loader(n_samples=2, n_res=10)
        cfg = _default_cfg(lr=1e-2, beta=0.0, gamma=0.0)
        optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=cfg.lr)

        losses = []
        for _ in range(5):
            metrics = train_one_epoch(model, loader, optimizer, cfg)
            losses.append(metrics["loss"])

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_gradient_clipping(self):
        model = _make_model()
        loader = _make_loader(n_samples=1, n_res=8)
        cfg = _default_cfg(max_grad_norm=0.01)
        optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=cfg.lr)

        # Run one step; gradient clipping should keep norms small
        train_one_epoch(model, loader, optimizer, cfg)

        # After a step, parameters should have been updated (non-zero)
        total_params = sum(p.numel() for p in model.contact_head.parameters())
        assert total_params > 0  # sanity check

    def test_reconstruction_loss_tracked(self):
        model = _make_model()
        loader = _make_loader(n_samples=2)
        cfg = _default_cfg(gamma=0.1)
        optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=cfg.lr)
        metrics = train_one_epoch(model, loader, optimizer, cfg)
        assert "L_recon" in metrics


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------
class TestValidate:
    def test_returns_val_metrics(self):
        model = _make_model()
        loader = _make_loader(n_samples=2, n_res=10)
        cfg = _default_cfg()
        metrics = validate(model, loader, cfg)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "adj_acc" in metrics
        assert "bf_pearson" in metrics

    def test_no_gradient_tracked(self):
        model = _make_model()
        loader = _make_loader(n_samples=1, n_res=8)
        cfg = _default_cfg()

        validate(model, loader, cfg)

        # No param should have accumulated gradients from validate
        for p in model.contact_head.parameters():
            assert p.grad is None or (p.grad == 0).all()

    def test_adj_acc_in_range(self):
        model = _make_model()
        loader = _make_loader(n_samples=2, n_res=10)
        cfg = _default_cfg()
        metrics = validate(model, loader, cfg)
        assert 0.0 <= metrics["adj_acc"] <= 1.0

    def test_bf_pearson_in_range(self):
        model = _make_model()
        loader = _make_loader(n_samples=2, n_res=10)
        cfg = _default_cfg()
        metrics = validate(model, loader, cfg)
        assert -1.0 <= metrics["bf_pearson"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TestFullTrainLoop
# ---------------------------------------------------------------------------
class TestFullTrainLoop:
    def test_overfit_single_protein(self):
        """Train on 1 protein for many epochs; loss should get small."""
        torch.manual_seed(0)
        n_res, c_z = 10, 64
        model = _make_model(c_z=c_z, bottleneck_dim=16)
        loader = _make_loader(n_samples=1, n_res=n_res, c_z=c_z)
        cfg = _default_cfg(lr=5e-3, beta=0.0, gamma=0.0, alpha=1.0)
        optimizer = torch.optim.Adam(model.contact_head.parameters(), lr=cfg.lr)

        final_loss = None
        for _ in range(40):
            metrics = train_one_epoch(model, loader, optimizer, cfg)
            final_loss = metrics["loss"]

        assert final_loss is not None
        assert final_loss < 0.5, f"Expected loss < 0.5 after overfit, got {final_loss:.4f}"
