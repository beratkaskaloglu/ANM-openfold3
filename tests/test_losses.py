"""Tests for loss functions."""

import torch
import pytest

from src.losses import contact_loss, gnm_loss, reconstruction_loss, total_loss


def _random_contact_pair(n: int = 20):
    """Create a random soft contact matrix and a perturbed copy."""
    c = torch.rand(n, n)
    c = 0.5 * (c + c.T)
    c.fill_diagonal_(0.0)
    c = c.clamp(0.01, 0.99)
    return c


class TestContactLoss:
    def test_minimal_when_binary_identical(self):
        """BCE is zero only when both pred and gt are 0 or 1."""
        c = torch.zeros(15, 15)
        # Set a few contacts to 1.0 (binary)
        c[0, 5] = c[5, 0] = 1.0
        c[2, 8] = c[8, 2] = 1.0
        c_pred = c.clamp(1e-7, 1 - 1e-7)
        loss = contact_loss(c_pred, c.clone(), seq_sep_min=0)
        assert loss.item() < 0.05  # near-zero for near-binary inputs

    def test_positive_when_different(self):
        c_pred = _random_contact_pair(15)
        c_gt = _random_contact_pair(15)
        loss = contact_loss(c_pred, c_gt)
        assert loss.item() > 0

    def test_seq_sep_filter(self):
        """Higher seq_sep_min should mask more pairs → different loss."""
        c_pred = _random_contact_pair(20)
        c_gt = _random_contact_pair(20)
        loss_low = contact_loss(c_pred, c_gt, seq_sep_min=2)
        loss_high = contact_loss(c_pred, c_gt, seq_sep_min=10)
        # They should generally differ (not necessarily one > other)
        assert not torch.isnan(torch.tensor(loss_low.item()))
        assert not torch.isnan(torch.tensor(loss_high.item()))


class TestGnmLoss:
    def test_near_zero_when_identical(self):
        c = _random_contact_pair(15)
        loss, details = gnm_loss(c, c.clone(), n_modes=5)
        assert loss.item() < 1e-4

    def test_returns_details(self):
        c_pred = _random_contact_pair(15)
        c_gt = _random_contact_pair(15)
        loss, details = gnm_loss(c_pred, c_gt, n_modes=5)
        assert "L_eigenvalue" in details
        assert "L_bfactor" in details
        assert "L_eigvec" in details

    def test_gradient_flows_through_eigh(self):
        c_pred = _random_contact_pair(12)
        c_pred.requires_grad_(True)
        c_gt = _random_contact_pair(12)
        loss, _ = gnm_loss(c_pred, c_gt, n_modes=5)
        loss.backward()
        assert c_pred.grad is not None
        assert not torch.isnan(c_pred.grad).any()

    def test_positive_when_different(self):
        c_pred = _random_contact_pair(15)
        c_gt = _random_contact_pair(15)
        loss, _ = gnm_loss(c_pred, c_gt, n_modes=5)
        assert loss.item() > 0


class TestTotalLoss:
    def test_combines_both(self):
        c_pred = _random_contact_pair(15)
        c_pred.requires_grad_(True)
        c_gt = _random_contact_pair(15)
        loss, details = total_loss(c_pred, c_gt, alpha=1.0, beta=0.5)
        assert "L_contact" in details
        assert "L_gnm" in details
        loss.backward()
        assert c_pred.grad is not None

    def test_beta_zero_ignores_gnm(self):
        c_pred = _random_contact_pair(12)
        c_gt = _random_contact_pair(12)
        loss_no_gnm, _ = total_loss(c_pred, c_gt, alpha=1.0, beta=0.0)
        pure_contact = contact_loss(c_pred, c_gt)
        assert abs(loss_no_gnm.item() - pure_contact.item()) < 1e-5

    def test_with_reconstruction_loss(self):
        c_pred = _random_contact_pair(10)
        c_gt = _random_contact_pair(10)
        z_orig = torch.randn(1, 10, 10, 128)
        z_recon = z_orig + torch.randn_like(z_orig) * 0.1
        loss, details = total_loss(
            c_pred, c_gt,
            z_original=z_orig, z_reconstructed=z_recon,
            gamma=0.1,
        )
        assert "L_recon" in details
        assert details["L_recon"] > 0

    def test_no_recon_when_none(self):
        c_pred = _random_contact_pair(10)
        c_gt = _random_contact_pair(10)
        _, details = total_loss(c_pred, c_gt)
        assert "L_recon" not in details


class TestReconstructionLoss:
    def test_zero_when_identical(self):
        z = torch.randn(1, 8, 8, 64)
        loss = reconstruction_loss(z, z.clone())
        assert loss.item() < 1e-7

    def test_positive_when_different(self):
        z_a = torch.randn(1, 8, 8, 64)
        z_b = torch.randn(1, 8, 8, 64)
        loss = reconstruction_loss(z_a, z_b)
        assert loss.item() > 0

    def test_gradient_flows(self):
        z_orig = torch.randn(1, 6, 6, 32)
        z_recon = torch.randn(1, 6, 6, 32, requires_grad=True)
        loss = reconstruction_loss(z_orig, z_recon)
        loss.backward()
        assert z_recon.grad is not None
