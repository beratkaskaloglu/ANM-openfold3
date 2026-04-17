"""Tests for ContactProjectionHead (invertible bottleneck)."""

import torch
import pytest

from src.contact_head import ContactProjectionHead


class TestContactProjectionHead:
    def test_output_shape(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        z = torch.randn(2, 10, 10, 128)
        out = head(z)
        assert out.shape == (2, 10, 10)

    def test_output_in_zero_one(self):
        head = ContactProjectionHead(c_z=64, bottleneck_dim=16)
        z = torch.randn(1, 15, 15, 64) * 5.0
        out = head(z)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_symmetry(self):
        head = ContactProjectionHead(c_z=128)
        z = torch.randn(1, 12, 12, 128)
        out = head(z)
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)

    def test_diagonal_is_zero(self):
        head = ContactProjectionHead(c_z=128)
        z = torch.randn(1, 8, 8, 128)
        out = head(z)
        diag = out[0].diag()
        assert torch.allclose(diag, torch.zeros(8), atol=1e-7)

    def test_gradient_flows(self):
        head = ContactProjectionHead(c_z=64, bottleneck_dim=16)
        z = torch.randn(1, 6, 6, 64, requires_grad=True)
        out = head(z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()

    def test_different_bottleneck_dims(self):
        for k in [8, 16, 32, 64]:
            head = ContactProjectionHead(c_z=128, bottleneck_dim=k)
            z = torch.randn(1, 5, 5, 128)
            out = head(z)
            assert out.shape == (1, 5, 5)

    def test_param_count(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        total = sum(p.numel() for p in head.parameters())
        # W_enc: 128*32=4096, v: 32, W_dec: 32*128=4096,
        # w_inv: Linear(1,32)=64 + Linear(32,128)=4224 → 4288
        # Total: 4096 + 32 + 4096 + 4288 = 12512
        assert total == 12512


class TestInversePath:
    def test_inverse_output_shape(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        c = torch.rand(10, 10) * 0.8 + 0.1  # avoid 0/1
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        pseudo_z = head.inverse(c)
        assert pseudo_z.shape == (10, 10, 128)

    def test_inverse_symmetry(self):
        """Symmetric C should produce symmetric pseudo_z."""
        head = ContactProjectionHead(c_z=64, bottleneck_dim=16)
        c = torch.rand(8, 8) * 0.8 + 0.1
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        pseudo_z = head.inverse(c)
        assert torch.allclose(
            pseudo_z, pseudo_z.transpose(0, 1), atol=1e-5
        )

    def test_inverse_differentiable(self):
        """Inverse path is differentiable through w_inv params."""
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        c = torch.rand(6, 6) * 0.8 + 0.1
        c = 0.5 * (c + c.T)
        pseudo_z = head.inverse(c)
        # w_inv makes output differentiable w.r.t. model params
        assert pseudo_z.requires_grad

    def test_inverse_no_grad_context(self):
        """Caller can suppress gradients with torch.no_grad()."""
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        c = torch.rand(6, 6) * 0.8 + 0.1
        c = 0.5 * (c + c.T)
        with torch.no_grad():
            pseudo_z = head.inverse(c)
        assert not pseudo_z.requires_grad


class TestEncodeBottleneck:
    def test_bottleneck_shape(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        z = torch.randn(1, 10, 10, 128)
        h = head.encode_bottleneck(z)
        assert h.shape == (1, 10, 10, 32)
