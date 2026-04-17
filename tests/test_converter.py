"""Tests for PairContactConverter."""

import pytest
import torch

from src.converter import PairContactConverter


@pytest.fixture
def converter():
    """Create a converter with random weights (no checkpoint)."""
    return PairContactConverter(checkpoint=None, device="cpu")


class TestZToContact:
    def test_3d_input(self, converter):
        z = torch.randn(10, 10, 128)
        c = converter.z_to_contact(z)
        assert c.shape == (10, 10)
        assert c.min() >= 0.0
        assert c.max() <= 1.0

    def test_4d_input_batch(self, converter):
        z = torch.randn(2, 10, 10, 128)
        c = converter.z_to_contact(z)
        assert c.shape == (2, 10, 10)

    def test_symmetric_output(self, converter):
        z = torch.randn(10, 10, 128)
        # Make z symmetric
        z = 0.5 * (z + z.transpose(0, 1))
        c = converter.z_to_contact(z)
        assert torch.allclose(c, c.T, atol=1e-5)


class TestContactToZ:
    def test_shape(self, converter):
        c = torch.rand(10, 10)
        c = 0.5 * (c + c.T)  # symmetric
        z = converter.contact_to_z(c)
        assert z.shape == (10, 10, 128)

    def test_output_is_finite(self, converter):
        c = torch.rand(10, 10).clamp(0.01, 0.99)
        c = 0.5 * (c + c.T)
        z = converter.contact_to_z(c)
        assert torch.isfinite(z).all()


class TestAnalyze:
    def test_from_contact(self, converter):
        c = torch.rand(10, 10)
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        result = converter.analyze(c, n_modes=5)
        assert "contact" in result
        assert "kirchhoff" in result
        assert "eigenvalues" in result
        assert "eigenvectors" in result
        assert "b_factors" in result
        assert result["eigenvalues"].shape[0] == 5

    def test_from_pair_repr(self, converter):
        z = torch.randn(10, 10, 128)
        result = converter.analyze(z, n_modes=5, is_contact=False)
        assert result["eigenvalues"].shape[0] == 5
        assert result["b_factors"].shape[0] == 10


class TestRoundtrip:
    def test_roundtrip_returns_finite_mse(self, converter):
        z = torch.randn(10, 10, 128)
        contact, z_recon, mse = converter.roundtrip(z)
        assert contact.shape == (10, 10)
        assert z_recon.shape == (10, 10, 128)
        assert mse >= 0.0
        assert mse == mse  # not NaN

    def test_roundtrip_4d(self, converter):
        z = torch.randn(1, 10, 10, 128)
        contact, z_recon, mse = converter.roundtrip(z)
        assert mse >= 0.0


class TestInitFromDict:
    def test_from_dict_checkpoint(self):
        """Test creating converter from a state dict."""
        from src.contact_head import ContactProjectionHead
        head = ContactProjectionHead(c_z=128, bottleneck_dim=64)
        ckpt = {
            "model_state_dict": head.state_dict(),
            "c_z": 128,
            "bottleneck_dim": 64,
        }
        conv = PairContactConverter(checkpoint=ckpt, device="cpu")
        z = torch.randn(10, 10, 128)
        c = conv.z_to_contact(z)
        assert c.shape == (10, 10)
