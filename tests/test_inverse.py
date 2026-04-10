"""Tests for inverse path: coords → pseudo pair representation."""

import torch
import pytest

from src.contact_head import ContactProjectionHead
from src.inverse import PairReprFromCoords


def _make_line_coords(n: int = 10, spacing: float = 3.8) -> torch.Tensor:
    coords = torch.zeros(n, 3)
    coords[:, 0] = torch.arange(n, dtype=torch.float32) * spacing
    return coords


class TestPairReprFromCoords:
    def test_output_shape(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        inv = PairReprFromCoords(head)
        coords = _make_line_coords(15)
        pseudo_z = inv(coords)
        assert pseudo_z.shape == (15, 15, 128)

    def test_different_coords_different_output(self):
        head = ContactProjectionHead(c_z=64, bottleneck_dim=16)
        inv = PairReprFromCoords(head)
        coords_a = _make_line_coords(10, spacing=3.8)
        coords_b = _make_line_coords(10, spacing=7.0)
        z_a = inv(coords_a)
        z_b = inv(coords_b)
        assert not torch.allclose(z_a, z_b, atol=1e-3)

    def test_varying_distance_gives_varying_repr(self):
        """Close and far pairs should produce different representations."""
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        inv = PairReprFromCoords(head)
        coords = _make_line_coords(20, spacing=3.8)
        pseudo_z = inv(coords)

        # Close pair (i=0, j=1, d=3.8 Å) vs far pair (i=0, j=19, d=72.2 Å)
        # Norms differ because logit magnitudes differ
        norm_close = pseudo_z[0, 1].norm().item()
        norm_far = pseudo_z[0, 19].norm().item()
        assert abs(norm_close - norm_far) > 1e-3

    def test_symmetry(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        inv = PairReprFromCoords(head)
        coords = torch.randn(12, 3) * 10.0
        pseudo_z = inv(coords)
        assert torch.allclose(
            pseudo_z, pseudo_z.transpose(0, 1), atol=1e-5
        )

    def test_no_grad_output(self):
        head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
        inv = PairReprFromCoords(head)
        coords = _make_line_coords(8)
        pseudo_z = inv(coords)
        assert not pseudo_z.requires_grad
