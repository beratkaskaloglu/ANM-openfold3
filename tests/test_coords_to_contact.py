"""Tests for coords_to_contact module."""

import torch
import pytest

from src.coords_to_contact import coords_to_contact


def _line_coords(n: int, spacing: float = 3.8) -> torch.Tensor:
    """Create a line of residues spaced evenly along x-axis."""
    coords = torch.zeros(n, 3)
    coords[:, 0] = torch.arange(n, dtype=torch.float32) * spacing
    return coords


class TestCoordsToContact:
    def test_output_shape(self):
        coords = torch.randn(15, 3) * 10.0
        C = coords_to_contact(coords)
        assert C.shape == (15, 15)

    def test_symmetric(self):
        coords = torch.randn(15, 3) * 10.0
        C = coords_to_contact(coords)
        assert torch.allclose(C, C.T, atol=1e-6)

    def test_diagonal_zero(self):
        coords = torch.randn(15, 3) * 10.0
        C = coords_to_contact(coords)
        assert torch.allclose(C.diag(), torch.zeros(15), atol=1e-7)

    def test_values_in_range(self):
        coords = torch.randn(20, 3) * 10.0
        C = coords_to_contact(coords)
        assert (C >= 0.0).all()
        assert (C <= 1.0).all()

    def test_close_residues_high(self):
        """Residues closer than r_cut should have contact > 0.5."""
        r_cut = 10.0
        # Two residues 5 Angstrom apart (well within cutoff)
        coords = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        C = coords_to_contact(coords, r_cut=r_cut, tau=1.5)
        assert C[0, 1].item() > 0.5

    def test_far_residues_low(self):
        """Residues much farther than r_cut should have contact near 0."""
        r_cut = 10.0
        # Two residues 50 Angstrom apart (well beyond cutoff)
        coords = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        C = coords_to_contact(coords, r_cut=r_cut, tau=1.5)
        assert C[0, 1].item() < 0.01

    def test_midpoint_at_rcut(self):
        """At distance = r_cut, sigmoid should give ~0.5."""
        r_cut = 10.0
        coords = torch.tensor([[0.0, 0.0, 0.0], [r_cut, 0.0, 0.0]])
        C = coords_to_contact(coords, r_cut=r_cut, tau=1.5)
        assert abs(C[0, 1].item() - 0.5) < 0.05

    def test_tau_sharpness(self):
        """Smaller tau → sharper transition around r_cut."""
        r_cut = 10.0
        # Just inside cutoff
        coords_near = torch.tensor([[0.0, 0.0, 0.0], [r_cut - 2.0, 0.0, 0.0]])
        # Just outside cutoff
        coords_far = torch.tensor([[0.0, 0.0, 0.0], [r_cut + 2.0, 0.0, 0.0]])

        # Sharp tau
        C_near_sharp = coords_to_contact(coords_near, r_cut=r_cut, tau=0.5)
        C_far_sharp = coords_to_contact(coords_far, r_cut=r_cut, tau=0.5)

        # Soft tau
        C_near_soft = coords_to_contact(coords_near, r_cut=r_cut, tau=3.0)
        C_far_soft = coords_to_contact(coords_far, r_cut=r_cut, tau=3.0)

        # Sharp transition: bigger difference between near and far
        diff_sharp = C_near_sharp[0, 1] - C_far_sharp[0, 1]
        diff_soft = C_near_soft[0, 1] - C_far_soft[0, 1]
        assert diff_sharp > diff_soft
