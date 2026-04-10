"""Tests for ground_truth module."""

import torch
import pytest

from src.ground_truth import compute_gt_probability_matrix


def _make_line_coords(n: int = 10, spacing: float = 3.8) -> torch.Tensor:
    """Residues in a straight line along x-axis, 3.8 Å apart."""
    coords = torch.zeros(n, 3)
    coords[:, 0] = torch.arange(n, dtype=torch.float32) * spacing
    return coords


class TestComputeGtProbabilityMatrix:
    def test_output_shape(self):
        coords = _make_line_coords(20)
        c = compute_gt_probability_matrix(coords)
        assert c.shape == (20, 20)

    def test_diagonal_is_zero(self):
        coords = _make_line_coords(15)
        c = compute_gt_probability_matrix(coords)
        assert torch.allclose(c.diag(), torch.zeros(15))

    def test_symmetry(self):
        coords = torch.randn(25, 3)
        c = compute_gt_probability_matrix(coords)
        assert torch.allclose(c, c.T, atol=1e-6)

    def test_values_in_zero_one(self):
        coords = torch.randn(30, 3) * 10.0
        c = compute_gt_probability_matrix(coords)
        assert (c >= 0.0).all()
        assert (c <= 1.0).all()

    def test_close_residues_high_prob(self):
        """Residues closer than r_cut should have p > 0.5."""
        coords = _make_line_coords(5, spacing=3.0)
        c = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)
        # i=0, j=1 → d=3.0 Å  ≪ 10 Å → should be high
        assert c[0, 1].item() > 0.5

    def test_far_residues_low_prob(self):
        """Residues much farther than r_cut should have p < 0.5."""
        coords = _make_line_coords(10, spacing=5.0)
        c = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)
        # i=0, j=9 → d=45.0 Å  ≫ 10 Å
        assert c[0, 9].item() < 0.01

    def test_sigmoid_midpoint(self):
        """At d == r_cut, sigmoid should return ≈ 0.5."""
        # Two atoms exactly r_cut apart
        coords = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        c = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)
        assert abs(c[0, 1].item() - 0.5) < 0.01
