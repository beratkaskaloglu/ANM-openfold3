"""Tests for kirchhoff module."""

import torch
import pytest

from src.kirchhoff import gnm_decompose, soft_kirchhoff


def _random_contact_matrix(n: int = 20) -> torch.Tensor:
    """Symmetric [0,1] matrix with zero diagonal."""
    c = torch.rand(n, n)
    c = 0.5 * (c + c.T)
    c.fill_diagonal_(0.0)
    return c


class TestSoftKirchhoff:
    def test_shape(self):
        c = _random_contact_matrix(15)
        gamma = soft_kirchhoff(c)
        assert gamma.shape == (15, 15)

    def test_symmetry(self):
        c = _random_contact_matrix(20)
        gamma = soft_kirchhoff(c)
        assert torch.allclose(gamma, gamma.T, atol=1e-6)

    def test_row_sums_near_eps(self):
        """Without eps, row sums of Kirchhoff should be 0.
        With eps, row sums ≈ eps * N (off-diagonal contribution)."""
        c = _random_contact_matrix(10)
        gamma = soft_kirchhoff(c, eps=0.0)
        row_sums = gamma.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(10), atol=1e-5)

    def test_positive_semidefinite(self):
        c = _random_contact_matrix(15)
        gamma = soft_kirchhoff(c, eps=1e-6)
        eigenvalues = torch.linalg.eigvalsh(gamma)
        assert (eigenvalues >= -1e-5).all()


class TestGnmDecompose:
    def test_output_shapes(self):
        c = _random_contact_matrix(30)
        gamma = soft_kirchhoff(c)
        vals, vecs, bf = gnm_decompose(gamma, n_modes=10)
        assert vals.shape == (10,)
        assert vecs.shape == (30, 10)
        assert bf.shape == (30,)

    def test_eigenvalues_positive(self):
        c = _random_contact_matrix(25)
        gamma = soft_kirchhoff(c)
        vals, _, _ = gnm_decompose(gamma, n_modes=15)
        assert (vals > 0).all()

    def test_bfactors_positive(self):
        c = _random_contact_matrix(20)
        gamma = soft_kirchhoff(c)
        _, _, bf = gnm_decompose(gamma, n_modes=10)
        assert (bf > 0).all()

    def test_n_modes_clamped(self):
        """If n_modes > N-1, should silently clamp."""
        c = _random_contact_matrix(5)
        gamma = soft_kirchhoff(c)
        vals, vecs, bf = gnm_decompose(gamma, n_modes=100)
        assert vals.shape[0] == 4  # N-1 = 4

    def test_gradient_flows(self):
        """Gradient must flow through eigh back to C."""
        c = _random_contact_matrix(10)
        c.requires_grad_(True)
        gamma = soft_kirchhoff(c)
        vals, vecs, bf = gnm_decompose(gamma, n_modes=5)
        loss = bf.sum()
        loss.backward()
        assert c.grad is not None
        assert not torch.isnan(c.grad).any()
