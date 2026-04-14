"""Tests for ANM module: Hessian, modes, collectivity, B-factors, displacement."""

import torch
import pytest

from src.anm import (
    anm_bfactors,
    anm_modes,
    build_hessian,
    collectivity,
    combo_collectivity,
    displace,
)


def _random_ca_coords(n: int = 20) -> torch.Tensor:
    """Spread-out synthetic CA coordinates for ANM."""
    torch.manual_seed(42)
    return torch.randn(n, 3) * 10.0


class TestBuildHessian:
    def test_shape(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        assert H.shape == (60, 60)

    def test_symmetric(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        assert torch.allclose(H, H.T, atol=1e-6)

    def test_row_sum_zero(self):
        coords = _random_ca_coords(15)
        H = build_hessian(coords)
        row_sums = H.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5)

    def test_positive_semidefinite(self):
        coords = _random_ca_coords(15)
        H = build_hessian(coords)
        eigenvalues = torch.linalg.eigvalsh(H.to(torch.float64))
        assert (eigenvalues >= -1e-6).all()

    def test_cutoff_effect(self):
        coords = _random_ca_coords(20)
        H_small = build_hessian(coords, cutoff=8.0)
        H_large = build_hessian(coords, cutoff=25.0)
        # Larger cutoff → more non-zero elements (stronger off-diagonal)
        nnz_small = (H_small.abs() > 1e-3).sum()
        nnz_large = (H_large.abs() > 1e-3).sum()
        assert nnz_large >= nnz_small

    def test_gamma_scaling(self):
        coords = _random_ca_coords(15)
        H1 = build_hessian(coords, gamma=1.0)
        H2 = build_hessian(coords, gamma=2.0)
        # Off-diagonal blocks should scale by gamma
        mask = ~torch.eye(H1.shape[0], dtype=torch.bool)
        ratio = H2[mask] / (H1[mask] + 1e-30)
        # Where H1 is non-negligible, ratio should be ~2
        significant = H1[mask].abs() > 1e-4
        assert torch.allclose(ratio[significant], torch.full_like(ratio[significant], 2.0), atol=1e-4)


class TestAnmModes:
    def test_eigenvalue_count(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, _ = anm_modes(H, n_modes=10)
        assert vals.shape == (10,)

    def test_eigenvector_shape(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=10)
        assert vecs.shape == (20, 10, 3)

    def test_trivial_modes_skipped(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, _ = anm_modes(H, n_modes=10)
        # All returned eigenvalues should be non-trivial (> 1e-4)
        assert (vals > 1e-4).all()

    def test_eigenvalues_positive(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, _ = anm_modes(H, n_modes=10)
        assert (vals > 0).all()

    def test_eigenvalues_ascending(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, _ = anm_modes(H, n_modes=10)
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6

    def test_n_modes_clamped(self):
        """If n_modes > available, should silently clamp."""
        coords = _random_ca_coords(5)
        H = build_hessian(coords)
        vals, vecs = anm_modes(H, n_modes=100)
        # 5 residues → 15 DOF → 15 - 6 = 9 non-trivial modes
        assert vals.shape[0] == 9
        assert vecs.shape == (5, 9, 3)


class TestCollectivity:
    def test_output_shape(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=10)
        kappa = collectivity(vecs)
        assert kappa.shape == (10,)

    def test_range_zero_one(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=10)
        kappa = collectivity(vecs)
        assert (kappa >= 0).all()
        assert (kappa <= 1.0 + 1e-6).all()

    def test_uniform_mode_high(self):
        """A mode where all residues move equally should have high collectivity."""
        N, n_modes = 20, 1
        # Uniform displacement in x direction for all residues
        vecs = torch.zeros(N, n_modes, 3)
        vecs[:, 0, 0] = 1.0 / (N ** 0.5)  # normalized uniform
        kappa = collectivity(vecs)
        # Should be close to 1.0 (maximally collective)
        assert kappa[0].item() > 0.9


class TestComboCollectivity:
    def test_returns_scalar(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=10)
        score = combo_collectivity(vecs, (0,))
        assert isinstance(score, float)

    def test_multi_mode(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=10)
        score = combo_collectivity(vecs, (0, 1, 2))
        assert isinstance(score, float)
        assert 0 <= score <= 1.0 + 1e-6


class TestAnmBfactors:
    def test_output_shape(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, vecs = anm_modes(H, n_modes=10)
        bf = anm_bfactors(vals, vecs)
        assert bf.shape == (20,)

    def test_all_positive(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        vals, vecs = anm_modes(H, n_modes=10)
        bf = anm_bfactors(vals, vecs)
        assert (bf > 0).all()


class TestDisplace:
    def test_output_shape(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=5)
        dfs = torch.ones(5)
        new_coords = displace(coords, vecs, dfs)
        assert new_coords.shape == (20, 3)

    def test_zero_df_no_change(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=5)
        dfs = torch.zeros(5)
        new_coords = displace(coords, vecs, dfs)
        assert torch.allclose(new_coords, coords, atol=1e-7)

    def test_displacement_proportional(self):
        coords = _random_ca_coords(20)
        H = build_hessian(coords)
        _, vecs = anm_modes(H, n_modes=5)
        dfs1 = torch.ones(5) * 0.5
        dfs2 = torch.ones(5) * 1.0
        disp1 = displace(coords, vecs, dfs1) - coords
        disp2 = displace(coords, vecs, dfs2) - coords
        assert torch.allclose(disp2, disp1 * 2.0, atol=1e-4)
