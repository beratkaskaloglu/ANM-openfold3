"""Tests for mode_combinator module."""

import torch
import pytest

from src.anm import anm_modes, build_hessian
from src.mode_combinator import (
    ModeCombo,
    collectivity_combinations,
    grid_combinations,
    random_combinations,
    targeted_combinations,
)


def _anm_fixtures(n: int = 15, n_modes: int = 8):
    """Build ANM modes from random coordinates for combinator tests."""
    torch.manual_seed(42)
    coords = torch.randn(n, 3) * 10.0
    H = build_hessian(coords)
    eigenvalues, eigenvectors = anm_modes(H, n_modes=n_modes)
    return coords, eigenvalues, eigenvectors


class TestModeCombo:
    def test_fields(self):
        mc = ModeCombo(
            mode_indices=(0, 2),
            dfs=(0.5, -0.3),
            label="test",
            collectivity_score=0.8,
        )
        assert mc.mode_indices == (0, 2)
        assert mc.dfs == (0.5, -0.3)
        assert mc.label == "test"
        assert mc.collectivity_score == 0.8


class TestCollectivityCombinations:
    def test_returns_list(self):
        _, eigenvalues, eigenvectors = _anm_fixtures()
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(eigenvectors, n_modes)
        assert isinstance(combos, list)
        assert all(isinstance(c, ModeCombo) for c in combos)

    def test_sorted_by_collectivity(self):
        _, eigenvalues, eigenvectors = _anm_fixtures()
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(eigenvectors, n_modes)
        for i in range(len(combos) - 1):
            assert combos[i].collectivity_score >= combos[i + 1].collectivity_score

    def test_max_combos_respected(self):
        _, eigenvalues, eigenvectors = _anm_fixtures()
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(eigenvectors, n_modes, max_combos=5)
        assert len(combos) <= 5

    def test_single_and_multi(self):
        _, eigenvalues, eigenvectors = _anm_fixtures()
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combo_size=3, max_combos=100,
        )
        sizes = {len(c.mode_indices) for c in combos}
        # Should have both single-mode and multi-mode combos
        assert 1 in sizes
        assert len(sizes) > 1


class TestGridCombinations:
    def test_returns_list(self):
        combos = grid_combinations(n_modes_available=5)
        assert isinstance(combos, list)
        assert all(isinstance(c, ModeCombo) for c in combos)

    def test_max_combos_respected(self):
        combos = grid_combinations(n_modes_available=5, max_combos=10)
        assert len(combos) <= 10


class TestRandomCombinations:
    def test_returns_list(self):
        combos = random_combinations(n_modes_available=8, n_combos=10, seed=0)
        assert isinstance(combos, list)
        assert all(isinstance(c, ModeCombo) for c in combos)

    def test_correct_count(self):
        combos = random_combinations(n_modes_available=8, n_combos=25, seed=0)
        assert len(combos) == 25

    def test_reproducible_with_seed(self):
        c1 = random_combinations(n_modes_available=8, n_combos=10, seed=123)
        c2 = random_combinations(n_modes_available=8, n_combos=10, seed=123)
        for a, b in zip(c1, c2):
            assert a.mode_indices == b.mode_indices
            assert a.dfs == b.dfs

    def test_mode_indices_valid(self):
        n_modes = 8
        combos = random_combinations(n_modes_available=n_modes, n_combos=20, seed=0)
        for c in combos:
            for idx in c.mode_indices:
                assert 0 <= idx < n_modes


class TestTargetedCombinations:
    def test_returns_list(self):
        coords, _, eigenvectors = _anm_fixtures()
        target = coords + torch.randn_like(coords) * 0.5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=10,
            seed=0,
        )
        assert isinstance(combos, list)
        assert all(isinstance(c, ModeCombo) for c in combos)

    def test_uses_projection(self):
        """Optimal combo should use the top-projecting modes onto displacement."""
        coords, _, eigenvectors = _anm_fixtures()
        N, n_modes, _ = eigenvectors.shape

        # Create a target displaced along a specific direction
        target = coords + torch.randn_like(coords) * 2.0

        top_modes = 5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=5,
            top_modes=top_modes,
            seed=0,
        )

        # Compute projections manually to verify
        delta = (target - coords).reshape(-1)
        modes_flat = eigenvectors.reshape(N * 3, n_modes)
        projections = modes_flat.T @ delta
        expected_top = projections.abs().argsort(descending=True)[:top_modes]
        expected_set = set(expected_top.tolist())

        # The optimal combo (first) should use exactly the top-projecting modes
        assert set(combos[0].mode_indices) == expected_set
