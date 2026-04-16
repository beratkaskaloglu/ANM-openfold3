"""Comprehensive tests for all 4 mode_combinator strategies + batch collectivity.

Uses a synthetic alpha-helix (~50 residues) for realistic ANM eigenmodes.
"""

from __future__ import annotations

import math

import torch
import pytest

from src.anm import anm_modes, build_hessian, combo_collectivity, batch_combo_collectivity
from src.mode_combinator import (
    ModeCombo,
    collectivity_combinations,
    grid_combinations,
    random_combinations,
    targeted_combinations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_helix_coords(n_residues: int = 50) -> torch.Tensor:
    """Generate ideal alpha-helix CA coordinates.

    Alpha-helix parameters: 1.5 A rise/residue, 2.3 A radius, 100 deg turn.
    """
    rise_per_res = 1.5  # Angstrom
    radius = 2.3  # Angstrom
    turn_angle = math.radians(100)  # ~100 degrees per residue

    coords = []
    for i in range(n_residues):
        x = radius * math.cos(i * turn_angle)
        y = radius * math.sin(i * turn_angle)
        z = i * rise_per_res
        coords.append([x, y, z])
    return torch.tensor(coords, dtype=torch.float32)


@pytest.fixture(scope="module")
def helix_anm():
    """Module-scoped ANM decomposition of a 50-residue helix."""
    coords = _build_helix_coords(50)
    H = build_hessian(coords, cutoff=15.0)
    eigenvalues, eigenvectors = anm_modes(H, n_modes=20)
    return coords, eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def _assert_valid_combos(
    combos: list[ModeCombo],
    n_modes: int,
    max_combos: int | None = None,
):
    """Shared invariants for any strategy's output."""
    assert isinstance(combos, list), "Expected list output"
    assert len(combos) > 0, "Expected non-empty combo list"
    for c in combos:
        assert isinstance(c, ModeCombo)
        # len(mode_indices) == len(dfs)
        assert len(c.mode_indices) == len(c.dfs), (
            f"mode_indices length {len(c.mode_indices)} != dfs length {len(c.dfs)}"
        )
        # mode indices in valid range
        for idx in c.mode_indices:
            assert 0 <= idx < n_modes, f"mode index {idx} out of range [0, {n_modes})"
        # label non-empty
        assert c.label, "Label must be non-empty"
    if max_combos is not None:
        assert len(combos) <= max_combos, (
            f"Got {len(combos)} combos, expected <= {max_combos}"
        )


# ===========================================================================
# 1. collectivity_combinations
# ===========================================================================

class TestCollectivityCombinations:
    """Tests for collectivity-ranked mode combos."""

    def test_basic_validity(self, helix_anm):
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combos=20, eigenvalues=eigenvalues,
        )
        _assert_valid_combos(combos, n_modes, max_combos=20)

    def test_sorted_collectivity_descending(self, helix_anm):
        """Pairs (+df, -df) share the same score; scores should be non-increasing."""
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combos=30, eigenvalues=eigenvalues,
        )
        scores = [c.collectivity_score for c in combos]
        for i in range(0, len(scores) - 2, 2):
            # Within a pair, scores are equal
            assert scores[i] == pytest.approx(scores[i + 1]), (
                f"Pair scores differ at {i}: {scores[i]} vs {scores[i+1]}"
            )
            # Across pairs, non-increasing
            if i + 2 < len(scores):
                assert scores[i] >= scores[i + 2] - 1e-6, (
                    f"Not descending at pair {i//2}: {scores[i]} < {scores[i+2]}"
                )

    def test_pos_neg_pairs(self, helix_anm):
        """Every +df combo should be followed by its -df counterpart."""
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combos=20, eigenvalues=eigenvalues,
        )
        assert len(combos) % 2 == 0, "Expected even number of combos (+df/-df pairs)"
        for i in range(0, len(combos), 2):
            pos = combos[i]
            neg = combos[i + 1]
            assert pos.mode_indices == neg.mode_indices
            assert "pos" in pos.label
            assert "neg" in neg.label
            for p, n in zip(pos.dfs, neg.dfs):
                assert p == pytest.approx(-n, abs=1e-7), (
                    f"Expected negated dfs: {p} vs {n}"
                )

    def test_max_combos_respected(self, helix_anm):
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combos=6, eigenvalues=eigenvalues,
        )
        _assert_valid_combos(combos, n_modes, max_combos=6)

    def test_without_eigenvalues(self, helix_anm):
        """Should work without eigenvalues (uniform weighting)."""
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combos=10, eigenvalues=None,
        )
        _assert_valid_combos(combos, n_modes, max_combos=10)

    def test_max_combo_size(self, helix_anm):
        """max_combo_size=1 should only produce single-mode combos."""
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = collectivity_combinations(
            eigenvectors, n_modes, max_combo_size=1, max_combos=10,
            eigenvalues=eigenvalues,
        )
        for c in combos:
            assert len(c.mode_indices) == 1


# ===========================================================================
# 2. grid_combinations
# ===========================================================================

class TestGridCombinations:
    """Tests for cartesian-product grid strategy."""

    def test_basic_validity(self):
        combos = grid_combinations(n_modes_available=6, select_modes=2, max_combos=50)
        _assert_valid_combos(combos, n_modes=6, max_combos=50)

    def test_df_range_respected(self):
        df_range = (-1.0, 1.0)
        combos = grid_combinations(
            n_modes_available=4, select_modes=2,
            df_range=df_range, df_steps=5, max_combos=500,
        )
        for c in combos:
            for d in c.dfs:
                assert df_range[0] - 1e-6 <= d <= df_range[1] + 1e-6, (
                    f"df {d} outside range {df_range}"
                )

    def test_cartesian_product_count(self):
        """Without max_combos cap, total = C(n,k) * df_steps^k."""
        n_modes = 4
        select_modes = 2
        df_steps = 3
        expected_subsets = math.comb(n_modes, select_modes)  # C(4,2) = 6
        expected_total = expected_subsets * (df_steps ** select_modes)  # 6 * 9 = 54
        combos = grid_combinations(
            n_modes_available=n_modes, select_modes=select_modes,
            df_steps=df_steps, max_combos=10000,
        )
        assert len(combos) == expected_total, (
            f"Expected {expected_total} combos, got {len(combos)}"
        )

    def test_max_combos_truncates(self):
        combos = grid_combinations(
            n_modes_available=10, select_modes=2,
            df_steps=5, max_combos=15,
        )
        assert len(combos) == 15

    def test_select_modes_clamped(self):
        """When n_modes_available < select_modes, should clamp."""
        combos = grid_combinations(
            n_modes_available=2, select_modes=5,
            df_steps=3, max_combos=100,
        )
        for c in combos:
            assert len(c.mode_indices) == 2

    def test_labels_present(self):
        combos = grid_combinations(n_modes_available=4, max_combos=10)
        for c in combos:
            assert c.label.startswith("grid_")


# ===========================================================================
# 3. random_combinations
# ===========================================================================

class TestRandomCombinations:
    """Tests for stochastic mode sampling."""

    def test_basic_validity(self, helix_anm):
        _, eigenvalues, _ = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = random_combinations(
            n_modes_available=n_modes, n_combos=30, seed=42,
            eigenvalues=eigenvalues,
        )
        _assert_valid_combos(combos, n_modes)
        assert len(combos) == 30

    def test_exact_count(self):
        combos = random_combinations(n_modes_available=10, n_combos=17, seed=0)
        assert len(combos) == 17

    def test_reproducibility(self):
        c1 = random_combinations(n_modes_available=10, n_combos=20, seed=99)
        c2 = random_combinations(n_modes_available=10, n_combos=20, seed=99)
        for a, b in zip(c1, c2):
            assert a.mode_indices == b.mode_indices
            assert a.dfs == b.dfs

    def test_mode_indices_in_range(self):
        n_modes = 12
        combos = random_combinations(n_modes_available=n_modes, n_combos=50, seed=7)
        for c in combos:
            for idx in c.mode_indices:
                assert 0 <= idx < n_modes

    def test_mode_count_in_range(self):
        lo, hi = 2, 4
        combos = random_combinations(
            n_modes_available=10, n_combos=100,
            select_modes_range=(lo, hi), seed=1,
        )
        for c in combos:
            assert lo <= len(c.mode_indices) <= hi

    def test_eigenvalue_weighting_bias(self, helix_anm):
        """With eigenvalue weighting, lower-freq modes should appear more often."""
        _, eigenvalues, _ = helix_anm
        n_modes = eigenvalues.shape[0]
        combos = random_combinations(
            n_modes_available=n_modes, n_combos=500,
            select_modes_range=(1, 1), seed=0,
            eigenvalues=eigenvalues,
        )
        counts = [0] * n_modes
        for c in combos:
            for idx in c.mode_indices:
                counts[idx] += 1
        # Mode 0 (lowest eigenvalue) should be sampled more than the last mode
        assert counts[0] > counts[-1], (
            f"Expected mode 0 ({counts[0]}) sampled more than mode {n_modes-1} "
            f"({counts[-1]}) with eigenvalue weighting"
        )

    def test_without_eigenvalues(self):
        combos = random_combinations(
            n_modes_available=8, n_combos=20, seed=0, eigenvalues=None,
        )
        _assert_valid_combos(combos, n_modes=8)

    def test_labels(self):
        combos = random_combinations(n_modes_available=5, n_combos=5, seed=0)
        for i, c in enumerate(combos):
            assert c.label == f"rand_{i:03d}"


# ===========================================================================
# 4. targeted_combinations
# ===========================================================================

class TestTargetedCombinations:
    """Tests for projection-based targeting strategy."""

    def test_basic_validity(self, helix_anm):
        coords, _, eigenvectors = helix_anm
        n_modes = eigenvectors.shape[1]
        target = coords + torch.randn_like(coords) * 0.5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=15, seed=42,
        )
        _assert_valid_combos(combos, n_modes)
        assert len(combos) == 15

    def test_optimal_combo_first(self, helix_anm):
        """First combo should be the exact projection onto top modes."""
        coords, _, eigenvectors = helix_anm
        N, n_modes, _ = eigenvectors.shape
        torch.manual_seed(123)
        target = coords + torch.randn_like(coords) * 2.0

        top_modes = 5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=10, top_modes=top_modes, seed=0,
        )

        # Manually compute expected optimal
        delta = (target - coords).reshape(-1)
        modes_flat = eigenvectors.reshape(N * 3, n_modes)
        projections = modes_flat.T @ delta
        expected_top_idx = projections.abs().argsort(descending=True)[:top_modes]
        expected_sorted = expected_top_idx.sort().values

        assert combos[0].label == "targeted_optimal"
        assert set(combos[0].mode_indices) == set(expected_sorted.tolist())
        # dfs should match exact projections
        for i, m in enumerate(combos[0].mode_indices):
            assert combos[0].dfs[i] == pytest.approx(
                projections[m].item(), abs=1e-4
            )

    def test_perturbed_combos_are_subsets(self, helix_anm):
        """Non-optimal combos should use subsets of the top modes."""
        coords, _, eigenvectors = helix_anm
        N, n_modes, _ = eigenvectors.shape
        target = coords + torch.randn_like(coords) * 1.0

        top_modes = 5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=10, top_modes=top_modes, seed=0,
        )
        optimal_modes = set(combos[0].mode_indices)
        for c in combos[1:]:
            assert set(c.mode_indices).issubset(optimal_modes), (
                f"Combo modes {c.mode_indices} not subset of optimal {optimal_modes}"
            )

    def test_top_modes_clamped(self, helix_anm):
        """top_modes > n_modes should be clamped."""
        coords, _, eigenvectors = helix_anm
        n_modes = eigenvectors.shape[1]
        target = coords + torch.randn_like(coords) * 0.5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=5, top_modes=999, seed=0,
        )
        for c in combos:
            assert len(c.mode_indices) <= n_modes

    def test_labels(self, helix_anm):
        coords, _, eigenvectors = helix_anm
        target = coords + torch.randn_like(coords) * 0.5
        combos = targeted_combinations(
            current_coords=coords,
            target_coords=target,
            mode_vectors=eigenvectors,
            n_combos=5, seed=0,
        )
        assert combos[0].label == "targeted_optimal"
        for i, c in enumerate(combos[1:], 1):
            assert c.label == f"targeted_{i:03d}"


# ===========================================================================
# 5. batch_combo_collectivity vs combo_collectivity
# ===========================================================================

class TestBatchVsSingleCollectivity:
    """batch_combo_collectivity should agree with combo_collectivity."""

    def test_agreement_without_eigenvalues(self, helix_anm):
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]

        subsets = [
            (0,),
            (1,),
            (0, 1),
            (0, 2, 4),
            (1, 3, 5, 7),
        ]
        # Filter subsets to valid range
        subsets = [s for s in subsets if all(i < n_modes for i in s)]

        batch_scores = batch_combo_collectivity(eigenvectors, subsets, eigenvalues=None)
        for i, s in enumerate(subsets):
            single = combo_collectivity(eigenvectors, s, eigenvalues=None)
            assert batch_scores[i].item() == pytest.approx(single, abs=1e-5), (
                f"Mismatch for subset {s}: batch={batch_scores[i].item()}, "
                f"single={single}"
            )

    def test_agreement_with_eigenvalues(self, helix_anm):
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]

        subsets = [
            (0,),
            (2,),
            (0, 3),
            (1, 2, 4),
            (0, 1, 3, 6),
        ]
        subsets = [s for s in subsets if all(i < n_modes for i in s)]

        batch_scores = batch_combo_collectivity(
            eigenvectors, subsets, eigenvalues=eigenvalues,
        )
        for i, s in enumerate(subsets):
            single = combo_collectivity(eigenvectors, s, eigenvalues=eigenvalues)
            assert batch_scores[i].item() == pytest.approx(single, abs=1e-5), (
                f"Mismatch for subset {s}: batch={batch_scores[i].item()}, "
                f"single={single}"
            )

    def test_single_mode_subsets(self, helix_anm):
        """For single-mode subsets, both should return identical values."""
        _, eigenvalues, eigenvectors = helix_anm
        n_modes = eigenvalues.shape[0]

        subsets = [(i,) for i in range(min(n_modes, 10))]
        batch_scores = batch_combo_collectivity(
            eigenvectors, subsets, eigenvalues=eigenvalues,
        )
        for i, s in enumerate(subsets):
            single = combo_collectivity(eigenvectors, s, eigenvalues=eigenvalues)
            assert batch_scores[i].item() == pytest.approx(single, abs=1e-6)

    def test_collectivity_range(self, helix_anm):
        """Collectivity should be in (0, 1]."""
        _, eigenvalues, eigenvectors = helix_anm
        N = eigenvectors.shape[0]
        n_modes = eigenvalues.shape[0]

        subsets = [(i,) for i in range(min(n_modes, 10))]
        scores = batch_combo_collectivity(eigenvectors, subsets, eigenvalues=eigenvalues)
        for s_val in scores:
            v = s_val.item()
            assert 0 < v <= 1.0 + 1e-6, f"Collectivity {v} outside (0, 1]"
