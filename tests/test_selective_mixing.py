"""Comprehensive test suite for selective_mixing.py.

Tests cover:
  - compute_change_score: identity, large displacement, symmetry, distance modes
  - compute_alpha_mask: cutoff behaviour, linear / sigmoid / step mappings
  - selective_blend_z: shape preservation, direction, zero-alpha pairs
  - Integration: ModeDrivePipeline with selective_mixing=True populates diagnostics
"""

from __future__ import annotations

import pytest
import torch

from src.selective_mixing import (
    compute_alpha_mask,
    compute_change_score,
    selective_blend_z,
)
from src.mode_drive import ModeDriveConfig, ModeDrivePipeline, StepResult
from src.mode_combinator import ModeCombo


# ── helpers ──────────────────────────────────────────────────────────────────

class MockConverter:
    """Lightweight mock for PairContactConverter — mirrors test_mode_drive_coverage.py."""

    def __init__(self):
        self.head = None

    def contact_to_z(self, C):
        N = C.shape[0]
        return torch.randn(N, N, 128)

    def z_to_contact(self, z):
        N = z.shape[-2]
        c = torch.rand(N, N)
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        return c


def _make_selective_pipeline(n: int = 15, **kw) -> ModeDrivePipeline:
    """Return a pipeline with selective_mixing=True and no diffusion fn."""
    cfg = ModeDriveConfig(
        n_steps=1,
        combination_strategy="collectivity",
        z_mixing_alpha=0.5,
        n_anm_modes=8,
        n_combinations=5,
        max_combo_size=2,
        df=0.6, df_min=0.3, df_max=1.5,
        confidence_rg_min=0.0, confidence_rg_max=100.0,
        selective_mixing=True,
        **kw,
    )
    return ModeDrivePipeline(converter=MockConverter(), config=cfg)


def _random_coords(n: int = 20, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, 3) * 10.0


# ── compute_change_score ──────────────────────────────────────────────────────

class TestChangeScoreIdentity:
    """When coords_before == coords_after the score should be (near) zero."""

    def test_zero_score_for_identical_inputs(self):
        coords = _random_coords(20, seed=1)
        score = compute_change_score(coords, coords, coords, r_cut=10.0, tau=1.5)
        assert score.shape == (20, 20)
        assert float(score.max().item()) < 1e-6

    def test_diagonal_always_zero(self):
        coords = _random_coords(15, seed=2)
        score = compute_change_score(coords, coords, coords, r_cut=10.0, tau=1.5)
        diag = score.diagonal()
        assert torch.all(diag == 0.0)

    def test_values_in_unit_interval(self):
        before = _random_coords(20, seed=3)
        after = before.clone()
        after[5:10] += 3.0
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        assert float(score.min().item()) >= -1e-6
        assert float(score.max().item()) <= 1.0 + 1e-6


class TestChangeScoreLargeDisplacement:
    """A subset of residues displaced by 5 A should yield higher scores there."""

    def test_moved_pairs_score_higher_than_static_pairs(self):
        torch.manual_seed(10)
        before = torch.randn(20, 3) * 5.0
        after = before.clone()
        # Residues 5-9 (inclusive) move 5 A in x
        after[5:10, 0] += 5.0

        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)

        # Pairs among moved residues (5-9 x 5-9)
        moved_block = score[5:10, 5:10]
        # Pairs entirely in static region (0-4 x 0-4)
        static_block = score[0:5, 0:5]

        assert float(moved_block.mean().item()) > float(static_block.mean().item())

    def test_score_strictly_positive_after_large_move(self):
        torch.manual_seed(11)
        before = torch.randn(10, 3) * 5.0
        after = before.clone()
        after[3:7] += 8.0   # big displacement

        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        assert float(score.max().item()) > 0.0


class TestChangeScoreSymmetry:
    """Score matrix must be symmetric and have a zero diagonal."""

    def test_symmetric(self):
        torch.manual_seed(20)
        before = torch.randn(15, 3)
        after = before.clone()
        after[2:6] += 3.0
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        assert torch.allclose(score, score.T, atol=1e-6)

    def test_zero_diagonal(self):
        torch.manual_seed(21)
        before = torch.randn(12, 3)
        after = before.clone()
        after[:4] += 2.0
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        assert torch.all(score.diagonal() == 0.0)


class TestChangeScoreDistanceModes:
    """'mean' and 'max' distance_mode both produce valid outputs."""

    def test_max_mode_output_shape(self):
        before = _random_coords(10, seed=30)
        after = before.clone(); after[3:7] += 2.0
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5,
                                      distance_mode="max")
        assert score.shape == (10, 10)
        assert torch.isfinite(score).all()

    def test_mean_mode_output_shape(self):
        before = _random_coords(10, seed=31)
        after = before.clone(); after[3:7] += 2.0
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5,
                                      distance_mode="mean")
        assert score.shape == (10, 10)
        assert torch.isfinite(score).all()

    def test_max_geq_mean_for_asymmetric_displacement(self):
        """max(d_i, d_j) >= mean(d_i, d_j) always holds element-wise."""
        torch.manual_seed(32)
        before = torch.randn(12, 3) * 5.0
        after = before.clone(); after[2:5] += 4.0

        score_max = compute_change_score(before, after, before, r_cut=10.0, tau=1.5,
                                          distance_mode="max")
        score_mean = compute_change_score(before, after, before, r_cut=10.0, tau=1.5,
                                           distance_mode="mean")
        # Off-diagonal: max-mode score >= mean-mode score (element-wise)
        mask = ~torch.eye(12, dtype=torch.bool)
        assert (score_max[mask] >= score_mean[mask] - 1e-6).all()


# ── compute_alpha_mask ────────────────────────────────────────────────────────

class TestAlphaMaskCutoff:
    """Scores below change_cutoff should collapse to alpha_base."""

    def test_below_cutoff_gets_alpha_base(self):
        score = torch.tensor([
            [0.0, 0.05, 0.5],
            [0.05, 0.0, 0.8],
            [0.5, 0.8, 0.0],
        ])
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="linear")
        # 0.05 < cutoff → alpha_base = 0.0
        assert float(mask[0, 1].item()) == pytest.approx(0.0, abs=1e-6)
        assert float(mask[1, 0].item()) == pytest.approx(0.0, abs=1e-6)

    def test_above_cutoff_gets_positive_alpha(self):
        score = torch.tensor([
            [0.0, 0.05, 0.5],
            [0.05, 0.0, 0.8],
            [0.5, 0.8, 0.0],
        ])
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="linear")
        # 0.5 >= cutoff → alpha > 0
        assert float(mask[0, 2].item()) > 0.0
        assert float(mask[2, 0].item()) > 0.0

    def test_all_zeros_score_produces_all_alpha_base(self):
        score = torch.zeros(8, 8)
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0)
        # All values should be alpha_base (0.0) or very close
        assert float(mask.max().item()) < 1e-6

    def test_diagonal_is_zero_regardless_of_score(self):
        score = torch.ones(6, 6)
        score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.0,
                                   alpha_base=0.0, alpha_max=1.0)
        assert torch.all(mask.diagonal() == 0.0)


class TestAlphaMaskLinear:
    """Linear mapping: alpha scales proportionally with normalised score."""

    def test_shape_preserved(self):
        score = torch.rand(7, 7); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.0, alpha_base=0.0,
                                   alpha_max=1.0, mapping="linear")
        assert mask.shape == (7, 7)

    def test_values_in_alpha_range(self):
        score = torch.rand(8, 8); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        alpha_base, alpha_max = 0.1, 0.8
        mask = compute_alpha_mask(score, change_cutoff=0.0,
                                   alpha_base=alpha_base, alpha_max=alpha_max,
                                   mapping="linear")
        # Off-diagonal values should lie in [alpha_base, alpha_max]
        off_diag = mask[~torch.eye(8, dtype=torch.bool)]
        assert float(off_diag.min().item()) >= alpha_base - 1e-6
        assert float(off_diag.max().item()) <= alpha_max + 1e-6

    def test_symmetric(self):
        torch.manual_seed(40)
        score = torch.rand(6, 6); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.0, alpha_base=0.0,
                                   alpha_max=1.0, mapping="linear")
        assert torch.allclose(mask, mask.T, atol=1e-6)


class TestAlphaMaskSigmoid:
    """Sigmoid mapping: smooth transition, values stay in [alpha_base, alpha_max]."""

    def test_values_bounded(self):
        torch.manual_seed(50)
        score = torch.rand(10, 10); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        alpha_base, alpha_max = 0.0, 0.9
        mask = compute_alpha_mask(score, change_cutoff=0.0,
                                   alpha_base=alpha_base, alpha_max=alpha_max,
                                   mapping="sigmoid")
        off_diag = mask[~torch.eye(10, dtype=torch.bool)]
        assert float(off_diag.min().item()) >= alpha_base - 1e-6
        assert float(off_diag.max().item()) <= alpha_max + 1e-6

    def test_shape_preserved(self):
        score = torch.rand(5, 5); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.0, alpha_base=0.0,
                                   alpha_max=1.0, mapping="sigmoid")
        assert mask.shape == (5, 5)

    def test_below_cutoff_gets_alpha_base(self):
        # Score uniformly small — all below cutoff
        score = torch.full((6, 6), 0.05)
        score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="sigmoid")
        off_diag = mask[~torch.eye(6, dtype=torch.bool)]
        assert float(off_diag.max().item()) < 1e-6

    def test_finite_values(self):
        torch.manual_seed(51)
        score = torch.rand(8, 8); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.1, alpha_base=0.0,
                                   alpha_max=1.0, mapping="sigmoid")
        assert torch.isfinite(mask).all()


class TestAlphaMaskStep:
    """Step mapping: binary — below cutoff → alpha_base, at/above → alpha_max."""

    def test_binary_values(self):
        score = torch.tensor([
            [0.0, 0.05, 0.3, 0.8],
            [0.05, 0.0, 0.6, 0.2],
            [0.3, 0.6, 0.0, 0.9],
            [0.8, 0.2, 0.9, 0.0],
        ])
        alpha_base, alpha_max = 0.0, 1.0
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=alpha_base, alpha_max=alpha_max,
                                   mapping="step")
        off_diag = mask[~torch.eye(4, dtype=torch.bool)]
        # Must be either alpha_base or alpha_max (binary)
        is_base = (off_diag - alpha_base).abs() < 1e-6
        is_max = (off_diag - alpha_max).abs() < 1e-6
        assert (is_base | is_max).all()

    def test_below_cutoff_is_alpha_base(self):
        score = torch.zeros(5, 5)
        score[0, 1] = score[1, 0] = 0.05  # below 0.1
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="step")
        assert float(mask[0, 1].item()) == pytest.approx(0.0, abs=1e-6)

    def test_above_cutoff_is_alpha_max(self):
        score = torch.zeros(5, 5)
        score[0, 2] = score[2, 0] = 0.5  # above 0.1
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="step")
        assert float(mask[0, 2].item()) == pytest.approx(1.0, abs=1e-6)

    def test_shape_preserved(self):
        score = torch.rand(9, 9); score = 0.5 * (score + score.T); score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.2,
                                   alpha_base=0.1, alpha_max=0.9, mapping="step")
        assert mask.shape == (9, 9)


# ── selective_blend_z ─────────────────────────────────────────────────────────

class TestSelectiveBlendShape:
    """Output tensor must have the same shape as inputs."""

    def test_preserves_shape(self):
        torch.manual_seed(60)
        N, C = 10, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask)
        assert result.shape == (N, N, C)

    def test_non_square_channel(self):
        """Works for any channel dimension."""
        N, C = 8, 64
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask, normalize=False)
        assert result.shape == (N, N, C)

    def test_output_finite(self):
        torch.manual_seed(61)
        N = 12
        z_pseudo = torch.randn(N, N, 128) * 100.0
        z_trunk = torch.randn(N, N, 128) * 0.1
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask, normalize=True)
        assert torch.isfinite(result).all()


class TestSelectiveBlendUniformEquivalence:
    """When alpha_mask is uniform and normalize=False, result equals uniform blend."""

    def test_uniform_alpha_equivalent_to_scalar(self):
        torch.manual_seed(70)
        N, C = 6, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha = 0.4
        alpha_mask = torch.full((N, N), alpha)

        result_selective = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                              normalize=False, direction="plus")
        result_uniform = z_trunk + alpha * (z_pseudo - z_trunk)
        torch.testing.assert_close(result_selective, result_uniform)


class TestSelectiveBlendZeroAlphaPairs:
    """Pairs where alpha_mask == 0 must preserve z_trunk exactly."""

    def test_zero_alpha_preserves_trunk(self):
        torch.manual_seed(80)
        N, C = 8, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)

        # alpha=0 for first 4 rows/cols, alpha=1 for rest
        alpha_mask = torch.zeros(N, N)
        alpha_mask[4:, 4:] = 1.0

        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=False, direction="plus")

        # Where alpha=0: z_result == z_trunk
        torch.testing.assert_close(result[:4, :4], z_trunk[:4, :4])

    def test_full_alpha_equals_z_pseudo(self):
        """When alpha_mask is all-ones, result should be z_pseudo (direction=plus)."""
        torch.manual_seed(81)
        N, C = 6, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.ones(N, N)

        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=False, direction="plus")
        torch.testing.assert_close(result, z_pseudo)


class TestSelectiveBlendDirectionMinus:
    """direction='minus' subtracts alpha * delta_z instead of adding."""

    def test_minus_differs_from_plus(self):
        torch.manual_seed(90)
        N, C = 8, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N) * 0.5

        plus_result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                         normalize=False, direction="plus")
        minus_result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                          normalize=False, direction="minus")
        assert not torch.allclose(plus_result, minus_result)

    def test_minus_formula_correct(self):
        """z_trunk - alpha * (z_pseudo - z_trunk) must match manual computation."""
        torch.manual_seed(91)
        N, C = 5, 32
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N)

        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=False, direction="minus")
        expected = z_trunk - alpha_mask.unsqueeze(-1) * (z_pseudo - z_trunk)
        torch.testing.assert_close(result, expected)

    def test_shape_preserved_minus(self):
        N, C = 10, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=False, direction="minus")
        assert result.shape == (N, N, C)


class TestSelectiveBlendNormalize:
    """normalize=True should match z_trunk statistics before blending."""

    def test_normalize_rescales_z_pseudo_to_trunk_magnitude(self):
        """After normalize, the blended output lives in z_trunk's magnitude range.

        Without normalisation, blending z_pseudo (scale ~0.001) at alpha=1 produces
        a result that is close to z_trunk (the raw z_pseudo is negligibly small, so
        z_trunk + 1*(z_pseudo-z_trunk) ~ z_pseudo ~ 0.001, far from z_trunk's
        ~50 scale).  With normalisation, z_pseudo is rescaled to z_trunk's mean
        and std, so the blended result has similar magnitude to z_trunk.
        """
        torch.manual_seed(100)
        N, C = 8, 128
        z_pseudo = torch.randn(N, N, C) * 0.001   # very small scale
        z_trunk = torch.randn(N, N, C) * 50.0      # large scale

        # Use alpha_mask=1 so the output is purely the (possibly normalised) z_pseudo
        alpha_mask = torch.ones(N, N)

        result_norm = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                         normalize=True, direction="plus")
        result_raw = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                        normalize=False, direction="plus")

        # Without normalize: output ~= z_pseudo (scale ~0.001), very small
        # With normalize: output has z_pseudo rescaled to z_trunk's distribution
        # So normalised output std should be much larger than raw output std
        assert result_norm.std().item() > result_raw.std().item() * 10

    def test_normalize_output_is_finite(self):
        """normalize=True with extreme scale difference must not produce NaN/Inf."""
        torch.manual_seed(101)
        N, C = 8, 128
        z_pseudo = torch.randn(N, N, C) * 1e-5
        z_trunk = torch.randn(N, N, C) * 1e3
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=True, direction="plus")
        assert torch.isfinite(result).all()


# ── End-to-end: pipeline diagnostics ─────────────────────────────────────────

class TestSelectiveMixingPipelineDiagnostics:
    """Test that ModeDrivePipeline populates StepResult diagnostics when selective_mixing=True."""

    def test_diagnostics_populated_with_selective_mixing(self):
        torch.manual_seed(200)
        n = 15
        pipe = _make_selective_pipeline(n=n)
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)

        assert isinstance(result, StepResult)
        assert result.change_score_mean is not None
        assert result.change_score_max is not None
        assert result.n_active_pairs is not None
        assert result.alpha_mask_mean is not None

    def test_diagnostics_none_without_selective_mixing(self):
        """With selective_mixing=False, diagnostic fields must remain None."""
        torch.manual_seed(201)
        n = 15
        cfg = ModeDriveConfig(
            n_steps=1,
            combination_strategy="collectivity",
            z_mixing_alpha=0.5,
            n_anm_modes=8,
            n_combinations=5,
            max_combo_size=2,
            df=0.6, df_min=0.3, df_max=1.5,
            confidence_rg_min=0.0, confidence_rg_max=100.0,
            selective_mixing=False,  # disabled
        )
        pipe = ModeDrivePipeline(converter=MockConverter(), config=cfg)
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)

        assert result.change_score_mean is None
        assert result.change_score_max is None
        assert result.n_active_pairs is None
        assert result.alpha_mask_mean is None

    def test_diagnostic_values_are_valid_floats(self):
        torch.manual_seed(202)
        n = 20
        pipe = _make_selective_pipeline(n=n)
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)

        assert isinstance(result.change_score_mean, float)
        assert isinstance(result.change_score_max, float)
        assert isinstance(result.n_active_pairs, int)
        assert isinstance(result.alpha_mask_mean, float)
        assert result.change_score_mean >= 0.0
        assert result.change_score_max >= result.change_score_mean
        assert result.n_active_pairs >= 0
        assert result.alpha_mask_mean >= 0.0

    def test_n_active_pairs_bounded_by_n_squared(self):
        torch.manual_seed(203)
        n = 12
        pipe = _make_selective_pipeline(n=n)
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)
        assert result.n_active_pairs <= n * n

    def test_pipeline_completes_successfully_with_selective_mixing(self):
        """Full run with selective_mixing=True must not raise."""
        torch.manual_seed(204)
        n = 15
        pipe = _make_selective_pipeline(n=n)
        pipe.config.n_steps = 2
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        from src.mode_drive import ModeDriveResult
        result = pipe.run(coords, zij, verbose=False)
        assert isinstance(result, ModeDriveResult)
        assert result.total_steps == 2

    def test_mapping_step_with_pipeline(self):
        """selective_mapping='step' must not crash the pipeline."""
        torch.manual_seed(205)
        n = 12
        pipe = _make_selective_pipeline(n=n, selective_mapping="step")
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)
        assert result.change_score_mean is not None

    def test_mapping_sigmoid_with_pipeline(self):
        """selective_mapping='sigmoid' must not crash the pipeline."""
        torch.manual_seed(206)
        n = 12
        pipe = _make_selective_pipeline(n=n, selective_mapping="sigmoid")
        coords = torch.randn(n, 3) * 10.0
        zij = torch.randn(n, n, 128)

        result = pipe.step(coords, coords, zij)
        assert result.change_score_mean is not None


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases: single residue, zero variance inputs."""

    def test_single_residue_change_score(self):
        coords = torch.zeros(1, 3)
        score = compute_change_score(coords, coords, coords, r_cut=10.0, tau=1.5)
        assert score.shape == (1, 1)
        assert score[0, 0] == 0.0

    def test_two_residue_change_score_symmetric(self):
        before = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        after = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # 2A move
        score = compute_change_score(before, after, before, r_cut=5.0, tau=1.0)
        assert score.shape == (2, 2)
        assert float(score[0, 1].item()) == pytest.approx(float(score[1, 0].item()), abs=1e-6)
        assert score[0, 0] == 0.0
        assert score[1, 1] == 0.0

    def test_alpha_mask_uniform_score(self):
        """Uniform score above cutoff → linear mask has uniform positive alpha."""
        score = torch.full((5, 5), 0.5)
        score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.1,
                                   alpha_base=0.0, alpha_max=1.0, mapping="linear")
        # All off-diagonal should be equal
        off = mask[~torch.eye(5, dtype=torch.bool)]
        assert off.std().item() < 1e-5

    def test_selective_blend_zero_alpha_mask(self):
        """alpha_mask=0 everywhere → result equals z_trunk."""
        torch.manual_seed(300)
        N, C = 7, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.zeros(N, N)

        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask,
                                    normalize=False, direction="plus")
        torch.testing.assert_close(result, z_trunk)

    def test_change_score_returns_tensor(self):
        before = _random_coords(8, seed=400)
        after = before + 0.1
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        assert isinstance(score, torch.Tensor)

    def test_alpha_mask_returns_tensor(self):
        score = torch.rand(6, 6)
        score = 0.5 * (score + score.T)
        score.fill_diagonal_(0.0)
        mask = compute_alpha_mask(score, change_cutoff=0.1, alpha_base=0.0, alpha_max=1.0)
        assert isinstance(mask, torch.Tensor)

    def test_selective_blend_returns_tensor(self):
        N, C = 5, 128
        z_pseudo = torch.randn(N, N, C)
        z_trunk = torch.randn(N, N, C)
        alpha_mask = torch.rand(N, N)
        result = selective_blend_z(z_pseudo, z_trunk, alpha_mask)
        assert isinstance(result, torch.Tensor)

    def test_large_n_performance(self):
        """compute_change_score should complete in reasonable time for N=100."""
        import time
        N = 100
        before = torch.randn(N, 3) * 10.0
        after = before.clone(); after[40:60] += 3.0
        start = time.time()
        score = compute_change_score(before, after, before, r_cut=10.0, tau=1.5)
        elapsed = time.time() - start
        assert score.shape == (N, N)
        assert elapsed < 10.0  # should be well under 10 seconds
