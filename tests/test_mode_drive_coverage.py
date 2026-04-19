"""Additional tests for mode_drive.py — targeting uncovered paths.

Covers: _blend_z, _confidence_ok, step() with various strategies,
step_with_fallback() L0-L5, run() with verbose/fallback, manual/targeted combos.
"""

import torch
import pytest

from src.mode_drive import (
    ModeDriveConfig,
    ModeDrivePipeline,
    ModeDriveResult,
    StepResult,
)
from src.mode_combinator import ModeCombo


class MockConverter:
    """Lightweight mock for PairContactConverter."""

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


def _make_pipeline(strategy="collectivity", alpha=0.3, n_steps=2, **kw):
    cfg = ModeDriveConfig(
        n_steps=n_steps,
        combination_strategy=strategy,
        z_mixing_alpha=alpha,
        n_anm_modes=8,
        n_combinations=5,
        max_combo_size=2,
        df=0.6, df_min=0.3, df_max=1.5,
        confidence_rg_min=0.0, confidence_rg_max=100.0,  # disable Rg filter for unit tests
        **kw,
    )
    return ModeDrivePipeline(converter=MockConverter(), config=cfg)


def _random_inputs(n=15):
    torch.manual_seed(42)
    coords = torch.randn(n, 3) * 10.0
    zij = torch.randn(n, n, 128)
    return coords, zij


# ── _blend_z ─────────────────────────────────────────────────────

class TestBlendZ:
    def test_plus_direction(self):
        pipe = _make_pipeline(alpha=0.5)
        pipe.config.z_direction = "plus"
        zij = torch.randn(5, 5, 128)
        z_pseudo = torch.randn(5, 5, 128)
        blended = pipe._blend_z(z_pseudo, zij)
        assert blended.shape == (5, 5, 128)
        # Not identical to either input
        assert not torch.allclose(blended, zij)

    def test_minus_direction(self):
        pipe = _make_pipeline(alpha=0.5)
        pipe.config.z_direction = "minus"
        zij = torch.randn(5, 5, 128)
        z_pseudo = torch.randn(5, 5, 128)
        blended = pipe._blend_z(z_pseudo, zij)
        assert blended.shape == (5, 5, 128)

    def test_alpha_zero_returns_trunk(self):
        pipe = _make_pipeline(alpha=0.0)
        pipe.config.normalize_z = False
        zij = torch.randn(5, 5, 128)
        z_pseudo = torch.randn(5, 5, 128)
        blended = pipe._blend_z(z_pseudo, zij)
        torch.testing.assert_close(blended, zij)

    def test_normalize_z_enabled(self):
        pipe = _make_pipeline(alpha=0.3)
        pipe.config.normalize_z = True
        zij = torch.randn(5, 5, 128) * 0.1
        z_pseudo = torch.randn(5, 5, 128) * 100.0  # very different scale
        blended = pipe._blend_z(z_pseudo, zij)
        assert blended.shape == (5, 5, 128)
        assert torch.isfinite(blended).all()

    def test_normalize_z_disabled(self):
        pipe = _make_pipeline(alpha=0.3)
        pipe.config.normalize_z = False
        zij = torch.randn(5, 5, 128)
        z_pseudo = torch.randn(5, 5, 128)
        blended = pipe._blend_z(z_pseudo, zij)
        assert blended.shape == (5, 5, 128)


# ── _confidence_ok ───────────────────────────────────────────────

class TestConfidenceOk:
    def _result(self, ptm=None, plddt=None, ranking=None):
        return StepResult(
            combo=ModeCombo(mode_indices=(0,), dfs=(0.5,)),
            displaced_ca=torch.zeros(5, 3),
            new_ca=torch.zeros(5, 3),
            z_modified=torch.zeros(5, 5, 128),
            contact_map=torch.zeros(5, 5),
            rmsd=1.0,
            eigenvalues=torch.ones(3),
            eigenvectors=torch.ones(5, 3, 3),
            b_factors=torch.ones(5),
            ptm=ptm,
            plddt=plddt,
            ranking_score=ranking,
        )

    def test_all_none_passes(self):
        pipe = _make_pipeline()
        assert pipe._confidence_ok(self._result()) is True

    def test_ptm_below_cutoff_fails(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ptm_cutoff = 0.5
        assert pipe._confidence_ok(self._result(ptm=0.3)) is False

    def test_ptm_above_cutoff_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ptm_cutoff = 0.5
        assert pipe._confidence_ok(self._result(ptm=0.7)) is True

    def test_plddt_below_cutoff_fails(self):
        pipe = _make_pipeline()
        pipe.config.confidence_plddt_cutoff = 70.0
        plddt = torch.full((5,), 50.0)
        assert pipe._confidence_ok(self._result(plddt=plddt)) is False

    def test_plddt_above_cutoff_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_plddt_cutoff = 70.0
        plddt = torch.full((5,), 80.0)
        assert pipe._confidence_ok(self._result(plddt=plddt)) is True

    def test_ranking_below_cutoff_fails(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ranking_cutoff = 0.5
        assert pipe._confidence_ok(self._result(ranking=0.3)) is False

    def test_ranking_above_cutoff_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ranking_cutoff = 0.5
        assert pipe._confidence_ok(self._result(ranking=0.7)) is True

    def test_and_logic_all_must_pass(self):
        """All three must pass, not just one."""
        pipe = _make_pipeline()
        pipe.config.confidence_ptm_cutoff = 0.5
        pipe.config.confidence_plddt_cutoff = 70.0
        pipe.config.confidence_ranking_cutoff = 0.5
        # ptm passes, plddt fails
        plddt = torch.full((5,), 50.0)
        assert pipe._confidence_ok(self._result(ptm=0.8, plddt=plddt, ranking=0.8)) is False


# ── step() with various strategies ───────────────────────────────

class TestStepStrategies:
    def test_random_strategy(self):
        pipe = _make_pipeline(strategy="random")
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij)
        assert isinstance(result, StepResult)
        assert result.new_ca.shape == coords.shape

    def test_grid_strategy(self):
        pipe = _make_pipeline(strategy="grid")
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij)
        assert isinstance(result, StepResult)

    def test_manual_strategy(self):
        pipe = _make_pipeline(strategy="manual", manual_modes=(0, 1))
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij)
        assert isinstance(result, StepResult)

    def test_manual_strategy_empty_modes_raises(self):
        pipe = _make_pipeline(strategy="manual")
        coords, zij = _random_inputs()
        with pytest.raises(ValueError, match="manual_modes"):
            pipe.step(coords, coords, zij)

    def test_targeted_strategy_requires_target(self):
        pipe = _make_pipeline(strategy="targeted")
        coords, zij = _random_inputs()
        with pytest.raises(ValueError, match="target_coords"):
            pipe.step(coords, coords, zij)

    def test_targeted_strategy_with_target(self):
        pipe = _make_pipeline(strategy="targeted")
        coords, zij = _random_inputs()
        target = coords + torch.randn_like(coords) * 2.0
        result = pipe.step(coords, coords, zij, target_coords=target)
        assert isinstance(result, StepResult)

    def test_collectivity_with_target(self):
        """Collectivity strategy with target uses negative RMSD-to-target."""
        pipe = _make_pipeline(strategy="collectivity")
        coords, zij = _random_inputs()
        target = coords + torch.randn_like(coords) * 2.0
        result = pipe.step(coords, coords, zij, target_coords=target)
        assert isinstance(result, StepResult)

    def test_step_idx_threaded(self):
        pipe = _make_pipeline(strategy="random")
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij, step_idx=5)
        assert isinstance(result, StepResult)


# ── step_with_fallback() ─────────────────────────────────────────

class TestStepWithFallback:
    def test_l0_passes_no_fallback(self):
        """Without confidence metrics, L0 always passes."""
        pipe = _make_pipeline()
        pipe.config.enable_confidence_fallback = True
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.fallback_level == 0
        assert result.rejected is False

    def test_forced_reject_when_all_fail(self):
        """Mock a pipeline where confidence always fails."""
        pipe = _make_pipeline()
        pipe.config.enable_confidence_fallback = True
        # Set impossible cutoffs
        pipe.config.confidence_ptm_cutoff = 999.0

        # Create a diffusion_fn that returns DiffusionResult-like with low pTM
        class FakeDiffResult:
            def __init__(self, ca):
                self.best_ca = ca
                self.all_ca = ca.unsqueeze(0)
                self.best_idx = 0
                self.plddt = torch.full((1, ca.shape[0]), 50.0)
                self.ptm = torch.tensor([0.1])
                self.ranking = torch.tensor([0.1])

        pipe.diffusion_fn = lambda z: FakeDiffResult(torch.randn(z.shape[0], 3))

        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.rejected is True

    def test_config_restored_after_fallback(self):
        """Config values must be restored even after fallback levels."""
        pipe = _make_pipeline(alpha=0.3)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0

        class FakeDiffResult:
            def __init__(self, ca):
                self.best_ca = ca
                self.all_ca = ca.unsqueeze(0)
                self.best_idx = 0
                self.plddt = torch.full((1, ca.shape[0]), 50.0)
                self.ptm = torch.tensor([0.1])
                self.ranking = torch.tensor([0.1])

        pipe.diffusion_fn = lambda z: FakeDiffResult(torch.randn(z.shape[0], 3))

        coords, zij = _random_inputs()
        orig_df = pipe.config.df
        orig_alpha = pipe.config.z_mixing_alpha
        orig_combo = pipe.config.max_combo_size

        pipe.step_with_fallback(coords, coords, zij)

        # Config should be restored
        assert pipe.config.df == orig_df
        assert pipe.config.z_mixing_alpha == orig_alpha
        assert pipe.config.max_combo_size == orig_combo


# ── run() ────────────────────────────────────────────────────────

class TestRun:
    def test_run_basic(self):
        pipe = _make_pipeline(n_steps=2)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        assert isinstance(result, ModeDriveResult)
        assert result.total_steps == 2
        assert len(result.step_results) == 2
        assert len(result.trajectory) == 3  # initial + 2 steps

    def test_run_with_verbose(self, capsys):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=True)
        captured = capsys.readouterr()
        assert "Mode-Drive Pipeline" in captured.out
        assert "Final RMSD" in captured.out

    def test_run_with_target_verbose(self, capsys):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        target = coords + torch.randn_like(coords) * 2.0
        result = pipe.run(coords, zij, target_coords=target, verbose=True)
        captured = capsys.readouterr()
        assert "TM_tgt" in captured.out or "TM-score" in captured.out

    def test_run_with_fallback_enabled(self):
        pipe = _make_pipeline(n_steps=1)
        pipe.config.enable_confidence_fallback = True
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        assert result.total_steps == 1

    def test_run_rejected_step_preserves_coords(self):
        """When a step is rejected, coords should not update."""
        pipe = _make_pipeline(n_steps=2)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0  # impossible

        class FakeDiffResult:
            def __init__(self, ca):
                self.best_ca = ca
                self.all_ca = ca.unsqueeze(0)
                self.best_idx = 0
                self.plddt = torch.full((1, ca.shape[0]), 50.0)
                self.ptm = torch.tensor([0.1])
                self.ranking = torch.tensor([0.1])

        pipe.diffusion_fn = lambda z: FakeDiffResult(torch.randn(z.shape[0], 3))

        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        # All steps should be rejected
        assert all(r.rejected for r in result.step_results)

    def test_run_zero_steps(self):
        pipe = _make_pipeline(n_steps=0)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        assert result.total_steps == 0
        assert len(result.step_results) == 0
        assert len(result.trajectory) == 1  # just initial


# ── DiffusionResult integration ──────────────────────────────────

class TestDiffusionFnIntegration:
    def test_with_diffusion_fn_tensor(self):
        """diffusion_fn returning plain tensor (backward compat)."""
        pipe = _make_pipeline(n_steps=1)
        pipe.diffusion_fn = lambda z: torch.randn(z.shape[0], 3)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        assert result.step_results[0].ptm is None

    def test_with_diffusion_fn_result_object(self):
        """diffusion_fn returning DiffusionResult-like."""
        pipe = _make_pipeline(n_steps=1)
        n = 15

        class FakeDiffResult:
            def __init__(self):
                self.best_ca = torch.randn(n, 3)
                self.all_ca = torch.randn(2, n, 3)
                self.best_idx = 0
                self.plddt = torch.rand(2, n)
                self.ptm = torch.tensor([0.7, 0.5])
                self.ranking = torch.tensor([0.75, 0.55])

        pipe.diffusion_fn = lambda z: FakeDiffResult()
        coords, zij = _random_inputs(n)
        result = pipe.run(coords, zij, verbose=False)
        sr = result.step_results[0]
        assert sr.ptm is not None
        assert sr.num_samples == 2
        assert sr.all_ptm is not None


# ── _manual_combo ─────────────────────────────────────────────────

class TestManualCombo:
    def test_single_mode(self):
        pipe = _make_pipeline(strategy="manual", manual_modes=(0,))
        eigenvalues = torch.tensor([1.0, 2.0, 3.0])
        combos = pipe._manual_combo(eigenvalues, df=1.0)
        assert len(combos) == 1
        assert combos[0].mode_indices == (0,)

    def test_multi_mode(self):
        pipe = _make_pipeline(strategy="manual", manual_modes=(0, 2))
        eigenvalues = torch.tensor([1.0, 2.0, 3.0])
        combos = pipe._manual_combo(eigenvalues, df=1.0)
        assert len(combos) == 1
        assert combos[0].mode_indices == (0, 2)
        assert len(combos[0].dfs) == 2


# ── step_with_fallback deeper L1-L5 coverage ─────────────────────

class _FakeDiffResult:
    """Configurable diffusion result for fallback tests."""

    def __init__(self, ca, ptm=0.1, plddt_mean=50.0,
                 mean_pae=None, has_clash=None, consensus_score=None,
                 contact_probs=None):
        n = ca.shape[0]
        self.best_ca = ca
        self.all_ca = ca.unsqueeze(0)
        self.best_idx = 0
        self.plddt = torch.full((1, n), plddt_mean)
        self.ptm = torch.tensor([ptm])
        self.ranking = torch.tensor([0.8 * ptm + 0.2 * plddt_mean / 100.0])
        # V2 fields
        self.mean_pae = mean_pae
        self.has_clash = has_clash
        self.consensus_score = consensus_score
        self.contact_probs = contact_probs
        self.pae = None
        self.sample_rmsd = None
        self.sample_rmsf = None


class TestFallbackLevels:
    """Test that fallback L1-L5 actually try different configurations."""

    def _make_failing_pipeline(self, ptm_cutoff=0.9, diff_ptm=0.1):
        """Pipeline with diffusion that always fails confidence."""
        pipe = _make_pipeline(n_steps=1, alpha=0.3)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = ptm_cutoff
        pipe.config.fallback_combo_tries = 2
        pipe.config.fallback_df_factor = 0.5
        pipe.config.fallback_max_combo_size = 1
        pipe.config.fallback_alpha_factor = 0.5
        pipe.config.fallback_extended_enabled = False  # disable L5 for speed

        pipe.diffusion_fn = lambda z: _FakeDiffResult(
            torch.randn(z.shape[0], 3), ptm=diff_ptm
        )
        return pipe

    def test_all_levels_exhausted(self):
        pipe = self._make_failing_pipeline()
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.rejected is True

    def test_l0_passes_when_ptm_ok(self):
        pipe = self._make_failing_pipeline(ptm_cutoff=0.05, diff_ptm=0.1)
        pipe.config.confidence_ranking_cutoff = 0.01  # low enough to pass
        pipe.config.confidence_plddt_cutoff = 10.0     # low enough to pass
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.fallback_level == 0
        assert result.rejected is False

    def test_extended_grid_l5(self):
        """Enable L5 extended grid and verify it runs."""
        pipe = self._make_failing_pipeline()
        pipe.config.fallback_extended_enabled = True
        pipe.config.fallback_extended_combo_count = 2
        pipe.config.fallback_extended_df_scales = (0.5,)
        pipe.config.fallback_extended_alpha_scales = (0.5,)
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        # Should reach L5 or beyond
        assert result.rejected is True  # still all fail
        assert result.ranking_score is not None

    def test_df_escalation_in_collectivity(self):
        """Collectivity strategy escalates df when no improvement."""
        pipe = _make_pipeline(strategy="collectivity", n_steps=1)
        pipe.config.df_min = 0.1
        pipe.config.df_max = 0.2
        pipe.config.df_escalation_factor = 1.5
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij, prev_rmsd=float("inf"))
        # Should complete without error even when no combo improves
        assert isinstance(result, StepResult)


# ── _generate_combos coverage ────────────────────────────────────

class TestGenerateCombos:
    def _pipe_with_modes(self, strategy, **kw):
        pipe = _make_pipeline(strategy=strategy, **kw)
        coords, _ = _random_inputs()
        from src.anm import build_hessian, anm_modes
        H = build_hessian(coords, 15.0, 1.0, 1.0)
        eigenvalues, eigenvectors = anm_modes(H, 8)
        return pipe, eigenvalues, eigenvectors, coords

    def test_collectivity_combos(self):
        pipe, evals, evecs, coords = self._pipe_with_modes("collectivity")
        combos = pipe._generate_combos(8, coords, evecs, evals, 0.6)
        assert len(combos) > 0

    def test_grid_combos(self):
        pipe, evals, evecs, coords = self._pipe_with_modes("grid")
        combos = pipe._generate_combos(8, coords, evecs, evals, 0.6)
        assert len(combos) > 0

    def test_random_combos(self):
        pipe, evals, evecs, coords = self._pipe_with_modes("random")
        combos = pipe._generate_combos(8, coords, evecs, evals, 0.6)
        assert len(combos) > 0

    def test_targeted_combos(self):
        pipe, evals, evecs, coords = self._pipe_with_modes("targeted")
        target = coords + torch.randn_like(coords)
        combos = pipe._generate_combos(8, coords, evecs, evals, 0.6, target)
        assert len(combos) > 0


# ── Fallback level-specific return paths ────────────────────────

class TestFallbackReturnPaths:
    """Test that specific fallback levels return when confidence passes."""

    def _stateful_pipe(self, pass_at_call):
        """Pipeline where diffusion succeeds at call number `pass_at_call`.

        Uses random strategy with n_combinations=1 so each step() makes
        exactly 1 diffusion call, keeping call counting predictable.
        Call pattern: L0=1, L1=+2(combo tries), L2=+1, L3=+1, L4=+1.
        Cumulative: L0@1, L1@2-3, L2@4, L3@5, L4@6.
        """
        cfg = ModeDriveConfig(
            n_steps=1,
            combination_strategy="random",
            z_mixing_alpha=0.3,
            n_anm_modes=8,
            n_combinations=1,
            max_combo_size=2,
            df=0.6, df_min=0.3, df_max=0.6,  # no escalation
            confidence_rg_min=0.0, confidence_rg_max=100.0,  # disable Rg for tests
            enable_confidence_fallback=True,
            confidence_ptm_cutoff=0.5,
            confidence_plddt_cutoff=10.0,
            confidence_ranking_cutoff=0.01,
            fallback_combo_tries=2,
            fallback_df_factor=0.5,
            fallback_max_combo_size=1,
            fallback_alpha_factor=0.5,
            fallback_extended_enabled=False,
        )
        pipe = ModeDrivePipeline(converter=MockConverter(), config=cfg)

        call_count = [0]

        def _diff(z):
            call_count[0] += 1
            n = z.shape[0]
            if call_count[0] >= pass_at_call:
                return _FakeDiffResult(torch.randn(n, 3), ptm=0.8, plddt_mean=80.0)
            return _FakeDiffResult(torch.randn(n, 3), ptm=0.1, plddt_mean=20.0)

        pipe.diffusion_fn = _diff
        return pipe

    def test_l2_df_fallback_returns(self):
        """L2: L0(1 call) + L1(0, single combo) = 1 fail, L2 at call 2."""
        pipe = self._stateful_pipe(pass_at_call=2)
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.fallback_level == 2
        assert result.rejected is False

    def test_l3_single_mode_returns(self):
        """L3: L0(1)+L2(1)=2 fail, L3 at call 3."""
        pipe = self._stateful_pipe(pass_at_call=3)
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.fallback_level == 3
        assert result.rejected is False

    def test_l4_alpha_fallback_returns(self):
        """L4: L0(1)+L2(1)+L3(1)=3 fail, L4 at call 4."""
        pipe = self._stateful_pipe(pass_at_call=4)
        coords, zij = _random_inputs()
        result = pipe.step_with_fallback(coords, coords, zij)
        assert result.fallback_level == 4
        assert result.rejected is False

    def test_run_verbose_with_fallback_info(self, capsys):
        """run() verbose mode prints fallback info."""
        pipe = self._stateful_pipe(pass_at_call=3)
        pipe.config.n_steps = 1
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=True)
        captured = capsys.readouterr()
        assert "Fallback triggered" in captured.out

    def test_run_verbose_rejected(self, capsys):
        """run() verbose shows rejected info."""
        pipe = _make_pipeline(n_steps=1)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0
        pipe.config.fallback_extended_enabled = False
        pipe.diffusion_fn = lambda z: _FakeDiffResult(
            torch.randn(z.shape[0], 3), ptm=0.1
        )
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=True)
        captured = capsys.readouterr()
        assert "Rejected" in captured.out


# ═══════════════════════════════════════════════════════════════════
# Confidence V2 Tests
# ═══════════════════════════════════════════════════════════════════

def _dummy_step_result(**overrides):
    """Minimal StepResult for _confidence_ok tests."""
    n = 10
    defaults = dict(
        combo=ModeCombo(mode_indices=(0,), dfs=(0.5,), label="test"),
        displaced_ca=torch.zeros(n, 3),
        new_ca=torch.zeros(n, 3),
        z_modified=torch.zeros(n, n, 128),
        contact_map=torch.zeros(n, n),
        rmsd=1.0,
        eigenvalues=torch.zeros(5),
        eigenvectors=torch.zeros(n, 5, 3),
        b_factors=torch.zeros(n),
        ptm=0.7,
        plddt=torch.full((n,), 80.0),
        ranking_score=0.6,
    )
    defaults.update(overrides)
    return StepResult(**defaults)


class TestConfidenceV2Warmup:
    """Test warmup period with relaxed cutoffs."""

    def test_warmup_uses_relaxed_ptm(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ptm_cutoff = 0.5
        pipe.config.confidence_warmup_steps = 3
        pipe.config.confidence_warmup_ptm_cutoff = 0.3
        # ptm=0.35 fails normal (0.5) but passes warmup (0.3)
        r = _dummy_step_result(ptm=0.35)
        assert pipe._confidence_ok(r, step_idx=0) is True   # in warmup
        assert pipe._confidence_ok(r, step_idx=2) is True   # still in warmup
        assert pipe._confidence_ok(r, step_idx=3) is False  # warmup over

    def test_warmup_uses_relaxed_ranking(self):
        pipe = _make_pipeline()
        pipe.config.confidence_ranking_cutoff = 0.5
        pipe.config.confidence_warmup_steps = 2
        pipe.config.confidence_warmup_ranking_cutoff = 0.3
        r = _dummy_step_result(ranking_score=0.35)
        assert pipe._confidence_ok(r, step_idx=0) is True   # warmup
        assert pipe._confidence_ok(r, step_idx=2) is False  # normal

    def test_no_warmup_when_disabled(self):
        pipe = _make_pipeline()
        pipe.config.confidence_warmup_steps = 0
        pipe.config.confidence_ptm_cutoff = 0.5
        r = _dummy_step_result(ptm=0.35)
        assert pipe._confidence_ok(r, step_idx=0) is False


class TestConfidenceV2RgFilter:
    """Test Rg ratio physical filter."""

    def test_rg_too_high_rejects(self):
        pipe = _make_pipeline()
        pipe.config.confidence_rg_max = 2.5
        pipe.config.confidence_rg_min = 0.3
        r = _dummy_step_result(rg_ratio=3.0)
        assert pipe._confidence_ok(r) is False

    def test_rg_too_low_rejects(self):
        pipe = _make_pipeline()
        pipe.config.confidence_rg_max = 2.5
        pipe.config.confidence_rg_min = 0.3
        r = _dummy_step_result(rg_ratio=0.1)
        assert pipe._confidence_ok(r) is False

    def test_rg_in_range_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_rg_max = 2.5
        pipe.config.confidence_rg_min = 0.3
        r = _dummy_step_result(rg_ratio=1.0)
        assert pipe._confidence_ok(r) is True

    def test_rg_none_skips_filter(self):
        pipe = _make_pipeline()
        pipe.config.confidence_rg_max = 2.5
        pipe.config.confidence_rg_min = 0.3
        r = _dummy_step_result(rg_ratio=None)
        assert pipe._confidence_ok(r) is True


class TestConfidenceV2ClashFilter:
    """Test clash rejection filter."""

    def test_clash_rejects_when_enabled(self):
        pipe = _make_pipeline()
        pipe.config.confidence_clash_reject = True
        r = _dummy_step_result(has_clash=True)
        assert pipe._confidence_ok(r) is False

    def test_clash_passes_when_disabled(self):
        pipe = _make_pipeline()
        pipe.config.confidence_clash_reject = False
        r = _dummy_step_result(has_clash=True)
        assert pipe._confidence_ok(r) is True

    def test_no_clash_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_clash_reject = True
        r = _dummy_step_result(has_clash=False)
        assert pipe._confidence_ok(r) is True

    def test_clash_none_passes(self):
        pipe = _make_pipeline()
        pipe.config.confidence_clash_reject = True
        r = _dummy_step_result(has_clash=None)
        assert pipe._confidence_ok(r) is True


class TestConfidenceV2Metrics:
    """Test V2 metric cutoffs (PAE, consensus, contact_recon, contact_of3)."""

    def test_mean_pae_rejects_high(self):
        pipe = _make_pipeline()
        pipe.config.confidence_mean_pae_cutoff = 10.0
        r = _dummy_step_result(mean_pae=15.0)
        assert pipe._confidence_ok(r) is False

    def test_mean_pae_passes_low(self):
        pipe = _make_pipeline()
        pipe.config.confidence_mean_pae_cutoff = 10.0
        r = _dummy_step_result(mean_pae=5.0)
        assert pipe._confidence_ok(r) is True

    def test_mean_pae_none_cutoff_disables(self):
        pipe = _make_pipeline()
        pipe.config.confidence_mean_pae_cutoff = None
        r = _dummy_step_result(mean_pae=999.0)
        assert pipe._confidence_ok(r) is True

    def test_consensus_rejects_low(self):
        pipe = _make_pipeline()
        pipe.config.confidence_consensus_cutoff = 0.5
        r = _dummy_step_result(consensus_score=0.3)
        assert pipe._confidence_ok(r) is False

    def test_consensus_passes_high(self):
        pipe = _make_pipeline()
        pipe.config.confidence_consensus_cutoff = 0.5
        r = _dummy_step_result(consensus_score=0.8)
        assert pipe._confidence_ok(r) is True

    def test_contact_recon_rejects_low(self):
        pipe = _make_pipeline()
        pipe.config.confidence_contact_recon_cutoff = 0.6
        r = _dummy_step_result(contact_recon=0.4)
        assert pipe._confidence_ok(r) is False

    def test_contact_recon_passes_high(self):
        pipe = _make_pipeline()
        pipe.config.confidence_contact_recon_cutoff = 0.6
        r = _dummy_step_result(contact_recon=0.8)
        assert pipe._confidence_ok(r) is True

    def test_contact_of3_rejects_low(self):
        pipe = _make_pipeline()
        pipe.config.confidence_contact_of3_cutoff = 0.5
        r = _dummy_step_result(contact_of3=0.3)
        assert pipe._confidence_ok(r) is False

    def test_contact_of3_passes_high(self):
        pipe = _make_pipeline()
        pipe.config.confidence_contact_of3_cutoff = 0.5
        r = _dummy_step_result(contact_of3=0.7)
        assert pipe._confidence_ok(r) is True


class TestConfidenceV2StallPrevention:
    """Test stall prevention: max_consecutive_rejected + alpha_decay."""

    def test_stall_stop_after_max_rejected(self, capsys):
        pipe = _make_pipeline(n_steps=5)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0  # always fail
        pipe.config.max_consecutive_rejected = 2
        pipe.config.fallback_extended_enabled = False
        pipe.diffusion_fn = lambda z: _FakeDiffResult(
            torch.randn(z.shape[0], 3), ptm=0.1
        )
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=True)
        # Should stop after 2 consecutive rejected
        assert result.total_steps == 2
        captured = capsys.readouterr()
        assert "STOP" in captured.out

    def test_alpha_decay_on_rejection(self):
        pipe = _make_pipeline(n_steps=3, alpha=0.5)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0
        pipe.config.max_consecutive_rejected = 3
        pipe.config.rejected_alpha_decay = 0.5
        pipe.config.fallback_extended_enabled = False
        pipe.diffusion_fn = lambda z: _FakeDiffResult(
            torch.randn(z.shape[0], 3), ptm=0.1
        )
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij, verbose=False)
        # After 3 rejected steps with 0.5 decay: 0.5 → 0.25 → 0.125
        # Alpha should be restored after run
        assert pipe.config.z_mixing_alpha == pytest.approx(0.5)

    def test_alpha_restored_after_run(self):
        pipe = _make_pipeline(n_steps=2, alpha=0.3)
        pipe.config.enable_confidence_fallback = True
        pipe.config.confidence_ptm_cutoff = 999.0
        pipe.config.rejected_alpha_decay = 0.8
        pipe.config.fallback_extended_enabled = False
        pipe.diffusion_fn = lambda z: _FakeDiffResult(
            torch.randn(z.shape[0], 3), ptm=0.1
        )
        coords, zij = _random_inputs()
        pipe.run(coords, zij)
        assert pipe.config.z_mixing_alpha == pytest.approx(0.3)

    def test_consecutive_counter_resets_on_success(self):
        pipe = _make_pipeline(n_steps=4, alpha=0.3)
        pipe.config.enable_confidence_fallback = False  # no fallback, just run-level rejection
        pipe.config.confidence_ptm_cutoff = 0.5
        pipe.config.max_consecutive_rejected = 3

        step_count = [0]
        def _diff(z):
            step_count[0] += 1
            n = z.shape[0]
            # Steps 1,2 fail (ptm<0.5), step 3 passes (ptm=0.8), step 4 fails
            if step_count[0] == 3:
                return _FakeDiffResult(torch.randn(n, 3), ptm=0.8)
            return _FakeDiffResult(torch.randn(n, 3), ptm=0.1)

        pipe.diffusion_fn = _diff
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        # Should complete all 4 steps: 2 rejected, 1 success (resets counter), 1 rejected
        assert result.total_steps == 4


class TestConfidenceV2ContactRecon:
    """Test contact_recon computation in _downstream_from_displaced."""

    def test_contact_recon_computed_with_diffusion(self):
        pipe = _make_pipeline(n_steps=1)
        n = 15
        ca = torch.randn(n, 3) * 10.0

        def _diff_v2(z):
            return _FakeDiffResult(
                ca + torch.randn(n, 3) * 0.1,  # slightly displaced
                ptm=0.8, mean_pae=5.0, has_clash=False,
                consensus_score=0.9,
            )

        pipe.diffusion_fn = _diff_v2
        coords, zij = _random_inputs(n=n)
        result = pipe.step(coords, coords, zij, prev_rmsd=0.0)
        # contact_recon should be computed (Pearson r)
        assert result.contact_recon is not None
        assert -1.0 <= result.contact_recon <= 1.0

    def test_contact_of3_computed_with_probs(self):
        pipe = _make_pipeline(n_steps=1)
        n = 15
        contact_p = torch.rand(n, n)
        contact_p = 0.5 * (contact_p + contact_p.T)

        def _diff_v2(z):
            return _FakeDiffResult(
                torch.randn(n, 3) * 10.0,
                ptm=0.8, contact_probs=contact_p,
                mean_pae=5.0, has_clash=False,
            )

        pipe.diffusion_fn = _diff_v2
        coords, zij = _random_inputs(n=n)
        result = pipe.step(coords, coords, zij, prev_rmsd=0.0)
        assert result.contact_of3 is not None
        assert -1.0 <= result.contact_of3 <= 1.0


class TestConfidenceV2RgComputation:
    """Test Rg ratio computation."""

    def test_rg_ratio_always_computed(self):
        """Rg ratio should be computed even without diffusion."""
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.step(coords, coords, zij, prev_rmsd=0.0)
        assert result.rg_ratio is not None
        assert result.rg_ratio > 0

    def test_rg_ratio_scaled_protein(self):
        """For realistic-scale coords, Rg ratio should be near 1.0."""
        pipe = _make_pipeline(n_steps=1)
        n = 50
        torch.manual_seed(99)
        # Create a chain-like structure with realistic Rg
        coords = torch.zeros(n, 3)
        for i in range(1, n):
            coords[i] = coords[i-1] + torch.randn(3) * 3.8  # ~3.8A CA spacing
        zij = torch.randn(n, n, 128)
        result = pipe.step(coords, coords, zij, prev_rmsd=0.0)
        # Rg_exp for N=50: 2.2 * 50^0.38 ≈ 9.7 Å
        # Rg_obs for random walk: sqrt(n*3.8^2/6) ≈ 17 → ratio ~1.7
        assert 0.3 < result.rg_ratio < 5.0


class TestSampleConsistency:
    """Test _compute_sample_consistency from of3_diffusion."""

    def test_single_sample_returns_none(self):
        from src.of3_diffusion import _compute_sample_consistency
        all_ca = torch.randn(1, 20, 3)
        rmsd, rmsf, cons = _compute_sample_consistency(all_ca)
        assert rmsd is None
        assert rmsf is None
        assert cons is None

    def test_identical_samples_perfect_consensus(self):
        from src.of3_diffusion import _compute_sample_consistency
        ca = torch.randn(1, 20, 3)
        all_ca = ca.repeat(3, 1, 1)  # 3 identical samples
        rmsd, rmsf, cons = _compute_sample_consistency(all_ca)
        assert cons == pytest.approx(1.0, abs=1e-5)
        assert rmsd is not None
        assert float(rmsd.max()) < 1e-5
        assert rmsf is not None
        assert float(rmsf.max()) < 1e-5

    def test_diverse_samples_low_consensus(self):
        from src.of3_diffusion import _compute_sample_consistency
        all_ca = torch.randn(5, 20, 3) * 10.0  # very diverse
        rmsd, rmsf, cons = _compute_sample_consistency(all_ca)
        assert cons is not None
        assert cons < 0.5  # low consensus expected
        assert rmsd.shape[0] == 10  # C(5,2) = 10 pairs
        assert rmsf.shape[0] == 20  # per-residue


class TestExtractHelpers:
    """Test V2 helper functions in of3_diffusion."""

    def test_extract_pae_none(self):
        from src.of3_diffusion import _extract_pae
        pae, mean = _extract_pae(None)
        assert pae is None
        assert mean is None

    def test_extract_pae_missing_key(self):
        from src.of3_diffusion import _extract_pae
        pae, mean = _extract_pae({"other": 1.0})
        assert pae is None

    def test_extract_pae_3d(self):
        from src.of3_diffusion import _extract_pae
        raw = torch.rand(1, 3, 10, 10)  # [1, K=3, N, N]
        pae, mean = _extract_pae({"pae": raw}, best_idx=1)
        assert pae.shape == (10, 10)
        assert mean == pytest.approx(float(raw[0, 1].mean()), abs=1e-4)

    def test_extract_pae_2d(self):
        from src.of3_diffusion import _extract_pae
        raw = torch.rand(1, 10, 10)  # [1, N, N]
        pae, mean = _extract_pae({"pae": raw})
        assert pae.shape == (10, 10)

    def test_extract_contact_probs_none(self):
        from src.of3_diffusion import _extract_contact_probs
        assert _extract_contact_probs(None) is None

    def test_extract_contact_probs(self):
        from src.of3_diffusion import _extract_contact_probs
        cp = torch.rand(1, 10, 10)
        result = _extract_contact_probs({"contact_probs": cp})
        assert result.shape == (10, 10)

    def test_extract_has_clash_none(self):
        from src.of3_diffusion import _extract_has_clash
        assert _extract_has_clash(None) is None

    def test_extract_has_clash_tensor(self):
        from src.of3_diffusion import _extract_has_clash
        assert _extract_has_clash({"has_clash": torch.tensor(True)}) is True
        assert _extract_has_clash({"has_clash": torch.tensor(False)}) is False

    def test_extract_has_clash_bool(self):
        from src.of3_diffusion import _extract_has_clash
        assert _extract_has_clash({"has_clash": True}) is True
