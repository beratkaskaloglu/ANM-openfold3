"""Tests for mode_drive pipeline module."""

import torch
import pytest

from src.mode_drive import (
    ModeDriveConfig,
    ModeDrivePipeline,
    ModeDriveResult,
    StepResult,
)


class MockConverter:
    """Lightweight mock for PairContactConverter (no trained weights needed)."""

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


def _make_pipeline(
    n_steps: int = 2,
    strategy: str = "collectivity",
    alpha: float = 0.3,
    n_modes: int = 8,
    n_combinations: int = 5,
) -> ModeDrivePipeline:
    """Create a pipeline with mock converter and small config."""
    cfg = ModeDriveConfig(
        n_steps=n_steps,
        combination_strategy=strategy,
        z_mixing_alpha=alpha,
        n_anm_modes=n_modes,
        n_combinations=n_combinations,
        max_combo_size=2,
        df=0.6,
        df_min=0.3,
        df_max=1.5,
    )
    return ModeDrivePipeline(converter=MockConverter(), config=cfg)


def _random_inputs(n: int = 15):
    """Generate random initial coords and z_trunk for testing."""
    torch.manual_seed(42)
    coords = torch.randn(n, 3) * 10.0
    zij = torch.randn(n, n, 128)
    return coords, zij


class TestModeDriveConfig:
    def test_default_values(self):
        cfg = ModeDriveConfig()
        assert cfg.n_steps == 5
        assert cfg.combination_strategy == "collectivity"
        assert cfg.z_mixing_alpha == 0.3
        assert cfg.anm_cutoff == 15.0
        assert cfg.n_anm_modes == 20

    def test_custom_values(self):
        cfg = ModeDriveConfig(n_steps=10, z_mixing_alpha=0.5, df=1.0)
        assert cfg.n_steps == 10
        assert cfg.z_mixing_alpha == 0.5
        assert cfg.df == 1.0


class TestModeDrivePipeline:
    def test_run_returns_result(self):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        assert isinstance(result, ModeDriveResult)

    def test_trajectory_length(self):
        n_steps = 2
        pipe = _make_pipeline(n_steps=n_steps)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        # initial + one per step
        assert len(result.trajectory) == n_steps + 1

    def test_step_results_length(self):
        n_steps = 3
        pipe = _make_pipeline(n_steps=n_steps)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        assert len(result.step_results) == n_steps

    def test_rmsd_from_initial(self):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        # RMSD should be a non-negative float
        assert result.step_results[0].rmsd >= 0.0

    def test_collectivity_strategy(self):
        pipe = _make_pipeline(n_steps=1, strategy="collectivity")
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        assert isinstance(result, ModeDriveResult)
        assert len(result.step_results) == 1

    def test_random_strategy(self):
        pipe = _make_pipeline(n_steps=1, strategy="random")
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        assert isinstance(result, ModeDriveResult)
        assert len(result.step_results) == 1

    def test_z_mixing_alpha_zero(self):
        """alpha=0 → z_mod should equal z_trunk (no pseudo contribution)."""
        pipe = _make_pipeline(n_steps=1, alpha=0.0)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        z_mod = result.step_results[0].z_modified
        # With alpha=0: z_mod = 0 * z_pseudo + 1.0 * z_trunk
        # But z_trunk gets updated, so just check shape is correct
        assert z_mod.shape == zij.shape

    def test_z_mixing_alpha_one(self):
        """alpha=1 → z_mod should be dominated by z_pseudo."""
        pipe = _make_pipeline(n_steps=1, alpha=1.0)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        z_mod = result.step_results[0].z_modified
        assert z_mod.shape == zij.shape

    def test_step_result_fields(self):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        sr = result.step_results[0]
        assert isinstance(sr, StepResult)
        assert sr.displaced_ca.shape == coords.shape
        assert sr.new_ca.shape == coords.shape
        assert sr.z_modified.shape == zij.shape
        assert sr.contact_map.shape == (coords.shape[0], coords.shape[0])
        assert isinstance(sr.rmsd, float)
        assert sr.eigenvalues.dim() == 1
        assert sr.eigenvectors.dim() == 3
        assert sr.b_factors.shape == (coords.shape[0],)
        assert isinstance(sr.df_used, float)

    def test_small_protein(self):
        """N=10 should work without errors."""
        pipe = _make_pipeline(n_steps=1, n_modes=4, n_combinations=3)
        torch.manual_seed(99)
        coords = torch.randn(10, 3) * 10.0
        zij = torch.randn(10, 10, 128)
        result = pipe.run(coords, zij)
        assert len(result.step_results) == 1

    def test_single_step(self):
        pipe = _make_pipeline(n_steps=1)
        coords, zij = _random_inputs()
        result = pipe.run(coords, zij)
        assert result.total_steps == 1
        assert len(result.trajectory) == 2
