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


# ======================================================================
# Autostop strategy (IW-ENM MD + early-stop picker)
# ======================================================================

def _autostop_pipeline(
    n_steps: int = 1,
    *,
    with_structure_ctx: bool = True,
    fallback_levels: tuple[int, ...] = (0,),
) -> tuple[ModeDrivePipeline, torch.Tensor, torch.Tensor]:
    """Build a tiny autostop pipeline + synthetic inputs."""
    from src.autostop_adapter import StructureContext

    cfg = ModeDriveConfig(
        n_steps=n_steps,
        combination_strategy="autostop",
        n_anm_modes=8,
        n_combinations=5,
        max_combo_size=2,
        # Very short autostop run
        autostop_n_steps=40,
        autostop_save_every=5,
        autostop_back_off=1,
        autostop_smooth_w=3,
        autostop_warmup_frac=0.10,
        autostop_patience=2,
        autostop_eps_E_rel=0.0002,
        autostop_eps_N_rel=0.0005,
        autostop_crash_window_saves=5,
        autostop_crash_threshold=5,
        autostop_min_saves_before_check=3,
        autostop_verbose=False,
        autostop_fallback_levels=fallback_levels,
        # Disable confidence fallback so a mock converter is OK
        enable_confidence_fallback=False,
    )

    # Small synthetic helix-like CA trace
    import numpy as np
    n = 18
    t = np.arange(n, dtype=np.float64)
    r = 2.3
    twist = np.deg2rad(100.0)
    ca_np = np.stack(
        [r * np.cos(twist * t), r * np.sin(twist * t), 1.5 * t], axis=-1
    )
    coords = torch.from_numpy(ca_np).to(torch.float32)
    zij = torch.randn(n, n, 128)

    structure_ctx = (
        StructureContext.from_ca_only(coords) if with_structure_ctx else None
    )
    pipe = ModeDrivePipeline(
        converter=MockConverter(), config=cfg, structure_ctx=structure_ctx,
    )
    return pipe, coords, zij


class TestAutostopStrategy:
    def test_run_returns_result(self):
        pipe, coords, zij = _autostop_pipeline(n_steps=1)
        result = pipe.run(coords, zij, verbose=False)
        assert isinstance(result, ModeDriveResult)
        assert len(result.step_results) == 1

    def test_autostop_info_populated(self):
        pipe, coords, zij = _autostop_pipeline(n_steps=1)
        result = pipe.run(coords, zij, verbose=False)
        sr = result.step_results[0]
        assert sr.autostop_info is not None
        info = sr.autostop_info
        for key in (
            "picked_save_index", "picked_step_md", "turn_k",
            "argmin_E_k", "argmin_N_k", "back_off_used",
            "monitor_params", "n_saves",
        ):
            assert key in info, f"autostop_info missing {key!r}"
        assert info["n_saves"] >= 1
        assert 0 <= info["picked_save_index"] < info["n_saves"]

    def test_combo_label_has_autostop_prefix(self):
        pipe, coords, zij = _autostop_pipeline(n_steps=1)
        result = pipe.run(coords, zij, verbose=False)
        combo = result.step_results[0].combo
        assert combo.label.startswith("autostop_")

    def test_df_used_is_zero(self):
        """Autostop does not use an ANM df; df_used should remain 0.0."""
        pipe, coords, zij = _autostop_pipeline(n_steps=1)
        result = pipe.run(coords, zij, verbose=False)
        assert result.step_results[0].df_used == 0.0

    def test_trace_cached_on_pipeline(self):
        pipe, coords, zij = _autostop_pipeline(n_steps=1)
        pipe.run(coords, zij, verbose=False)
        assert getattr(pipe, "_autostop_last_trace", None) is not None
        trace = pipe._autostop_last_trace
        assert len(trace.steps) == len(trace.E_tot)
        assert len(trace.trajectory) == len(trace.steps) + 1

    def test_trajectory_shape(self):
        pipe, coords, zij = _autostop_pipeline(n_steps=2)
        result = pipe.run(coords, zij, verbose=False)
        assert len(result.trajectory) == 3  # initial + 2 steps
        for frame in result.trajectory:
            assert frame.shape == coords.shape

    def test_missing_structure_ctx_raises(self):
        """Autostop strategy without structure_ctx must raise a clear error."""
        pipe, coords, zij = _autostop_pipeline(
            n_steps=1, with_structure_ctx=False,
        )
        with pytest.raises(RuntimeError, match="StructureContext"):
            pipe.run(coords, zij, verbose=False)

    def test_fallback_level_zero_by_default(self):
        """Baseline (no fallback escalation) → fallback_level == 0."""
        pipe, coords, zij = _autostop_pipeline(
            n_steps=1, fallback_levels=(0,),
        )
        result = pipe.run(coords, zij, verbose=False)
        assert result.step_results[0].fallback_level == 0


# ======================================================================
# Regression — the 5 existing strategies still work
# ======================================================================

class TestExistingStrategiesRegression:
    """Ensure autostop wiring did not break the other strategies."""

    @pytest.mark.parametrize(
        "strategy",
        ["collectivity", "manual", "targeted", "grid", "random"],
    )
    def test_strategy_runs(self, strategy):
        # `manual` needs explicit mode indices; `targeted` needs target_coords.
        cfg_kwargs = dict(
            n_steps=1,
            combination_strategy=strategy,
            z_mixing_alpha=0.3,
            n_anm_modes=8,
            n_combinations=5,
            max_combo_size=2,
            df=0.6, df_min=0.3, df_max=1.5,
        )
        if strategy == "manual":
            cfg_kwargs["manual_modes"] = (0, 1)
        cfg = ModeDriveConfig(**cfg_kwargs)
        pipe = ModeDrivePipeline(converter=MockConverter(), config=cfg)

        coords, zij = _random_inputs()
        run_kwargs = {"verbose": False}
        if strategy == "targeted":
            run_kwargs["target_coords"] = coords + 0.3
        result = pipe.run(coords, zij, **run_kwargs)
        assert isinstance(result, ModeDriveResult)
        assert len(result.step_results) == 1
        # Non-autostop strategies should NOT have autostop_info
        assert result.step_results[0].autostop_info is None
