"""Unit tests for src.autostop_adapter.

Covers:
 - StructureContext.from_ca_only / rebuild_from_ca invariants
 - run_autostop_from_tensor: Pick/Trace shapes, device/dtype preservation,
   synthetic-structure regime produces a well-formed trace.
 - replay_monitor parity (same knobs → same pick) and back_off sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.autostop_adapter import (
    AutostopParams,
    AutostopPick,
    AutostopTrace,
    StructureContext,
    replay_monitor,
    run_autostop_from_tensor,
)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def _helix_ca(n: int = 24, seed: int = 0) -> torch.Tensor:
    """Synthetic α-helix-ish CA trace (not a real helix, just spaced points)."""
    rng = np.random.default_rng(seed)
    # Rough helix: radius 2.3, rise 1.5 Å/res, 100° twist — gives ~3.8 Å CA–CA.
    t = np.arange(n, dtype=np.float64)
    r = 2.3
    rise = 1.5
    twist = np.deg2rad(100.0)
    xs = r * np.cos(twist * t)
    ys = r * np.sin(twist * t)
    zs = rise * t
    ca = np.stack([xs, ys, zs], axis=-1)
    # Tiny jitter so packing count isn't degenerate
    ca = ca + 0.02 * rng.standard_normal(ca.shape)
    return torch.from_numpy(ca).to(torch.float32)


def _tiny_params(
    n_steps: int = 60,
    save_every: int = 5,
    warmup_frac: float = 0.10,
    patience: int = 2,
) -> AutostopParams:
    """Small, fast params — ≤ 12 saves, deterministic with a seeded ENM."""
    return AutostopParams(
        R_bb=11.0,
        R_sc=2.0,
        K_0=0.8,
        d_0=3.8,
        n_ref=10.0,
        dt=0.01,
        mass=1.0,
        damping=0.0,
        v_mode="breathing",
        v_magnitude=1.0,
        n_steps=n_steps,
        save_every=save_every,
        back_off=2,
        crash_threshold_distance=0.5,
        smooth_w=3,
        warmup_frac=warmup_frac,
        patience=patience,
        eps_E_rel=0.0002,
        eps_N_rel=0.0005,
        crash_window_saves=5,
        crash_threshold=5,
        min_saves_before_check=3,
        verbose=False,
    )


# ----------------------------------------------------------------------
# StructureContext
# ----------------------------------------------------------------------

class TestStructureContext:
    def test_from_ca_only_shapes(self):
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        assert ctx.struct.N == 20
        assert ctx.struct.coords_ca.shape == (20, 3)
        assert ctx.struct.coords_cb.shape == (20, 3)
        # At least CB per residue
        assert ctx.struct.n_atoms >= 20

    def test_from_ca_only_default_res_names(self):
        ca = _helix_ca(10)
        ctx = StructureContext.from_ca_only(ca)
        assert all(r == "GLY" for r in ctx.struct.res_names)

    def test_from_ca_only_custom_res_names(self):
        ca = _helix_ca(5)
        ctx = StructureContext.from_ca_only(ca, res_names=["ALA"] * 5)
        assert ctx.struct.res_names == ["ALA", "ALA", "ALA", "ALA", "ALA"]

    def test_rebuild_from_ca_preserves_atom_count(self):
        ca0 = _helix_ca(15)
        ctx = StructureContext.from_ca_only(ca0)
        n_atoms_before = ctx.struct.n_atoms

        # Rigid translate
        ca1 = (ca0 + 1.0).numpy().astype(np.float64)
        new_struct = ctx.rebuild_from_ca(ca1)
        assert new_struct.N == ctx.struct.N
        assert new_struct.n_atoms == n_atoms_before
        assert new_struct.coords_ca.shape == (15, 3)

    def test_from_ca_only_accepts_numpy(self):
        """Verify from_ca_only handles np.ndarray inputs too."""
        ca = _helix_ca(12).numpy()
        ctx = StructureContext.from_ca_only(ca)
        assert ctx.struct.N == 12


# ----------------------------------------------------------------------
# run_autostop_from_tensor
# ----------------------------------------------------------------------

class TestRunAutostopFromTensor:
    def test_returns_pick_and_trace(self):
        ca = _helix_ca(18)
        ctx = StructureContext.from_ca_only(ca)
        pick, trace = run_autostop_from_tensor(ca, ctx, _tiny_params())
        assert isinstance(pick, AutostopPick)
        assert isinstance(trace, AutostopTrace)

    def test_picked_ca_shape_matches_input(self):
        ca = _helix_ca(16)
        ctx = StructureContext.from_ca_only(ca)
        pick, _ = run_autostop_from_tensor(ca, ctx, _tiny_params())
        assert pick.picked_ca.shape == ca.shape

    def test_picked_ca_preserves_dtype_and_device(self):
        ca = _helix_ca(16).to(torch.float64)
        ctx = StructureContext.from_ca_only(ca)
        pick, _ = run_autostop_from_tensor(ca, ctx, _tiny_params())
        assert pick.picked_ca.dtype == torch.float64
        assert pick.picked_ca.device == ca.device

    def test_trace_arrays_consistent_length(self):
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        _, trace = run_autostop_from_tensor(ca, ctx, _tiny_params(n_steps=40, save_every=5))
        S = len(trace.steps)
        assert S >= 1
        assert trace.E_tot.shape == (S,)
        assert trace.n_springs.shape == (S,)
        assert trace.crashes_cum_at_save.shape == (S,)
        # trajectory holds initial + S saves
        assert len(trace.trajectory) == S + 1

    def test_trace_steps_are_monotone(self):
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        _, trace = run_autostop_from_tensor(ca, ctx, _tiny_params())
        steps = trace.steps
        assert np.all(np.diff(steps) > 0)  # strictly increasing

    def test_pick_indices_in_range(self):
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        pick, trace = run_autostop_from_tensor(ca, ctx, _tiny_params())
        S = len(trace.steps)
        assert 0 <= pick.picked_save_index < S
        assert 0 <= pick.turn_k < S
        assert 0 <= pick.argmin_E_k < S
        assert 0 <= pick.argmin_N_k < S
        assert pick.back_off_used == 2

    def test_rejects_bad_shape(self):
        ctx = StructureContext.from_ca_only(_helix_ca(10))
        bad = torch.randn(10, 4)  # wrong last dim
        with pytest.raises(ValueError, match=r"\[N, 3\]"):
            run_autostop_from_tensor(bad, ctx, _tiny_params())

    def test_monitor_params_recorded(self):
        ca = _helix_ca(16)
        ctx = StructureContext.from_ca_only(ca)
        params = _tiny_params()
        pick, _ = run_autostop_from_tensor(ca, ctx, params)
        # Monitor snapshot should round-trip the knobs used
        assert pick.monitor_params["smooth_w"] == params.smooth_w
        assert pick.monitor_params["patience"] == params.patience
        assert pick.monitor_params["eps_E_rel"] == pytest.approx(params.eps_E_rel)


# ----------------------------------------------------------------------
# replay_monitor
# ----------------------------------------------------------------------

class TestReplayMonitor:
    def test_parity_same_params(self):
        """Same monitor knobs + back_off → same picked frame."""
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        params = _tiny_params()
        pick_orig, trace = run_autostop_from_tensor(ca, ctx, params)

        pick_replay = replay_monitor(
            trace=trace,
            monitor_params=params.monitor_only(),
            back_off=params.back_off,
            device=pick_orig.picked_ca.device,
            dtype=pick_orig.picked_ca.dtype,
        )
        assert pick_replay.picked_save_index == pick_orig.picked_save_index
        assert pick_replay.picked_step_md == pick_orig.picked_step_md
        assert pick_replay.turn_k == pick_orig.turn_k
        assert pick_replay.argmin_E_k == pick_orig.argmin_E_k
        assert torch.allclose(pick_replay.picked_ca, pick_orig.picked_ca)

    def test_back_off_shifts_pick(self):
        """Larger back_off → earlier or equal picked save index."""
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        params = _tiny_params()
        _, trace = run_autostop_from_tensor(ca, ctx, params)

        pick_small = replay_monitor(trace, params.monitor_only(), back_off=0)
        pick_large = replay_monitor(trace, params.monitor_only(), back_off=4)
        assert pick_large.picked_save_index <= pick_small.picked_save_index
        assert pick_large.back_off_used == 4
        assert pick_small.back_off_used == 0

    def test_no_md_reintegration(self):
        """Replay reuses trace.trajectory — picked_ca must match a cached frame."""
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        params = _tiny_params()
        _, trace = run_autostop_from_tensor(ca, ctx, params)

        pick = replay_monitor(trace, params.monitor_only(), back_off=2)
        traj_idx = min(pick.picked_save_index + 1, len(trace.trajectory) - 1)
        expected = trace.trajectory[traj_idx]
        got = pick.picked_ca.detach().cpu().numpy().astype(np.float64)
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_different_monitor_knobs_produce_valid_pick(self):
        """Swapping monitor knobs must not crash and must yield a valid pick."""
        ca = _helix_ca(20)
        ctx = StructureContext.from_ca_only(ca)
        params = _tiny_params()
        _, trace = run_autostop_from_tensor(ca, ctx, params)

        relaxed = dict(params.monitor_only())
        relaxed["eps_E_rel"] = params.eps_E_rel * 4.0
        relaxed["eps_N_rel"] = params.eps_N_rel * 4.0
        pick = replay_monitor(trace, relaxed, back_off=params.back_off)
        S = len(trace.steps)
        assert 0 <= pick.picked_save_index < S
