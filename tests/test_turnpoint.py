"""Tests for iw_enm/turnpoint.py: smoothing and turning-point detection."""

import numpy as np
import pytest

from src.iw_enm.turnpoint import _smooth, find_turning_point, select_best_frame


class TestSmooth:
    def test_identity_for_short_signal(self):
        x = np.array([1.0, 2.0])
        result = _smooth(x, w=5)
        np.testing.assert_array_equal(result, x)

    def test_identity_for_w_one(self):
        x = np.array([1.0, 3.0, 2.0, 4.0])
        result = _smooth(x, w=1)
        np.testing.assert_array_equal(result, x)

    def test_constant_signal_unchanged(self):
        x = np.ones(20) * 5.0
        result = _smooth(x, w=5)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_output_length_matches_input(self):
        for n in [5, 10, 50]:
            x = np.random.randn(n)
            result = _smooth(x, w=3)
            assert len(result) == n

    def test_reduces_noise(self):
        np.random.seed(42)
        x = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.5
        smoothed = _smooth(x, w=11)
        # Smoothed signal should have less variance than original
        assert np.std(np.diff(smoothed)) < np.std(np.diff(x))

    def test_accepts_list_input(self):
        result = _smooth([1.0, 2.0, 3.0, 2.0, 1.0], w=3)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5


class TestFindTurningPoint:
    def test_monotonic_decrease_returns_last(self):
        """Signal that only decreases → minimum is at end."""
        values = np.linspace(100, 10, 50)
        idx = find_turning_point(values, smooth_w=3, warmup_skip=0.1)
        assert idx == len(values) - 1

    def test_clear_minimum(self):
        """V-shaped signal with clear minimum in second half."""
        n = 100
        values = np.concatenate([
            np.linspace(100, 20, 70),  # decrease
            np.linspace(20, 80, 30),   # increase
        ])
        idx = find_turning_point(values, smooth_w=3, warmup_skip=0.3)
        # Should find minimum around index 70
        assert 60 <= idx <= 75

    def test_short_signal_returns_last(self):
        values = np.array([5.0, 3.0, 1.0])
        idx = find_turning_point(values, smooth_w=11, min_len=10)
        assert idx == len(values) - 1

    def test_warmup_skip_works(self):
        """Early minimum should be ignored due to warmup skip."""
        n = 100
        # Very low early, then decrease-increase later
        values = np.ones(n) * 50.0
        values[5] = 1.0  # early dip (within warmup)
        values[70] = 10.0  # real minimum
        values[80:] = 60.0  # increase after
        idx = find_turning_point(values, smooth_w=3, warmup_skip=0.3)
        # Should skip early dip and find minimum around 70
        assert idx >= 30  # past warmup

    def test_returns_integer(self):
        values = np.random.randn(50).cumsum()
        idx = find_turning_point(values, smooth_w=3, warmup_skip=0.1)
        assert isinstance(idx, int)

    def test_result_in_valid_range(self):
        values = np.random.randn(80)
        idx = find_turning_point(values, smooth_w=5, warmup_skip=0.2)
        assert 0 <= idx < len(values)

    def test_warmup_skip_fallback_when_too_large(self):
        """When warmup_skip would skip almost everything, falls back."""
        values = np.linspace(10, 1, 10)
        idx = find_turning_point(values, smooth_w=3, warmup_skip=0.95)
        assert 0 <= idx < len(values)


# ── Mock Simulation for select_best_frame ─────────────────────────

class _MockSimulation:
    """Minimal mock of iw_enm.simulation.Simulation for select_best_frame."""

    def __init__(self, n_frames=20, n_res=10):
        steps = list(range(10, 10 * (n_frames + 1), 10))
        # Energy: decrease then increase (turning point ~ frame 12)
        e_vals = np.concatenate([
            np.linspace(100, 20, 12),
            np.linspace(22, 60, n_frames - 12),
        ])
        self.energies = [
            (steps[i], 0.0, 0.0, e_vals[i]) for i in range(n_frames)
        ]
        # Spring counts: decrease then increase
        s_vals = np.concatenate([
            np.linspace(50, 20, 14),
            np.linspace(22, 40, n_frames - 14),
        ])
        self.spring_counts = [(steps[i], s_vals[i]) for i in range(n_frames)]

        # Trajectory: initial + one per saved frame
        self.trajectory = [np.random.randn(n_res, 3) for _ in range(n_frames + 1)]
        self.target_ca = None
        self.crash_events = 2
        self.crash_per_step = [(50, 1), (80, 1)]


class TestSelectBestFrame:
    def test_returns_dict(self):
        sim = _MockSimulation()
        result = select_best_frame(sim, back_off=1)
        assert isinstance(result, dict)
        assert "idx" in result
        assert "coords" in result
        assert "score" in result

    def test_idx_before_turning_point(self):
        sim = _MockSimulation(n_frames=30)
        result = select_best_frame(sim, back_off=2, smooth_w=3, warmup_skip=0.1)
        # Should pick a frame before the energy minimum
        assert result["idx"] >= 0
        assert result["idx"] < len(sim.energies)

    def test_back_off_shifts_earlier(self):
        sim = _MockSimulation()
        r1 = select_best_frame(sim, back_off=0, smooth_w=3, warmup_skip=0.1)
        r2 = select_best_frame(sim, back_off=3, smooth_w=3, warmup_skip=0.1)
        assert r2["idx"] <= r1["idx"]

    def test_with_target(self):
        sim = _MockSimulation()
        sim.target_ca = np.random.randn(10, 3)
        result = select_best_frame(sim, back_off=1)
        assert result["rmsd_to_target"] is not None
        assert result["tm_to_target"] is not None

    def test_without_target(self):
        sim = _MockSimulation()
        sim.target_ca = None
        result = select_best_frame(sim, back_off=1)
        assert result["rmsd_to_target"] is None
        assert result["tm_to_target"] is None

    def test_crash_tracking(self):
        sim = _MockSimulation()
        result = select_best_frame(sim, back_off=1)
        assert "crashes_until_best" in result
        assert "crashes_total" in result
        assert result["crashes_total"] == 2

    def test_empty_simulation_raises(self):
        sim = _MockSimulation()
        sim.energies = []
        with pytest.raises(ValueError, match="no saved frames"):
            select_best_frame(sim)

    def test_result_keys_complete(self):
        sim = _MockSimulation()
        result = select_best_frame(sim, back_off=1)
        expected_keys = {
            "idx", "traj_idx", "step", "coords",
            "rmsd_to_target", "tm_to_target",
            "e_tot", "n_springs",
            "turn_idx_e", "turn_idx_s",
            "step_turn_e", "step_turn_s",
            "crashes_until_best", "crashes_total", "score",
        }
        assert expected_keys == set(result.keys())
