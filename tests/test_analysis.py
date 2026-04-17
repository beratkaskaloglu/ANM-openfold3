"""Tests for iw_enm/analysis.py: numpy-based RMSD, RMSF, TM-score, Kabsch."""

import numpy as np
import pytest

from src.iw_enm.analysis import (
    compute_kinetic_energy,
    compute_rmsd,
    compute_rmsf,
    compute_bfactors,
    kabsch_align,
    compute_rmsd_aligned,
    compute_tm_score,
)


class TestKineticEnergy:
    def test_zero_velocity(self):
        v = np.zeros((10, 3))
        assert compute_kinetic_energy(v) == 0.0

    def test_unit_velocity(self):
        v = np.ones((1, 3))
        # 0.5 * 1.0 * (1+1+1) = 1.5
        assert np.isclose(compute_kinetic_energy(v, mass=1.0), 1.5)

    def test_mass_scaling(self):
        v = np.ones((1, 3))
        e1 = compute_kinetic_energy(v, mass=1.0)
        e2 = compute_kinetic_energy(v, mass=2.0)
        assert np.isclose(e2, 2 * e1)


class TestRMSD:
    def test_identical_coords(self):
        c = np.random.randn(10, 3)
        assert np.isclose(compute_rmsd(c, c), 0.0)

    def test_known_shift(self):
        c = np.zeros((4, 3))
        r = np.ones((4, 3))
        rmsd = compute_rmsd(c, r)
        expected = np.sqrt(3.0)  # sqrt(mean(3*1^2))
        assert np.isclose(rmsd, expected)

    def test_positive(self):
        c1 = np.random.randn(20, 3)
        c2 = np.random.randn(20, 3)
        assert compute_rmsd(c1, c2) > 0


class TestRMSF:
    def test_static_trajectory(self):
        frame = np.random.randn(10, 3)
        traj = [frame.copy() for _ in range(5)]
        rmsf = compute_rmsf(traj)
        np.testing.assert_allclose(rmsf, 0.0, atol=1e-10)

    def test_shape(self):
        traj = [np.random.randn(8, 3) for _ in range(10)]
        rmsf = compute_rmsf(traj)
        assert rmsf.shape == (8,)

    def test_all_positive_with_motion(self):
        traj = [np.random.randn(5, 3) * (i + 1) for i in range(5)]
        rmsf = compute_rmsf(traj)
        assert (rmsf > 0).all()

    def test_custom_ref(self):
        traj = [np.random.randn(5, 3) for _ in range(4)]
        ref = np.zeros((5, 3))
        rmsf = compute_rmsf(traj, ref_coords=ref)
        assert rmsf.shape == (5,)


class TestBFactors:
    def test_zero_rmsf(self):
        rmsf = np.zeros(10)
        bf = compute_bfactors(rmsf)
        np.testing.assert_allclose(bf, 0.0)

    def test_proportional(self):
        rmsf = np.array([1.0, 2.0])
        bf = compute_bfactors(rmsf)
        # B ~ RMSF^2
        assert bf[1] / bf[0] == pytest.approx(4.0)


class TestKabschAlign:
    def test_identity(self):
        c = np.random.randn(10, 3)
        aligned, R, _ = kabsch_align(c, c)
        np.testing.assert_allclose(aligned, c, atol=1e-10)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_translation_only(self):
        c = np.random.randn(10, 3)
        shifted = c + np.array([5.0, -3.0, 2.0])
        aligned, _, _ = kabsch_align(shifted, c)
        np.testing.assert_allclose(aligned, c, atol=1e-10)

    def test_rotation(self):
        """90-degree rotation around z-axis."""
        c = np.array([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        # Rotate 90° around z
        R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = (c - c.mean(0)) @ R90.T + c.mean(0)
        aligned, _, _ = kabsch_align(rotated, c)
        np.testing.assert_allclose(aligned, c, atol=1e-8)


class TestRMSDAligned:
    def test_same_structure(self):
        c = np.random.randn(15, 3)
        assert np.isclose(compute_rmsd_aligned(c, c), 0.0, atol=1e-10)

    def test_translated_structure(self):
        c = np.random.randn(15, 3)
        shifted = c + 100.0
        assert np.isclose(compute_rmsd_aligned(shifted, c), 0.0, atol=1e-8)


class TestTMScore:
    def test_identical(self):
        c = np.random.randn(50, 3)
        tm = compute_tm_score(c, c)
        assert np.isclose(tm, 1.0, atol=1e-6)

    def test_range_zero_one(self):
        c1 = np.random.randn(30, 3) * 5
        c2 = np.random.randn(30, 3) * 5
        tm = compute_tm_score(c1, c2)
        assert 0.0 < tm <= 1.0

    def test_custom_L(self):
        c = np.random.randn(20, 3)
        tm = compute_tm_score(c, c, L=20)
        assert np.isclose(tm, 1.0, atol=1e-6)

    def test_short_chain_clamps_d0(self):
        """Very short chain should clamp d0 to 0.5."""
        c = np.random.randn(5, 3) * 0.1
        tm = compute_tm_score(c, c)
        assert tm > 0
