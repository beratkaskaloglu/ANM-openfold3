"""Energy, RMSD, RMSF analysis (numpy backend for IW-ENM MD trajectories).

Note: src/mode_drive_utils.py has torch-tensor equivalents of kabsch/RMSD/TM-score
for the Mode-Drive pipeline.  These numpy versions are used by the IW-ENM simulation
and autostop paths which operate on numpy arrays.
"""

import numpy as np


def compute_kinetic_energy(velocities, mass=1.0):
    """E_kin = 0.5 * sum m_i * |v_i|^2"""
    return 0.5 * mass * np.sum(velocities ** 2)


def compute_rmsd(coords, ref_coords):
    """RMSD between two coordinate sets."""
    diff = coords - ref_coords
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_rmsf(trajectory, ref_coords=None):
    """Per-residue RMSF over trajectory."""
    coords_stack = np.array(trajectory)  # (n_frames, N, 3)
    if ref_coords is None:
        ref_coords = coords_stack.mean(axis=0)
    deviations = coords_stack - ref_coords[np.newaxis, :, :]
    msf = np.mean(np.sum(deviations ** 2, axis=2), axis=0)
    return np.sqrt(msf)


def compute_bfactors(rmsf):
    """B_i = (8*pi^2/3) * RMSF_i^2"""
    return (8.0 * np.pi ** 2 / 3.0) * rmsf ** 2


def kabsch_align(mobile, target):
    """
    Kabsch algorithm: align mobile onto target (least-RMSD superposition).
    Returns aligned_mobile, rotation_matrix, translation.
    Both inputs: (N, 3) arrays.
    """
    # Center both
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_c = mobile - mobile_center
    target_c = target - target_center

    # Covariance matrix
    H = mobile_c.T @ target_c

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    # Rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply
    aligned = (mobile - mobile_center) @ R.T + target_center

    return aligned, R, (mobile_center, target_center)


def compute_rmsd_aligned(coords, ref_coords):
    """RMSD after Kabsch alignment."""
    aligned, _, _ = kabsch_align(coords, ref_coords)
    diff = aligned - ref_coords
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_tm_score(coords, ref_coords, L=None):
    """
    TM-score: template modeling score.
    L = length of target protein.
    TM-score ∈ (0, 1], >0.5 means same fold.
    """
    aligned, _, _ = kabsch_align(coords, ref_coords)
    if L is None:
        L = len(ref_coords)
    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(d0, 0.5)
    di = np.sqrt(np.sum((aligned - ref_coords) ** 2, axis=1))
    tm = np.sum(1.0 / (1.0 + (di / d0) ** 2)) / L
    return tm
