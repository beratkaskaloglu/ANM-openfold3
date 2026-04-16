"""Turning-point based best-frame selector.

Insight: in IW-ENM trajectories, E_total and spring-count both DECREASE while
the structure is converging toward the target conformation. When either signal
reverses (from decreasing to increasing), the structure starts deteriorating
and RMSD to target begins to rise. This module detects that reversal and
returns the frame just before it as the "model" output.
"""

import numpy as np

from .analysis import compute_rmsd_aligned, compute_tm_score


def _smooth(x, w=3):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < w or w <= 1:
        return x
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def find_turning_point(values, smooth_w=11, warmup_skip=0.30, min_len=10):
    """Find the strongest minimum of a heavily-smoothed signal, after skipping
    the initial warmup transient.

    Early IW-ENM steps always drop E_tot/n_springs as the system relaxes
    kinetic→potential — this is NOT the meaningful turning point. The real
    reversal we care about happens later, after the structure has converged
    near the target. We therefore:
      (1) skip the first `warmup_skip` fraction of saved frames
      (2) smooth with a larger window to suppress oscillation
      (3) return the global minimum of the smoothed signal within the
          post-warmup region

    Returns the saved-frame index of the signal's minimum.
    """
    n = len(values)
    if n < max(min_len, smooth_w + 2):
        return n - 1
    s = _smooth(values, smooth_w)
    skip = max(1, int(n * warmup_skip))
    if skip >= n - 2:
        skip = n // 3
    idx_rel = int(np.argmin(s[skip:]))
    return skip + idx_rel


def select_best_frame(simulation, back_off=1, weight_crash=0.30,
                      smooth_w=11, warmup_skip=0.30):
    """Pick the best frame from a completed Simulation using turning-point logic.

    Args:
        simulation: completed iw_enm.simulation.Simulation instance
        back_off: saved-frame count to rewind from the turning point
        weight_crash: penalty coefficient for log(1+crashes) in composite score
        smooth_w: moving-average window for signal smoothing
        patience: # consecutive non-negative diffs required to confirm reversal

    Returns:
        dict with the selected frame's coordinates, step, RMSD/TM to target,
        crash-aware composite score, and diagnostic info.
    """
    if len(simulation.energies) == 0:
        raise ValueError("Simulation has no saved frames.")

    e_tot = np.array([e[3] for e in simulation.energies], dtype=np.float64)
    n_spr = np.array([c[1] for c in simulation.spring_counts], dtype=np.float64)

    idx_e = find_turning_point(e_tot, smooth_w=smooth_w, warmup_skip=warmup_skip)
    idx_s = find_turning_point(n_spr, smooth_w=smooth_w, warmup_skip=warmup_skip)
    # Prefer the earlier of the two minima — structure starts deteriorating
    # as soon as EITHER signal reverses.
    idx_turn = min(idx_e, idx_s)
    idx_best = max(0, idx_turn - back_off)

    # trajectory[0] = initial (pre-step); trajectory[i+1] aligns with energies[i]
    traj_idx = idx_best + 1
    if traj_idx >= len(simulation.trajectory):
        traj_idx = len(simulation.trajectory) - 1
    coords_best = simulation.trajectory[traj_idx]
    step_best = int(simulation.energies[idx_best][0])

    # Score on target if available, otherwise use raw energy
    rmsd_tgt = None
    tm_tgt = None
    if simulation.target_ca is not None:
        rmsd_tgt = float(compute_rmsd_aligned(coords_best, simulation.target_ca))
        tm_tgt = float(compute_tm_score(coords_best, simulation.target_ca))

    # Crashes observed up to step_best (approximate — crash_per_step is per-step)
    crashes_until = sum(
        n for (s, n) in simulation.crash_per_step if s <= step_best
    )

    base = rmsd_tgt if rmsd_tgt is not None else float(e_tot[idx_best])
    score = base + weight_crash * float(np.log1p(crashes_until))

    return {
        "idx": idx_best,
        "traj_idx": traj_idx,
        "step": step_best,
        "coords": coords_best,
        "rmsd_to_target": rmsd_tgt,
        "tm_to_target": tm_tgt,
        "e_tot": float(e_tot[idx_best]),
        "n_springs": int(n_spr[idx_best]),
        "turn_idx_e": int(idx_e),
        "turn_idx_s": int(idx_s),
        "step_turn_e": int(simulation.energies[idx_e][0]),
        "step_turn_s": int(simulation.spring_counts[idx_s][0]),
        "crashes_until_best": int(crashes_until),
        "crashes_total": int(simulation.crash_events),
        "score": float(score),
    }


def export_model_pdb(simulation, path, selected=None, **kwargs):
    """Write the selected best frame as a PDB file."""
    if selected is None:
        selected = select_best_frame(simulation, **kwargs)
    simulation.structure.to_pdb(path, coords=selected["coords"])
    return selected
