"""Composite scoring function shared by surrogate and turnpoint modules."""

import numpy as np


# Default composite-loss weights (tune per experiment).
DEFAULT_WEIGHTS = {
    "alpha": 0.30,  # crash penalty weight       (log(1+crashes))
    "beta":  0.01,  # positive-drift penalty     (max(0, E_drift))
    "gamma": 0.10,  # plateau-noise penalty      (log(1+E_plateau_std))
}


def composite_loss(delta_rmsd, crash_events, e_drift, e_plateau_std,
                   alpha=None, beta=None, gamma=None):
    """Combined scalar loss (LOWER = BETTER).

        loss = -delta_rmsd
             + alpha * log(1 + max(0, crashes))
             + beta  * max(0, e_drift)
             + gamma * log(1 + plateau_std)

    Inputs can be scalars or numpy arrays of equal length.
    """
    a = DEFAULT_WEIGHTS["alpha"] if alpha is None else alpha
    b = DEFAULT_WEIGHTS["beta"]  if beta  is None else beta
    g = DEFAULT_WEIGHTS["gamma"] if gamma is None else gamma

    delta = np.asarray(delta_rmsd, dtype=np.float32)
    crash = np.maximum(np.asarray(crash_events, dtype=np.float32), 0.0)
    drift = np.asarray(e_drift, dtype=np.float32)
    pstd  = np.asarray(e_plateau_std, dtype=np.float32)

    return (
        -delta
        + a * np.log1p(crash)
        + b * np.maximum(drift, 0.0)
        + g * np.log1p(pstd)
    )
