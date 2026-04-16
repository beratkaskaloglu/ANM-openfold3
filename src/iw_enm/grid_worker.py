"""Grid search worker function (must be importable module for multiprocessing)."""

import sys
import io
import numpy as np

from .config import SimulationConfig
from .structure import ProteinStructure
from .network import InteractionWeightedENM
from .integrator import VelocityVerletIntegrator
from .simulation import Simulation
from .analysis import compute_rmsd_aligned, compute_tm_score


def grid_worker_tuple(args):
    """Wrapper so imap_unordered can pass a single tuple."""
    return grid_worker(*args)


def grid_worker(combo, keys, fixed_params, initial_ca, initial_cb,
                initial_res_names, initial_res_ids, initial_chain_ids,
                target_ca, baseline_rmsd):
    """Run a single grid point in a separate process."""
    params = dict(zip(keys, combo))

    try:
        structure = ProteinStructure(
            initial_ca, initial_cb, initial_res_names,
            initial_res_ids, initial_chain_ids,
        )

        cfg = SimulationConfig(output_prefix="gs", **{**fixed_params, **params})
        enm = InteractionWeightedENM(
            R_bb=cfg.R_bb, R_sc=cfg.R_sc, K_0=cfg.K_0,
            d_0=cfg.d_0, n_ref=cfg.n_ref,
        )
        integrator = VelocityVerletIntegrator(
            mass=cfg.mass, dt=cfg.dt, damping=cfg.damping,
        )
        sim = Simulation(structure, enm, integrator, cfg)

        # Suppress stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sim.run()
        finally:
            sys.stdout = old_stdout

        rmsds = [compute_rmsd_aligned(f, target_ca) for f in sim.trajectory]
        tms = [compute_tm_score(f, target_ca) for f in sim.trajectory]

        # Energy features
        e_pot = np.array([e[1] for e in sim.energies], dtype=np.float64)
        e_kin = np.array([e[2] for e in sim.energies], dtype=np.float64)
        e_tot = np.array([e[3] for e in sim.energies], dtype=np.float64)
        # Drift: last 20% mean − first 20% mean (plateau metric)
        k = max(1, len(e_tot) // 5)
        e_drift = float(e_tot[-k:].mean() - e_tot[:k].mean())
        # Plateau quality: std of last 20% of E_tot (small = flat)
        e_plateau_std = float(e_tot[-k:].std()) if len(e_tot) > 1 else 0.0

        return {
            "params": params,
            "min_rmsd": min(rmsds),
            "final_rmsd": rmsds[-1],
            "best_tm": max(tms),
            "final_tm": tms[-1],
            "delta_rmsd": baseline_rmsd - min(rmsds),
            "rmsd_trace": rmsds,
            "tm_trace": tms,
            # Crash metrics
            "crash_events": int(sim.crash_events),
            "crashed_pairs": int(len(sim.crashed_pairs)),
            "min_sc_distance": float(sim.min_sc_distance),
            "crash_threshold": float(cfg.crash_threshold),
            # Energy metrics
            "e_drift": e_drift,
            "e_tot_init": float(e_tot[0]),
            "e_tot_final": float(e_tot[-1]),
            "e_kin_max": float(e_kin.max()),
            "e_plateau_std": e_plateau_std,
        }
    except Exception as e:
        return {
            "params": params,
            "min_rmsd": 999,
            "final_rmsd": 999,
            "best_tm": 0,
            "final_tm": 0,
            "delta_rmsd": -999,
            "crash_events": -1,
            "crashed_pairs": -1,
            "min_sc_distance": -1,
            "e_drift": 999,
            "e_tot_init": 0,
            "e_tot_final": 999,
            "e_kin_max": 999,
            "e_plateau_std": 999,
            "error": str(e),
        }
