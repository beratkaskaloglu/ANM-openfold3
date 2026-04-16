"""High-level one-shot runner: build Simulation with given params and run it.

Used by both surrogate-to-sim validation and turnpoint demos.
"""

import numpy as np

from ..structure import ProteinStructure
from ..network import InteractionWeightedENM
from ..integrator import VelocityVerletIntegrator
from ..simulation import Simulation
from ..config import SimulationConfig
from ..analysis import compute_rmsd_aligned
from .grid import DEFAULT_FIXED


def build_simulation(pdb_path, target_path, params, *,
                     chain_id="A", fixed_params=None,
                     output_prefix="run"):
    """Load structures, build Simulation with `params` merged over fixed defaults."""
    fixed = dict(fixed_params or DEFAULT_FIXED)
    fixed.update(params)
    fixed.setdefault("chain_id", chain_id)

    structure = ProteinStructure.from_pdb(pdb_path, chain_id=chain_id)
    if target_path.lower().endswith(".cif"):
        target = ProteinStructure.from_cif(target_path, chain_id=chain_id)
    else:
        target = ProteinStructure.from_pdb(target_path, chain_id=chain_id)
    target_ca = target.coords_ca
    baseline = compute_rmsd_aligned(structure.coords_ca, target_ca)

    cfg = SimulationConfig(output_prefix=output_prefix, **fixed)
    enm = InteractionWeightedENM(
        R_bb=cfg.R_bb, R_sc=cfg.R_sc, K_0=cfg.K_0,
        d_0=cfg.d_0, n_ref=cfg.n_ref,
    )
    integrator = VelocityVerletIntegrator(
        mass=cfg.mass, dt=cfg.dt, damping=cfg.damping,
    )
    sim = Simulation(structure, enm, integrator, cfg, target_ca=target_ca)
    return sim, baseline


def run_with_params(pdb_path, target_path, params, *,
                    chain_id="A", fixed_params=None,
                    output_prefix="run", verbose=False):
    """Build + run a single simulation, returning (sim, baseline)."""
    sim, baseline = build_simulation(
        pdb_path, target_path, params,
        chain_id=chain_id, fixed_params=fixed_params,
        output_prefix=output_prefix,
    )
    if verbose:
        print(f"Baseline: {baseline:.3f} Å", flush=True)
    else:
        import sys, io
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sim.run()
        finally:
            sys.stdout = old
        return sim, baseline
    sim.run()
    return sim, baseline
