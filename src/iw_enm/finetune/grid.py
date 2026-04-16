"""Grid-search runner for IW-ENM hyperparameters."""

import itertools
import multiprocessing as mp
import time

from ..structure import ProteinStructure
from ..analysis import compute_rmsd_aligned
from ..grid_worker import grid_worker_tuple


DEFAULT_GRID = {
    "R_bb":        [9.0, 10.0, 11.0, 12.0],
    "K_0":         [0.6, 0.8, 1.0, 1.3, 1.6],
    "n_ref":       [5.0, 7.0, 10.0],
    "v_magnitude": [0.7, 1.0, 1.3],
}

DEFAULT_FIXED = dict(
    R_sc=2.0, d_0=3.8, dt=0.01, mass=1.0,
    n_steps=5000, save_every=50,
    damping=0.0, v_mode="breathing",
    crash_threshold=1.0, chain_id="A",
)


def run_grid_search(pdb_path, target_path, *,
                    chain_id="A",
                    grid=None, fixed_params=None,
                    n_workers=None, progress_every=5,
                    verbose=True):
    """Run a parallel grid search over ENM hyperparameters.

    Args:
        pdb_path: path to source PDB (e.g. "1AKE.pdb")
        target_path: path to target CIF or PDB (e.g. "4ake.cif")
        chain_id: chain selector
        grid: dict[str, list] — hyperparameter sweep. Defaults to DEFAULT_GRID.
        fixed_params: dict of SimulationConfig params held constant.
                      Defaults to DEFAULT_FIXED.
        n_workers: int or None (auto = min(8, cpu_count))
        progress_every: print a progress line every N completed combos
        verbose: print status lines

    Returns:
        list[dict] — raw grid_worker results (same order as completion, not grid order)
    """
    grid = dict(grid or DEFAULT_GRID)
    fixed_params = dict(fixed_params or DEFAULT_FIXED)

    # Load structures
    structure = ProteinStructure.from_pdb(pdb_path, chain_id=chain_id)
    if target_path.lower().endswith(".cif"):
        target = ProteinStructure.from_cif(target_path, chain_id=chain_id)
    else:
        target = ProteinStructure.from_pdb(target_path, chain_id=chain_id)
    target_ca = target.coords_ca

    baseline = compute_rmsd_aligned(structure.coords_ca, target_ca)
    if verbose:
        print(f"Baseline RMSD ({pdb_path}→{target_path}): {baseline:.3f} Å",
              flush=True)

    # Build combo list
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    n_total = len(combos)

    args_list = [
        (combo, keys, fixed_params,
         structure.coords_ca, structure.coords_cb,
         structure.res_names, structure.res_ids, structure.chain_ids,
         target_ca, baseline)
        for combo in combos
    ]

    if n_workers is None:
        n_workers = min(8, mp.cpu_count())

    if verbose:
        print(f"\n{n_workers} worker x {n_total} combos — starting...",
              flush=True)

    t0 = time.time()
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(n_workers) as pool:
        for i, res in enumerate(
            pool.imap_unordered(grid_worker_tuple, args_list, chunksize=1), 1
        ):
            results.append(res)
            if verbose and (i % progress_every == 0 or i == n_total or i <= 3):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (n_total - i) / rate if rate > 0 else 0
                print(f"  [{i:3d}/{n_total}]  {elapsed:6.1f}s  "
                      f"|  rate={rate:.2f}/s  |  eta {eta:5.1f}s",
                      flush=True)

    if verbose:
        print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)
    return results
