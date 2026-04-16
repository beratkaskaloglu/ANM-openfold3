"""Tek-dosya early-stopping IW-ENM runner.

Matematiksel çekirdek
---------------------
Her `save_every` adımda smoothed sinyaller üretilir:

    Ẽ[k] = (1/w) Σ_{j=k-w+1..k} E_tot[j]       (w = smooth_w, replicate-padded)
    Ñ[k] = (1/w) Σ_{j=k-w+1..k} n_spr[j]

Running minima (warmup'tan sonra):
    E_min[k] = min_{S ≤ j ≤ k} Ẽ[j],   argmin_E[k] = arg-min konumu
    N_min[k] = min_{S ≤ j ≤ k} Ñ[j],   argmin_N[k]

Reversal (dönüş doğrulama):
    rev_E[k] = 1  ⇔  Ẽ[k] ≥ E_min[k] + ε_E   ve bu, patience_rev ardışık frame boyu sürer
    rev_N[k] = 1  ⇔  Ñ[k] ≥ N_min[k] · (1+ε_N)      (aynı sürede)

Crash-onset (son W frame penceresinde birikim):
    c_recent[k] = Σ_{step ∈ [step_k - W·save_every, step_k]} crashes_at_step
    crash_trig[k] = 1  ⇔  c_recent[k] ≥ crash_thr

Stop ⇔  rev_E[k] ∧ rev_N[k] ∧ crash_trig[k]
Turnpoint ⇔  i* = min(argmin_E[k*], argmin_N[k*])
Output frame ⇔  trajectory[ max(0, i* - back_off) ]

Kullanım
--------
    python run_autostop.py 1AKE.pdb --target 4ake.cif --out model.pdb \
        --R_bb 11 --K_0 0.8 --n_ref 10 --v_magnitude 1.0

Target CIF/PDB zorunlu değil — sadece final-frame değerlendirmesi için.
"""

import argparse
import sys
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from iw_enm.structure import ProteinStructure
from iw_enm.network import InteractionWeightedENM
from iw_enm.integrator import VelocityVerletIntegrator
from iw_enm.config import SimulationConfig
from iw_enm.analysis import (
    compute_kinetic_energy,
    compute_rmsd_aligned,
    compute_tm_score,
)


# ----------------------------- Utilities -----------------------------

def smooth_window(buf, w):
    """Causal box filter: Ẽ[k] = mean(last w samples). Replicate-pads if short."""
    if len(buf) == 0:
        return 0.0
    take = list(buf)[-w:]
    if len(take) < w:
        take = [take[0]] * (w - len(take)) + take
    return float(np.mean(take))


# --------------------------- Signal Monitor --------------------------

class EarlyStopMonitor:
    """Saves (step, E_tot, n_spr, crashes_cum) on each save and decides stop.

    Stop logic = reversal_E AND reversal_N AND crash_onset, all held for
    `patience` consecutive saves AFTER the warmup fraction has elapsed.
    """

    def __init__(self,
                 total_steps,
                 save_every,
                 smooth_w=11,
                 warmup_frac=0.20,
                 patience=3,
                 eps_E_rel=0.002,      # Ẽ must exceed E_min by 0.2% of |E_min|
                 eps_N_rel=0.005,      # Ñ must exceed N_min by 0.5%
                 crash_window_saves=20,
                 crash_threshold=5,
                 min_saves_before_check=15):
        self.save_every = save_every
        self.smooth_w = smooth_w
        self.patience = patience
        self.eps_E_rel = eps_E_rel
        self.eps_N_rel = eps_N_rel
        self.crash_window_saves = crash_window_saves
        self.crash_threshold = crash_threshold
        self.min_saves_before_check = min_saves_before_check
        self.warmup_saves = max(min_saves_before_check,
                                int((total_steps / save_every) * warmup_frac))

        # Rolling buffers of smoothed values and raw values
        self.steps = []
        self.E_raw = deque(maxlen=4 * smooth_w)
        self.N_raw = deque(maxlen=4 * smooth_w)
        self.E_smooth = []   # smoothed trajectory (one per save)
        self.N_smooth = []
        self.crashes_cum = []   # cumulative crash count at each save

        self.E_min = np.inf
        self.N_min = np.inf
        self.argmin_E = 0
        self.argmin_N = 0

        # Reversal hold counters
        self.hold_E = 0
        self.hold_N = 0
        self.hold_C = 0

        self.stopped = False
        self.stop_reason = None

    def update(self, step, E_tot, n_spr, crashes_cum):
        """Called at each save. Returns True if stop triggered on this save."""
        self.E_raw.append(E_tot)
        self.N_raw.append(n_spr)
        Es = smooth_window(self.E_raw, self.smooth_w)
        Ns = smooth_window(self.N_raw, self.smooth_w)

        k = len(self.steps)
        self.steps.append(step)
        self.E_smooth.append(Es)
        self.N_smooth.append(Ns)
        self.crashes_cum.append(crashes_cum)

        # Running minima (after warmup only)
        if k >= self.warmup_saves:
            if Es < self.E_min:
                self.E_min = Es
                self.argmin_E = k
            if Ns < self.N_min:
                self.N_min = Ns
                self.argmin_N = k

        # Wait until we have enough history past warmup
        if k < self.warmup_saves + self.min_saves_before_check:
            return False

        # --- Reversal tests (require holding for `patience` saves) ---
        eps_E_abs = self.eps_E_rel * max(abs(self.E_min), 1.0)
        reversed_E = Es >= self.E_min + eps_E_abs
        reversed_N = Ns >= self.N_min * (1.0 + self.eps_N_rel)
        self.hold_E = self.hold_E + 1 if reversed_E else 0
        self.hold_N = self.hold_N + 1 if reversed_N else 0

        # --- Crash-onset: recent window accumulation ---
        w = self.crash_window_saves
        lo = max(0, k - w)
        recent_crashes = crashes_cum - self.crashes_cum[lo]
        crash_onset = recent_crashes >= self.crash_threshold
        self.hold_C = self.hold_C + 1 if crash_onset else 0

        # --- Combined trigger ---
        if (self.hold_E >= self.patience and
                self.hold_N >= self.patience and
                self.hold_C >= self.patience):
            self.stopped = True
            self.stop_reason = (
                f"reversal(E:{self.hold_E},N:{self.hold_N}) + "
                f"crash_onset({recent_crashes} in last {w*self.save_every} steps)"
            )
            return True
        return False

    def turnpoint_index(self):
        """Saved-frame index of the real turn (earlier of the two minima)."""
        return min(self.argmin_E, self.argmin_N)


# ---------------------- Streaming runner ----------------------------

def run_with_autostop(pdb_path,
                      target_path=None,
                      params=None,
                      chain_id="A",
                      n_steps=5000,
                      save_every=10,
                      back_off=2,
                      verbose=True,
                      # monitor params
                      smooth_w=11,
                      warmup_frac=0.20,
                      patience=3,
                      eps_E_rel=0.002,
                      eps_N_rel=0.005,
                      crash_window_saves=20,
                      crash_threshold=5):
    """Run one simulation, stop on reversal+crash, return dict with picked frame."""

    params = dict(params or {})
    # Default fixed params (matches iw_enm.finetune.grid.DEFAULT_FIXED)
    fixed = dict(
        R_sc=2.0, d_0=3.8, dt=0.01, mass=1.0,
        n_steps=n_steps, save_every=save_every,
        damping=0.0, v_mode="breathing",
        crash_threshold=0.5, output_prefix="autostop",
        chain_id=chain_id,
    )
    fixed.update(params)

    cfg = SimulationConfig(**fixed)

    # --- Load structures ---
    struct = ProteinStructure.from_pdb(pdb_path, chain_id=chain_id)
    target_ca = None
    if target_path is not None:
        if target_path.lower().endswith(".cif"):
            tgt = ProteinStructure.from_cif(target_path, chain_id=chain_id)
        else:
            tgt = ProteinStructure.from_pdb(target_path, chain_id=chain_id)
        target_ca = np.asarray(tgt.coords_ca, dtype=np.float64)
    baseline = (compute_rmsd_aligned(struct.coords_ca, target_ca)
                if target_ca is not None else None)

    # --- Build physics objects ---
    enm = InteractionWeightedENM(
        R_bb=cfg.R_bb, R_sc=cfg.R_sc, K_0=cfg.K_0,
        d_0=cfg.d_0, n_ref=cfg.n_ref,
    )
    integrator = VelocityVerletIntegrator(
        mass=cfg.mass, dt=cfg.dt, damping=cfg.damping,
    )

    coords = struct.coords_ca.copy()
    coords_sc = struct.coords_cb.copy()
    atom_coords = struct.atom_coords.copy()
    atom_res_idx = struct.atom_res_idx
    enm.set_equilibrium_distances(coords)
    velocities = integrator.initialize_velocities(
        struct.N, mode=cfg.v_mode, magnitude=cfg.v_magnitude, coords=coords,
    )

    trajectory = [coords.copy()]       # trajectory[0] = initial
    energies = []                       # list of (step, E_pot, E_kin, E_tot)
    springs = []                        # list of (step, n_springs)
    crash_events = 0
    crash_per_step = []

    monitor = EarlyStopMonitor(
        total_steps=cfg.n_steps, save_every=cfg.save_every,
        smooth_w=smooth_w, warmup_frac=warmup_frac, patience=patience,
        eps_E_rel=eps_E_rel, eps_N_rel=eps_N_rel,
        crash_window_saves=crash_window_saves,
        crash_threshold=crash_threshold,
    )

    thr = cfg.crash_threshold
    if verbose:
        print(f"[autostop] N={struct.N}  params={fixed}")
        if baseline is not None:
            print(f"[autostop] baseline RMSD = {baseline:.3f} Å")

    stop_step = None

    for step in range(1, cfg.n_steps + 1):
        coords, velocities, coords_sc, atom_coords, forces, network_info = (
            integrator.step(coords, velocities, coords_sc, enm,
                            struct.res_names, atom_coords, atom_res_idx)
        )
        neighbors, K_matrix, n_springs, interaction_counts = network_info

        # --- Crash tracking (atomic, non-adjacent residues) ---
        tree = cKDTree(atom_coords)
        pairs_near = tree.query_pairs(r=max(thr * 2.0, 1.0), output_type="ndarray")
        if len(pairs_near) > 0:
            ri = atom_res_idx[pairs_near[:, 0]]
            rj = atom_res_idx[pairs_near[:, 1]]
            nonadj = np.abs(ri - rj) > 1
            if nonadj.any():
                pp = pairs_near[nonadj]
                diffs = atom_coords[pp[:, 0]] - atom_coords[pp[:, 1]]
                d_inter = np.linalg.norm(diffs, axis=1)
                n_below = int((d_inter < thr).sum())
                if n_below > 0:
                    crash_events += n_below
                    crash_per_step.append((step, n_below))

        if step % cfg.save_every == 0 or step == 1:
            E_pot = enm.compute_energy(coords, neighbors, K_matrix)
            E_kin = compute_kinetic_energy(velocities, cfg.mass)
            E_tot = E_pot + E_kin

            trajectory.append(coords.copy())
            energies.append((step, E_pot, E_kin, E_tot))
            springs.append((step, n_springs))

            triggered = monitor.update(step, E_tot, n_springs, crash_events)

            if verbose and (step % (cfg.save_every * 20) == 0 or step == 1):
                tag = ""
                if target_ca is not None:
                    r = compute_rmsd_aligned(coords, target_ca)
                    tag = f"  RMSDt={r:6.3f}"
                print(f"  step {step:6d}  E_tot={E_tot:10.2f}  "
                      f"n_spr={n_springs:5d}  crashes={crash_events:6d}{tag}")

            if triggered:
                stop_step = step
                if verbose:
                    print(f"\n[autostop] STOP at step {step}: {monitor.stop_reason}")
                break

    if stop_step is None and verbose:
        print(f"\n[autostop] n_steps exhausted ({cfg.n_steps}), "
              f"no reversal+crash trigger. Falling back to saved-min.")

    # --- Pick frame: back off from turnpoint index ---
    if len(energies) == 0:
        raise RuntimeError("No saves produced — save_every > n_steps?")

    k_turn = monitor.turnpoint_index() if monitor.argmin_E != 0 else 0
    k_best = max(0, k_turn - back_off)

    # trajectory[0] = initial; trajectory[i+1] aligns with energies[i]
    traj_idx = min(k_best + 1, len(trajectory) - 1)
    picked_coords = trajectory[traj_idx]
    picked_step = energies[k_best][0]

    result = {
        "picked_step": int(picked_step),
        "picked_coords": picked_coords,
        "stop_step": stop_step,
        "turn_k": int(k_turn),
        "argmin_E_k": int(monitor.argmin_E),
        "argmin_N_k": int(monitor.argmin_N),
        "step_argmin_E": int(energies[monitor.argmin_E][0]) if energies else None,
        "step_argmin_N": int(springs[monitor.argmin_N][0]) if springs else None,
        "crashes_total": int(crash_events),
        "baseline_rmsd": baseline,
        "structure": struct,
        "trajectory": trajectory,
        "energies": energies,
        "springs": springs,
    }
    if target_ca is not None:
        result["rmsd_to_target"] = float(compute_rmsd_aligned(picked_coords, target_ca))
        result["tm_to_target"] = float(compute_tm_score(picked_coords, target_ca))
        # Oracle for diagnostic
        traj_rmsds = np.array(
            [compute_rmsd_aligned(c, target_ca) for c in trajectory[1:]]
        )
        result["oracle_rmsd"] = float(traj_rmsds.min())
        result["oracle_k"] = int(np.argmin(traj_rmsds))
    return result


# ------------------------------- CLI --------------------------------

def _parse_params(s):
    out = {}
    for pair in (s or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Early-stopping IW-ENM runner")
    ap.add_argument("pdb", help="initial structure (PDB)")
    ap.add_argument("--target", default=None, help="target structure (PDB or CIF)")
    ap.add_argument("--chain", default="A")
    ap.add_argument("--out", default="model_autostop.pdb")

    # physics params
    ap.add_argument("--R_bb", type=float, default=11.0)
    ap.add_argument("--K_0", type=float, default=0.8)
    ap.add_argument("--n_ref", type=float, default=10.0)
    ap.add_argument("--v_magnitude", type=float, default=1.0)
    ap.add_argument("--params", default="",
                    help="extra 'k=v,k=v' overrides (e.g. 'dt=0.01,damping=0')")

    # loop control
    ap.add_argument("--n_steps", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--back_off", type=int, default=2,
                    help="saved-frame count to rewind from turnpoint")

    # monitor
    ap.add_argument("--smooth_w", type=int, default=11)
    ap.add_argument("--warmup_frac", type=float, default=0.20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eps_E_rel", type=float, default=0.002)
    ap.add_argument("--eps_N_rel", type=float, default=0.005)
    ap.add_argument("--crash_window_saves", type=int, default=20)
    ap.add_argument("--crash_threshold", type=int, default=5,
                    help="total crashes within the window to trigger crash_onset")

    args = ap.parse_args(argv)

    params = {
        "R_bb": args.R_bb, "K_0": args.K_0,
        "n_ref": args.n_ref, "v_magnitude": args.v_magnitude,
    }
    params.update(_parse_params(args.params))

    res = run_with_autostop(
        args.pdb,
        target_path=args.target,
        params=params,
        chain_id=args.chain,
        n_steps=args.n_steps,
        save_every=args.save_every,
        back_off=args.back_off,
        smooth_w=args.smooth_w,
        warmup_frac=args.warmup_frac,
        patience=args.patience,
        eps_E_rel=args.eps_E_rel,
        eps_N_rel=args.eps_N_rel,
        crash_window_saves=args.crash_window_saves,
        crash_threshold=args.crash_threshold,
    )

    # Export model PDB
    res["structure"].to_pdb(args.out, coords=res["picked_coords"])

    print("\n=== RESULT ===")
    print(f"  stop_step        : {res['stop_step']}")
    print(f"  argmin_E step    : {res['step_argmin_E']}")
    print(f"  argmin_N step    : {res['step_argmin_N']}")
    print(f"  picked step      : {res['picked_step']}")
    print(f"  total crashes    : {res['crashes_total']}")
    if "rmsd_to_target" in res:
        print(f"  baseline RMSD    : {res['baseline_rmsd']:.3f} Å")
        print(f"  picked RMSDt     : {res['rmsd_to_target']:.3f} Å")
        print(f"  picked TMt       : {res['tm_to_target']:.3f}")
        print(f"  oracle best RMSDt: {res['oracle_rmsd']:.3f} Å "
              f"(frame {res['oracle_k']+1})")
        gap = res['rmsd_to_target'] - res['oracle_rmsd']
        print(f"  gap vs oracle    : {gap:+.3f} Å "
              f"({'WITHIN' if abs(gap) < 0.3 else 'OFF'})")
    print(f"  model PDB        : {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
