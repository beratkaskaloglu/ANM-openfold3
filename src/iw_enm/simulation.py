"""Main simulation loop."""

import os
import csv
import numpy as np

from .structure import ProteinStructure
from .network import InteractionWeightedENM
from .integrator import VelocityVerletIntegrator
from .analysis import compute_kinetic_energy, compute_rmsd_aligned, compute_rmsf, compute_tm_score
from .config import SimulationConfig


class Simulation:
    """Main simulation manager."""

    def __init__(self, structure, enm, integrator, config, target_ca=None):
        self.structure = structure
        self.enm = enm
        self.integrator = integrator
        self.config = config
        self.target_ca = (
            np.asarray(target_ca, dtype=np.float64) if target_ca is not None else None
        )

        self.trajectory = []
        self.energies = []
        self.spring_counts = []
        self.spring_networks = []
        self.forces_history = []
        self.velocities_history = []

        # RMSD/TM to target (if provided)
        self.rmsd_to_ref = []       # (step, rmsd_vs_initial)
        self.rmsd_to_target = []    # (step, rmsd_vs_target)
        self.tm_to_target = []      # (step, tm_vs_target)
        self.best_rmsd_target = np.inf
        self.best_tm_target = 0.0
        self.best_step = 0

        # Crash tracking (sidechain-sidechain clashes)
        self.crash_events = 0          # total (pair x step) below threshold
        self.crashed_pairs = set()     # unique (i,j) pairs that crashed
        self.min_sc_distance = np.inf  # smallest sc-sc distance encountered
        self.crash_per_step = []       # (step, n_pairs_below_threshold)

    def run(self):
        """Run the simulation."""
        cfg = self.config
        struct = self.structure

        coords = struct.coords_ca.copy()
        coords_sc = struct.coords_cb.copy()
        atom_coords = struct.atom_coords.copy()
        atom_res_idx = struct.atom_res_idx
        ref_coords = coords.copy()

        # Set native distances as equilibrium
        self.enm.set_equilibrium_distances(coords)

        velocities = self.integrator.initialize_velocities(
            struct.N, mode=cfg.v_mode, magnitude=cfg.v_magnitude, coords=coords
        )

        # Save initial state
        self.trajectory.append(coords.copy())
        self.velocities_history.append(velocities.copy())

        print(f"IW-ENM Simulation: N={struct.N}, atoms={struct.n_atoms} (heavy, no CA), steps={cfg.n_steps}")
        print(f"  R_bb={cfg.R_bb}, R_sc={cfg.R_sc}, K_0={cfg.K_0}, dt={cfg.dt}")
        print(f"  damping={cfg.damping}, v_mode={cfg.v_mode}, v_mag={cfg.v_magnitude}")
        print(f"  crash_threshold={cfg.crash_threshold} Å (atomic)")
        print()

        thr = cfg.crash_threshold

        from scipy.spatial import cKDTree

        for step in range(1, cfg.n_steps + 1):
            coords, velocities, coords_sc, atom_coords, forces, network_info = (
                self.integrator.step(coords, velocities, coords_sc, self.enm,
                                     struct.res_names, atom_coords, atom_res_idx)
            )
            neighbors, K_matrix, n_springs, interaction_counts = network_info

            # --- Atomic crash tracking (non-adjacent residues only) ---
            tree = cKDTree(atom_coords)
            pairs_near = tree.query_pairs(r=max(thr * 2.0, 1.0), output_type="ndarray")
            if len(pairs_near) > 0:
                ri = atom_res_idx[pairs_near[:, 0]]
                rj = atom_res_idx[pairs_near[:, 1]]
                # Exclude same-residue and sequential neighbors (|i-j|<=1)
                nonadj = np.abs(ri - rj) > 1
                if nonadj.any():
                    pp = pairs_near[nonadj]
                    diffs = atom_coords[pp[:, 0]] - atom_coords[pp[:, 1]]
                    d_inter = np.linalg.norm(diffs, axis=1)
                    min_d = float(d_inter.min())
                    if min_d < self.min_sc_distance:
                        self.min_sc_distance = min_d
                    below = d_inter < thr
                    n_below = int(below.sum())
                    if n_below > 0:
                        self.crash_events += n_below
                        self.crash_per_step.append((step, n_below))
                        for a, b in pp[below]:
                            self.crashed_pairs.add((int(atom_res_idx[a]),
                                                    int(atom_res_idx[b])))

            if step % cfg.save_every == 0 or step == 1:
                E_pot = self.enm.compute_energy(coords, neighbors, K_matrix)
                E_kin = compute_kinetic_energy(velocities, cfg.mass)
                E_tot = E_pot + E_kin
                rmsd = compute_rmsd_aligned(coords, ref_coords)
                self.rmsd_to_ref.append((step, rmsd))

                # RMSD/TM to target (if available)
                rmsd_tgt = None
                tm_tgt = None
                if self.target_ca is not None:
                    rmsd_tgt = compute_rmsd_aligned(coords, self.target_ca)
                    tm_tgt = compute_tm_score(coords, self.target_ca)
                    self.rmsd_to_target.append((step, rmsd_tgt))
                    self.tm_to_target.append((step, tm_tgt))
                    if rmsd_tgt < self.best_rmsd_target:
                        self.best_rmsd_target = rmsd_tgt
                        self.best_step = step
                    if tm_tgt > self.best_tm_target:
                        self.best_tm_target = tm_tgt

                self.trajectory.append(coords.copy())
                self.energies.append((step, E_pot, E_kin, E_tot))
                self.spring_counts.append((step, n_springs))
                self.spring_networks.append((neighbors, K_matrix.copy()))
                self.forces_history.append(forces.copy())

                if step % (cfg.save_every * 10) == 0 or step == 1:
                    tgt_str = (
                        f"  RMSDt={rmsd_tgt:6.3f}  TMt={tm_tgt:5.3f}"
                        if rmsd_tgt is not None else ""
                    )
                    print(
                        f"  Step {step:6d}: E_pot={E_pot:9.2f}  E_kin={E_kin:9.2f}  "
                        f"E_tot={E_tot:9.2f}  RMSD={rmsd:6.3f}{tgt_str}  "
                        f"springs={n_springs}  "
                        f"crashes={self.crash_events:6d}  min_sc_d={self.min_sc_distance:.2f}Å"
                    )

        print(f"\nDone. {len(self.trajectory)} frames saved.")
        if self.target_ca is not None:
            baseline = compute_rmsd_aligned(ref_coords, self.target_ca)
            baseline_tm = compute_tm_score(ref_coords, self.target_ca)
            print(
                f"  Target RMSD: baseline={baseline:.3f} Å   "
                f"best={self.best_rmsd_target:.3f} Å (step {self.best_step})   "
                f"Δ={baseline - self.best_rmsd_target:+.3f} Å"
            )
            print(
                f"  Target TM:   baseline={baseline_tm:.3f}       "
                f"best={self.best_tm_target:.3f}"
            )

    def save_results(self, output_dir, prefix=None):
        """Save all outputs."""
        if prefix is None:
            prefix = self.config.output_prefix
        os.makedirs(output_dir, exist_ok=True)

        base = os.path.join(output_dir, prefix)
        struct = self.structure

        # Equilibrium structure
        struct.to_pdb(f"{base}_equilibrium.pdb")

        # Trajectory
        struct.write_trajectory_pdb(f"{base}_trajectory.pdb", self.trajectory)

        # Initial spring network
        if self.spring_networks:
            n0, K0 = self.spring_networks[0]
            struct.write_springs_pdb(
                f"{base}_springs_t0.pdb", self.trajectory[0], n0, K0
            )
            # Final spring network
            nf, Kf = self.spring_networks[-1]
            struct.write_springs_pdb(
                f"{base}_springs_final.pdb", self.trajectory[-1], nf, Kf
            )

        # Initial velocities
        if self.velocities_history:
            struct.write_vectors_pdb(
                f"{base}_velocities_t0.pdb",
                self.trajectory[0],
                self.velocities_history[0],
                label="VEL"
            )

        # Force vectors (last saved frame)
        if self.forces_history:
            struct.write_vectors_pdb(
                f"{base}_forces_final.pdb",
                self.trajectory[-1],
                self.forces_history[-1],
                label="FRC"
            )

        # Energy CSV
        with open(f"{base}_energy.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "E_pot", "E_kin", "E_tot"])
            for row in self.energies:
                w.writerow(row)

        # RMSF CSV
        if len(self.trajectory) > 1:
            rmsf = compute_rmsf(self.trajectory)
            with open(f"{base}_rmsf.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["res_id", "res_name", "rmsf"])
                for i in range(struct.N):
                    w.writerow([struct.res_ids[i], struct.res_names[i], f"{rmsf[i]:.4f}"])

        # Spring count CSV
        with open(f"{base}_spring_counts.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "n_springs"])
            for row in self.spring_counts:
                w.writerow(row)

        print(f"Results saved to {output_dir}/{prefix}_*")
        return base
