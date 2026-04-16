"""Velocity Verlet integrator."""

import numpy as np


class VelocityVerletIntegrator:
    """Velocity Verlet integration with network rebuild at each step."""

    def __init__(self, mass=1.0, dt=0.01, damping=0.0):
        self.mass = mass
        self.dt = dt
        self.damping = damping

    def step(self, coords, velocities, coords_sc, enm, res_names,
             atom_coords=None, atom_res_idx=None):
        """
        Single Velocity Verlet step with network rebuild.

        If atom_coords + atom_res_idx given, uses atomic packing count.
        All residue-bound atoms translate rigidly with their CA.

        Returns:
            r_new, v_new, coords_sc_new, atom_coords_new, forces_new, network_info
        """
        dt = self.dt
        m = self.mass

        # Current network and forces
        neighbors, K_matrix, n_springs, ic = enm.build_network(
            coords, coords_sc, res_names, atom_coords, atom_res_idx
        )
        forces = enm.compute_forces(coords, neighbors, K_matrix)
        if self.damping > 0:
            forces -= self.damping * velocities

        # Half-step velocity
        v_half = velocities + (dt / (2.0 * m)) * forces

        # Full-step position
        r_new = coords + dt * v_half

        # Update sidechain positions (CB and all atoms move rigidly with CA)
        delta = r_new - coords
        coords_sc_new = coords_sc + delta
        if atom_coords is not None:
            atom_coords_new = atom_coords + delta[atom_res_idx]
        else:
            atom_coords_new = None

        # Rebuild network with new positions
        neighbors_new, K_new, n_springs_new, ic_new = enm.build_network(
            r_new, coords_sc_new, res_names, atom_coords_new, atom_res_idx
        )
        forces_new = enm.compute_forces(r_new, neighbors_new, K_new)
        if self.damping > 0:
            forces_new -= self.damping * v_half

        # Full-step velocity
        v_new = v_half + (dt / (2.0 * m)) * forces_new

        network_info = (neighbors_new, K_new, n_springs_new, ic_new)
        return r_new, v_new, coords_sc_new, atom_coords_new, forces_new, network_info

    def initialize_velocities(self, N, mode="random", magnitude=1.0, coords=None):
        """
        Initialize velocities.

        Modes:
            'random':    random directions, uniform magnitude
            'breathing': outward from center of mass
            'zero':      zero velocities
        """
        if mode == "zero":
            return np.zeros((N, 3), dtype=np.float64)

        if mode == "breathing" and coords is not None:
            com = coords.mean(axis=0)
            directions = coords - com
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            directions = directions / norms
            return directions * magnitude

        # Default: random
        v = np.random.randn(N, 3)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        v = v / norms * magnitude
        return v
