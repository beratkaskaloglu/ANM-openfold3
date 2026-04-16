"""ENM network construction and spring constant calculation."""

import numpy as np
from scipy.spatial.distance import cdist

from .structure import ProteinStructure


class InteractionWeightedENM:
    """Sidechain interaction-weighted elastic network model."""

    def __init__(self, R_bb=10.0, R_sc=6.0, K_0=1.0, d_0=3.8, n_ref=7.0,
                 use_native_distances=True, use_atomic_packing=False):
        self.R_bb = R_bb
        self.R_sc = R_sc
        self.K_0 = K_0
        self.d_0 = d_0  # used only if use_native_distances=False
        self.n_ref = n_ref
        self.use_native_distances = use_native_distances
        self.use_atomic_packing = use_atomic_packing
        self.volume_table = ProteinStructure.VOLUME_TABLE
        self.V_ref = ProteinStructure.V_REF
        self._vol_cache = {}
        self._eq_distances = None  # native distance matrix (set from initial structure)

    def set_equilibrium_distances(self, coords_ca):
        """Store initial Cα distances as equilibrium lengths."""
        self._eq_distances = cdist(coords_ca, coords_ca)

    def _get_volumes(self, res_names):
        key = tuple(res_names)
        if key not in self._vol_cache:
            self._vol_cache[key] = np.array(
                [self.volume_table.get(r, self.V_ref) for r in res_names],
                dtype=np.float64,
            )
        return self._vol_cache[key]

    def count_interactions(self, coords_sc):
        """Count sidechain interactions within R_sc for each residue (CB-based)."""
        dist_matrix = cdist(coords_sc, coords_sc)
        contact_matrix = (dist_matrix < self.R_sc) & (dist_matrix > 0.0)
        return contact_matrix.sum(axis=1).astype(np.float64)

    def count_interactions_atomic(self, atom_coords, atom_res_idx, N):
        """
        Per-residue atomic contact count: for each atom in residue i,
        count atoms in OTHER residues within R_sc.
        Returns per-residue total counts (N,).
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(atom_coords)
        pairs = tree.query_pairs(r=self.R_sc, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros(N, dtype=np.float64)
        res_i = atom_res_idx[pairs[:, 0]]
        res_j = atom_res_idx[pairs[:, 1]]
        # Keep only inter-residue pairs
        mask = res_i != res_j
        res_i, res_j = res_i[mask], res_j[mask]
        # Each pair contributes +1 to both residues' counts
        counts = np.bincount(res_i, minlength=N).astype(np.float64)
        counts += np.bincount(res_j, minlength=N).astype(np.float64)
        return counts

    def build_network(self, coords_ca, coords_sc, res_names,
                      atom_coords=None, atom_res_idx=None):
        """
        Build interaction-weighted spring network (vectorized).

        If atom_coords + atom_res_idx are given, packing n_i is computed
        at the atomic level (all heavy atoms except CA).
        Otherwise falls back to CB-based single-point count.

        Returns:
            neighbors: list[list[int]]
            K_matrix: np.ndarray (N, N)
            n_springs: int
            interaction_counts: np.ndarray (N,)
        """
        N = len(coords_ca)
        if self.use_atomic_packing and atom_coords is not None and atom_res_idx is not None:
            interaction_counts = self.count_interactions_atomic(
                atom_coords, atom_res_idx, N
            )
        else:
            interaction_counts = self.count_interactions(coords_sc)
        interaction_counts = np.maximum(interaction_counts, 1.0)

        dist_matrix = cdist(coords_ca, coords_ca)

        mask = (dist_matrix < self.R_bb)
        np.fill_diagonal(mask, False)

        # Vectorized K_ij
        ic_outer = np.outer(interaction_counts, interaction_counts)
        phi_matrix = np.sqrt(ic_outer) / self.n_ref

        volumes = self._get_volumes(res_names)
        w_matrix = (volumes[:, None] + volumes[None, :]) / (2.0 * self.V_ref)

        K_matrix = self.K_0 * phi_matrix * w_matrix * mask

        # Build neighbor lists
        neighbors = [[] for _ in range(N)]
        rows, cols = np.where(np.triu(mask, k=1))
        n_springs = len(rows)
        for i, j in zip(rows, cols):
            neighbors[i].append(j)
            neighbors[j].append(i)

        return neighbors, K_matrix, n_springs, interaction_counts

    def _get_d0_matrix(self, dist_matrix):
        """Get equilibrium distance matrix."""
        if self.use_native_distances and self._eq_distances is not None:
            return self._eq_distances
        # Uniform d_0
        return np.full_like(dist_matrix, self.d_0)

    def compute_forces(self, coords, neighbors, K_matrix):
        """
        F_i = sum_j K_ij * (1 - d0_ij/d_ij) * (r_j - r_i)
        """
        N = len(coords)
        dr = coords[None, :, :] - coords[:, None, :]  # (N, N, 3)
        dist = np.linalg.norm(dr, axis=2)
        dist = np.maximum(dist, 0.5)

        d0 = self._get_d0_matrix(dist)
        scalar = K_matrix * (1.0 - d0 / dist)

        forces = np.sum(scalar[:, :, None] * dr, axis=1)
        return forces

    def compute_energy(self, coords, neighbors, K_matrix):
        """U = 0.5 * sum_{i<j} K_ij * (d_ij - d0_ij)^2"""
        dist_matrix = cdist(coords, coords)
        d0 = self._get_d0_matrix(dist_matrix)
        K_upper = np.triu(K_matrix, k=1)
        d_upper = np.triu(dist_matrix, k=1)
        d0_upper = np.triu(d0, k=1)
        active = K_upper > 0
        return 0.5 * np.sum(K_upper[active] * (d_upper[active] - d0_upper[active]) ** 2)
