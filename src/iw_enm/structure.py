"""PDB parsing and writing."""

import numpy as np


class ProteinStructure:
    """Reads protein structure from PDB file."""

    # Sidechain volumes table (Å³)
    VOLUME_TABLE = {
        "GLY": 66, "ALA": 92, "VAL": 142, "LEU": 168, "ILE": 169,
        "PRO": 129, "PHE": 203, "TRP": 228, "MET": 171, "SER": 99,
        "THR": 122, "CYS": 106, "TYR": 204, "HIS": 167, "ASP": 125,
        "GLU": 155, "ASN": 135, "GLN": 161, "LYS": 171, "ARG": 202,
    }
    V_REF = 137.5  # average volume

    def __init__(self, coords_ca, coords_cb, res_names, res_ids, chain_ids,
                 atom_coords=None, atom_res_idx=None, atom_names=None):
        self.coords_ca = np.array(coords_ca, dtype=np.float64)
        self.coords_cb = np.array(coords_cb, dtype=np.float64)
        self.res_names = list(res_names)
        self.res_ids = list(res_ids)
        self.chain_ids = list(chain_ids)
        self.N = len(self.coords_ca)

        # All heavy atoms (excluding CA, excluding H). Flat arrays.
        if atom_coords is not None:
            self.atom_coords = np.array(atom_coords, dtype=np.float64)
            self.atom_res_idx = np.array(atom_res_idx, dtype=np.int32)
            self.atom_names = list(atom_names) if atom_names is not None else []
        else:
            # Fallback: single CB per residue
            self.atom_coords = self.coords_cb.copy()
            self.atom_res_idx = np.arange(self.N, dtype=np.int32)
            self.atom_names = ["CB"] * self.N
        self.n_atoms = len(self.atom_coords)

    @classmethod
    def from_pdb(cls, path, chain_id="A"):
        """Parse PDB file, extract CA, CB and all heavy atoms for given chain."""
        ca_atoms = {}
        cb_atoms = {}
        res_atoms = {}  # res_seq -> list of (atom_name, coord) excluding CA and H

        with open(path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom_name = line[12:16].strip()
                chain = line[21].strip()
                if chain != chain_id:
                    continue
                res_name = line[17:20].strip()
                res_seq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coord = np.array([x, y, z])

                # Skip hydrogen atoms
                if atom_name.startswith("H") or atom_name.startswith("D"):
                    continue

                if atom_name == "CA":
                    ca_atoms[res_seq] = (coord, res_name, chain)
                elif atom_name in ("N", "C", "O", "OXT"):
                    # Backbone atoms — skip (not part of sidechain)
                    pass
                else:
                    if atom_name == "CB":
                        cb_atoms[res_seq] = coord
                    res_atoms.setdefault(res_seq, []).append((atom_name, coord))

        # Build arrays sorted by residue number
        sorted_keys = sorted(ca_atoms.keys())
        coords_ca, coords_cb, res_names, res_ids, chain_ids = [], [], [], [], []
        atom_coords, atom_res_idx, atom_names = [], [], []

        for idx, res_seq in enumerate(sorted_keys):
            ca_coord, res_name, ch = ca_atoms[res_seq]
            coords_ca.append(ca_coord)
            res_names.append(res_name)
            res_ids.append(res_seq)
            chain_ids.append(ch)
            coords_cb.append(cb_atoms.get(res_seq, ca_coord.copy()))

            # Add all non-CA heavy atoms for this residue
            for aname, acoord in res_atoms.get(res_seq, []):
                atom_coords.append(acoord)
                atom_res_idx.append(idx)
                atom_names.append(aname)

        return cls(coords_ca, coords_cb, res_names, res_ids, chain_ids,
                   atom_coords=atom_coords, atom_res_idx=atom_res_idx,
                   atom_names=atom_names)

    @classmethod
    def from_cif(cls, path, chain_id="A"):
        """Parse mmCIF file, extract CA, CB and all heavy atoms for given chain."""
        ca_atoms = {}
        cb_atoms = {}
        res_atoms = {}
        in_atom_site = False
        columns = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    columns.append(line.split(".")[1])
                    continue
                if in_atom_site and not line.startswith("_") and not line.startswith("#") and line:
                    if line.startswith("loop_"):
                        in_atom_site = False
                        continue
                    parts = line.split()
                    if len(parts) < len(columns):
                        continue
                    row = dict(zip(columns, parts))

                    if row.get("group_PDB") != "ATOM":
                        continue
                    auth_chain = row.get("auth_asym_id", "")
                    if auth_chain != chain_id:
                        continue
                    atom_name = row.get("auth_atom_id", row.get("label_atom_id", ""))
                    res_name = row.get("auth_comp_id", row.get("label_comp_id", ""))
                    res_seq = int(row.get("auth_seq_id", row.get("label_seq_id", 0)))
                    x = float(row["Cartn_x"])
                    y = float(row["Cartn_y"])
                    z = float(row["Cartn_z"])

                    # Skip hydrogens
                    if atom_name.startswith("H") or atom_name.startswith("D"):
                        pass
                    elif atom_name == "CA":
                        ca_atoms[res_seq] = (np.array([x, y, z]), res_name, auth_chain)
                    elif atom_name in ("N", "C", "O", "OXT"):
                        # Backbone atoms — skip (not part of sidechain)
                        pass
                    else:
                        coord = np.array([x, y, z])
                        if atom_name == "CB":
                            cb_atoms[res_seq] = coord
                        res_atoms.setdefault(res_seq, []).append((atom_name, coord))
                elif in_atom_site and (line.startswith("#") or line.startswith("loop_")):
                    in_atom_site = False

        sorted_keys = sorted(ca_atoms.keys())
        coords_ca, coords_cb, res_names, res_ids, chain_ids = [], [], [], [], []
        atom_coords, atom_res_idx, atom_names = [], [], []
        for idx, res_seq in enumerate(sorted_keys):
            ca_coord, res_name, ch = ca_atoms[res_seq]
            coords_ca.append(ca_coord)
            res_names.append(res_name)
            res_ids.append(res_seq)
            chain_ids.append(ch)
            coords_cb.append(cb_atoms.get(res_seq, ca_coord.copy()))
            for aname, acoord in res_atoms.get(res_seq, []):
                atom_coords.append(acoord)
                atom_res_idx.append(idx)
                atom_names.append(aname)

        return cls(coords_ca, coords_cb, res_names, res_ids, chain_ids,
                   atom_coords=atom_coords, atom_res_idx=atom_res_idx,
                   atom_names=atom_names)

    def get_sidechain_representatives(self):
        """Return CB coordinates (or CA for GLY)."""
        return self.coords_cb.copy()

    def to_pdb(self, path, coords=None, model_num=None):
        """Write coordinates as PDB."""
        if coords is None:
            coords = self.coords_ca
        lines = []
        if model_num is not None:
            lines.append(f"MODEL     {model_num:4d}")
        for i in range(self.N):
            lines.append(
                f"ATOM  {i+1:5d}  CA  {self.res_names[i]:3s} "
                f"{self.chain_ids[i]}{self.res_ids[i]:4d}    "
                f"{coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}"
                f"  1.00  0.00           C"
            )
        if model_num is not None:
            lines.append("ENDMDL")
        else:
            lines.append("END")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_trajectory_pdb(self, path, trajectory):
        """Write multi-model PDB from trajectory frames."""
        lines = []
        for frame_idx, coords in enumerate(trajectory):
            lines.append(f"MODEL     {frame_idx+1:4d}")
            for i in range(self.N):
                lines.append(
                    f"ATOM  {i+1:5d}  CA  {self.res_names[i]:3s} "
                    f"{self.chain_ids[i]}{self.res_ids[i]:4d}    "
                    f"{coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}"
                    f"  1.00  0.00           C"
                )
            lines.append("ENDMDL")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_springs_pdb(self, path, coords, neighbors, K_matrix):
        """Write spring network as PDB with CONECT records."""
        lines = []
        for i in range(self.N):
            lines.append(
                f"ATOM  {i+1:5d}  CA  {self.res_names[i]:3s} "
                f"{self.chain_ids[i]}{self.res_ids[i]:4d}    "
                f"{coords[i,0]:8.3f}{coords[i,1]:8.3f}{coords[i,2]:8.3f}"
                f"  1.00{K_matrix[i].sum():6.2f}           C"
            )
        for i in range(self.N):
            for j in neighbors[i]:
                if j > i:
                    lines.append(f"CONECT{i+1:5d}{j+1:5d}")
        lines.append("END")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_vectors_pdb(self, path, coords, vectors, label="VEC"):
        """Write vectors as pseudoatom pairs (for PyMOL CGO arrows)."""
        lines = []
        scale = 5.0  # scale factor for visualization
        atom_idx = 1
        for i in range(self.N):
            start = coords[i]
            end = coords[i] + vectors[i] * scale
            lines.append(
                f"ATOM  {atom_idx:5d}  CA  {label:3s} "
                f"V{i+1:4d}    "
                f"{start[0]:8.3f}{start[1]:8.3f}{start[2]:8.3f}"
                f"  1.00  0.00           C"
            )
            atom_idx += 1
            lines.append(
                f"ATOM  {atom_idx:5d}  CB  {label:3s} "
                f"V{i+1:4d}    "
                f"{end[0]:8.3f}{end[1]:8.3f}{end[2]:8.3f}"
                f"  1.00  0.00           C"
            )
            atom_idx += 1
            lines.append(f"CONECT{atom_idx-2:5d}{atom_idx-1:5d}")
        lines.append("END")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
