"""Tests for iw_enm/structure.py: ProteinStructure parsing and writing."""

import tempfile
import numpy as np
import pytest

from src.iw_enm.structure import ProteinStructure


# ── Fixtures ───────────────────────────────────────────────────────

def _make_structure(n=10):
    """Create a minimal ProteinStructure for testing."""
    coords_ca = np.random.randn(n, 3) * 10.0
    coords_cb = coords_ca + np.random.randn(n, 3) * 0.5
    res_names = ["ALA"] * n
    res_ids = list(range(1, n + 1))
    chain_ids = ["A"] * n
    return ProteinStructure(coords_ca, coords_cb, res_names, res_ids, chain_ids)


def _write_minimal_pdb(path, n=5, chain="A"):
    """Write a minimal valid PDB file for parsing tests."""
    lines = []
    res_names = ["ALA", "GLY", "VAL", "LEU", "ILE"]
    for i in range(n):
        x, y, z = i * 3.8, 0.0, 0.0
        rn = res_names[i % len(res_names)]
        # CA atom
        lines.append(
            f"ATOM  {i*3+1:5d}  CA  {rn} {chain}{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        # CB atom (not for GLY but we add it anyway for simplicity)
        lines.append(
            f"ATOM  {i*3+2:5d}  CB  {rn} {chain}{i+1:4d}    "
            f"{x+0.5:8.3f}{y+0.5:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        # N atom (backbone, should be skipped for sidechain)
        lines.append(
            f"ATOM  {i*3+3:5d}  N   {rn} {chain}{i+1:4d}    "
            f"{x-1.0:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           N"
        )
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Constructor ────────────────────────────────────────────────────

class TestConstructor:
    def test_basic_attributes(self):
        s = _make_structure(8)
        assert s.N == 8
        assert s.coords_ca.shape == (8, 3)
        assert s.coords_cb.shape == (8, 3)
        assert len(s.res_names) == 8
        assert len(s.res_ids) == 8

    def test_fallback_atom_coords(self):
        """Without explicit atom_coords, falls back to CB."""
        s = _make_structure(5)
        assert s.n_atoms == 5
        np.testing.assert_array_equal(s.atom_coords, s.coords_cb)
        assert s.atom_names == ["CB"] * 5

    def test_explicit_atom_coords(self):
        n = 4
        ca = np.random.randn(n, 3)
        cb = ca + 0.1
        atom_c = np.random.randn(10, 3)
        atom_idx = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
        atom_nm = ["CB", "CG"] * 5
        s = ProteinStructure(
            ca, cb, ["ALA"] * n, list(range(1, n + 1)), ["A"] * n,
            atom_coords=atom_c, atom_res_idx=atom_idx, atom_names=atom_nm,
        )
        assert s.n_atoms == 10
        assert len(s.atom_names) == 10

    def test_coords_are_float64(self):
        s = _make_structure(3)
        assert s.coords_ca.dtype == np.float64
        assert s.coords_cb.dtype == np.float64


# ── PDB Parsing ────────────────────────────────────────────────────

class TestFromPDB:
    def test_parse_basic_pdb(self):
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            _write_minimal_pdb(f.name, n=5)
            path = f.name
        s = ProteinStructure.from_pdb(path, chain_id="A")
        assert s.N == 5
        assert s.coords_ca.shape == (5, 3)
        # First CA should be at (0, 0, 0)
        np.testing.assert_allclose(s.coords_ca[0], [0.0, 0.0, 0.0], atol=0.01)

    def test_chain_filter(self):
        """Only atoms from the specified chain should be parsed."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            path = f.name
        # Write chain A and chain B
        lines = []
        for ch, offset in [("A", 0), ("B", 100)]:
            for i in range(3):
                x = offset + i * 3.8
                lines.append(
                    f"ATOM  {i+1:5d}  CA  ALA {ch}{i+1:4d}    "
                    f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
                )
        lines.append("END")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        s = ProteinStructure.from_pdb(path, chain_id="B")
        assert s.N == 3
        assert s.coords_ca[0, 0] > 90  # chain B starts at x=100

    def test_cb_fallback_to_ca(self):
        """Residues without CB (like GLY) should use CA coords."""
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            # GLY with only CA, no CB
            f.write(
                "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00  0.00           C\n"
                "END\n"
            )
            path = f.name
        s = ProteinStructure.from_pdb(path, chain_id="A")
        assert s.N == 1
        np.testing.assert_allclose(s.coords_ca[0], s.coords_cb[0])

    def test_hydrogen_atoms_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
                "ATOM      3  H   ALA A   1       0.000   1.000   0.000  1.00  0.00           H\n"
                "ATOM      4  HA  ALA A   1       0.000   0.000   1.000  1.00  0.00           H\n"
                "END\n"
            )
            path = f.name
        s = ProteinStructure.from_pdb(path, chain_id="A")
        # H and HA should be skipped; only CB in atom list
        assert s.N == 1
        assert "H" not in s.atom_names
        assert "HA" not in s.atom_names


# ── PDB Writing ────────────────────────────────────────────────────

class TestToPDB:
    def test_roundtrip_write_read(self):
        """Write then re-parse should recover same coordinates."""
        s = _make_structure(6)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.to_pdb(path)
        s2 = ProteinStructure.from_pdb(path, chain_id="A")
        assert s2.N == 6
        np.testing.assert_allclose(s2.coords_ca, s.coords_ca, atol=0.001)

    def test_model_number(self):
        s = _make_structure(3)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.to_pdb(path, model_num=1)
        with open(path) as fh:
            content = fh.read()
        assert "MODEL" in content
        assert "ENDMDL" in content

    def test_custom_coords(self):
        s = _make_structure(4)
        custom = np.zeros((4, 3))
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.to_pdb(path, coords=custom)
        s2 = ProteinStructure.from_pdb(path, chain_id="A")
        np.testing.assert_allclose(s2.coords_ca, custom, atol=0.001)


class TestTrajectoryPDB:
    def test_write_trajectory(self):
        s = _make_structure(5)
        traj = [np.random.randn(5, 3) for _ in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.write_trajectory_pdb(path, traj)
        with open(path) as fh:
            content = fh.read()
        assert content.count("MODEL") == 3
        assert content.count("ENDMDL") == 3


class TestFromCIF:
    def _write_minimal_cif(self, path, n=3):
        """Write a minimal mmCIF file for parsing."""
        lines = [
            "data_test",
            "_atom_site.group_PDB",
            "_atom_site.auth_atom_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]
        for i in range(n):
            x = i * 3.8
            lines.append(f"ATOM CA  ALA A {i+1} {x:.3f} 0.000 0.000")
            lines.append(f"ATOM CB  ALA A {i+1} {x+0.5:.3f} 0.500 0.000")
        lines.append("#")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_parse_basic_cif(self):
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            path = f.name
        self._write_minimal_cif(path, n=4)
        s = ProteinStructure.from_cif(path, chain_id="A")
        assert s.N == 4
        assert s.coords_ca.shape == (4, 3)

    def test_cif_chain_filter(self):
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            path = f.name
        lines = [
            "data_test",
            "_atom_site.group_PDB",
            "_atom_site.auth_atom_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "ATOM CA  ALA A 1 0.000 0.000 0.000",
            "ATOM CA  ALA B 1 10.000 0.000 0.000",
            "#",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        s = ProteinStructure.from_cif(path, chain_id="B")
        assert s.N == 1
        assert s.coords_ca[0, 0] > 9.0

    def test_cif_hydrogen_skipped(self):
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as f:
            path = f.name
        lines = [
            "data_test",
            "_atom_site.group_PDB",
            "_atom_site.auth_atom_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "ATOM CA  ALA A 1 0.000 0.000 0.000",
            "ATOM CB  ALA A 1 1.000 0.000 0.000",
            "ATOM H   ALA A 1 0.000 1.000 0.000",
            "#",
        ]
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        s = ProteinStructure.from_cif(path, chain_id="A")
        assert "H" not in s.atom_names


class TestWriteSpringsPDB:
    def test_write_springs(self):
        s = _make_structure(5)
        neighbors = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
        K = np.ones((5, 5)) * 0.5
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.write_springs_pdb(path, s.coords_ca, neighbors, K)
        with open(path) as fh:
            content = fh.read()
        assert "ATOM" in content
        assert "CONECT" in content
        assert "END" in content


class TestWriteVectorsPDB:
    def test_write_vectors(self):
        s = _make_structure(4)
        vectors = np.random.randn(4, 3)
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            path = f.name
        s.write_vectors_pdb(path, s.coords_ca, vectors)
        with open(path) as fh:
            content = fh.read()
        assert "ATOM" in content
        assert "CONECT" in content
        # Should have 2 atoms per residue (start + end of vector)
        assert content.count("ATOM") == 8


class TestHelpers:
    def test_get_sidechain_representatives(self):
        s = _make_structure(5)
        sc = s.get_sidechain_representatives()
        np.testing.assert_array_equal(sc, s.coords_cb)
        # Should be a copy
        sc[0, 0] = 999.0
        assert s.coords_cb[0, 0] != 999.0

    def test_volume_table(self):
        assert "ALA" in ProteinStructure.VOLUME_TABLE
        assert "GLY" in ProteinStructure.VOLUME_TABLE
        assert len(ProteinStructure.VOLUME_TABLE) == 20
