"""Tests for dataset utilities."""

import numpy as np
import torch
import pytest

from src.data import ShardedPairReprDataset


# ---------------------------------------------------------------------------
# BioPython-dependent tests
# ---------------------------------------------------------------------------
try:
    from Bio.PDB import PDBParser

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


@pytest.mark.skipif(not HAS_BIOPYTHON, reason="BioPython not installed")
class TestExtractCaCoords:
    def test_basic_shape(self, tmp_path):
        """Write a minimal PDB with 3 residues and check output shape."""
        from src.data import extract_ca_coords

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  VAL A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(pdb_content)

        coords = extract_ca_coords(pdb_file)
        assert coords.shape == (3, 3)
        assert coords.dtype == torch.float32

    def test_coordinate_values(self, tmp_path):
        from src.data import extract_ca_coords

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "one.pdb"
        pdb_file.write_text(pdb_content)

        coords = extract_ca_coords(pdb_file)
        assert coords.shape == (1, 3)
        assert torch.allclose(coords, torch.tensor([[1.5, 2.5, 3.5]]), atol=0.1)

    def test_raises_on_no_ca(self, tmp_path):
        from src.data import extract_ca_coords

        pdb_content = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
END
"""
        pdb_file = tmp_path / "no_ca.pdb"
        pdb_file.write_text(pdb_content)

        with pytest.raises(ValueError, match="No C.*atoms found"):
            extract_ca_coords(pdb_file)


# ---------------------------------------------------------------------------
# ProteinContactDataset tests (require BioPython + real PDB)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_BIOPYTHON, reason="BioPython not installed")
class TestProteinContactDataset:
    def test_length_matches_paths(self, tmp_path):
        from src.data import ProteinContactDataset

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  VAL A   3       7.000   8.000   9.000  1.00  0.00           C
END
"""
        paths = []
        for i in range(3):
            p = tmp_path / f"prot_{i}.pdb"
            p.write_text(pdb_content)
            paths.append(p)

        ds = ProteinContactDataset(paths)
        assert len(ds) == 3

    def test_item_keys(self, tmp_path):
        from src.data import ProteinContactDataset

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
END
"""
        p = tmp_path / "test.pdb"
        p.write_text(pdb_content)

        ds = ProteinContactDataset([p])
        item = ds[0]
        assert "coords_ca" in item
        assert "c_gt" in item
        assert "pdb_id" in item

    def test_c_gt_shape_and_symmetry(self, tmp_path):
        from src.data import ProteinContactDataset

        pdb_content = """\
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  CA  VAL A   3       7.000   8.000   9.000  1.00  0.00           C
ATOM      4  CA  LEU A   4      10.000  11.000  12.000  1.00  0.00           C
END
"""
        p = tmp_path / "four.pdb"
        p.write_text(pdb_content)

        ds = ProteinContactDataset([p])
        item = ds[0]
        c_gt = item["c_gt"]
        assert c_gt.shape == (4, 4)
        assert torch.allclose(c_gt, c_gt.T, atol=1e-6)


# ---------------------------------------------------------------------------
# ShardedPairReprDataset tests
# ---------------------------------------------------------------------------
def _make_shard(path, pdb_ids, n_residues_list, c_z=128):
    """Create a .npz shard file with synthetic data."""
    data = {"pdb_ids": np.array(pdb_ids)}
    for i, n in enumerate(n_residues_list):
        data[f"pair_repr_{i}"] = np.random.randn(n, n, c_z).astype(np.float32)
        data[f"coords_ca_{i}"] = np.random.randn(n, 3).astype(np.float32)
    np.savez(str(path), **data)


class TestShardedPairReprDataset:
    def test_length_single_shard(self, tmp_path):
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["1abc", "2def", "3ghi"], [8, 10, 12], c_z=64)

        ds = ShardedPairReprDataset([shard], r_cut=8.0, tau=1.0)
        assert len(ds) == 3

    def test_length_multi_shard(self, tmp_path):
        s0 = tmp_path / "shard_0.npz"
        s1 = tmp_path / "shard_1.npz"
        _make_shard(s0, ["a", "b"], [6, 8], c_z=64)
        _make_shard(s1, ["c"], [10], c_z=64)

        ds = ShardedPairReprDataset([s0, s1], r_cut=8.0, tau=1.0)
        assert len(ds) == 3

    def test_item_keys(self, tmp_path):
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["1abc"], [10], c_z=64)

        ds = ShardedPairReprDataset([shard])
        item = ds[0]
        assert "pair_repr" in item
        assert "c_gt" in item
        assert "pdb_id" in item

    def test_item_shapes(self, tmp_path):
        n, c_z = 12, 64
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["test"], [n], c_z=c_z)

        ds = ShardedPairReprDataset([shard])
        item = ds[0]
        assert item["pair_repr"].shape == (n, n, c_z)
        assert item["c_gt"].shape == (n, n)

    def test_item_dtypes(self, tmp_path):
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["test"], [8], c_z=32)

        ds = ShardedPairReprDataset([shard])
        item = ds[0]
        assert item["pair_repr"].dtype == torch.float32
        assert item["c_gt"].dtype == torch.float32

    def test_variable_size_proteins(self, tmp_path):
        """Proteins with different sizes should return correct per-item shapes."""
        sizes = [6, 10, 15]
        c_z = 32
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["a", "b", "c"], sizes, c_z=c_z)

        ds = ShardedPairReprDataset([shard])
        for i, n in enumerate(sizes):
            item = ds[i]
            assert item["pair_repr"].shape == (n, n, c_z)
            assert item["c_gt"].shape == (n, n)

    def test_c_gt_symmetry_and_range(self, tmp_path):
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["sym"], [10], c_z=32)

        ds = ShardedPairReprDataset([shard])
        c_gt = ds[0]["c_gt"]
        assert torch.allclose(c_gt, c_gt.T, atol=1e-6)
        assert (c_gt >= 0.0).all()
        assert (c_gt <= 1.0).all()

    def test_pdb_id_string(self, tmp_path):
        shard = tmp_path / "shard_0.npz"
        _make_shard(shard, ["my_protein"], [8], c_z=32)

        ds = ShardedPairReprDataset([shard])
        assert ds[0]["pdb_id"] == "my_protein"
