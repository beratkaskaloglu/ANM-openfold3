"""Dataset utilities: PDB → Cα coordinates + OpenFold3 features."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .ground_truth import compute_gt_probability_matrix

logger = logging.getLogger(__name__)


def extract_ca_coords(pdb_path: str | Path) -> torch.Tensor:
    """Extract Cα atom coordinates from a PDB/mmCIF file.

    Uses BioPython's PDBParser / MMCIFParser.

    Args:
        pdb_path: path to PDB or mmCIF file.

    Returns:
        coords: [N_res, 3] float32 tensor of Cα positions (Å).
    """
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(pdb_path)
    if path.suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("protein", str(path))

    coords: list[list[float]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(list(residue["CA"].get_vector()))
        break  # first model only

    if not coords:
        raise ValueError(f"No Cα atoms found in {pdb_path}")

    return torch.tensor(coords, dtype=torch.float32)


class ProteinContactDataset(Dataset):
    """Dataset that returns Cα coords and (optionally) cached pair repr.

    Each item provides:
        - ``coords_ca``: [N, 3]
        - ``c_gt``:      [N, N] ground-truth soft contact map
        - ``pair_repr``: [N, N, c_z] if cache dir is available, else None
        - ``pdb_id``:    str identifier
    """

    def __init__(
        self,
        pdb_paths: list[str | Path],
        cache_dir: Optional[str | Path] = None,
        r_cut: float = 10.0,
        tau: float = 1.5,
    ) -> None:
        self.pdb_paths = [Path(p) for p in pdb_paths]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.r_cut = r_cut
        self.tau = tau

    def __len__(self) -> int:
        return len(self.pdb_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pdb_path = self.pdb_paths[idx]
        pdb_id = pdb_path.stem

        coords_ca = extract_ca_coords(pdb_path)
        c_gt = compute_gt_probability_matrix(
            coords_ca, r_cut=self.r_cut, tau=self.tau
        )

        item: Dict[str, Any] = {
            "coords_ca": coords_ca,
            "c_gt": c_gt,
            "pdb_id": pdb_id,
        }

        # Load cached pair representation if available
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{pdb_id}_pair.pt"
            if cache_file.exists():
                item["pair_repr"] = torch.load(
                    cache_file, map_location="cpu", weights_only=True
                )

        return item


class ShardedPairReprDataset(Dataset):
    """Dataset that lazily loads .npz shards containing pair representations.

    Each shard contains multiple proteins with variable-size tensors.
    Shards are loaded on demand and cached in memory (one shard at a time).

    Shard format:
        pdb_ids: list of PDB identifiers
        pair_repr_{i}: [N_i, N_i, c_z] pair representation for protein i
        coords_ca_{i}: [N_i, 3] Cα coordinates for protein i
    """

    def __init__(
        self,
        shard_paths: List[Path],
        r_cut: float = 8.0,
        tau: float = 1.0,
    ) -> None:
        self.shard_paths = sorted(shard_paths)
        self.r_cut = r_cut
        self.tau = tau

        # Build index: (shard_idx, protein_idx_within_shard)
        self._index: List[tuple[int, int]] = []
        self._shard_sizes: List[int] = []

        for shard_idx, sp in enumerate(self.shard_paths):
            with np.load(sp, allow_pickle=True) as data:
                pdb_ids = data["pdb_ids"]
                # Count actually available proteins (not just pdb_ids length)
                n = 0
                for j in range(len(pdb_ids)):
                    if f"pair_repr_{j}" in data:
                        n += 1
                    else:
                        break
            self._shard_sizes.append(n)
            for j in range(n):
                self._index.append((shard_idx, j))

        # Pre-load ALL shards into memory (avoids disk re-reads during training)
        self._all_data: List[dict] = []
        for sp in self.shard_paths:
            self._all_data.append(dict(np.load(sp, allow_pickle=True)))

        logger.info(
            "ShardedDataset: %d proteins across %d shards (all in memory)",
            len(self._index),
            len(self.shard_paths),
        )

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> dict:
        return self._all_data[shard_idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_idx, protein_idx = self._index[idx]
        data = self._load_shard(shard_idx)

        # Guard against partial shards
        pdb_ids = data["pdb_ids"]
        if protein_idx >= len(pdb_ids) or f"pair_repr_{protein_idx}" not in data:
            # Fallback to first valid protein in this shard
            protein_idx = 0

        pdb_id = str(data["pdb_ids"][protein_idx])
        pair_repr = torch.from_numpy(
            data[f"pair_repr_{protein_idx}"]
        ).float()
        coords_ca = torch.from_numpy(
            data[f"coords_ca_{protein_idx}"]
        ).float()

        c_gt = compute_gt_probability_matrix(
            coords_ca, r_cut=self.r_cut, tau=self.tau
        )

        # Align sizes (n_token may differ from n_ca)
        n_tok = pair_repr.shape[0]
        n_ca = coords_ca.shape[0]
        n = min(n_tok, n_ca)

        pair_repr = pair_repr[:n, :n, :]
        c_gt = c_gt[:n, :n]

        return {
            "pair_repr": pair_repr,   # [N, N, c_z] — DataLoader adds batch dim
            "c_gt": c_gt,             # [N, N]
            "pdb_id": pdb_id,
        }
