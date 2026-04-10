"""Dataset utilities: PDB → Cα coordinates + OpenFold3 features."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .ground_truth import compute_gt_probability_matrix


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
