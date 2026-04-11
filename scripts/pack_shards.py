#!/usr/bin/env python3
"""Phase 3: Pack individual .pt files into .npz shards.

Groups proteins into shards of N (default 50) for efficient I/O.

Usage:
    python scripts/pack_shards.py
    python scripts/pack_shards.py --shard-size 100 --delete-pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def find_valid_proteins(
    pair_repr_dir: Path,
    coords_dir: Path,
) -> list[str]:
    """Find proteins that have both pair_repr and coords files."""
    pair_files = {p.stem.replace("_pair_repr", "") for p in pair_repr_dir.glob("*_pair_repr.pt")}
    coord_files = {p.stem.replace("_ca", "") for p in coords_dir.glob("*_ca.pt")}

    valid = sorted(pair_files & coord_files)
    logger.info(
        "Found %d pair_repr, %d coords, %d valid (intersection)",
        len(pair_files),
        len(coord_files),
        len(valid),
    )
    return valid


def pack_shard(
    shard_idx: int,
    pdb_ids: list[str],
    pair_repr_dir: Path,
    coords_dir: Path,
    shard_dir: Path,
) -> Path:
    """Pack a group of proteins into a single .npz file."""
    shard_path = shard_dir / f"shard_{shard_idx:04d}.npz"

    if shard_path.exists():
        logger.info("  Shard %04d: already exists, skipping", shard_idx)
        return shard_path

    arrays = {"pdb_ids": np.array(pdb_ids, dtype=object)}
    skipped = []

    for i, pdb_id in enumerate(pdb_ids):
        try:
            # Load pair representation
            pr_data = torch.load(
                pair_repr_dir / f"{pdb_id}_pair_repr.pt",
                map_location="cpu",
                weights_only=False,
            )
            pair_repr = pr_data if isinstance(pr_data, torch.Tensor) else pr_data["pair_repr"]
            if pair_repr.dim() == 4:
                pair_repr = pair_repr.squeeze(0)  # [N, N, c_z]

            # Load coordinates
            coords = torch.load(
                coords_dir / f"{pdb_id}_ca.pt",
                map_location="cpu",
                weights_only=True,
            )

            arrays[f"pair_repr_{i}"] = pair_repr.numpy()
            arrays[f"coords_ca_{i}"] = coords.numpy()

        except Exception as e:
            logger.warning("  %s: failed to load — %s", pdb_id, e)
            skipped.append(pdb_id)
            # Store empty placeholders
            arrays[f"pair_repr_{i}"] = np.zeros((1, 1, 1), dtype=np.float32)
            arrays[f"coords_ca_{i}"] = np.zeros((1, 3), dtype=np.float32)

    np.savez_compressed(shard_path, **arrays)

    size_mb = shard_path.stat().st_size / (1024 * 1024)
    logger.info(
        "  Shard %04d: %d proteins, %.1f MB%s",
        shard_idx,
        len(pdb_ids),
        size_mb,
        f" (skipped: {skipped})" if skipped else "",
    )
    return shard_path


def verify_shard(shard_path: Path) -> tuple[int, int]:
    """Verify a shard loads correctly. Returns (ok_count, total_count)."""
    data = np.load(shard_path, allow_pickle=True)
    pdb_ids = data["pdb_ids"]
    ok = 0
    for i in range(len(pdb_ids)):
        pr = data[f"pair_repr_{i}"]
        ca = data[f"coords_ca_{i}"]
        if pr.shape[0] > 1 and ca.shape[0] > 1:
            ok += 1
    return ok, len(pdb_ids)


def main():
    parser = argparse.ArgumentParser(description="Pack .pt files into .npz shards")
    parser.add_argument(
        "--pair-repr-dir",
        type=str,
        default="/content/pair_reprs",
        help="Directory with pair_repr .pt files",
    )
    parser.add_argument(
        "--coords-dir",
        type=str,
        default="/content/coords",
        help="Directory with coords .pt files",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="data/shards",
        help="Output directory for .npz shards",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=50,
        help="Number of proteins per shard",
    )
    parser.add_argument(
        "--delete-pt",
        action="store_true",
        help="Delete individual .pt files after packing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify shards after packing",
    )
    args = parser.parse_args()

    pair_repr_dir = Path(args.pair_repr_dir)
    coords_dir = Path(args.coords_dir)
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Find valid proteins
    valid_ids = find_valid_proteins(pair_repr_dir, coords_dir)
    if not valid_ids:
        logger.error("No valid proteins found!")
        sys.exit(1)

    # Pack shards
    logger.info("Packing %d proteins into shards of %d...", len(valid_ids), args.shard_size)
    shard_paths = []

    for i in range(0, len(valid_ids), args.shard_size):
        chunk = valid_ids[i : i + args.shard_size]
        shard_idx = i // args.shard_size
        sp = pack_shard(shard_idx, chunk, pair_repr_dir, coords_dir, shard_dir)
        shard_paths.append(sp)

    # Verify
    if args.verify:
        logger.info("\nVerifying %d shards...", len(shard_paths))
        total_ok = 0
        total_n = 0
        for sp in shard_paths:
            ok, n = verify_shard(sp)
            total_ok += ok
            total_n += n
        logger.info("Verification: %d/%d proteins OK", total_ok, total_n)

    # Optionally delete .pt files
    if args.delete_pt:
        logger.info("Deleting individual .pt files...")
        for pdb_id in valid_ids:
            (pair_repr_dir / f"{pdb_id}_pair_repr.pt").unlink(missing_ok=True)
            (coords_dir / f"{pdb_id}_ca.pt").unlink(missing_ok=True)
        logger.info("Deleted %d × 2 .pt files", len(valid_ids))

    # Summary
    total_size = sum(sp.stat().st_size for sp in shard_paths) / (1024 * 1024)
    logger.info(
        "\n=== DONE ===\n"
        "  Shards: %d files in %s\n"
        "  Proteins: %d\n"
        "  Total size: %.1f MB",
        len(shard_paths),
        shard_dir,
        len(valid_ids),
        total_size,
    )


if __name__ == "__main__":
    main()
