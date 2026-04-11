#!/usr/bin/env python3
"""Phase 2+3 combined: Chunked OpenFold3 inference with inline shard packing.

Every SHARD_SIZE proteins (default 50):
  1. Run inference in chunks of 10 → save .pt
  2. Download PDB → extract Cα coords → save .pt
  3. Pack completed .pt files into .npz shard
  4. Delete .pt files to free disk

Resume-safe: skips shards with existing .npz + .ok marker.
Disk usage stays under ~2 GB at all times.

Usage:
    python scripts/extract_pairs.py --pdb-list data/pdb_2000.json
    python scripts/extract_pairs.py --start 500 --end 1000
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# Ensure openfold3 and src/ are importable when run as subprocess
_script_dir = Path(__file__).resolve().parent.parent
_of3_dir = _script_dir / "openfold3-repo"
for _p in [str(_script_dir), str(_of3_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── OpenFold3 setup ──────────────────────────────────────────────

def setup_openfold3():
    """Import OpenFold3 classes for inference."""
    try:
        from openfold3.entry_points.validator import InferenceExperimentConfig
        from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
        from openfold3.projects.of3_all_atom.config.inference_query_format import (
            InferenceQuerySet,
        )
        return InferenceExperimentConfig, InferenceExperimentRunner, InferenceQuerySet
    except ImportError as e:
        logger.error("OpenFold3 import failed: %s", e)
        logger.error("Ensure openfold3-repo is installed: pip install -e openfold3-repo")
        sys.exit(1)


# ── Single protein inference ─────────────────────────────────────

def write_query_json(pdb_id: str, sequence: str, query_dir: Path) -> Path:
    """Write OpenFold3 query JSON for a single protein."""
    query = {
        "queries": {
            pdb_id: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": sequence,
                    }
                ]
            }
        }
    }
    query_file = query_dir / f"{pdb_id}.json"
    query_file.write_text(json.dumps(query, indent=2))
    return query_file


def run_single_protein(
    pdb_id: str,
    sequence: str,
    query_dir: Path,
    output_dir: Path,
    pair_repr_dir: Path,
    InferenceExperimentConfig,
    InferenceExperimentRunner,
    InferenceQuerySet,
) -> bool:
    """Run OpenFold3 inference for one protein and save pair repr .pt."""
    pair_file = pair_repr_dir / f"{pdb_id}_pair_repr.pt"
    if pair_file.exists():
        logger.info("  %s: cached", pdb_id)
        return True

    msa_dir = Path(f"/tmp/of3_msa_{pdb_id}")
    if msa_dir.exists():
        shutil.rmtree(msa_dir, ignore_errors=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    runner = None
    try:
        query_file = write_query_json(pdb_id, sequence, query_dir)

        config = InferenceExperimentConfig(
            output_writer_settings={
                "structure_format": "cif",
                "write_latent_outputs": True,
                "write_full_confidence_scores": False,
                "write_features": False,
            },
            model_update={
                "custom": {
                    "settings": {
                        "memory": {
                            "eval": {
                                "use_deepspeed_evo_attention": False,
                                "use_cueq_triangle_kernels": False,
                                "use_triton_triangle_kernels": False,
                            }
                        }
                    }
                }
            },
            msa_computation_settings={
                "msa_output_directory": str(msa_dir),
                "cleanup_msa_dir": True,
                "save_mappings": False,
            },
        )

        out_dir = output_dir / pdb_id
        out_dir.mkdir(parents=True, exist_ok=True)

        runner = InferenceExperimentRunner(
            config,
            num_diffusion_samples=1,
            num_model_seeds=1,
            use_msa_server=True,
            use_templates=False,
            output_dir=out_dir,
        )

        query_set = InferenceQuerySet.from_json(str(query_file))
        runner.setup()
        runner.run(query_set)

        # Find latent output with pair representation
        latent_files = list(out_dir.rglob("*_latent_output.pt"))
        if latent_files:
            latent = torch.load(latent_files[0], map_location="cpu", weights_only=False)
            zij = latent.get("pair_repr", latent.get("zij_trunk"))
            if zij is not None:
                torch.save({"pair_repr": zij}, pair_file)
                logger.info("  %s: OK zij=%s", pdb_id, list(zij.shape))
                return True

        # Fallback: search all .pt files
        for pt_file in out_dir.rglob("*.pt"):
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                for key in ["pair_repr", "zij_trunk", "zij"]:
                    if key in data:
                        torch.save({"pair_repr": data[key]}, pair_file)
                        logger.info("  %s: OK zij=%s", pdb_id, list(data[key].shape))
                        return True

        logger.warning("  %s: no pair repr found in output", pdb_id)
        return False

    except Exception as e:
        logger.error("  %s: FAILED — %s", pdb_id, e)
        return False
    finally:
        if msa_dir.exists():
            shutil.rmtree(msa_dir, ignore_errors=True)
        if runner is not None:
            try:
                runner.cleanup()
            except Exception:
                pass


def download_coords(pdb_id: str, coords_dir: Path) -> bool:
    """Download PDB and extract Cα coordinates."""
    coord_file = coords_dir / f"{pdb_id}_ca.pt"
    if coord_file.exists():
        return True

    try:
        import Bio.PDB as bpdb

        parser = bpdb.PDBParser(QUIET=True)
        pdbl = bpdb.PDBList()

        pdb_cache = Path("/tmp/pdb_cache")
        pdb_cache.mkdir(exist_ok=True)

        pdb_file = pdbl.retrieve_pdb_file(
            pdb_id, pdir=str(pdb_cache), file_format="pdb"
        )
        structure = parser.get_structure(pdb_id, pdb_file)
        first_chain = list(structure[0].get_chains())[0]

        ca_coords = []
        for res in first_chain:
            if res.get_id()[0] == " " and "CA" in res:
                ca_coords.append(res["CA"].get_vector().get_array())

        if not ca_coords:
            logger.warning("  %s: no Ca atoms found", pdb_id)
            return False

        coords = torch.tensor(np.array(ca_coords), dtype=torch.float32)
        torch.save(coords, coord_file)
        return True

    except Exception as e:
        logger.error("  %s: coord download failed — %s", pdb_id, e)
        return False


# ── Shard packing ────────────────────────────────────────────────

def pack_shard(
    shard_idx: int,
    pdb_ids: list[str],
    pair_repr_dir: Path,
    coords_dir: Path,
    shard_dir: Path,
) -> tuple[Path, int]:
    """Pack completed .pt files into one .npz shard. Returns (path, ok_count)."""
    shard_path = shard_dir / f"shard_{shard_idx:04d}.npz"

    valid_ids = []
    arrays: dict = {}

    for pdb_id in pdb_ids:
        pr_file = pair_repr_dir / f"{pdb_id}_pair_repr.pt"
        ca_file = coords_dir / f"{pdb_id}_ca.pt"
        if not pr_file.exists() or not ca_file.exists():
            continue

        try:
            pr_data = torch.load(pr_file, map_location="cpu", weights_only=False)
            pair_repr = pr_data if isinstance(pr_data, torch.Tensor) else pr_data["pair_repr"]
            if pair_repr.dim() == 4:
                pair_repr = pair_repr.squeeze(0)

            coords = torch.load(ca_file, map_location="cpu", weights_only=True)

            idx = len(valid_ids)
            arrays[f"pair_repr_{idx}"] = pair_repr.numpy()
            arrays[f"coords_ca_{idx}"] = coords.numpy()
            valid_ids.append(pdb_id)

        except Exception as e:
            logger.warning("  %s: skip packing — %s", pdb_id, e)

    if not valid_ids:
        logger.warning("  Shard %04d: no valid proteins, skipping", shard_idx)
        return shard_path, 0

    arrays["pdb_ids"] = np.array(valid_ids, dtype=object)
    np.savez_compressed(shard_path, **arrays)

    size_mb = shard_path.stat().st_size / (1024 * 1024)
    logger.info(
        "  Shard %04d: %d proteins, %.1f MB",
        shard_idx, len(valid_ids), size_mb,
    )
    return shard_path, len(valid_ids)


def cleanup_pt_files(pdb_ids: list[str], pair_repr_dir: Path, coords_dir: Path):
    """Delete .pt files after shard packing to free disk."""
    for pdb_id in pdb_ids:
        (pair_repr_dir / f"{pdb_id}_pair_repr.pt").unlink(missing_ok=True)
        (coords_dir / f"{pdb_id}_ca.pt").unlink(missing_ok=True)


# ── Main pipeline ────────────────────────────────────────────────

def process_shard_group(
    shard_idx: int,
    proteins: list[dict],
    query_dir: Path,
    output_dir: Path,
    pair_repr_dir: Path,
    coords_dir: Path,
    shard_dir: Path,
    progress_dir: Path,
    InferenceExperimentConfig,
    InferenceExperimentRunner,
    InferenceQuerySet,
    inference_chunk_size: int = 10,
) -> tuple[int, int]:
    """Process a shard group: inference → pack → cleanup.

    Returns (success_count, total_count).
    """
    marker = progress_dir / f"shard_{shard_idx:04d}.ok"
    shard_path = shard_dir / f"shard_{shard_idx:04d}.npz"

    if marker.exists() and shard_path.exists():
        info = json.loads(marker.read_text())
        logger.info(
            "Shard %04d: already completed (%d/%d), skipping",
            shard_idx, info["success"], info["total"],
        )
        return info["success"], info["total"]

    logger.info(
        "============================================================\n"
        "  Shard %04d: %d proteins (%s ... %s)\n"
        "============================================================",
        shard_idx, len(proteins),
        proteins[0]["pdb_id"], proteins[-1]["pdb_id"],
    )

    # Step 1: Run inference + download coords (in sub-chunks of 10)
    success_ids = []
    failed_ids = []

    for i in range(0, len(proteins), inference_chunk_size):
        sub_chunk = proteins[i : i + inference_chunk_size]
        for p in sub_chunk:
            pdb_id = p["pdb_id"]
            sequence = p["sequence"]

            ok = run_single_protein(
                pdb_id, sequence, query_dir, output_dir, pair_repr_dir,
                InferenceExperimentConfig, InferenceExperimentRunner,
                InferenceQuerySet,
            )

            if ok:
                ok = download_coords(pdb_id, coords_dir)

            if ok:
                success_ids.append(pdb_id)
            else:
                failed_ids.append(pdb_id)

        # GPU cleanup between sub-chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 2: Pack into .npz shard
    all_ids = [p["pdb_id"] for p in proteins]
    _, n_packed = pack_shard(shard_idx, all_ids, pair_repr_dir, coords_dir, shard_dir)

    # Step 3: Delete .pt files to free disk
    cleanup_pt_files(all_ids, pair_repr_dir, coords_dir)

    # Clean OF3 output for these proteins
    for p in proteins:
        pdb_dir = output_dir / p["pdb_id"]
        if pdb_dir.exists():
            shutil.rmtree(pdb_dir, ignore_errors=True)
        # Also check for directories containing the PDB ID
        for d in output_dir.glob(f"*{p['pdb_id']}*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

    # Write marker
    marker.write_text(
        json.dumps({
            "success": len(success_ids),
            "total": len(proteins),
            "packed": n_packed,
            "failed": failed_ids,
        })
    )

    logger.info(
        "Shard %04d done: %d/%d extracted, %d packed, failed: %s",
        shard_idx, len(success_ids), len(proteins),
        n_packed, failed_ids or "none",
    )

    return len(success_ids), len(proteins)


def main():
    parser = argparse.ArgumentParser(
        description="Chunked OpenFold3 inference + inline shard packing"
    )
    parser.add_argument(
        "--pdb-list", type=str, default="data/pdb_2000.json",
        help="Path to curated PDB list JSON",
    )
    parser.add_argument(
        "--shard-size", type=int, default=50,
        help="Proteins per .npz shard (default 50)",
    )
    parser.add_argument(
        "--inference-chunk-size", type=int, default=10,
        help="Proteins per inference sub-chunk (default 10)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start protein index")
    parser.add_argument("--end", type=int, default=-1, help="End index (-1 = all)")
    parser.add_argument(
        "--pair-repr-dir", type=str, default="/content/pair_reprs",
        help="Temp dir for pair repr .pt files",
    )
    parser.add_argument(
        "--coords-dir", type=str, default="/content/coords",
        help="Temp dir for coord .pt files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/content/of3_output",
        help="OpenFold3 raw output dir",
    )
    parser.add_argument(
        "--query-dir", type=str, default="/content/queries",
        help="Dir for query JSON files",
    )
    parser.add_argument(
        "--shard-dir", type=str, default="data/shards",
        help="Output dir for .npz shards",
    )
    parser.add_argument(
        "--progress-dir", type=str, default="data/progress",
        help="Dir for completion markers",
    )
    args = parser.parse_args()

    # Load PDB list
    pdb_list_path = Path(args.pdb_list)
    if not pdb_list_path.exists():
        logger.error("PDB list not found: %s", pdb_list_path)
        sys.exit(1)

    proteins = json.loads(pdb_list_path.read_text())
    end = args.end if args.end > 0 else len(proteins)
    proteins = proteins[args.start : end]
    logger.info(
        "Processing proteins %d–%d (%d total)",
        args.start, args.start + len(proteins), len(proteins),
    )

    # Create dirs
    dirs = {
        "pair_repr": Path(args.pair_repr_dir),
        "coords": Path(args.coords_dir),
        "output": Path(args.output_dir),
        "query": Path(args.query_dir),
        "shard": Path(args.shard_dir),
        "progress": Path(args.progress_dir),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Setup OpenFold3
    InferenceExperimentConfig, InferenceExperimentRunner, InferenceQuerySet = (
        setup_openfold3()
    )

    # Process shard groups
    total_success = 0
    total_processed = 0
    t0 = time.time()

    for i in range(0, len(proteins), args.shard_size):
        group = proteins[i : i + args.shard_size]
        shard_idx = (args.start + i) // args.shard_size

        success, total = process_shard_group(
            shard_idx=shard_idx,
            proteins=group,
            query_dir=dirs["query"],
            output_dir=dirs["output"],
            pair_repr_dir=dirs["pair_repr"],
            coords_dir=dirs["coords"],
            shard_dir=dirs["shard"],
            progress_dir=dirs["progress"],
            InferenceExperimentConfig=InferenceExperimentConfig,
            InferenceExperimentRunner=InferenceExperimentRunner,
            InferenceQuerySet=InferenceQuerySet,
            inference_chunk_size=args.inference_chunk_size,
        )

        total_success += success
        total_processed += total

        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        eta = (len(proteins) - total_processed) / rate if rate > 0 else 0
        logger.info(
            "Overall: %d/%d success (%.1f%%), %.1f prot/min, ETA %.0f min",
            total_success, total_processed,
            100 * total_success / max(total_processed, 1),
            rate * 60, eta / 60,
        )

    # Collect failed list
    failed_all = []
    for mf in sorted(dirs["progress"].glob("shard_*.ok")):
        info = json.loads(mf.read_text())
        failed_all.extend(info.get("failed", []))

    failed_path = Path("data/failed_pdbs.json")
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(json.dumps(failed_all, indent=2))

    # Shard stats
    shard_files = sorted(dirs["shard"].glob("shard_*.npz"))
    total_size_mb = sum(f.stat().st_size for f in shard_files) / (1024 * 1024)

    elapsed = time.time() - t0
    logger.info(
        "\n=== DONE ===\n"
        "  Extracted: %d/%d (%.1f%%)\n"
        "  Shards: %d files (%.1f MB total)\n"
        "  Failed: %d (saved to %s)\n"
        "  Time: %.1f min",
        total_success, total_processed,
        100 * total_success / max(total_processed, 1),
        len(shard_files), total_size_mb,
        len(failed_all), failed_path,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
