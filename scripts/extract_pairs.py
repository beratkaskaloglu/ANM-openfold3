#!/usr/bin/env python3
"""Phase 2: Chunked OpenFold3 inference — 10 proteins at a time.

Extracts pair representations (zij_trunk) and Cα coordinates.
Resume-safe: skips chunks with .ok markers.

Usage:
    python scripts/extract_pairs.py --pdb-list data/pdb_2000.json --chunk-size 10
    python scripts/extract_pairs.py --start 500 --end 1000  # resume from protein 500
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/extract_pairs.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_openfold3():
    """Import and configure OpenFold3 for inference."""
    try:
        from openfold3.entry_points.inference import (
            InferenceExperimentConfig,
            run_inference,
        )
        return InferenceExperimentConfig, run_inference
    except ImportError:
        logger.error(
            "OpenFold3 not found. Ensure openfold3-repo is in sys.path or installed."
        )
        sys.exit(1)


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
    run_inference,
) -> bool:
    """Run OpenFold3 inference for a single protein and extract pair repr."""
    # Check if already cached
    pair_file = pair_repr_dir / f"{pdb_id}_pair_repr.pt"
    if pair_file.exists():
        logger.info("  %s: cached", pdb_id)
        return True

    # Per-protein isolated MSA directory
    msa_dir = Path(f"/tmp/of3_msa_{pdb_id}")
    if msa_dir.exists():
        shutil.rmtree(msa_dir, ignore_errors=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    try:
        query_file = write_query_json(pdb_id, sequence, query_dir)

        config_args = {
            "output_writer_settings": {
                "structure_format": "cif",
                "write_latent_outputs": True,
                "write_full_confidence_scores": False,
                "write_features": False,
            },
            "model_update": {
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
            "msa_computation_settings": {
                "msa_output_directory": str(msa_dir),
                "cleanup_msa_dir": True,
                "save_mappings": False,
            },
        }

        config = InferenceExperimentConfig(**config_args)
        run_inference(
            queries_json=str(query_file),
            output_dir=str(output_dir),
            experiment_config=config,
        )

        # Find and extract pair representation
        zij_path = None
        for p in output_dir.rglob(f"*{pdb_id}*/**/zij_trunk*.pt"):
            zij_path = p
            break
        if zij_path is None:
            for p in output_dir.rglob(f"*{pdb_id}*/**/*latent*.pt"):
                zij_path = p
                break

        if zij_path is None:
            # Try reading from CIF output
            for p in output_dir.rglob(f"*{pdb_id}*"):
                if p.is_dir():
                    for f in p.rglob("*.pt"):
                        data = torch.load(f, map_location="cpu", weights_only=False)
                        if isinstance(data, dict) and "pair_repr" in data:
                            torch.save(
                                {"pair_repr": data["pair_repr"]}, pair_file
                            )
                            logger.info(
                                "  %s: OK zij=%s",
                                pdb_id,
                                list(data["pair_repr"].shape),
                            )
                            return True

            logger.warning("  %s: no pair repr found in output", pdb_id)
            return False

        zij = torch.load(zij_path, map_location="cpu", weights_only=False)
        if isinstance(zij, dict):
            zij = zij.get("pair_repr", zij.get("zij", list(zij.values())[0]))
        torch.save({"pair_repr": zij}, pair_file)
        logger.info("  %s: OK zij=%s", pdb_id, list(zij.shape))
        return True

    except Exception as e:
        logger.error("  %s: FAILED — %s", pdb_id, e)
        return False
    finally:
        # Cleanup MSA cache
        if msa_dir.exists():
            shutil.rmtree(msa_dir, ignore_errors=True)


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
            logger.warning("  %s: no Cα atoms found", pdb_id)
            return False

        coords = torch.tensor(np.array(ca_coords), dtype=torch.float32)
        torch.save(coords, coord_file)
        return True

    except Exception as e:
        logger.error("  %s: coord download failed — %s", pdb_id, e)
        return False


def process_chunk(
    chunk_idx: int,
    proteins: list[dict],
    query_dir: Path,
    output_dir: Path,
    pair_repr_dir: Path,
    coords_dir: Path,
    progress_dir: Path,
    InferenceExperimentConfig,
    run_inference,
) -> tuple[int, int]:
    """Process a chunk of proteins. Returns (success_count, total_count)."""
    marker = progress_dir / f"chunk_{chunk_idx:04d}.ok"
    if marker.exists():
        logger.info("Chunk %d: already completed, skipping", chunk_idx)
        return len(proteins), len(proteins)

    logger.info(
        "=== Chunk %d: %d proteins (%s ... %s) ===",
        chunk_idx,
        len(proteins),
        proteins[0]["pdb_id"],
        proteins[-1]["pdb_id"],
    )

    success = 0
    failed_ids = []

    for p in proteins:
        pdb_id = p["pdb_id"]
        sequence = p["sequence"]

        # Extract pair representation
        ok = run_single_protein(
            pdb_id,
            sequence,
            query_dir,
            output_dir,
            pair_repr_dir,
            InferenceExperimentConfig,
            run_inference,
        )

        # Download coordinates
        if ok:
            ok = download_coords(pdb_id, coords_dir)

        if ok:
            success += 1
        else:
            failed_ids.append(pdb_id)

    # GPU cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write marker
    marker.write_text(
        json.dumps(
            {
                "success": success,
                "total": len(proteins),
                "failed": failed_ids,
            }
        )
    )

    logger.info(
        "Chunk %d: %d/%d OK, failed: %s",
        chunk_idx,
        success,
        len(proteins),
        failed_ids or "none",
    )

    return success, len(proteins)


def main():
    parser = argparse.ArgumentParser(
        description="Chunked OpenFold3 inference for pair representation extraction"
    )
    parser.add_argument(
        "--pdb-list",
        type=str,
        default="data/pdb_2000.json",
        help="Path to curated PDB list JSON",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10, help="Proteins per chunk"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index (protein number)"
    )
    parser.add_argument(
        "--end", type=int, default=-1, help="End index (-1 = all)"
    )
    parser.add_argument(
        "--pair-repr-dir",
        type=str,
        default="/content/pair_reprs",
        help="Directory for pair representation .pt files",
    )
    parser.add_argument(
        "--coords-dir",
        type=str,
        default="/content/coords",
        help="Directory for Cα coordinate .pt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/of3_output",
        help="OpenFold3 raw output directory",
    )
    parser.add_argument(
        "--query-dir",
        type=str,
        default="/content/queries",
        help="Directory for query JSON files",
    )
    parser.add_argument(
        "--progress-dir",
        type=str,
        default="data/progress",
        help="Directory for chunk completion markers",
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
        args.start,
        args.start + len(proteins),
        len(proteins),
    )

    # Create directories
    pair_repr_dir = Path(args.pair_repr_dir)
    coords_dir = Path(args.coords_dir)
    output_dir = Path(args.output_dir)
    query_dir = Path(args.query_dir)
    progress_dir = Path(args.progress_dir)

    for d in [pair_repr_dir, coords_dir, output_dir, query_dir, progress_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup OpenFold3
    InferenceExperimentConfig, run_inference_fn = setup_openfold3()

    # Process chunks
    total_success = 0
    total_processed = 0
    failed_all = []

    t0 = time.time()

    for i in range(0, len(proteins), args.chunk_size):
        chunk = proteins[i : i + args.chunk_size]
        chunk_idx = (args.start + i) // args.chunk_size

        success, total = process_chunk(
            chunk_idx=chunk_idx,
            proteins=chunk,
            query_dir=query_dir,
            output_dir=output_dir,
            pair_repr_dir=pair_repr_dir,
            coords_dir=coords_dir,
            progress_dir=progress_dir,
            InferenceExperimentConfig=InferenceExperimentConfig,
            run_inference=run_inference_fn,
        )

        total_success += success
        total_processed += total

        elapsed = time.time() - t0
        rate = total_processed / elapsed if elapsed > 0 else 0
        eta = (len(proteins) - total_processed) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d success (%.1f%%), %.1f prot/min, ETA %.0f min",
            total_success,
            total_processed,
            100 * total_success / max(total_processed, 1),
            rate * 60,
            eta / 60,
        )

    # Save failed list
    failed_path = Path("data/failed_pdbs.json")
    marker_files = sorted(progress_dir.glob("chunk_*.ok"))
    for mf in marker_files:
        info = json.loads(mf.read_text())
        failed_all.extend(info.get("failed", []))

    failed_path.write_text(json.dumps(failed_all, indent=2))

    elapsed = time.time() - t0
    logger.info(
        "\n=== DONE ===\n"
        "  Total: %d/%d extracted (%.1f%%)\n"
        "  Failed: %d proteins (saved to %s)\n"
        "  Time: %.1f min",
        total_success,
        total_processed,
        100 * total_success / max(total_processed, 1),
        len(failed_all),
        failed_path,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
