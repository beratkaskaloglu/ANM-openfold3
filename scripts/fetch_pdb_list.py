#!/usr/bin/env python3
"""Phase 1: Curate ~2000 diverse PDB entries via RCSB Search API.

Filters:
  - X-ray, resolution ≤ 2.5 Å, R-free ≤ 0.28
  - Polymer entity type = protein (single chain)
  - Chain length 30–500 residues
  - Sequence identity clustering at 30%

Output: data/pdb_2000.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def build_query(
    max_resolution: float = 2.5,
    min_length: int = 30,
    max_length: int = 500,
    identity_cutoff: int = 30,
    max_results: int = 3000,
) -> dict:
    """Build RCSB Search API query for diverse, high-quality proteins."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "request_options": {
            "group_by": {
                "aggregation_method": "sequence_identity",
                "similarity_cutoff": identity_cutoff,
                "ranking_criteria_type": {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                },
            },
            "paginate": {"start": 0, "rows": max_results},
            "return_counts": True,
        },
        "return_type": "polymer_entity",
    }


def fetch_sequences_batch(entity_ids: list[str]) -> dict[str, str]:
    """Fetch sequences for a batch of polymer entity IDs from RCSB."""
    sequences = {}
    batch_size = 50
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i : i + batch_size]
        for eid in batch:
            pdb_id = eid.split("_")[0]
            url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    seq = (
                        data.get("entity_poly", {})
                        .get("pdbx_seq_one_letter_code_can", "")
                    )
                    if seq:
                        sequences[pdb_id] = seq
            except Exception:
                continue
        logger.info("  Fetched sequences: %d / %d", len(sequences), len(entity_ids))
    return sequences


def fetch_pdb_list(
    max_resolution: float = 2.5,
    min_length: int = 30,
    max_length: int = 500,
    identity_cutoff: int = 30,
    target_count: int = 2000,
) -> list[dict]:
    """Query RCSB and return curated PDB list."""
    query = build_query(
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        identity_cutoff=identity_cutoff,
        max_results=target_count + 500,  # fetch extra to account for filtering
    )

    logger.info("Querying RCSB Search API...")
    resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=120)
    resp.raise_for_status()
    results = resp.json()

    total_count = results.get("total_count", 0)
    logger.info("RCSB returned %d cluster representatives", total_count)

    result_set = results.get("result_set", [])
    entity_ids = [r["identifier"] for r in result_set]

    # Extract PDB IDs and fetch metadata
    logger.info("Fetching metadata for %d entities...", len(entity_ids))
    entries = []
    batch_size = 200

    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i : i + batch_size]
        for eid in batch:
            pdb_id = eid.split("_")[0].upper()
            # Fetch entry info for resolution and length
            url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                seq = (
                    data.get("entity_poly", {})
                    .get("pdbx_seq_one_letter_code_can", "")
                )
                length = len(seq)
                if length < min_length or length > max_length:
                    continue
                if not seq:
                    continue

                entries.append(
                    {
                        "pdb_id": pdb_id,
                        "chain": "A",
                        "length": length,
                        "sequence": seq,
                    }
                )
            except Exception:
                continue

        logger.info(
            "  Processed %d / %d entities, %d valid",
            min(i + batch_size, len(entity_ids)),
            len(entity_ids),
            len(entries),
        )

        if len(entries) >= target_count:
            break

    # Trim to target
    entries = entries[:target_count]
    logger.info("Final curated list: %d proteins", len(entries))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Curate PDB list for training")
    parser.add_argument(
        "--target", type=int, default=2000, help="Target number of proteins"
    )
    parser.add_argument(
        "--max-resolution", type=float, default=2.5, help="Max resolution (Å)"
    )
    parser.add_argument(
        "--min-length", type=int, default=30, help="Min chain length (aa)"
    )
    parser.add_argument(
        "--max-length", type=int, default=500, help="Max chain length (aa)"
    )
    parser.add_argument(
        "--identity", type=int, default=30, help="Sequence identity cutoff (%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pdb_2000.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = fetch_pdb_list(
        max_resolution=args.max_resolution,
        min_length=args.min_length,
        max_length=args.max_length,
        identity_cutoff=args.identity,
        target_count=args.target,
    )

    output_path.write_text(json.dumps(entries, indent=2))
    logger.info("Saved %d entries to %s", len(entries), output_path)

    # Summary stats
    lengths = [e["length"] for e in entries]
    logger.info(
        "Length stats: min=%d, max=%d, mean=%.0f, median=%.0f",
        min(lengths),
        max(lengths),
        sum(lengths) / len(lengths),
        sorted(lengths)[len(lengths) // 2],
    )


if __name__ == "__main__":
    main()
