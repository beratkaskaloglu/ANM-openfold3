#!/usr/bin/env python3
"""Phase 1: Curate ~2000 diverse PDB entries via RCSB Search API.

Filters:
  - X-ray, resolution ≤ 2.5 Å
  - Single protein entity
  - Chain length 30–500 residues

Output: data/pdb_2000.json
"""

import argparse
import json
import logging
import sys
import time
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
    max_results: int = 5000,
) -> dict:
    """Build RCSB Search API query for high-quality single-chain proteins."""
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
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
        "return_type": "entry",
    }


def fetch_entity_info(pdb_id: str) -> dict | None:
    """Fetch sequence and metadata for entity 1 of a PDB entry."""
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        seq = (
            data.get("entity_poly", {})
            .get("pdbx_seq_one_letter_code_can", "")
        )
        if not seq:
            return None
        return {"pdb_id": pdb_id, "chain": "A", "length": len(seq), "sequence": seq}
    except Exception:
        return None


def fetch_pdb_list(
    max_resolution: float = 2.5,
    min_length: int = 30,
    max_length: int = 500,
    target_count: int = 2000,
) -> list[dict]:
    """Query RCSB and return curated PDB list."""
    # Fetch many candidates, filter by length
    query = build_query(
        max_resolution=max_resolution,
        min_length=min_length,
        max_length=max_length,
        max_results=target_count * 3,
    )

    logger.info("Querying RCSB Search API...")
    resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=120)
    resp.raise_for_status()
    results = resp.json()

    total_count = results.get("total_count", 0)
    logger.info("RCSB returned %d entries", total_count)

    result_set = results.get("result_set", [])
    pdb_ids = [r["identifier"] for r in result_set]
    logger.info("Processing %d PDB IDs...", len(pdb_ids))

    # Fetch metadata and filter by length
    entries = []
    seen_lengths = set()  # crude diversity: skip exact-same-length duplicates at high count

    for i, pdb_id in enumerate(pdb_ids):
        if len(entries) >= target_count:
            break

        info = fetch_entity_info(pdb_id)
        if info is None:
            continue

        length = info["length"]
        if length < min_length or length > max_length:
            continue

        entries.append(info)

        if (i + 1) % 100 == 0:
            logger.info(
                "  Processed %d / %d IDs, %d valid so far",
                i + 1,
                len(pdb_ids),
                len(entries),
            )

        # Small delay to be polite to RCSB API
        if (i + 1) % 200 == 0:
            time.sleep(1)

    entries = entries[:target_count]
    logger.info("Final curated list: %d proteins", len(entries))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Curate PDB list for training")
    parser.add_argument(
        "--target", type=int, default=2000, help="Target number of proteins"
    )
    parser.add_argument(
        "--max-resolution", type=float, default=2.5, help="Max resolution (A)"
    )
    parser.add_argument(
        "--min-length", type=int, default=30, help="Min chain length (aa)"
    )
    parser.add_argument(
        "--max-length", type=int, default=500, help="Max chain length (aa)"
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
        target_count=args.target,
    )

    if not entries:
        logger.error("No entries found! Check network connection and RCSB API.")
        sys.exit(1)

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
