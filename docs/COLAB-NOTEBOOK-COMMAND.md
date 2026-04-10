# Colab Pro Notebook: Full Pipeline

> **Ana notebook:** `notebooks/full_pipeline.ipynb` — direkt Colab'a upload et ve calistir.
>
> Asagidaki markdown versiyonu referans icin korunmustur.

OpenFold3'ün gerçek API'si kullanılarak pair representation çıkarma + eğitim + evaluation.

---

## Cell 1: GPU Check & Install

```python
# ═══ CELL 1: Environment Setup ═══
!nvidia-smi
print("="*60)

# OpenFold3 kurulumu
!pip install -q openfold3

# Model weights indir (~10GB, ~/.openfold3/)
!setup_openfold

# Verify
import openfold3
print(f"OpenFold3 version: {openfold3.__version__}")
```

## Cell 2: Mount Drive & Dirs

```python
# ═══ CELL 2: Google Drive Mount ═══
from google.colab import drive
drive.mount('/content/drive')

import os
from pathlib import Path

BASE_DIR = Path('/content/drive/MyDrive/ANM-openfold3')
PAIR_REPR_DIR = BASE_DIR / 'pair_reprs'
COORDS_DIR = BASE_DIR / 'coords'
QUERY_DIR = Path('/content/queries')

for d in [PAIR_REPR_DIR, COORDS_DIR, QUERY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Pair reprs: {PAIR_REPR_DIR}")
print(f"Coords:     {COORDS_DIR}")
```

## Cell 3: Training PDB List

```python
# ═══ CELL 3: Training PDB Set ═══
# High-quality single chain proteins, diverse folds
# Resolution < 2.5Å, 50-400 residues

TRAINING_PDBS = [
    # Small (50-100 aa)
    "1UBQ", "1L2Y", "2JOF", "1CRN", "1VII",
    "2F4K", "1PRB", "1ENH", "2GB1", "1FME",
    # Medium (100-200 aa)
    "4AKE", "1AKE", "2LZM", "3LZM", "1HHP",
    "1MBN", "1HEL", "2RN2", "1RIS", "3CLN",
    "1CTF", "1SN3", "2CI2", "1LZ1", "1BPI",
    # Large (200-400 aa)
    "1TIM", "1GFL", "3GFP", "1HBS", "1MBO",
    "2CGA", "1CHD", "1PPT", "4HHB", "1LYZ",
    "2SOD", "1SUP", "3PGK", "1PFK", "1ADK",
    # More diverse
    "1A2P", "1BRS", "1CSE", "1DFN", "1ECA",
    "1FKJ", "1GDN", "1HMV", "1IFC", "1JWE",
]

print(f"Training set: {len(TRAINING_PDBS)} proteins")
```

## Cell 4: Fetch Sequences from RCSB

```python
# ═══ CELL 4: PDB → Sequence ═══
import requests
import json

def fetch_sequence_from_pdb(pdb_id: str) -> str:
    """PDB ID → protein sequence via RCSB API."""
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        seq = data.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
        return seq.replace("\n", "")
    # Fallback: FASTA
    url2 = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
    resp2 = requests.get(url2, timeout=10)
    if resp2.status_code == 200:
        lines = resp2.text.strip().split("\n")
        return "".join(l for l in lines if not l.startswith(">"))
    return ""

# Pre-fetch all sequences
sequences = {}
for pdb_id in TRAINING_PDBS:
    seq = fetch_sequence_from_pdb(pdb_id)
    if seq:
        sequences[pdb_id] = seq
        print(f"  {pdb_id}: {len(seq)} aa")
    else:
        print(f"  {pdb_id}: FAILED")

print(f"\nFetched: {len(sequences)}/{len(TRAINING_PDBS)}")
```

## Cell 5: Runner YAML (write_latent_outputs: true)

```python
# ═══ CELL 5: Create runner.yml with write_latent_outputs ═══
# Bu, OpenFold3'e zij_trunk (pair repr) kaydetmesini söyler

RUNNER_YAML = "/content/runner_latent.yml"

runner_config = """
output_writer_settings:
  structure_format: cif
  write_latent_outputs: true
  write_full_confidence_scores: true
  write_features: false
"""

with open(RUNNER_YAML, 'w') as f:
    f.write(runner_config.strip())

print(f"Runner YAML: {RUNNER_YAML}")
print("write_latent_outputs: true  →  zij_trunk saved as .pt")
```

## Cell 6: Generate Query JSONs

```python
# ═══ CELL 6: Create OpenFold3 query JSONs ═══
# Format: https://github.com/aqlaboratory/openfold-3 inference docs

query_files = {}

for pdb_id, seq in sequences.items():
    query = {
        "queries": {
            pdb_id: {
                "chains": [{
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": seq,
                }]
            }
        }
    }

    query_path = QUERY_DIR / f"{pdb_id}_query.json"
    with open(query_path, 'w') as f:
        json.dump(query, f, indent=2)
    query_files[pdb_id] = query_path

print(f"Created {len(query_files)} query JSON files")
```

## Cell 7: Run Inference & Extract Pair Repr

```python
# ═══ CELL 7: OpenFold3 Inference with Latent Output ═══
# run_openfold predict → MSA server (no JackHMMER) → zij_trunk saved
#
# Key flags:
#   --use-msa-server=True   → ColabFold MMseqs2 server (~30s vs 30min JackHMMER)
#   --runner-yaml            → write_latent_outputs: true
#   --num-diffusion-samples=1 → tek sample yeterli (sadece pair repr lazım)
#   --num-model-seeds=1      → tek seed yeterli

import subprocess
import time

OUTPUT_BASE = Path('/content/of3_output')
RUNNER_YAML_PATH = "/content/runner_latent.yml"

extracted = 0
failed = []

for pdb_id, query_path in query_files.items():
    # Skip if already extracted
    pair_repr_file = PAIR_REPR_DIR / f"{pdb_id}_pair_repr.pt"
    if pair_repr_file.exists():
        print(f"  {pdb_id}: cached ✓")
        extracted += 1
        continue

    output_dir = OUTPUT_BASE / pdb_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {pdb_id}: running inference...", end="", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run([
            'run_openfold', 'predict',
            f'--query-json={query_path}',
            f'--output-dir={output_dir}',
            f'--runner-yaml={RUNNER_YAML_PATH}',
            '--use-msa-server=True',
            '--use-templates=True',
            '--num-diffusion-samples=1',
            '--num-model-seeds=1',
        ], capture_output=True, text=True, timeout=600)

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f" FAILED ({elapsed:.0f}s)")
            print(f"    stderr: {result.stderr[:200]}")
            failed.append(pdb_id)
            continue

        # Find latent output file
        # Pattern: {output_dir}/{pdb_id}/seed_*/{ pdb_id}_seed_*_latent_output.pt
        latent_files = list(output_dir.rglob("*_latent_output.pt"))

        if latent_files:
            import torch
            latent = torch.load(latent_files[0], map_location='cpu')

            # Extract zij_trunk (pair representation)
            zij = latent.get('zij_trunk')
            si = latent.get('si_trunk')

            if zij is not None:
                # Save pair repr separately (smaller file)
                torch.save({
                    'pair_repr': zij,        # [1, N, N, C_z] typically C_z=128
                    'single_repr': si,       # [1, N, C_s] typically C_s=384
                    'pdb_id': pdb_id,
                    'n_tokens': zij.shape[1],
                    'sequence': sequences[pdb_id],
                }, pair_repr_file)

                extracted += 1
                print(f" done ✓ ({elapsed:.0f}s, N={zij.shape[1]}, "
                      f"zij={list(zij.shape)})")
            else:
                print(f" latent found but no zij_trunk key!")
                print(f"    keys: {list(latent.keys())}")
                failed.append(pdb_id)
        else:
            print(f" no latent output file found!")
            failed.append(pdb_id)

    except subprocess.TimeoutExpired:
        print(f" TIMEOUT (>600s)")
        failed.append(pdb_id)
    except Exception as e:
        print(f" ERROR: {e}")
        failed.append(pdb_id)

print(f"\n{'='*60}")
print(f"Extracted: {extracted}/{len(query_files)}")
if failed:
    print(f"Failed: {failed}")
```

## Cell 8: Download Ground Truth Cα Coords

```python
# ═══ CELL 8: PDB → Cα Coordinates ═══
!pip install -q biopython

import numpy as np
import torch
import Bio.PDB as bpdb

parser = bpdb.PDBParser(QUIET=True)
pdbl = bpdb.PDBList()

PDB_CACHE = Path('/content/pdb_cache')
PDB_CACHE.mkdir(exist_ok=True)

saved_coords = 0
for pdb_id in sequences:
    coord_file = COORDS_DIR / f"{pdb_id}_ca.pt"
    if coord_file.exists():
        saved_coords += 1
        continue

    try:
        pdb_file = pdbl.retrieve_pdb_file(
            pdb_id, pdir=str(PDB_CACHE), file_format='pdb'
        )
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]

        # Extract Cα coords from first chain
        ca_coords = []
        first_chain = list(model.get_chains())[0]
        for residue in first_chain:
            if residue.get_id()[0] == ' ' and 'CA' in residue:
                ca_coords.append(residue['CA'].get_vector().get_array())

        if ca_coords:
            coords_tensor = torch.tensor(
                np.array(ca_coords), dtype=torch.float32
            )
            torch.save(coords_tensor, coord_file)
            saved_coords += 1
            print(f"  {pdb_id}: {len(ca_coords)} Cα atoms")

    except Exception as e:
        print(f"  {pdb_id}: {e}")

print(f"\nCoords saved: {saved_coords}/{len(sequences)}")
```

## Cell 9: Verify Extracted Data

```python
# ═══ CELL 9: Sanity Check ═══
import torch

pair_files = sorted(PAIR_REPR_DIR.glob("*_pair_repr.pt"))
coord_files = sorted(COORDS_DIR.glob("*_ca.pt"))

print(f"Pair repr files: {len(pair_files)}")
print(f"Coord files:     {len(coord_files)}")
print()

# Check a few samples
for pf in pair_files[:5]:
    data = torch.load(pf, map_location='cpu')
    pdb_id = data['pdb_id']
    zij = data['pair_repr']
    si = data.get('single_repr')

    coord_file = COORDS_DIR / f"{pdb_id}_ca.pt"
    ca = torch.load(coord_file, map_location='cpu') if coord_file.exists() else None

    print(f"  {pdb_id}:")
    print(f"    pair_repr:   {list(zij.shape)}  (dtype={zij.dtype})")
    if si is not None:
        print(f"    single_repr: {list(si.shape)}")
    if ca is not None:
        print(f"    Cα coords:   {list(ca.shape)}")
    print(f"    sequence:    {data['sequence'][:30]}... ({len(data['sequence'])} aa)")
    print()

# Match check
pair_ids = {f.stem.replace('_pair_repr', '') for f in pair_files}
coord_ids = {f.stem.replace('_ca', '') for f in coord_files}
matched = pair_ids & coord_ids
print(f"Matched (pair + coords): {len(matched)} proteins")
print(f"  → Ready for training!")
```

## Cell 10: Quick Training Test (Optional)

```python
# ═══ CELL 10: Training Sanity Check on Real Data ═══
# Bu cell opsiyonel — lokalde de çalıştırılabilir
# Sadece ilk 5 protein ile hızlı overfit testi

import torch
import sys

# src/ dosyalarını import et (Drive'dan veya upload)
# Option A: Drive'a kopyaladıysanız
sys.path.insert(0, str(BASE_DIR))

# Option B: GitHub'dan çekin
# !git clone <your-repo> /content/anm-openfold3
# sys.path.insert(0, '/content/anm-openfold3')

from src.contact_head import ContactProjectionHead
from src.ground_truth import compute_gt_probability_matrix
from src.losses import total_loss

# Load first few proteins
pair_files = sorted(PAIR_REPR_DIR.glob("*_pair_repr.pt"))[:5]
print(f"Using {len(pair_files)} proteins for quick test\n")

head = ContactProjectionHead(c_z=128, bottleneck_dim=32).cuda()
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(20):
    epoch_loss = 0
    n = 0

    for pf in pair_files:
        data = torch.load(pf, map_location='cpu')
        pdb_id = data['pdb_id']
        zij = data['pair_repr'].cuda()  # [1, N, N, 128]

        # Load coords
        coord_file = COORDS_DIR / f"{pdb_id}_ca.pt"
        if not coord_file.exists():
            continue
        coords = torch.load(coord_file, map_location='cpu').cuda()

        # Truncate if needed (pair_repr N_token might differ from N_ca)
        N_ca = coords.shape[0]
        N_tok = zij.shape[1]
        N = min(N_ca, N_tok)
        zij = zij[:, :N, :N, :]
        coords = coords[:N]

        C_gt = compute_gt_probability_matrix(coords)
        C_pred = head(zij).squeeze(0)

        loss, details = total_loss(C_pred, C_gt, alpha=1.0, beta=0.5, gamma=0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n += 1

    if n > 0 and epoch % 5 == 0:
        print(f"Epoch {epoch:3d}: avg_loss={epoch_loss/n:.4f}")

print("\nQuick training test complete!")
print("Download pair_reprs/ and coords/ to your local machine for full training.")
```

---

## Özet

| Cell | Ne Yapıyor | Süre |
|------|-----------|------|
| 1 | OpenFold3 install + weights | ~10 dk |
| 2 | Drive mount | 5 sn |
| 3 | PDB listesi tanımla | anında |
| 4 | Sequence fetch (RCSB) | ~1 dk |
| 5 | Runner YAML (write_latent_outputs) | anında |
| 6 | Query JSON'lar oluştur | anında |
| 7 | **Inference + pair repr extraction** | **~2-5 dk/protein** |
| 8 | Cα coordinate download | ~5 dk |
| 9 | Verify | anında |
| 10 | Quick training test (opsiyonel) | ~2 dk |

**Toplam:** ~3-4 saat (50 protein icin)

## Key Points

- **MSA:** `--use-msa-server=True` → ColabFold MMseqs2 server (~30s/protein, JackHMMER yok)
- **Pair repr:** `write_latent_outputs: true` → `zij_trunk` [N, N, 128] otomatik kaydedilir
- **Output:** `{pdb_id}_seed_0_latent_output.pt` → `zij_trunk`, `si_trunk`, confidence scores
- **API:** `run_openfold predict --query-json=... --runner-yaml=... --use-msa-server=True`

## Sonraki Adım

1. Pair repr'leri Drive'a kaydet
2. Lokale indir: `pair_reprs/*.pt` + `coords/*.pt`
3. Lokal M1'de `python -m src.train --data_dir pair_reprs/ --device mps`

## Related
- [[architecture/07-training-plan]] - Tam eğitim stratejisi
- [[architecture/05-gnm-contact-learner]] - GNM-Contact Learner detayları
- [[architecture/06-gnm-math-detail]] - Matematik detayları

#colab #pair-repr #extraction #openfold3
