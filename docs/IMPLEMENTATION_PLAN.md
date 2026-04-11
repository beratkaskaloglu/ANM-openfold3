# Scale-Up Implementation Plan: 2000 PDB Fine-Tuning

## 1. Problem Statement

Current pipeline trains on 45 proteins with 100 epochs. We need:
- **2000+ diverse proteins** for robust generalization
- **Many more epochs** (500-1000) for deeper convergence
- **Batch-10 chunked inference** to avoid Colab OOM / MSA cache corruption
- **Hyperparameter tuning** for lower loss
- Standalone `.py` script (not notebook) for reproducibility

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1: PDB Curation (run once)                                │
│  fetch_pdb_list.py → pdb_2000.json                               │
│  PISCES-based: ≤2.5Å, 30-500 aa, ≤30% seq identity              │
└───────────────────────┬──────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 2: Chunked Inference (10-at-a-time)                       │
│  extract_pairs.py                                                │
│                                                                  │
│  for chunk_i in range(0, 2000, 10):                              │
│      pdb_ids = pdb_list[chunk_i : chunk_i+10]                    │
│      run OpenFold3 inference → save pair_repr .pt                │
│      download PDB → extract Cα coords → save .pt                │
│      clear MSA cache + GPU cache                                 │
│      save chunk_i.ok marker                                      │
│                                                                  │
│  Resume-safe: skips chunks with .ok marker                       │
└───────────────────────┬──────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 3: Pack into .npz shards                                  │
│  pack_shards.py                                                  │
│                                                                  │
│  Group 50 proteins per .npz shard → shard_0000.npz               │
│  Each shard: {pdb_id: {pair_repr, coords_ca, c_gt, n_res}}      │
│  Delete individual .pt files after verification                  │
└───────────────────────┬──────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Phase 4: Training                                               │
│  train_large.py                                                  │
│                                                                  │
│  ShardedDataset: streams .npz shards lazily                      │
│  Enhanced training loop:                                         │
│    - Focal loss (replaces BCE)                                   │
│    - OneCycleLR scheduler                                        │
│    - Warmup + cosine decay                                       │
│    - Train/val/test split (80/10/10)                             │
│    - Early stopping (patience=50)                                │
│    - Checkpoint every 50 epochs                                  │
│    - WandB logging                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. PDB Curation Strategy

### Source: PISCES + PDB REST API

```
Filters:
  - Resolution: ≤ 2.5 Å
  - R-value: ≤ 0.25
  - Chain length: 30-500 amino acids (GNM sweet spot)
  - Max seq identity: 30% (non-redundant)
  - Experimental method: X-ray only
  - Exclude: membrane proteins, DNA/RNA complexes
  - Single chain per entry (chain A preferred)
```

### Why these filters?
- **30-500 aa**: GNM eigendecomposition is O(N³). Above 500, Kirchhoff matrices get expensive. Below 30, too few modes.
- **≤30% identity**: Prevents data leakage between train/val/test.
- **≤2.5 Å**: Ensures reliable Cα coordinates for ground truth contacts.

### Implementation
- Use RCSB PDB Search API (GraphQL) for bulk query
- Fallback: download PISCES pre-curated list from Dunbrack lab
- Store as `data/pdb_2000.json`: `[{"pdb_id": "1UBQ", "chain": "A", "length": 76, "resolution": 1.8}, ...]`

---

## 4. Chunked Inference (10-at-a-time)

### Why 10?
- OpenFold3 MSA server is the bottleneck (~60-120s per protein)
- A100 40GB can hold ~10 concurrent MSA computations
- Per-protein isolated MSA dirs prevent cross-contamination (lesson from 48-protein run)
- After each chunk: `torch.cuda.empty_cache()` + `shutil.rmtree(msa_dir)`

### Resume Safety
```python
# Each chunk writes a marker file on completion
marker = PROGRESS_DIR / f"chunk_{chunk_i:04d}.ok"
if marker.exists():
    continue  # skip completed chunk

# ... run inference ...
marker.write_text(f"{len(successful)}/{len(chunk_ids)} OK")
```

### Error Handling
- Individual protein failures don't stop the chunk
- Failed proteins logged to `data/failed_pdbs.json`
- Retry logic: 1 retry with fresh MSA dir
- Expected: ~5-10% failure rate (template issues, MSA timeouts)

---

## 5. .npz Shard Format

### Why .npz instead of individual .pt?
- **I/O efficiency**: 50 proteins per file = 40 shards for 2000 proteins
- **Colab-friendly**: Fewer files = faster Google Drive sync
- **Streaming**: Load one shard at a time, no need to hold 2000 tensors in RAM

### Shard Structure
```python
# shard_0000.npz
{
    "pdb_ids": ["1UBQ", "1L2Y", ...],           # (50,) string array
    "pair_reprs_0": np.array([N, N, 128]),       # per-protein (variable size)
    "coords_ca_0": np.array([N, 3]),
    "pair_reprs_1": np.array([M, M, 128]),
    ...
}
```

Note: Variable-size tensors can't be stacked, so each protein is stored separately within the shard with indexed keys.

---

## 6. Hyperparameter Recommendations

### Current vs Proposed

| Parameter             | Current (48 PDB) | Proposed (2000 PDB) | Rationale                                   |
| --------------------- | ---------------- | ------------------- | ------------------------------------------- |
| `epochs`              | 100              | 500-1000            | More data needs more passes for convergence |
| `lr`                  | 1e-4             | 3e-4 (peak)         | OneCycleLR allows higher peak with warmup   |
| `weight_decay`        | 1e-2             | 1e-2                | Keep same                                   |
| `batch_size`          | 1                | 1                   | Variable protein sizes, can't batch easily  |
| `bottleneck_dim`      | 32               | 64                  | More capacity for diverse protein space     |
| `n_modes`             | 20               | 20                  | Literature standard for GNM                 |
| `r_cut`               | 10.0 Å           | 8.0 Å               | GNM optimal: 7-8 Å (Bahar lab literature)   |
| `tau`                 | 1.5              | 1.0                 | Sharper sigmoid = closer to binary contacts |
| `alpha` (L_contact)   | 1.0              | 1.0                 | Keep dominant                               |
| `beta` (L_gnm)        | 0.5              | 0.3                 | Reduce - GNM loss is noisy early on         |
| `gamma` (L_recon)     | 0.1              | 0.05                | Reduce - reconstruction is auxiliary        |
| `seq_sep_min`         | 6                | 6                   | Standard for medium/long-range contacts     |
| Loss function         | BCE              | **Focal Loss**      | Better for class-imbalanced contacts        |
| Scheduler             | CosineAnnealing  | **OneCycleLR**      | Better convergence for longer training      |
| Gradient accumulation | -                | **4 steps**         | Effective batch size = 4                    |
| Early stopping        | -                | **patience=50**     | Prevent overfitting                         |

### Key Changes Explained

**1. Focal Loss (gamma=2.0, alpha=0.75)**
Contact maps are highly imbalanced: ~10-15% contacts, ~85-90% non-contacts.
Focal loss down-weights easy negatives and focuses on hard boundary cases:
```
FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
```

**2. r_cut: 10.0 → 8.0 Å**
Literature (Bahar, Atilgan et al.) shows GNM optimal cutoff is 7-8 Å for Cα-Cα contacts. 10 Å is too generous and includes many non-physical contacts. Sharper cutoff = cleaner signal.

**3. bottleneck_dim: 32 → 64**
With 2000 diverse proteins spanning all CATH classes, 32 dims may be too constrained for the encoder. 64 doubles parameters from ~8K to ~16K — still tiny.

**4. OneCycleLR**
- Warmup: first 5% of epochs (lr: 3e-5 → 3e-4)
- Cosine decay: remaining 95% (lr: 3e-4 → 3e-6)
- Better than flat CosineAnnealing for long training

**5. Gradient Accumulation (4 steps)**
Since batch_size=1 (variable protein sizes), accumulating gradients over 4 proteins gives smoother updates.

---

## 7. File Structure

```
scripts/
├── fetch_pdb_list.py      # Phase 1: Curate 2000 PDBs
├── extract_pairs.py       # Phase 2: Chunked OpenFold3 inference
├── pack_shards.py         # Phase 3: .pt → .npz shards
└── train_large.py         # Phase 4: Full training

src/
├── contact_head.py        # (existing, unchanged)
├── kirchhoff.py           # (existing, eigh CPU fix)
├── losses.py              # UPDATE: add FocalLoss
├── data.py                # UPDATE: add ShardedDataset
├── ground_truth.py        # (existing, unchanged)
├── train.py               # UPDATE: add OneCycleLR, grad accum, early stopping
├── model.py               # (existing, unchanged)
└── inverse.py             # (existing, unchanged)

data/
├── pdb_2000.json          # Curated PDB list with metadata
├── failed_pdbs.json       # Proteins that failed inference
├── shards/                # .npz shard files
│   ├── shard_0000.npz
│   ├── shard_0001.npz
│   └── ...
└── progress/              # Chunk completion markers
    ├── chunk_0000.ok
    └── ...
```

---

## 8. Implementation Phases

### Phase 1: PDB Curation (`fetch_pdb_list.py`)
- Query RCSB PDB Search API
- Filter by resolution, length, identity
- Save curated list
- **Output**: `data/pdb_2000.json`
- **Time**: ~2 min (API query)

### Phase 2: Chunked Inference (`extract_pairs.py`)
- Load PDB list
- Process in chunks of 10
- Per-protein: OpenFold3 inference → zij_trunk → save .pt
- Per-protein: PDB download → Cα coords → save .pt
- Clean MSA cache after each chunk
- **Output**: `pair_reprs/*.pt`, `coords/*.pt`
- **Time**: ~100s/protein × 2000 = ~55 hours (can parallelize with multiple Colab sessions)

### Phase 3: Pack Shards (`pack_shards.py`)
- Group completed .pt files into .npz shards (50/shard)
- Verify all data loads correctly
- **Output**: `data/shards/shard_*.npz`
- **Time**: ~5 min

### Phase 4: Training (`train_large.py`)
- ShardedDataset loads .npz lazily
- 80/10/10 train/val/test split (by shard, not by protein)
- Focal loss + OneCycleLR + gradient accumulation
- Early stopping + best model checkpointing
- **Output**: `checkpoints/best_model.pt`, `training_curves.png`
- **Time**: ~2-4 hours for 500 epochs on A100

---

## 9. Colab Execution Plan

```bash
# Session 1: PDB curation + start inference
!cd /content/ANM-openfold3 && git pull
!python scripts/fetch_pdb_list.py
!python scripts/extract_pairs.py --chunk-size 10 --start 0 --end 500

# Session 2-4: Continue inference (resume-safe)
!python scripts/extract_pairs.py --chunk-size 10 --start 0 --end 2000

# Session 5: Pack + Train
!python scripts/pack_shards.py
!python scripts/train_large.py --epochs 500 --lr 3e-4 --bottleneck-dim 64 --r-cut 8.0
```

---

## 10. Risk Mitigation

| Risk                            | Mitigation                                                |
| ------------------------------- | --------------------------------------------------------- |
| Colab disconnects mid-inference | Resume markers (.ok files)                                |
| MSA cache corruption            | Per-protein isolated dirs (proven fix)                    |
| GPU OOM during inference        | `torch.cuda.empty_cache()` after each chunk               |
| Disk space on Colab             | .npz compression + delete .pt after packing               |
| Too many failed proteins        | Retry logic + fallback to 1500-1800 successful            |
| Overfitting on small val set    | 200 proteins in val, early stopping                       |
| Long training time              | Gradient accumulation + OneCycleLR for faster convergence |
