# Benchmark Fix Implementation Plan

**Date:** 2026-05-06
**Prerequisite:** [benchmark_issues_analysis.md](benchmark_issues_analysis.md)

---

## Fix 1: token_mask — Singleton Model + Per-Protein Batch

### Problem
`load_of3_diffusion()` creates a fresh `InferenceExperimentRunner` per call. After the first protein succeeds, subsequent calls produce batches missing `token_mask` due to global state corruption in the OF3 data pipeline (ColabFold MSA cache, runner internals).

### Solution
Split `load_of3_diffusion` into modular parts:

```python
# ONCE per session:
of3_model, of3_config = load_of3_model(device)

# ONCE per protein (same sequence for both directions):
diffusion_fn, zij_trunk = prepare_of3_for_sequence(
    model=of3_model,
    config=of3_config,
    sequence=sequence,
    device=device,
)
```

**Key changes to `src/of3_diffusion.py`:**

1. New function `load_of3_model(device)` — loads model + checkpoint ONCE
2. New function `prepare_of3_for_sequence(model, sequence, device)`:
   - Creates query JSON
   - Creates a FRESH InferenceExperimentRunner for data pipeline only
   - Gets batch from predict_dataloader
   - Adds `token_mask` fallback if missing
   - Runs trunk, returns (diffusion_fn, zij_trunk)
3. The existing `load_of3_diffusion()` becomes a convenience wrapper (backward compat)

**Fallback approach (simpler, if above is too invasive):**
- After `cached_batch = next(iter(predict_dl))`, explicitly check and create `token_mask`:

```python
if "token_mask" not in cached_batch:
    # Find N_token from restype or any token-level tensor
    for key in ("restype", "residue_index", "ref_pos", "ref_charge"):
        if key in cached_batch and isinstance(cached_batch[key], torch.Tensor):
            N_tok = cached_batch[key].shape[-1]  # [batch, N_token] or [batch, N_token, ...]
            break
    cached_batch["token_mask"] = torch.ones((1, N_tok), dtype=torch.float32, device=device)
```

### Files Modified
- `src/of3_diffusion.py` — add token_mask fallback + model caching API
- `notebooks/benchmark_open_closed.ipynb` — use cached model

---

## Fix 2: Size Mismatch — Sequence Alignment + Common Core

### Problem
Open and closed PDB structures have different numbers of resolved residues.

### Solution
Add a `align_and_trim_ca()` function that:
1. Extracts sequences from both PDBs
2. Performs pairwise sequence alignment
3. Identifies common positions (matched, no gaps)
4. Returns trimmed CA coordinates for both structures (same length)

```python
def align_and_trim_ca(
    ca_a: Tensor, seq_a: str,
    ca_b: Tensor, seq_b: str,
) -> tuple[Tensor, Tensor, str]:
    """Align sequences and return common-core CA coords.

    Returns:
        ca_a_trimmed, ca_b_trimmed, common_sequence
    """
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(seq_a, seq_b)[0]

    # Extract matched positions (no gaps in either)
    idx_a, idx_b = [], []
    pos_a, pos_b = 0, 0
    for char_a, char_b in zip(alignment[0], alignment[1]):
        gap_a = (char_a == '-')
        gap_b = (char_b == '-')
        if not gap_a and not gap_b:
            idx_a.append(pos_a)
            idx_b.append(pos_b)
        if not gap_a:
            pos_a += 1
        if not gap_b:
            pos_b += 1

    ca_a_trim = ca_a[idx_a]
    ca_b_trim = ca_b[idx_b]
    common_seq = ''.join(seq_a[i] for i in idx_a)

    return ca_a_trim, ca_b_trim, common_seq
```

**Important:** When using trimmed coordinates:
- Pipeline runs with `initial_ca` from one PDB (full length)
- TM/RMSD comparison uses trimmed common core for both
- OF3 sequence uses the INITIAL PDB's full sequence (it predicts full structure)
- After OF3 prediction, trim predicted CA to common core for comparison

### Files Modified
- `src/mode_drive_utils.py` — add `align_and_trim_ca()`
- `notebooks/benchmark_open_closed.ipynb` — use alignment in `run_single_direction`

---

## Fix 3: Chain ID Validation

### Problem
Some PDBs don't have the expected chain A.

### Solution
Update `fetch_ca()` with auto-detection fallback:

```python
def fetch_ca(pdb_id: str, chain_id: str):
    ...
    chains = [c for c in structure[0].get_chains() if c.id == chain_id]
    if not chains:
        available = [c.id for c in structure[0].get_chains()]
        # Try first available chain
        if available:
            print(f'  WARNING: chain {chain_id} not found in {pdb_id}, using {available[0]}')
            chain = [c for c in structure[0].get_chains() if c.id == available[0]][0]
        else:
            raise ValueError(f'No chains in {pdb_id}')
    else:
        chain = chains[0]
    ...
```

Also update benchmark table with verified chain IDs:
- PKA (3FJQ): chain E
- PKA (1SYK): chain E

### Files Modified
- `notebooks/benchmark_open_closed.ipynb` — fix `fetch_ca` + update table

---

## Fix 4: Benchmark Notebook Structure (Performance)

### Current Flow (slow)
```
for each protein:
    for each direction:
        load_of3_model()          # ~30s, repeated
        prepare_batch()           # ~30s (MSA)
        run_trunk()               # ~20s
        run_pipeline()            # ~10min
```

### Optimized Flow
```
load_of3_model()                  # ONCE (~30s)
for each protein:
    prepare_batch(sequence)       # ONCE per protein (~30s MSA)
    run_trunk(batch)              # ONCE per protein (~20s)
    for each direction:
        run_pipeline()            # ~10min (only diffusion repeated)
```

**Savings:** ~60s per direction (model loading), ~30s per protein (MSA reuse)

---

## Implementation Order

1. **Fix 1 (token_mask)** — unblocks 8 runs immediately
2. **Fix 2 (size mismatch)** — unblocks 6 more runs
3. **Fix 3 (chain ID)** — unblocks 2 runs
4. **Fix 4 (performance)** — nice-to-have

**After all fixes:** Expected 16-18/18 success rate.

---

## Notebook Changes Summary

The benchmark notebook needs these updates in cell 4 (helpers):

1. `fetch_ca()` — chain fallback + better error messages
2. `align_and_trim_ca()` — new helper for size mismatch
3. `setup_of3()` — token_mask fallback (already in src)
4. `run_single_direction()` — use alignment when sizes differ
5. Model loading — move outside the loop (load once)
