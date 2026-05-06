# Benchmark Issues Analysis

**Date:** 2026-05-06
**Source:** benchmark_open_closed.ipynb first run on Colab (A100)

## Results Summary

| # | Protein | N | Open->Closed | Closed->Open | Issue |
|---|---------|---|:---:|:---:|-------|
| 1 | Adenylate kinase | 214 | TM 0.57->0.93 | TM 0.57->0.74 | None |
| 2 | Maltose-binding | 370 | FAIL | FAIL | `token_mask` KeyError |
| 3 | Glutamine-binding | 220/223 | FAIL | FAIL | Size mismatch |
| 4 | Citrate synthase | 437 | FAIL | FAIL | `token_mask` KeyError |
| 5 | Lactoferrin | 691 | FAIL | FAIL | `token_mask` KeyError |
| 6 | Src kinase | 452/449 | FAIL | FAIL | Size mismatch |
| 7 | PKA-Calpha | ? | FAIL | FAIL | `list index out of range` (no chain A) |
| 8 | GGBP | 305/309 | FAIL | FAIL | Size mismatch |
| 9 | LAO-binding | 238 | FAIL | FAIL | `token_mask` KeyError |

**Success rate:** 2/18 runs (11%)

---

## Issue 1: `token_mask` KeyError (Critical)

**Affected:** Maltose-binding, Citrate synthase, Lactoferrin, LAO-binding (8 runs)

**Observation:** Only Adenylate kinase (first protein, N=214) works. ALL subsequent proteins fail with `KeyError: 'token_mask'` even though the batch creation prints `[OF3] Input batch ready.`

**Root Cause Analysis:**

The OF3 `InferenceExperimentRunner` is re-created for each protein via `setup_of3()`. However, global state corruption occurs:

1. The ColabFold MSA server cache persists between calls (`/tmp/of3-of-root/colabfold_msas/`). Warning messages show "Mapping file already exists. Appending new sequences." for all subsequent proteins.

2. The `InferenceExperimentRunner` creates a `lightning_data_module` which internally calls `prepare_data()` and `setup(stage="predict")`. After the first successful run, some internal state (dataset config, seeds, datapoint cache) may not be properly reset.

3. The `predict_dataloader()` returns a DataLoader. The `next(iter(predict_dl))` call triggers the full data pipeline — featurization, MSA, templates. If the MSA cache mapping returns stale/corrupt data (due to reusing sequences from previous proteins), the featurization could produce an incomplete batch.

4. **Key insight:** The batch IS produced (no exception in data pipeline), but `token_mask` is missing from the tensor dict. This means `create_all_features()` either:
   - Returned an incomplete features dict due to a silent error in structure featurization
   - The collator dropped `token_mask` during padding (unlikely for a 1D mask)
   - The `tensor_tree_map` to device skipped it (also unlikely)

**Most likely cause:** The `openfold_batch_collator` uses `dict_multimap(pad_feat_fn, samples)` which calls `torch.nn.utils.rnn.pad_sequence` on every tensor value. If `token_mask` is somehow not a tensor (e.g., became a list or was dropped by the dataset's `__getitem__`), it would be silently excluded.

**Proposed Fix:**
- Add explicit `token_mask` creation as fallback in `load_of3_diffusion` (already done locally)
- Better: **Reuse the same OF3 runner across proteins** instead of recreating it. Cache the model, only rerun trunk with new batch.
- Best: Debug what the batch actually contains after collation for the failing proteins (print all keys).

---

## Issue 2: Size Mismatch (Medium)

**Affected:** Glutamine-binding (220 vs 223), Src kinase (452 vs 449), GGBP (305 vs 309)

**Root Cause:** Different PDB structures of the same protein have different numbers of resolved residues (missing loops, terminal truncation, alternate conformations). The CA extraction uses:

```python
residues = [r for r in chain if r.get_id()[0] == ' ' and 'CA' in r]
```

This picks all standard residues with a CA atom, but different crystal structures may have different resolved regions.

**Proposed Fix:**
- **Sequence alignment approach:** Align sequences from both PDBs, find common residue range, use only matched positions.
- Implementation: Use pairwise sequence alignment (BioPython `pairwise2`) to find the mapping, then extract only common positions.

---

## Issue 3: Chain Not Found (Low)

**Affected:** PKA-Calpha (3FJQ, 1SYK)

**Root Cause:** `list index out of range` when doing:
```python
chain = [c for c in structure[0].get_chains() if c.id == chain_id][0]
```

The PDB file either doesn't have chain A, or uses a different chain ID. Need to verify actual chain IDs:
- 3FJQ: May use chain E or other
- 1SYK: May not have chain A

**Proposed Fix:**
- Add chain ID auto-detection or validation with clear error message
- Update the benchmark table with correct chain IDs

---

## Issue 4: RMSD Calculation (Already Correct)

**User concern:** "yapilari align edip oyle hesaplasin"

**Status:** `compute_rmsd()` in `mode_drive_utils.py` already performs Kabsch superimposition before calculating RMSD. This is correct.

**However:** The `StepResult.rmsd` field represents RMSD from **initial structure** (not from target). In the benchmark notebook's `run_single_direction`, we separately compute:
- `sr.rmsd` = RMSD from initial (used for exploration scoring)
- `cur_rmsd = compute_rmsd(sr.new_ca, target_ca)` = RMSD to target (this is what we report)

Both use Kabsch alignment. No fix needed.

---

## Issue 5: Model Reloading Overhead

**Observation:** `load_of3_diffusion()` is called for EVERY direction, meaning:
- Model loaded 18 times (9 proteins x 2 directions)
- MSA computed fresh each time (even for same protein)
- Each load takes ~30-60s

**Proposed Fix:**
- Cache the model between runs
- For same protein (both directions), reuse the MSA and only rerun trunk with different initial coordinates
- Split `load_of3_diffusion` into: `load_of3_model()` (once) + `prepare_of3_batch(sequence)` (per protein) + `run_of3_trunk(batch)` (per direction)

---

## Priority Order

1. **token_mask fix** — Critical, blocks 80% of benchmark
2. **Size mismatch fix** — Medium, blocks 3 proteins (6 runs)
3. **Chain ID fix** — Low, blocks 1 protein (manual table update)
4. **Model caching** — Performance, not correctness
