# Implementation Plan: autostop Strategy for ANM Mode-Drive Pipeline

## 1. Requirements Restatement

### 1.1 What "like collectivity" means

In `src/mode_drive.py`, "collectivity" is a value of `ModeDriveConfig.combination_strategy`. Selecting it:

- Switches the branch inside `ModeDrivePipeline._generate_combos` / `step`
- Produces `list[ModeCombo]` (mode indices + per-mode df)
- Downstream: displace → contact → z_pseudo → blend → OF3 → confidence → fallback

"Like collectivity" therefore means:
1. Selectable via `combination_strategy="autostop"`
2. Slotted into the same `step()` contract
3. Interoperates with existing z-blend → OF3 → confidence → fallback

### 1.2 What autostop actually does

Autostop **replaces the displacement sub-step** (current steps 1–3: ANM modes → combo → `displace()`).
Everything downstream (coords → contact → z_pseudo → blend → OF3 → confidence → fallback) is untouched.

## 2. Architectural Decisions

### 2.1 Where autostop slots in

**Decision: Replace the displacement sub-step only.**

```
Current:  coords_ca → ANM → combo → displace → contact → z → blend → OF3 → QC
Autostop: coords_ca → run_with_autostop → picked_coords → contact → z → blend → OF3 → QC
```

### 2.2 Abstraction seam

Introduce `DisplacementProposal` dataclass:
```python
displaced_ca:  torch.Tensor [N, 3]
combo:         ModeCombo          # synthetic for autostop
df_used:       float              # 0.0 for autostop
diagnostics:   dict | None        # autostop trace
```

`_evaluate_combo` is split into:
- `_displace_proposal(...) -> DisplacementProposal` (strategy-specific)
- `_evaluate_proposal(proposal) -> StepResult` (common downstream)

### 2.3 Synthetic ModeCombo for autostop

```python
mode_indices = ()
dfs = ()
label = f"autostop_step{step_idx}_pick{picked_step}_turn{turn_k}"
collectivity_score = 0.0
```

### 2.4 Fallback adaptation

Current levels tune df/combo/alpha. For autostop, the fallback sweep expands to cover:
- **Displacement-side** (autostop kinetics + picker): `v_magnitude`, `back_off`,
  and the 7 monitor params `smooth_w`, `warmup_frac`, `patience`, `eps_E_rel`,
  `eps_N_rel`, `crash_window_saves`, `crash_threshold`
- **Blending-side** (unchanged semantics): `alpha`

Rationale: if the picked frame still produces a bad OF3 confidence, the issue can
be (a) kick too strong → earlier back_off / smaller v_magnitude, (b) monitor
stopped too late or too early → relax/tighten `eps_E_rel` / `eps_N_rel` /
`patience` / `smooth_w`, (c) crash detector over/undersensitive → sweep
`crash_window_saves` / `crash_threshold`, (d) warmup mask misaligned →
`warmup_frac`, or (e) downstream z-blend too aggressive → smaller alpha.

Level ladder (each level re-runs autostop when it changes displacement-side
params; only alpha-only changes skip autostop re-run):

| Level | Changes                                                                                              | Rerun autostop? |
|-------|------------------------------------------------------------------------------------------------------|-----------------|
| L0    | baseline autostop (user defaults)                                                                    | —               |
| L1    | `back_off += Δ`  (pick earlier frame on cached trajectory)                                           | no (reuse)      |
| L2    | `v_magnitude × 0.5` (softer kick)                                                                    | yes             |
| L3    | monitor relaxation: `eps_E_rel × 2`, `eps_N_rel × 2`, `patience - 1` (stop earlier, accept noisier)  | yes             |
| L4    | `alpha × 0.5` (z-blend only; does NOT re-run autostop)                                               | no              |
| L5    | monitor tightening + smoothing: `eps_E_rel × 0.5`, `smooth_w + 4`, `warmup_frac × 1.5`               | yes             |
| L6    | crash detector sweep: `crash_window_saves × 2`, `crash_threshold + 2` (more tolerant of transients)  | yes             |
| L7    | grid search over `(v_magnitude, back_off, alpha)` from coarse scales                                 | yes (per cell)  |
| L8    | extended grid over `(eps_E_rel, eps_N_rel, patience, smooth_w)`                                      | yes (per cell)  |
| L9    | skip step (no displacement applied this step)                                                        | —               |

Notes:
- Levels that only mutate post-trajectory picking (L1) or only mutate α (L4)
  reuse the cached autostop trajectory — no re-integration cost.
- Level 7–8 grids are bounded tight: default 2 × 2 × 2 and 2 × 2 × 2 × 2, else
  budget explodes per step. Grid bounds are user-configurable.
- `clamp()` invariants: `back_off ∈ [0, len(trajectory)-1]`,
  `smooth_w ≥ 3` (odd), `warmup_frac ∈ [0, 0.5]`, `patience ≥ 1`,
  `eps_E_rel, eps_N_rel > 0`, `crash_window_saves ≥ 1`, `crash_threshold ≥ 1`.
- try/finally restores all mutated config fields after each level, same
  discipline as current `step_with_fallback` (see commit 399565e).

### 2.5 Torch ↔ NumPy bridge

`run_autostop.py` is NumPy/SciPy. `mode_drive.py` is PyTorch.
Bridge in `src/autostop_adapter.py`:
- in: `coords_ca: torch.Tensor` (any device/dtype)
- body: `.detach().cpu().numpy()` → `run_with_autostop` → tensor
- out: `(displaced_ca_tensor, diagnostics_dict)`

### 2.6 Residue metadata problem

`run_with_autostop` needs a PDB file — mid-pipeline we only have CA tensor.
**Decision: Option B (pragmatic).** Cache initial `ProteinStructure`; regenerate CB + atom_coords from CA via idealized placement. Escalate to Option A (iw_enm helper) only if crash detection degrades.

## 3. Phase-by-Phase Breakdown

### Phase 0 — Unblock iw_enm dependency (BLOCKER)

`iw_enm` package is not in repo. Options:
- (a) pip install from PyPI / private index
- (b) git submodule
- (c) vendor under `src/iw_enm/`
- (d) reimplement subset (multi-day — NOT RECOMMENDED)

**Exit criteria:** `python run_autostop.py <sample.pdb>` runs from repo root.

### Phase 1 — Adapter layer

| # | Step | File |
|---|------|------|
| 1.1 | Create `src/autostop_adapter.py` with `run_autostop_from_tensor(coords_ca, structure_ctx, params) -> (tensor, dict)` | NEW |
| 1.2 | Define `StructureContext` dataclass (cached `ProteinStructure`, res_names, initial atom layout) | same |
| 1.3 | Rebuild CB/atoms from CA via idealized placement | same |
| 1.4 | Return tensor on input's device/dtype | same |
| 1.5 | Diagnostics dict: stop_step, picked_step, turn_k, argmin_E_k, argmin_N_k, crashes_total, energies, springs | same |
| 1.6 | Parity test: adapter matches CLI bit-for-bit (fixed seed) | `tests/test_autostop_adapter.py` NEW |
| 1.7 | Expose `replay_monitor(trajectory, energies, springs, crash_steps, new_monitor_params) -> (picked_coords, diagnostics)` for cheap fallback (see §6.1) | same |
| 1.8 | Parity test: `replay_monitor` with original params matches original `run_with_autostop` output | same |

### Phase 2 — Wire into ModeDriveConfig + Pipeline

| # | Step |
|---|------|
| 2.1 | Add `"autostop"` to allowed `combination_strategy` values |
| 2.2 | Add 15 `autostop_*` config fields (R_bb, K_0, n_ref, v_magnitude, n_steps, save_every, back_off, smooth_w, warmup_frac, patience, eps_E_rel, eps_N_rel, crash_window_saves, crash_threshold, verbose). Defaults: smooth_w=11, warmup_frac=0.20, patience=3, eps_E_rel=0.002, eps_N_rel=0.005, crash_window_saves=20, crash_threshold=5 |
| 2.3 | Add `StepResult.autostop_info: dict \| None = None` |
| 2.4 | Add `ModeDrivePipeline.__init__(structure_ctx=None)` |
| 2.5 | Add `_autostop_proposal(coords_ca) -> DisplacementProposal` |
| 2.6 | Refactor `_evaluate_combo` → `_displace_proposal` + `_evaluate_proposal` |
| 2.7 | Branch in `step()`: autostop → single-shot proposal |
| 2.8 | Autostop bypasses df-escalation loop |

### Phase 3 — Fallback adaptation

| # | Step |
|---|------|
| 3.1 | Add fallback scale config fields:<br>• `autostop_fallback_v_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)`<br>• `autostop_fallback_back_off_adds: tuple[int, ...] = (0, 2, 4, 8)`<br>• `autostop_fallback_eps_E_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)`<br>• `autostop_fallback_eps_N_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)`<br>• `autostop_fallback_patience_deltas: tuple[int, ...] = (0, -1, +1, +2)`<br>• `autostop_fallback_smooth_w_deltas: tuple[int, ...] = (0, +4, -2, +8)` (clamped odd ≥ 3)<br>• `autostop_fallback_warmup_frac_scales: tuple[float, ...] = (1.0, 1.5, 0.5)`<br>• `autostop_fallback_crash_window_scales: tuple[float, ...] = (1.0, 2.0, 0.5)`<br>• `autostop_fallback_crash_threshold_adds: tuple[int, ...] = (0, +2, -2)` (clamped ≥ 1)<br>• `autostop_fallback_levels: tuple[int, ...] = tuple(range(10))`  # which L0–L9 to enable |
| 3.2 | Implement `ModeDrivePipeline._mutate_autostop_config(level, combo_of_scales) -> ConfigPatch` with `try/finally` restore (mirror commit 399565e discipline) |
| 3.3 | New `step_with_autostop_fallback()` implementing the L0→L9 ladder from §2.4. Each level rebuilds a `DisplacementProposal` (reusing cached trajectory when possible per §2.4 "Rerun autostop?" column) |
| 3.4 | L1 (back_off-only) and L4 (alpha-only) MUST reuse cached trajectory — do not re-integrate MD |
| 3.5 | L7/L8 grids use `itertools.product` over config-supplied scale tuples; cap total cells via `autostop_fallback_grid_cap` (default 8) |
| 3.6 | Dispatch in `run()`: if `strategy == "autostop"` → `step_with_autostop_fallback()`, else existing `step_with_fallback()` |
| 3.7 | try/finally config restore after every level (applies to all 9 mutated fields) |
| 3.8 | Record per-level params into `StepResult.autostop_info["fallback_trace"]` for notebook plotting |

### Phase 4 — Notebook integration

- Add autostop CONFIGURABLE PARAMETERS cell
- Guard κ plot when strategy=autostop
- Add autostop trace plot (E_tot, n_springs, crashes with markers at argmin/turn/pick/stop)
- Demo cell for end-to-end autostop

### Phase 5 — Tests

- Adapter parity test
- `TestAutostopStrategy` in `tests/test_mode_drive.py`
- Golden regression for other 5 strategies
- Fallback test (force confidence failure)
- Coverage ≥ 80%

### Phase 6 — Documentation

- `docs/architecture/09-anm-mode-drive.md` new autostop section
- `docs/modules/autostop-adapter.md` new
- Cross-links in `docs/00-project-index.md`

## 4. File List

### New files
- `src/autostop_adapter.py`
- `tests/test_autostop_adapter.py`
- `docs/modules/autostop-adapter.md`
- `docs/plans/autostop_integration.md` (this file)
- Possibly `src/iw_enm/` (if vendoring)

### Modified files
- `src/mode_drive.py` (config, StepResult, refactor, new branches)
- `run_autostop.py` (if vendor path → update imports)
- `tests/test_mode_drive.py` (TestAutostopStrategy)
- `notebooks/test_mode_drive.ipynb` (config, plots, demo)
- `docs/architecture/09-anm-mode-drive.md`
- `docs/00-project-index.md`
- `requirements.txt` / `pyproject.toml` (if pip)

## 5. Risk Assessment

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | iw_enm not findable | CRITICAL | Hard-block merge; user must resolve Q1 |
| R2 | Mid-pipeline has only CA tensor, no PDB | HIGH | StructureContext cache + idealized atom rebuild |
| R3 | Refactoring `_evaluate_combo` breaks existing strategies | HIGH | Golden regression before refactor |
| R4 | NumPy/Torch dtype drift | MEDIUM | Pin output dtype; roundtrip assertion |
| R5 | 5000 step autostop per pipeline step → slow | MEDIUM | Document runtime; notebook default 1000 |
| R6 | Fallback back_off clamping | LOW | Clamp to trajectory length |
| R7 | GPU↔CPU thrashing | MEDIUM | Single .cpu() transfer per step |
| R8 | Idealized atoms corrupt crash detection | MEDIUM | Validate on known-good case; escalate if needed |
| R9 | Notebook plots assume non-empty mode_indices | LOW | Guards |
| R10 | CLI vs library divergence | LOW | CLI imports from adapter post-refactor |

## 6. Open Questions (BLOCKING)

1. **Q1 (BLOCKER):** Where is iw_enm? pip? private repo? local dir? paste source?
2. **Q2:** Autostop single-shot per step, or rerun with v_magnitude variations like df-escalation? *(RESOLVED: single-shot at L0, re-run only on fallback levels L2+)*
3. **Q3:** Rebuild atoms from CA via idealized geometry (B), or add iw_enm-side `from_ca_only` helper (A)?
4. **Q4:** ~~Fallback L1 = back_off bump (cheap) or v_magnitude reduction (expensive, full rerun)?~~ *(RESOLVED: L1=back_off (cheap, cache reuse), L2=v_magnitude (rerun), monitor params in L3/L5/L6, extended grids at L7/L8 — see §2.4)*
5. **Q5:** `autostop_n_steps` default: 5000 (library) or 1000 (notebook-friendly)?
6. **Q6:** κ plot in notebook: hide for autostop or show "N/A" placeholder?
7. **Q7:** Future hybrid `autostop_then_modes`? Affects abstraction seam design.
8. **Q8 (NEW):** Grid cap `autostop_fallback_grid_cap` default 8 acceptable? (L7 = 2×2×2 = 8 cells; L8 = 2×2×2×2 = 16 cells — exceeds cap unless user opts in)
9. **Q9 (NEW):** Should L3/L5/L6 monitor mutations re-use cached autostop trajectory + only re-run the `EarlyStopMonitor` over it (cheap), or do they require full re-integration? *(recommendation: cheap re-monitor on cached trajectory — save N_steps of MD per fallback level)*

### 6.1 Critical insight — cheap re-monitor path

If `EarlyStopMonitor` state can be rebuilt from the cached `(energies, springs)` arrays already produced by a prior autostop run, then **all monitor-parameter fallback levels (L3, L5, L6, L8) can avoid re-running MD**. They would only:
1. Rebuild monitor with new `(smooth_w, warmup_frac, patience, eps_E_rel, eps_N_rel, crash_window_saves, crash_threshold)`
2. Re-scan cached `energies`/`springs`/crash arrays
3. Emit a new `(stop_step, picked_step, turn_k)` → different picked frame from cached trajectory

This makes the fallback ladder dramatically cheaper:
- **Cheap (cache-only)**: L0 (once), L1, L3, L4, L5, L6, L8 → reuses ONE MD integration
- **Expensive (re-integrate)**: L2, L7 (v_magnitude changes) → require new MD runs

Phase 1 adapter MUST expose `replay_monitor(trajectory, energies, springs, crash_steps, new_params) -> (picked_coords, diagnostics)` for this to work.

## 7. Acceptance Criteria

- [ ] `python run_autostop.py <sample.pdb>` runs from repo root
- [ ] `ModeDriveConfig(combination_strategy="autostop")` validates + executes
- [ ] `pipeline.run(...)` produces `len(trajectory) == n_steps + 1`, all coords finite
- [ ] `StepResult.autostop_info` has all expected keys
- [ ] `StepResult.combo.label` starts with `"autostop_"`
- [ ] Adapter parity test passes (bit-for-bit vs CLI under fixed seed)
- [ ] All pre-existing tests pass
- [ ] Fallback path exercised by at least one test
- [ ] Notebook runs top-to-bottom with `strategy="autostop"`
- [ ] Coverage ≥ 80% on new code
- [ ] Docs updated
- [ ] No regression in 5 existing strategies (golden comparison)
