# Autostop Adapter Module

> IW-ENM MD + early-stop monitor'i `ModeDrivePipeline`'e baglayan torch ↔ numpy bridge.

## Amac

`run_autostop.py` icerisindeki saf NumPy IW-ENM MD kodunu mode-drive pipeline'ina entegre eder. Tek bir MD calistirmasi + kasa (cached) bir `AutostopTrace` dondurur; monitor-only fallback seviyelerinde MD yeniden cozulmez, ayni trace uzerinden `replay_monitor` ile baska pick uretilir.

## Kaynak Dosyalar

| Dosya | Icerik |
|-------|--------|
| `src/autostop_adapter.py` | Adapter — parameters, StructureContext, trace/pick, entry points |
| `src/iw_enm/structure.py` | `ProteinStructure` (vendored from run_autostop) |
| `src/iw_enm/network.py` | `InteractionWeightedENM` — IW spring network |
| `src/iw_enm/integrator.py` | `VelocityVerletIntegrator` |
| `src/iw_enm/config.py` | `SimulationConfig` |
| `src/iw_enm/analysis.py` | `compute_kinetic_energy`, yardimcilar |

## Public API

### `AutostopParams` (dataclass)

Tek bir autostop MD + monitor calistirmasinin TUM ayarlarini gruplar. Fallback seviyeleri bu dataclass'i per-level mutasyona ugratir.

```python
@dataclass
class AutostopParams:
    # --- ENM physics ---
    R_bb: float = 11.0           # backbone contact cutoff (Å)
    R_sc: float = 2.0            # sidechain contact cutoff (Å)
    K_0: float = 0.8             # base spring constant
    d_0: float = 3.8             # reference CA-CA distance (Å)
    n_ref: float = 10.0          # reference packing count (volume weighting)

    # --- Integration ---
    dt: float = 0.01
    mass: float = 1.0
    damping: float = 0.0
    v_mode: str = "breathing"    # initial velocity mode
    v_magnitude: float = 1.0

    # --- Run control ---
    n_steps: int = 5000
    save_every: int = 10
    back_off: int = 2
    crash_threshold_distance: float = 0.5

    # --- Monitor (early-stop) ---
    smooth_w: int = 11
    warmup_frac: float = 0.20
    patience: int = 3
    eps_E_rel: float = 0.002
    eps_N_rel: float = 0.005
    crash_window_saves: int = 20
    crash_threshold: int = 5
    min_saves_before_check: int = 15

    verbose: bool = False

    def monitor_only(self) -> dict: ...  # just the monitor knobs
```

> **Kullanici spec baseline** (`params = dict(R_bb=11.0, K_0=0.8, n_ref=10.0, v_magnitude=1.0)` + monitor knoblari ile):
> `warmup_frac=0.40, eps_E_rel=0.0002, eps_N_rel=0.0005, verbose=True` — `ModeDriveConfig` default'lari bunu yansitir.

### `StructureContext`

Pipeline basinda BIR KEZ insa edilir. Residue isimleri + CA'ya goreli atom offset'lerini cache'ler; her autostop adiminda yalnizca CA degisir, atomlar rigid-translate edilir.

```python
@classmethod
def from_pdb(cls, path: str, chain_id: str = "A") -> StructureContext: ...

@classmethod
def from_cif(cls, path: str, chain_id: str = "A") -> StructureContext: ...

@classmethod
def from_ca_only(
    cls,
    coords_ca: Tensor | ndarray,
    res_names: Sequence[str] | None = None,
    res_ids: Sequence[int] | None = None,
    chain_id: str = "A",
) -> StructureContext: ...

def rebuild_from_ca(self, coords_ca_np: ndarray) -> ProteinStructure: ...
```

**Kullanim sirasi (onerilen):**

1. PDB varsa → `StructureContext.from_pdb(...)` (gercek sidechain geometrisi).
2. CIF varsa → `StructureContext.from_cif(...)`.
3. Hic yapi dosyasi yoksa → `StructureContext.from_ca_only(...)` (idealize CB offset; volume weighting icin all-GLY varsayilir).

### `AutostopTrace`

Tek bir MD calistirmasinin raw sinyalleri. Ucuz fallback replay'i icin yeterli:

```python
@dataclass
class AutostopTrace:
    steps: np.ndarray                # shape (S,), raw MD step at each save
    E_tot: np.ndarray                # shape (S,)
    n_springs: np.ndarray            # shape (S,)
    crashes_cum_at_save: np.ndarray  # shape (S,), cumulative crash count
    trajectory: list[np.ndarray]     # length S+1; traj[0]=initial, traj[i+1]=save i
    total_mdsteps_requested: int
    save_every: int
    stop_step_md: int | None
```

### `AutostopPick`

Secilen frame + monitor diagnostikleri. Mode-drive pipeline'inin tuketecegi cikti:

```python
@dataclass
class AutostopPick:
    picked_ca: torch.Tensor   # (N,3), input device/dtype ile ayni
    picked_save_index: int    # 0-based save index
    picked_step_md: int       # raw MD step
    turn_k: int               # save-index of turnpoint (back_off oncesi)
    argmin_E_k: int
    argmin_N_k: int
    stop_step_md: int | None
    crashes_total: int
    back_off_used: int
    monitor_params: dict      # bu pick icin kullanilan monitor knoblari
    stop_reason: str | None
```

### `run_autostop_from_tensor`

```python
pick, trace = run_autostop_from_tensor(
    coords_ca: torch.Tensor,  # (N, 3), herhangi bir device/dtype
    ctx: StructureContext,
    params: AutostopParams,
)
# → (AutostopPick, AutostopTrace)
```

**Device/dtype contract:**
- Input tensor herhangi bir device/dtype olabilir.
- Adapter dahili olarak CPU float64'e cekip NumPy fizigini kosturur.
- `pick.picked_ca` ORIJINAL device/dtype'ta donerilir.
- `trace.trajectory` NumPy float64 olarak saklanir (replay'de round-trip maliyeti yok).

### `replay_monitor`

Ayni trace uzerinde yeni monitor knoblari ile yeniden pick uretir. MD YENIDEN COZULMEZ.

```python
new_pick = replay_monitor(
    trace: AutostopTrace,
    monitor_params: dict,    # AutostopParams.monitor_only() cikti format
    back_off: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
)
```

**Parity garantisi:** Ayni knoblarla ayni trace → ayni pick (test: `test_autostop_adapter.py::TestReplayMonitor::test_parity_same_params`).

## Mimari Akis

```
ModeDrivePipeline.step()
  │
  ├── strategy == "autostop"?
  │     │
  │     ├── (L0 baseline)  run_autostop_from_tensor(coords_ca, ctx, params)
  │     │       → pick_0, trace_0    (trace cache'lenir)
  │     │
  │     ├── (L1 monitor-only)  replay_monitor(trace_0, relaxed_monitor, back_off)
  │     │       → pick_1    (MD yok, ucuz)
  │     │
  │     └── (L2 physics tweak)  run_autostop_from_tensor(coords_ca, ctx, new_params)
  │             → pick_2, trace_2    (tam MD, pahali)
  │
  └── pick.picked_ca → z-blend → OF3 diffusion → confidence gate
```

## Kullanim Ornegi

```python
import torch
from src.autostop_adapter import StructureContext, AutostopParams, run_autostop_from_tensor
from src.mode_drive import ModeDriveConfig, ModeDrivePipeline
from src.converter import PairContactConverter

# 1. Structure context BIR KEZ
structure_ctx = StructureContext.from_pdb("1AKE.pdb", chain_id="A")

# 2. Pipeline
converter = PairContactConverter("checkpoints/best.pt")
config = ModeDriveConfig(
    combination_strategy="autostop",
    autostop_R_bb=11.0, autostop_K_0=0.8, autostop_n_ref=10.0,
    autostop_v_magnitude=1.0,
    autostop_n_steps=5000, autostop_save_every=10,
    autostop_back_off=2, autostop_smooth_w=11,
    autostop_warmup_frac=0.40, autostop_patience=3,
    autostop_eps_E_rel=0.0002, autostop_eps_N_rel=0.0005,
    autostop_crash_window_saves=20, autostop_crash_threshold=5,
    autostop_verbose=True,
    # Fallback ladder
    autostop_fallback_levels=(0, 1, 3, 5),
    enable_confidence_fallback=False,
)
pipeline = ModeDrivePipeline(
    converter=converter,
    config=config,
    structure_ctx=structure_ctx,
)

# 3. Calistir
result = pipeline.run(initial_coords_ca, zij_trunk)

# 4. Autostop diagnostikleri
for i, step in enumerate(result.step_results):
    info = step.autostop_info
    print(
        f"Step {i+1}: RMSD={step.rmsd:.2f}Å  "
        f"picked_save={info['picked_save_index']}/{info['n_saves']}  "
        f"turn_k={info['turn_k']}  fallback_L={step.fallback_level}"
    )

# 5. Cached trace (plotting, ablation vb.)
trace = pipeline._autostop_last_trace
# trace.E_tot, trace.n_springs, trace.crashes_cum_at_save, trace.trajectory
```

## Tek Basina Kullanim (pipeline disinda)

```python
from src.autostop_adapter import (
    AutostopParams, StructureContext, run_autostop_from_tensor, replay_monitor,
)

ctx = StructureContext.from_pdb("1AKE.pdb")
params = AutostopParams(R_bb=11.0, K_0=0.8, n_ref=10.0, v_magnitude=1.0)

# Tek MD calistirmasi
pick, trace = run_autostop_from_tensor(coords_ca, ctx, params)
print(pick.picked_save_index, pick.turn_k, pick.stop_reason)

# Ayni trace uzerinde daha gevsek monitor ile yeniden pick
relaxed = dict(params.monitor_only())
relaxed["eps_E_rel"] *= 4.0
relaxed["eps_N_rel"] *= 4.0
pick_relaxed = replay_monitor(trace, relaxed, back_off=4)
```

## Early-Stop Semantigi

`_EarlyStopMonitor` ucl kriterin HEPSINI `patience` kadar art arda gozlemlemelidir:

| Kriter | Detay |
|--------|-------|
| Enerji reversal | `E_smooth >= E_min + eps_E_rel * max(|E_min|, 1)` |
| Spring-count reversal | `N_smooth >= N_min * (1 + eps_N_rel)` |
| Crash onset | Son `crash_window_saves` save icinde `>= crash_threshold` yeni crash |

Warmup: ilk `max(min_saves_before_check, (n_steps/save_every) * warmup_frac)` save'de kontrol yok — argmin buffer'i oturuyor.

## Turn Point + back_off

```
k_turn  = min(argmin_E, argmin_N)          # en erken "geri donus" point'i
k_best  = max(0, k_turn - back_off)        # conservative: turn_point'ten back_off save geri
traj_idx = k_best + 1                      # trajectory offset (trajectory[0] = initial)
picked_ca = trajectory[traj_idx]
```

`back_off >= 1` conservatism onerilir — turnpoint tam enerji minimum oldugu icin, gradyanin henuz tersine donmedigi birkaç save geride frame secmek daha stabil sonuc verir.

## Test Kapsami

| Test | Icerik |
|------|--------|
| `tests/test_autostop_adapter.py::TestStructureContext` | from_pdb, from_cif, from_ca_only (torch & numpy), rebuild_from_ca atom-count preservation |
| `tests/test_autostop_adapter.py::TestRunAutostopFromTensor` | Pick/Trace shape, dtype/device preservation (float64), monotone steps, bad-shape rejection, monitor_params round-trip |
| `tests/test_autostop_adapter.py::TestReplayMonitor` | Parity (ayni knob → ayni pick), back_off sensitivity, no-MD-reintegration, relaxed knoblar |
| `tests/test_mode_drive.py::TestAutostopStrategy` | Pipeline entegrasyonu, fallback_level, autostop_info, trace cache |

Toplam: 17 + 8 = 25 autostop-spesifik test. Hepsi sentetik helix CA ile gercek PDB gerektirmeden calisir.

## Iliskili Dokumanlar

- [[architecture/09-anm-mode-drive]] §12-13 — Autostop stratejisi ve konfigurasyon
- [[architecture/13-confidence-guided-pipeline]] — Fallback ladder
- [[modules/anm-mode-drive]] — Ana pipeline modulu
