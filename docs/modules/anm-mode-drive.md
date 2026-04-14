# ANM Mode-Drive Module

> OF3 diffusion ile ANM normal modlarini birlestiren iteratif konformasyonel kesif modulu.

## Kaynak Dosyalar

| Dosya | Icerik |
|-------|--------|
| `src/anm.py` | `build_hessian`, `anm_modes`, `displace`, `anm_bfactors`, `collectivity`, `combo_collectivity` |
| `src/coords_to_contact.py` | `coords_to_contact` |
| `src/mode_combinator.py` | `ModeCombo`, `collectivity_combinations`, `grid_combinations`, `random_combinations`, `targeted_combinations` |
| `src/mode_drive.py` | `ModeDriveConfig`, `ModeDrivePipeline`, `StepResult`, `ModeDriveResult` |
| `src/of3_diffusion.py` | `load_of3_diffusion` — OF3 trunk once, SampleDiffusion per step |

## API Referansi

### ANM Core (`src/anm.py`)

```python
build_hessian(coords: [N,3], cutoff=15.0, gamma=1.0, tau=1.0) -> [3N, 3N]
anm_modes(hessian: [3N,3N], n_modes=20) -> (eigenvalues: [k], eigenvectors: [N,k,3])
displace(coords: [N,3], mode_vectors: [N,k,3], dfs: [k]) -> [N, 3]
anm_bfactors(eigenvalues: [k], eigenvectors: [N,k,3]) -> [N]
collectivity(eigenvectors: [N,k,3]) -> [k]           # per-mode collectivity
combo_collectivity(eigvecs: [N,k,3], indices) -> float  # multi-mode collectivity
```

### Combinators (`src/mode_combinator.py`)

```python
# Varsayilan strateji — collectivity'ye gore siralanmis
collectivity_combinations(
    eigenvectors: [N,k,3],
    n_modes_available: int,
    max_combo_size: int = 3,       # maks 3-lu veya 5-li
    df: float = 0.6,               # global deplasman faktoru
    max_combos: int = 50,
) -> list[ModeCombo]               # κ descending

# Diger stratejiler
grid_combinations(n_modes, select_modes=3, df_range, df_steps, max_combos) -> [ModeCombo]
random_combinations(n_modes, n_combos=50, select_modes_range, df_scale, eigenvalues) -> [ModeCombo]
targeted_combinations(current, target, modes, n_combos=20, top_modes=5) -> [ModeCombo]
```

### Pipeline (`src/mode_drive.py`)

```python
pipeline = ModeDrivePipeline(converter, config, diffusion_fn)
result = pipeline.run(initial_coords_ca, zij_trunk, target_coords=None)

# result.trajectory:    list of [N, 3] (n_steps+1 yapi: initial + her step)
# result.step_results:  list of StepResult
# result.total_steps:   int
```

**StepResult alanlari:**
```python
step.combo           # ModeCombo (mode_indices, dfs, label, collectivity_score)
step.displaced_ca    # [N, 3] ANM ile hareket ettirilmis CA
step.new_ca          # [N, 3] diffusion'dan cikan (veya displaced) CA
step.z_modified      # [N, N, 128] blend edilmis z_ij
step.contact_map     # [N, N] displaced'den hesaplanan contact map
step.rmsd            # float — RMSD from INITIAL (yuksek = iyi)
step.eigenvalues     # [n_modes]
step.eigenvectors    # [N, n_modes, 3]
step.b_factors       # [N] ANM B-faktorler
step.df_used         # float — bu adimda kullanilan df
```

## Kullanim Ornegi

```python
from src.converter import PairContactConverter
from src.mode_drive import ModeDrivePipeline, ModeDriveConfig

# 1. Egitilmis head'i yukle
converter = PairContactConverter("checkpoints/finetune/best_model.pt")

# 2. Pipeline konfigurasyonu
config = ModeDriveConfig(
    n_anm_modes=20,
    n_steps=5,                       # sabit 5 adim, erken durma yok
    combination_strategy="collectivity",
    n_combinations=30,
    z_mixing_alpha=0.3,
    # Collectivity parametreleri
    df=0.6,
    df_min=0.3,
    df_max=3.0,
    df_escalation_factor=1.5,
    max_combo_size=3,                # maks 3-lu kombinasyon
)

# 3. Pipeline olustur (diffusion_fn olmadan → displaced coords kullanilir)
pipeline = ModeDrivePipeline(converter, config)

# 4. Calistir
result = pipeline.run(
    initial_coords_ca=coords_ca,      # [N, 3]
    zij_trunk=zij_trunk,               # [N, N, 128]
)

# 5. Sonuclari incele
for i, step in enumerate(result.step_results):
    print(f"Step {i+1}: RMSD={step.rmsd:.2f}Å  df={step.df_used:.2f}  "
          f"combo={step.combo.label}  κ={step.combo.collectivity_score:.3f}")
```

**Ornek cikti:**
```
Step 1: RMSD=1.20Å  df=0.30  combo=coll_000_m0_1  κ=0.823
Step 2: RMSD=1.54Å  df=0.30  combo=coll_002_m0_2  κ=0.751
Step 3: RMSD=1.83Å  df=0.45  combo=coll_000_m0_1  κ=0.810
Step 4: RMSD=2.31Å  df=0.45  combo=coll_001_m0    κ=0.784
Step 5: RMSD=2.67Å  df=0.68  combo=coll_003_m1_2  κ=0.698
```

### OF3 Diffusion Entegrasyonu (`src/of3_diffusion.py`)

```python
from src.of3_diffusion import load_of3_diffusion

# Trunk ONCE calisir, diffusion her step'te tekrar calisir
diffusion_fn, zij_trunk = load_of3_diffusion(
    query_json="query.json",   # OF3 inference query
    device="cuda",
)

# Pipeline ile kullanim
pipeline = ModeDrivePipeline(converter, config, diffusion_fn=diffusion_fn)
result = pipeline.run(initial_coords_ca, zij_trunk)
```

**Akis:** z_mod → SampleDiffusion (DiffusionConditioning → AtomAttentionEncoder → DiffusionTransformer → AtomAttentionDecoder) → all-atom coords → CA extraction

**Gereksinimler:** A100/V100 GPU (~16GB VRAM), openfold3-repo clone + pip install

### Bidirectional Displacement

Her mode combo icin hem +df hem -df yonunde deplasman denenir:
- `coll_000_m0_1_pos` — pozitif yon
- `coll_000_m0_1_neg` — negatif yon

Target varsa: RMSD-to-target en dusuk olan secilir.
Target yoksa: RMSD-from-initial en yuksek olan secilir.

## Temel Prensipler

1. **RMSD initial'den olculur** — yuksek = daha fazla kesif = iyi
2. **n_steps sabit** — erken durma yok, verilen adim sayisi kadar calisir
3. **Collectivity-first** — en kollektif combo once denenir
4. **df eskalasyonu** — RMSD artmiyorsa df otomatik artar
5. **Iteratif z_ij** — her adimda hem coords hem z_ij guncellenir
6. **Bidirectional ±df** — her combo +/- iki yonde denenir
7. **Real OF3 diffusion** — pseudo-diffusion degil, gercek SampleDiffusion

## Mimari Dokumanlar

- [[architecture/08-anm-theory]] — ANM matematigi, collectivity formulu
- [[architecture/09-anm-mode-drive]] — Pipeline diyagramlari ve pseudocode
- [[architecture/10-iterative-refinement]] — df eskalasyonu, failure modlari
- [[architecture/05-gnm-contact-learner]] — ContactProjectionHead (z ↔ C)
- [[architecture/11-pipeline-mathematics]] — End-to-end matematiksel referans
