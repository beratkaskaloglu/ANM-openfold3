# Selective Z-Mixing Pipeline — Full Architecture Document

**Tarih:** 2026-05-05
**Versiyon:** V5 (son hal)
**Durum:** Aktif — Grid search devam ediyor

---

## Executive Summary

ANM-openfold3, protein yapisindaki konformasyonel degisimi tahmin etmek icin **Anisotropic Network Model (ANM)** titresim modlarini, **OpenFold3 (OF3)** diffusion modeliyle birlestiren bir pipeline'dir. **Selective Z-Mixing**, bu pipeline'in en son eklenen ve en kritik bilesenidir: ANM'nin urettigi deplasman bilgisini OF3'un anlayacagi pair representation formatina donusturur — ama bunu **spatially-adaptive** (mekansal olarak uyarlanabilir) bir sekilde yapar.

**Temel fikir**: Protein icindeki her (i,j) residue cifti icin, "ne kadar hareket etti?" sorusunu sorup, buna gore z-blending agresifligini ayarla.

---

## 1. Pipeline Genel Bakis

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         ANM-OPENFOLD3 PIPELINE                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  YAPI [N,3]                                                                │
│     │                                                                      │
│     ▼                                                                      │
│  ┌──────────────────┐                                                      │
│  │ ANM Eigenvectors │  H = build_hessian(coords)                           │
│  │   λ[k], v[N,k,3] │  λ,v = anm_modes(H, n_modes=20)                    │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │  Mode Combinator │  combos = collectivity_combinations(v, λ, ...)       │
│  │  (indices, dfs)  │  Ranked by eigenvalue-weighted collectivity           │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │  ANM Displace    │  displaced = coords + Σ(df_k * amp_k * v_k)          │
│  │  [N, 3]          │  amp_k = 1/√λ_k (eigenvalue weighting)              │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ Contact Map      │  C_ij = σ(-(d_ij - r_cut) / τ)                      │
│  │  [N, N]          │  Soft sigmoid contact                                │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ Converter        │  z_pseudo = contact_to_z(contact)                    │
│  │ (Inverse Head)   │  Learned MLP: logit(C) → [N, N, 128]                │
│  │  [N, N, 128]     │                                                      │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ╔══════════════════════════════════════╗                                   │
│  ║  SELECTIVE Z-MIXING (V5 yenilik)    ║                                   │
│  ╠══════════════════════════════════════╣                                   │
│  ║                                      ║                                   │
│  ║  change_score[N,N] ←                 ║                                   │
│  ║    compute_change_score(             ║                                   │
│  ║      coords_before, displaced,       ║                                   │
│  ║      initial_coords)                 ║                                   │
│  ║                                      ║                                   │
│  ║  alpha_mask[N,N] ←                   ║                                   │
│  ║    compute_alpha_mask(               ║                                   │
│  ║      change_score, cutoff,           ║                                   │
│  ║      alpha_base, alpha_max)          ║                                   │
│  ║                                      ║                                   │
│  ║  z_modified[N,N,128] ←              ║                                   │
│  ║    selective_blend_z(                ║                                   │
│  ║      z_pseudo, z_trunk, alpha_mask)  ║                                   │
│  ║                                      ║                                   │
│  ╚═══════════════╤══════════════════════╝                                   │
│                  │                                                          │
│                  ▼                                                          │
│  ┌──────────────────┐                                                      │
│  │ OF3 Diffusion    │  z_modified → SampleDiffusion → new_ca [N, 3]        │
│  │ (trunk cached)   │  + pTM, pLDDT, PAE, ranking                          │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  ┌──────────────────┐                                                      │
│  │ Confidence Gate  │  composite_score >= threshold?                        │
│  │  + Fallback      │  Hard rejects: Rg, clash, rmsd_init                  │
│  └────────┬─────────┘                                                      │
│           │                                                                │
│           ▼                                                                │
│  KABUL → new_ca sonraki iterasyonun girdisi                                │
│  RED → fallback ladder (df↓, alpha↓, combo↓) veya STALL                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Katman Katman Detay

### 2.1 ANM Eigenvector Hesaplama

**Dosya:** `src/anm.py`

ANM, proteini **elastik ag modeli** olarak ele alir. Her CA atomu bir dugum, yakin atomlar arasi yay baglantilari vardir.

```python
# Hessian matrisi [3N x 3N]
H_ij = -γ * w_ij * (e_ij ⊗ e_ij)   # off-diagonal 3x3 block
H_ii = -Σ_{j≠i} H_ij               # satir toplami

# w_ij: mesafeye dayali yumusak cutoff
w_ij = sigmoid(-(d_ij - cutoff) / tau)

# Eigendecomposition
λ[k], v[N, k, 3] = eigh(H)  # ilk 6 trivial mod atlanir
```

**Cikti:** `n_modes` adet eigenvector (titresim yonleri) ve eigenvalue (frekans^2).

**Fiziksel anlam:**
- Dusuk eigenvalue = dusuk frekans = buyuk amplitud = kolektif hareket
- Yuksek eigenvalue = yuksek frekans = kucuk amplitud = lokal titresim
- `1/√λ` ile agirliklandirma fiziksel amplitudu verir

### 2.2 Mode Kombinasyonu

**Dosya:** `src/mode_combinator.py`

Tek bir eigenvector yerine, birden fazla modu birlestirerek daha zengin konformasyonel degisiklikler uretilir.

**Strateji: Collectivity-ranked combinations**

```python
# Tum 1-3 mod kombinasyonlari enumerate et
subsets = all_combinations(modes, max_size=3)

# Her subset icin collectivity hesapla
κ = (1/N) * exp(Shannon_entropy(||v_i||²))

# Eigenvalue-weighted collectivity ile sirala
score = κ * Σ(1/λ_k)  # dusuk frekansli kolektif modlar one cikar

# ±df yonleri ile cift
combos = top_N_combos * 2  # her biri + ve - yonde
```

**Displacement formula:**
```python
new_coords = coords + Σ_k(df_k * (1/√λ_k) * v_k)
```

### 2.3 Contact Map Hesaplama

**Dosya:** `src/coords_to_contact.py`

Displaced koordinatlardan **yumusak contact map** uretir:

```python
d_ij = ||displaced_i - displaced_j||₂        # [N, N] mesafe matrisi
C_ij = sigmoid(-(d_ij - r_cut) / tau)         # [N, N] soft contact

# Parametreler:
# r_cut = 10.0 Å (contact esigi)
# tau = 1.5 (sigmoid sertligi)
```

**Neden soft sigmoid?**
- Hard cutoff turevlenebilir degil → gradient-based optimizasyon icin uygun degil
- Sigmoid, esik civarinda yumusak gecis saglar
- tau kuculdukce hard cutoff'a yaklasir

### 2.4 Pair Contact Converter

**Dosya:** `src/converter.py`

Contact map'i OF3'un pair representation uzayina donusturur.

**Ileri yon (training icin):**
```python
z [N, N, 128] → ContactProjectionHead → contact [N, N]
```

**Ters yon (inference — pipeline icin kritik):**
```python
contact [N, N] → logit(C) → W_inv (learned MLP) → z_pseudo [N, N, 128]
```

**Neden onemli?** OF3, yapilari pair representation (`z`) uzerinden uretir. ANM ise koordinat uzayinda calisir. Converter, bu iki uzay arasindaki kopruyu kurar.

### 2.5 Selective Z-Mixing (V5 Core Innovation)

**Dosya:** `src/selective_mixing.py`

#### Problem: Uniform Alpha

Eski yakklasim (V1-V4):
```python
z_mod = z_trunk + alpha * (z_pseudo - z_trunk)   # tum pair'lere ayni alpha
```

Sorun: Loop bolgeler cok hareket ederken, helix/core bolgeler hic hareket etmez. Uniform alpha ile:
- Cok hareket eden bölgelere yetersiz guncelleme
- Hareketsiz bolgelere gereksiz gurultu

#### Cozum: Per-Pair Adaptive Alpha

Uc adimli hesaplama:

**Adim 1: Change Score — "Ne kadar degisti?"**

```python
def compute_change_score(coords_before, coords_after, initial_coords, ...):
    # Signal 1: Topolojik degisim
    C_before = coords_to_contact(coords_before)
    C_after  = coords_to_contact(coords_after)
    ΔC = |C_after - C_before|                    # [N, N]

    # Signal 2: Konformasyonel degisim
    aligned = kabsch_superimpose(initial, coords_after)
    d_i = ||aligned_i - before_i||               # [N] per-residue displacement
    D_ij = max(d_i, d_j)                         # [N, N] pairwise

    # Birlesim: Weighted geometric mean
    S_ij = (ΔC_norm)^w_c * (D_norm)^w_d         # [N, N] ∈ [0, 1]
```

**Neden iki sinyal birlikte?**
| Durum | Sadece ΔC | Sadece ΔD | İkisi birlikte |
|-------|-----------|-----------|----------------|
| Uzak residue'lar arasinda kucuk mesafe degisikligi → buyuk sigmoid etkisi | False positive ↑ | OK | Filtrelenir |
| Iki komsu residue birlikte hareket → aralari degismez ama cevre ile temas kopar | OK | False negative ↑ | Yakalanir |
| Gercek buyuk hareket + temas kopusu | Pozitif | Pozitif | Guclendirme |

**Adim 2: Alpha Mask — "Ne kadar blend edilsin?"**

```python
def compute_alpha_mask(change_score, cutoff=0.1, base=0.0, max=1.0, mapping="linear"):
    # Cutoff altindakiler sifirla
    S[S < cutoff] = 0

    # Mapping: score → alpha
    if mapping == "linear":
        alpha_ij = base + (max - base) * (S / S.max())
    elif mapping == "sigmoid":
        alpha_ij = base + (max - base) * sigmoid((S_norm - 0.5) * 10)
    elif mapping == "step":
        alpha_ij = where(S > cutoff, max, base)

    return alpha_ij  # [N, N] ∈ [base, max]
```

**Adim 3: Selective Blend — "Per-pair alpha ile karistir"**

```python
def selective_blend_z(z_pseudo, z_trunk, alpha_mask):
    # Normalize z_pseudo stats to match z_trunk
    z_pseudo = (z_pseudo - mean) / std * trunk_std + trunk_mean

    delta_z = z_pseudo - z_trunk                   # [N, N, 128]
    alpha_expanded = alpha_mask.unsqueeze(-1)       # [N, N, 1]
    z_blended = z_trunk + alpha_expanded * delta_z  # [N, N, 128]
```

**Gorsel ozet:**

```
                    Change Score Heat Map
                    ┌───────────────────┐
                    │ ░░░░░████░░░░░░░░ │  █ = yuksek change (loop/hinge)
                    │ ░░░░████░░░░░░░░░ │  ░ = dusuk change (core/helix)
                    │ ░░░████░░░░░░░░░░ │
                    │ ███████░░░░░░░░░░ │
                    │ ██████░░░░░████░░ │
                    │ ░░░░░░░░░░████░░░ │
                    │ ░░░░░░░░░████░░░░ │
                    └───────────────────┘
                              ↓
                    Alpha Mask (per-pair)
                    ┌───────────────────┐
                    │ 0 0 0 0 .8 .9 0 0 │  Sadece hareket eden
                    │ 0 0 0 .7 .9 .8 0 0 │  bölgelere yuksek alpha
                    │ 0 0 .6 .8 .7 0 0 0 │  → OF3 sadece "asil"
                    │ .8 .9 .8 .6 0 0 0 0 │  degisimi gorur
                    │ .9 .8 0 0 0 0 .7 0 │
                    │ 0 0 0 0 0 0 .8 .7 │
                    │ 0 0 0 0 0 .7 .6 0 │
                    └───────────────────┘
```

### 2.6 OF3 Diffusion

**Dosya:** `src/of3_diffusion.py`

OF3'un pair representation'i (`z_trunk`) bir kez hesaplanir (trunk inference, pahali). Sonra her adimda sadece diffusion sampling yapilir (ucuz).

```python
# Bir kere (baslatmada):
si_input, si_trunk, zij_trunk = model.run_trunk(batch)  # [N, N, 128]

# Her adimda:
z_modified = selective_blend_z(z_pseudo, zij_trunk, alpha_mask)  # pipeline'dan gelen
atom_positions = model.sample_diffusion(z_modified, ...)         # yeni yapi
confidence = aux_heads(atom_positions)                            # pTM, pLDDT, PAE
```

**Ranking formula:** `ranking = 0.8 * pTM + 0.2 * mean(pLDDT / 100)`

**Coklu sample:** K sample uretilir, en yuksek ranking secilir.

### 2.7 Composite Confidence Scoring

**Dosya:** `src/composite_confidence.py`

Tek bir metrik yerine agirlikli bilesik skor:

```python
score = w_ptm    * normalize_ptm(ptm)           # [0, 1]
      + w_plddt  * normalize_plddt(plddt_mean)   # [0, 1]
      + w_pae    * normalize_pae(mean_pae)       # [0, 1]
      + w_rg     * normalize_rg(rg_ratio)        # [0, 1]
      + w_cr     * normalize_contact_recon(cr)   # [0, 1]
```

**Hard reject (composite'den bagimsiz):**
- Rg ratio > 2.5 (yapi sismis)
- Clash detected (atomlar carpisir)
- RMSD_init > 10 Å (cok buyuk adim)

**Kabul kriteri:** `composite_score >= threshold` (tipik: 0.50)

### 2.8 Fallback Ladder

Confidence basarisiz olursa kademeli geri cekilme:

| Level | Strateji | Etki |
|-------|----------|------|
| L0 | Normal step | - |
| L1 | Sonraki N combo dene | Alternatif hareket |
| L2 | df azalt (×factor) | Daha kucuk adim |
| L3 | Tek mod (en kolektif) | En guvenli hareket |
| L4 | Alpha azalt | Daha az z degisiklligi |
| L5+ | Grid: combo × df × alpha | Exhaustive search |

### 2.9 Drift Korumalari (V4+)

| Mekanizma | Tetikleme | Etki |
|-----------|-----------|------|
| Best-so-far rollback | TM %40+ duserse | En iyi yapiya geri don |
| Adaptive early stop | 3 ardisik TM dususu | Pipeline'i durdur |
| Alpha decay | Ardisik reject | alpha *= 0.8 (kucult) |
| RMSD_init hard cutoff | > 10 Å | Adimi hard-reject |

---

## 3. Tensor Shapes & Data Flow

| Adim | Girdi | Cikti | Shape |
|------|-------|-------|-------|
| ANM Hessian | coords [N,3] | H | [3N, 3N] |
| Eigendecomp | H [3N,3N] | λ, v | [k], [N, k, 3] |
| Displace | coords + v + df | displaced | [N, 3] |
| Contact | displaced [N,3] | C | [N, N] |
| Inverse head | C [N,N] | z_pseudo | [N, N, 128] |
| Change score | before, after [N,3] | S | [N, N] |
| Alpha mask | S [N,N] | α | [N, N] |
| Selective blend | z_pseudo, z_trunk, α | z_mod | [N, N, 128] |
| OF3 diffusion | z_mod [N,N,128] | new_ca | [N, 3] |
| Confidence | new_ca + metrics | score | scalar |

---

## 4. Konfigürasyon Parametreleri (V5)

### 4.1 ANM & Contact

| Parametre | Default | Açiklama |
|-----------|---------|----------|
| `anm_cutoff` | 15.0 Å | Hessian icin mesafe cutoff |
| `n_anm_modes` | 20 | Hesaplanan mod sayisi |
| `contact_r_cut` | 10.0 Å | Soft contact mesafe esigi |
| `contact_tau` | 1.5 | Sigmoid sicakligi |
| `df` | 0.6 | Baslangic displacement factor |
| `df_min` / `df_max` | 0.3 / 3.0 | DF escalation araligi |

### 4.2 Z-Mixing (Uniform)

| Parametre | Default | Açiklama |
|-----------|---------|----------|
| `z_mixing_alpha` | 0.7 | Global blend gücü |
| `normalize_z` | True | z_pseudo → z_trunk stats'a normalize |
| `z_direction` | "plus" | + veya - delta_z |

### 4.3 Selective Mixing (V5)

| Parametre | Default | Açiklama |
|-----------|---------|----------|
| `selective_mixing` | False | Selective mixing aktif mi |
| `selective_w_connectivity` | 0.5 | ΔC agirligi (topolojik degisim) |
| `selective_w_distance` | 0.5 | ΔD agirligi (konformasyonel degisim) |
| `selective_change_cutoff` | 0.1 | Bu esik alti → alpha_base |
| `selective_alpha_base` | 0.0 | Hareketsiz pair'ler icin alpha |
| `selective_alpha_max` | 1.0 | Max hareket eden pair'ler icin alpha |
| `selective_mapping` | "linear" | "linear", "sigmoid", "step" |
| `selective_distance_mode` | "max" | "max" veya "mean" (pairwise) |

### 4.4 Confidence & Safety

| Parametre | Default | Açiklama |
|-----------|---------|----------|
| `confidence_ptm_cutoff` | 0.30 | pTM safety net |
| `confidence_plddt_cutoff` | 65.0 | pLDDT minimum |
| `confidence_ranking_cutoff` | 0.45 | PRIMARY gate |
| `confidence_rg_max` | 2.5 | Max Rg orani |
| `confidence_rmsd_init_max` | 10.0 | Max adim buyuklugu |
| `confidence_clash_reject` | True | Clash → hard reject |
| `enable_best_rollback` | True | TM drop → rollback |
| `best_rollback_tm_drop` | 0.40 | %40 TM dususunde tetikle |
| `enable_adaptive_stop` | True | Ardisik dususte dur |
| `adaptive_stop_window` | 3 | Kac ardisik dusus |

---

## 5. Alternatif Strateji: Autostop (IW-ENM MD)

**Dosya:** `src/autostop_adapter.py`

Mode-combo yerine **Isotropic Weighted ENM Molecular Dynamics** kullanir:

```
CA coords → IW-ENM kuvvet alani → Velocity Verlet integrasyon
  → Enerji izleme (E_tot, N_springs, crashes)
  → Early-stop (enerji reversal + crash onset)
  → Picked frame [N, 3]
  → Downstream pipeline (contact → z → blend → OF3)
```

**Avantajlari:**
- Mode secimi gereksiz (MD tum modlari dogal olarak olusturur)
- Fiziksel enerji minimizasyonu
- Early-stop ile optimal frame secimi

**Dezavantajlari:**
- Hesaplama maliyeti (N_steps=5000 MD adimi)
- Yonlendirme zor (MD nereye gidecegini bilemezsin)

---

## 6. Grid Search Fazlari (V5 Notebook)

| Faz | Degisken | Deger Araligi | Run Sayisi |
|-----|----------|---------------|------------|
| A | Uniform vs Selective | 2 config | 2 |
| B | change_cutoff | [0.05, 0.1, 0.15, 0.2, 0.3] | 5 |
| C | cutoff × alpha_base × alpha_max | [0.05,0.10] × [0.0,0.05] × [0.5,0.7,1.0] | 12 |
| D | mapping × weights | [linear,sigmoid,step] × 3 weight pair | 9 |
| **Toplam** | | | **28** |

**Hedef:** TM-score'u V4 baseline'in uzerine cikarmak.

---

## 7. Neden Selective Mixing Calismali?

### Fiziksel Motivasyon

Proteinlerde hareket **homojen degildir**:
- **Loop/hinge** bölgeleri: Buyuk amplitud, cok hareket
- **Helix/sheet** core: Rijit, az hareket
- **Active site**: Ozel hareket profili

Uniform alpha, bu heterojenliği ignor eder. Selective mixing, ANM'nin verdigi hareket bilgisini **filtreleyerek** OF3'a iletir:

1. **Hedefli guncelleme**: OF3 sadece gercekten degisen bolgeleri gorur
2. **Daha yuksek alpha kullanabilme**: alpha_max=1.0 bile guvenli (sadece degisen pair'lere)
3. **Dogal drift korumasi**: Hareketsiz bolgelerin z_ij'si korunur → yapi stabilitesi
4. **Daha az reject**: Gereksiz z perturbasyonu olmadan OF3 daha tutarli yapilar uretir

### Beklenen Sonuc

```
V4 (uniform alpha=0.7):
  - Tum pair'lere ayni alpha → core bozulur → confidence dusturur → reject artar

V5 (selective alpha_base=0, alpha_max=1.0):
  - Core pair'ler: alpha=0 → z_trunk korunur → OF3 stable
  - Loop pair'ler: alpha≈0.8 → buyuk guncelleme → OF3 hareketi gorur
  - Sonuc: Daha yuksek accept rate + daha iyi TM-score
```

---

## 8. Dosya Indeksi

| Dosya | Satir | Rol |
|-------|-------|-----|
| `src/anm.py` | ~250 | ANM Hessian, eigenvectors, displacement |
| `src/mode_combinator.py` | ~260 | Mode combination strategies |
| `src/mode_drive.py` | ~1700 | Ana pipeline orchestration |
| `src/mode_drive_config.py` | ~300 | Tum konfigürasyon |
| `src/coords_to_contact.py` | ~25 | Soft contact hesaplama |
| `src/converter.py` | ~150 | z ↔ contact donusum |
| `src/contact_head.py` | ~100 | Neural network head |
| `src/selective_mixing.py` | ~190 | Per-pair adaptive blending |
| `src/of3_diffusion.py` | ~500 | OF3 wrapper + confidence |
| `src/autostop_adapter.py` | ~600 | IW-ENM MD alternative |
| `src/composite_confidence.py` | ~200 | Weighted scoring |
| `src/mode_drive_utils.py` | ~150 | Kabsch, RMSD, TM-score |

---

## 9. Ilgili Dokumanlar

- [08-anm-theory.md](08-anm-theory.md) — ANM matematigi
- [09-anm-mode-drive.md](09-anm-mode-drive.md) — Mode-drive tasarimi
- [11-pipeline-mathematics.md](11-pipeline-mathematics.md) — Pipeline matematigi
- [13-confidence-guided-pipeline.md](13-confidence-guided-pipeline.md) — Confidence sistemi
- [plans/selective_mixing_v1.md](../plans/selective_mixing_v1.md) — Selective mixing tasarim plani
- [plans/selective_mixing_v5_resilient.md](../plans/selective_mixing_v5_resilient.md) — V5 notebook plani
