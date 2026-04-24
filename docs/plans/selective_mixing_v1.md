# Selective Z-Mixing: Spatially-Adaptive Pair Representation Blending

**Tarih:** 2026-04-25
**Durum:** Tasarım
**İlgili:** `mode_drive.py:_blend_z()`, `coords_to_contact.py`, `mode_drive_utils.py`

---

## 1. Problem

Mevcut `_blend_z()` **uniform alpha** kullanıyor:

```python
z_mod = z_trunk + alpha * (z_pseudo - z_trunk)   # tüm (i,j) pairleri aynı alpha
```

Bu yaklaşımın sorunu: MD hareketi sırasında **bazı bölgeler çok hareket ediyor** (loop, hinge),
**bazıları hiç hareket etmiyor** (core, helix). Uniform alpha ile:

- Çok hareket eden bölgelerin z_ij güncellemesi **yetersiz** kalıyor (alpha=0.7 bile küçük)
- Hiç hareket etmeyen bölgelere **gereksiz gürültü** ekleniyor
- Sonuç: OF3 hem "asıl değişimi" yeterince görmüyor, hem de "sabit bölgelerde" konfüze oluyor

## 2. Çözüm: Per-Pair Adaptive Alpha Mask

MD sonrası her (i,j) pair'i için **ne kadar değiştiğini** ölçüp, buna göre **pair-specific alpha** uygula:

```
alpha_ij = f(connectivity_change_ij, distance_change_ij)
z_mod_ij = z_trunk_ij + alpha_ij * (z_pseudo_ij - z_trunk_ij)
```

- Çok değişen pair → yüksek alpha (daha fazla yeni z_ij)
- Az değişen pair → düşük alpha veya 0 (eski z_ij'yi koru)

## 3. Değişim Ölçüm Metrikleri

### 3.1 Connectivity Farkı (ΔC)

```python
C_before = coords_to_contact(coords_current, r_cut, tau)   # [N, N]
C_after  = coords_to_contact(displaced_ca, r_cut, tau)      # [N, N]
delta_C  = |C_after - C_before|                              # [N, N], 0-1 arası
```

Connectivity matrix farkı her (i,j) pair'in **topolojik değişimini** ölçer.
Bir temas koptuğunda veya yeni temas oluştuğunda delta_C yüksek olur.

### 3.2 Mesafe Farkı (ΔD)

```python
# İlk yapıya (initial_coords) Kabsch align et
displaced_aligned, _ = kabsch_superimpose(initial_coords, displaced_ca)

# Per-residue displacement magnitude
d_i = ||displaced_aligned_i - coords_current_i||   # [N] her residue ne kadar hareket etti

# Pairwise displacement map: (i,j) pair'inin iki residue'sunun hareketinin max'ı
D_ij = max(d_i, d_j)   # [N, N] — ya da ortalama: (d_i + d_j) / 2
```

Bu ölçüm **konformasyon değişikliğinin en fazla olduğu bölgeleri** yakalar.

### 3.3 Birleşik Değişim Skoru (S_ij)

İki metriği birleştir:

```python
# Normalize et [0, 1] aralığına
delta_C_norm = delta_C / (delta_C.max() + eps)
D_norm = D_ij / (D_ij.max() + eps)

# Birleşik skor (ağırlıklı geometrik ortalama)
S_ij = (delta_C_norm ** w_c) * (D_norm ** w_d)   # w_c=0.5, w_d=0.5 default
```

**Neden ikisi birlikte?**
- Sadece ΔC: Uzak residue'lar arasında küçük mesafe değişiklikleri sigmoid'den büyük contact değişikliğine yol açabilir (false positive)
- Sadece ΔD: İki komşu residue birlikte hareket ederse mesafe değişmez ama çevreyle teması kopar (false negative)
- İkisinin çarpımı: Hem gerçekten hareket eden HEM DE topolojik değişim gösteren pair'leri seçer

### 3.4 Cutoff ve Alpha Mapping

```python
# Cutoff: belirli bir eşiğin altındaki pair'leri filtrele
S_ij[S_ij < change_cutoff] = 0.0   # default change_cutoff = 0.1

# Alpha mapping: S_ij -> alpha_ij
# Seçenek A: Doğrusal
alpha_ij = alpha_base + (alpha_max - alpha_base) * S_ij

# Seçenek B: Sigmoid (yumuşak geçiş)
alpha_ij = alpha_base + (alpha_max - alpha_base) * sigmoid((S_ij - midpoint) / temperature)

# Seçenek C: Step function (sert geçiş)
alpha_ij = where(S_ij > change_cutoff, alpha_max, alpha_base)
```

**Konfigürasyon parametreleri:**

| Parametre | Default | Açıklama |
|-----------|---------|----------|
| `selective_mixing` | `False` | Selective mixing'i aktifleştir |
| `selective_w_connectivity` | `0.5` | ΔC ağırlığı |
| `selective_w_distance` | `0.5` | ΔD ağırlığı |
| `selective_change_cutoff` | `0.1` | Bu eşiğin altı → alpha_base |
| `selective_alpha_base` | `0.0` | Değişmeyen pair'ler için alpha |
| `selective_alpha_max` | `1.0` | Max değişim olan pair'ler için alpha |
| `selective_mapping` | `"linear"` | `"linear"`, `"sigmoid"`, `"step"` |
| `selective_distance_mode` | `"max"` | `"max"` veya `"mean"` (pair displacement) |

## 4. Implementasyon Planı

### 4.1 Dosya Değişiklikleri

#### A. `src/mode_drive_config.py` — Yeni config alanları
```python
# Selective mixing
selective_mixing: bool = False
selective_w_connectivity: float = 0.5
selective_w_distance: float = 0.5
selective_change_cutoff: float = 0.1
selective_alpha_base: float = 0.0
selective_alpha_max: float = 1.0
selective_mapping: str = "linear"          # "linear", "sigmoid", "step"
selective_distance_mode: str = "max"       # "max", "mean"
```

#### B. `src/selective_mixing.py` — Yeni modül (core logic)

```python
def compute_change_score(
    coords_before: Tensor,    # [N, 3] mevcut coords
    coords_after: Tensor,     # [N, 3] displaced coords
    initial_coords: Tensor,   # [N, 3] alignment reference
    r_cut: float,
    tau: float,
    w_c: float = 0.5,
    w_d: float = 0.5,
    distance_mode: str = "max",
) -> Tensor:
    """Her (i,j) pair'i için değişim skoru hesapla → [N, N]"""

def compute_alpha_mask(
    change_score: Tensor,     # [N, N]
    change_cutoff: float,
    alpha_base: float,
    alpha_max: float,
    mapping: str = "linear",
) -> Tensor:
    """Değişim skorundan per-pair alpha mask üret → [N, N]"""

def selective_blend_z(
    z_pseudo: Tensor,         # [N, N, 128]
    z_trunk: Tensor,          # [N, N, 128]
    alpha_mask: Tensor,       # [N, N]
    normalize: bool = True,
    direction: str = "plus",
) -> Tensor:
    """Per-pair alpha ile z blending → [N, N, 128]"""
```

#### C. `src/mode_drive.py` — `_blend_z()` güncelleme

Mevcut `_blend_z()` → `_blend_z_uniform()` olarak yeniden adlandır.
Yeni `_blend_z()` selective_mixing config'e göre yönlendirir:

```python
def _blend_z(self, z_pseudo, zij_trunk, coords_before=None,
             displaced=None, initial_coords=None):
    if self.config.selective_mixing and coords_before is not None:
        return self._blend_z_selective(
            z_pseudo, zij_trunk, coords_before, displaced, initial_coords
        )
    return self._blend_z_uniform(z_pseudo, zij_trunk)
```

`_downstream_from_displaced()` güncelle: coords_before ve initial_coords parametrelerini geçir.

#### D. `src/__init__.py` — Export güncelleme

#### E. `tests/test_selective_mixing.py` — Yeni test dosyası

### 4.2 Downstream Entegrasyonu

```
    coords_current ──┐
                     │
    autostop/ANM ──► displaced_ca ──┐
                                    │
    ┌───────────────────────────────┘
    │
    ▼
    compute_change_score(coords_current, displaced_ca, initial_coords)
    │
    ▼
    compute_alpha_mask(change_score, cutoff, alpha_base, alpha_max)
    │                                        [N, N]
    ▼
    selective_blend_z(z_pseudo, z_trunk, alpha_mask)
    │                                        [N, N, 128]
    ▼
    OF3 diffusion
```

### 4.3 StepResult'a Ekleme

```python
# Diagnostic fields
change_score_mean: float | None = None    # ortalama change score
change_score_max: float | None = None     # max change score
n_active_pairs: int | None = None         # cutoff üstü pair sayısı
alpha_mask_mean: float | None = None      # ortalama alpha (selective)
```

## 5. Test Planı

### 5.1 Unit Tests (`test_selective_mixing.py`)

1. **test_change_score_identity** — Aynı coords → S_ij = 0
2. **test_change_score_large_displacement** — Büyük hareket → S_ij > 0
3. **test_alpha_mask_cutoff** — Cutoff altı → alpha_base, üstü → interpolated
4. **test_alpha_mask_mappings** — linear, sigmoid, step fonksiyonları
5. **test_selective_blend_preserves_shape** — Output shape = input shape
6. **test_selective_vs_uniform** — selective_mixing=False → uniform alpha ile aynı sonuç
7. **test_only_changed_pairs_updated** — S_ij=0 olan pair'ler z_trunk değerini koruyor
8. **test_symmetric** — alpha_mask simetrik (S_ij = S_ji)

### 5.2 Integration Tests (`test_mode_drive_coverage.py`)

9. **test_selective_mixing_pipeline** — Tam pipeline selective_mixing=True ile çalışıyor
10. **test_selective_mixing_with_autostop** — Autostop + selective mixing uyumlu

### 5.3 Colab Grid Search (V5)

11. **Uniform vs Selective** — Aynı paramterlerle A/B test
12. **change_cutoff sweep** — [0.05, 0.1, 0.15, 0.2, 0.3]
13. **alpha_base/alpha_max sweep** — base: [0.0, 0.05], max: [0.5, 0.7, 1.0]
14. **Mapping comparison** — linear vs sigmoid vs step
15. **w_c/w_d balance** — connectivity vs distance ağırlık dengeleri

## 6. Beklenen Faydalar

1. **Hedefli güncelleme:** OF3 sadece gerçekten değişen bölgeleri görür, sabit bölgelerde gürültü yok
2. **Daha yüksek alpha kullanabilme:** alpha_max=1.0 bile güvenli çünkü sadece değişen pair'lere uygulanıyor
3. **Drift koruması doğal:** Değişmeyen bölgelerin z_ij'si korunduğu için yapı daha stabil
4. **Daha az rejected step:** Gereksiz z_ij pertürbasyonu olmadığı için OF3 daha tutarlı yapılar üretir

## 7. Subagent Komutları

### Agent 1: Core Implementation (Implementor)

```
GÖREV: Selective mixing modülünü implement et.

DOSYALAR:
- OKU: src/mode_drive.py (özellikle _blend_z, _downstream_from_displaced, _autostop_downstream_from_pick)
- OKU: src/mode_drive_config.py (config yapısı)
- OKU: src/coords_to_contact.py (contact hesaplama)
- OKU: src/mode_drive_utils.py (kabsch_superimpose)
- OKU: docs/plans/selective_mixing_v1.md (bu plan)

YAPILACAKLAR:
1. src/selective_mixing.py oluştur:
   - compute_change_score() — connectivity delta + distance delta → [N,N] skor
   - compute_alpha_mask() — skor → per-pair alpha, linear/sigmoid/step mapping
   - selective_blend_z() — per-pair alpha ile z blending
   Tüm fonksiyonlar pure tensor ops, side-effect yok.

2. src/mode_drive_config.py'ye selective mixing config alanlarını ekle:
   selective_mixing, selective_w_connectivity, selective_w_distance,
   selective_change_cutoff, selective_alpha_base, selective_alpha_max,
   selective_mapping, selective_distance_mode

3. src/mode_drive.py güncellemeleri:
   - _blend_z() → selective_mixing=True ise selective path kullan
   - _downstream_from_displaced() → coords_before (hareket öncesi coords) parametresi ekle
   - _autostop_downstream_from_pick() → aynı şekilde coords_before geçir
   - step() ve autostop step fonksiyonlarında coords_before'u doğru yere bağla

4. StepResult'a diagnostic fields ekle:
   change_score_mean, change_score_max, n_active_pairs, alpha_mask_mean

5. src/__init__.py export güncelle

KISITLAR:
- selective_mixing=False (default) ise davranış HİÇ değişmemeli
- Mevcut testler KIRMIZI olmamalı
- Yeni fonksiyonlar torch.no_grad() ile çalışmalı
- alpha_mask simetrik olmalı (i,j) = (j,i)
```

### Agent 2: Test Suite (TDD)

```
GÖREV: Selective mixing için kapsamlı test suite yaz.

DOSYALAR:
- OKU: docs/plans/selective_mixing_v1.md (plan ve test listesi)
- OKU: tests/test_mode_drive_coverage.py (mevcut test yapısı ve helper'lar)
- OKU: src/selective_mixing.py (implement edildikten sonra)

YAPILACAKLAR:
1. tests/test_selective_mixing.py oluştur:
   Unit testler:
   - test_change_score_identity: aynı coords → score=0
   - test_change_score_large_displacement: büyük hareket → score>0
   - test_alpha_mask_cutoff: cutoff davranışı
   - test_alpha_mask_linear/sigmoid/step: 3 mapping modu
   - test_selective_blend_preserves_shape: output shape
   - test_selective_vs_uniform: disabled → uniform ile aynı
   - test_only_changed_pairs_updated: değişmeyen pair korunuyor
   - test_symmetric: alpha_mask simetrik

2. tests/test_mode_drive_coverage.py'ye integration testler ekle:
   - test_selective_mixing_pipeline: tam pipeline selective_mixing=True
   - test_selective_mixing_with_autostop: autostop uyumu

3. Tüm testleri çalıştır, hepsinin geçtiğini doğrula
4. Coverage >= 80% kontrol et

KISITLAR:
- Mock/pseudo-diffusion kullan (OF3 yok)
- Mevcut testlere dokunma (kırma)
- pytest convention'larını takip et
```

### Agent 3: V5 Notebook (Colab Search Hazırlığı)

```
GÖREV: Selective mixing A/B testi için V5 notebook hazırla.

DOSYALAR:
- OKU: notebooks/optimized_search_v4.ipynb (mevcut V4 yapısı)
- OKU: docs/plans/selective_mixing_v1.md (plan)
- OKU: src/selective_mixing.py (implement edildikten sonra)

YAPILACAKLAR:
1. notebooks/selective_mixing_v5.ipynb oluştur:
   Phase A: Uniform vs Selective (kontrol grubu)
   - V4 best config + selective_mixing=False (baseline)
   - V4 best config + selective_mixing=True (default params)

   Phase B: change_cutoff sweep
   - cutoff: [0.05, 0.1, 0.15, 0.2, 0.3]

   Phase C: alpha range sweep
   - alpha_base: [0.0, 0.05]
   - alpha_max: [0.5, 0.7, 1.0]

   Phase D: Mapping & weight sweep
   - mapping: linear, sigmoid, step
   - w_c/w_d: [(0.3,0.7), (0.5,0.5), (0.7,0.3)]

   Analysis: TM-score, RMSD, alpha_mask heatmap visualization

KISITLAR:
- V4 notebook yapısını takip et (config, converter, runner, analysis)
- Drive'a kaydetme hücreleri ekle
- Markdown açıklamalar Türkçe
```
