# 13 - Confidence-Guided Adaptive Pipeline

> Mode-Drive pipeline'ına OF3 confidence metrikleri (pLDDT, pTM) entegrasyonu,
> multi-sample diffusion, ve adaptive fallback stratejisi.

## 1. Motivasyon

Mevcut pipeline her step'te tek bir diffusion sample üretir ve sadece RMSD'ye bakar.
Sorunlar:
- **Kalite kontrolü yok**: Diffusion çıktısı fiziksel olarak anlamsız olabilir (clash, broken chain)
- **Tek sample**: OF3 stochastic — aynı z_mod'dan 5 farklı yapı çıkabilir, en iyisini seçmeliyiz
- **Kör eskalasyon**: df artarsa ya da mode subset çok agresifse yapı bozulur ama pipeline bunu anlamaz

Çözüm: **Confidence-guided adaptive fallback** — her step'te confidence ölç, düşükse parametreleri kademeli düşür.

## 2. Mimari Genel Bakış

```
Step i:
  coords_i → ANM modes → displace → contact → z_pseudo → blend(z_trunk, alpha)
                                                              ↓
                                                     z_modified [N,N,128]
                                                              ↓
                                              ┌───────────────────────────────┐
                                              │  Multi-Sample Diffusion (K=5) │
                                              │  z_mod → 5x [N,3] + logits   │
                                              └───────────────────────────────┘
                                                              ↓
                                              ┌───────────────────────────────┐
                                              │  Confidence Scoring           │
                                              │  pLDDT, pTM per sample        │
                                              └───────────────────────────────┘
                                                              ↓
                                              ┌───────────────────────────────┐
                                              │  Selection & Fallback         │
                                              │  Best sample by ranking_score │
                                              │  If below cutoff → fallback   │
                                              └───────────────────────────────┘
                                                              ↓
                                                    coords_{i+1} veya retry
```

## 3. Confidence Metrikleri

### 3.1 OF3'ten Confidence Çıkarımı

OF3 forward pipeline:
```
trunk → (si_input, si_trunk, zij_trunk)
sample_diffusion → atom_positions [1, K, N_atom, 3]
aux_heads(si_input, output={si_trunk, zij_trunk, atom_positions}) → logits
get_confidence_scores(batch, outputs) → {pLDDT, pTM, ipTM, PAE, PDE}
```

**Kritik**: `sample_diffusion()` sadece koordinat döner.
Confidence için `aux_heads` ayrıca çalıştırılmalı.
`aux_heads` girdileri:
- `batch`: cached (trunk'tan)
- `si_input`: cached (trunk'tan)
- `output`: `{si_trunk, zij_trunk, atom_positions_predicted}`
  - `zij_trunk` → biz z_modified kullanacağız
  - `si_trunk` → cached
  - `atom_positions_predicted` → diffusion çıktısı

### 3.2 Metrikler

| Metrik | Boyut | Aralık | Anlam |
|--------|-------|--------|-------|
| pLDDT | [N_token] | 0-1 | Per-residue güvenilirlik |
| pTM | scalar | 0-1 | Global yapı kalitesi (TM-score tahmini) |
| ipTM | scalar | 0-1 | Interface kalitesi (multimer) |
| PAE | [N, N] | Å | Predicted aligned error |

### 3.3 Ranking Score

Sample seçimi için tek skor:

```python
ranking_score = 0.8 * pTM + 0.2 * mean(pLDDT)
```

AF3 ile aynı formül. ipTM multimer için eklenir:
```python
# multimer:
ranking_score = 0.8 * (0.8 * ipTM + 0.2 * pTM) + 0.2 * mean(pLDDT)
```

## 4. Multi-Sample Diffusion

### 4.1 Değişiklik: `of3_diffusion.py`

Mevcut: `no_rollout_samples=1` → tek yapı
Yeni: `no_rollout_samples=K` (default 5) → K yapı

```python
def diffusion_fn(z_mod: Tensor) -> DiffusionResult:
    """
    Args:
        z_mod: [N, N, 128] modified pair representation

    Returns:
        DiffusionResult:
            all_ca:     [K, N, 3]       K adet CA koordinatları
            best_ca:    [N, 3]          En iyi sample
            plddt:      [K, N]          Per-sample pLDDT
            ptm:        [K]             Per-sample pTM
            ranking:    [K]             Per-sample ranking score
            best_idx:   int             En iyi sample indeksi
    """
```

### 4.2 Implementasyon Detayı

```python
@dataclass
class DiffusionResult:
    all_ca: torch.Tensor        # [K, N, 3]
    best_ca: torch.Tensor       # [N, 3]
    plddt: torch.Tensor         # [K, N]
    ptm: torch.Tensor           # [K]
    ranking: torch.Tensor       # [K]
    best_idx: int
```

`load_of3_diffusion()` değişiklikleri:
1. `no_rollout_samples` parametresi ekle (default=5)
2. Diffusion sonrası `aux_heads` çalıştır
3. `get_confidence_scores` ile pLDDT/pTM hesapla
4. Per-sample ranking score hesapla
5. En iyi sample'ı seç, tüm metrikleri dön

```python
def load_of3_diffusion(
    query_json, device="cuda",
    num_rollout_steps=None,
    num_samples=5,           # YENİ: multi-sample
    use_msa_server=False,
    use_templates=False,
) -> tuple[Callable, Tensor]:
```

### 4.3 aux_heads Çağrısı

```python
# diffusion_fn içinde:
atom_positions = model.sample_diffusion(
    batch=cached_batch,
    si_input=si_input_cached,
    si_trunk=si_trunk_cached,
    zij_trunk=zij_modified,      # z_mod kullan
    noise_schedule=noise_schedule,
    no_rollout_samples=K,
    use_conditioning=True,
)
# atom_positions: [1, K, N_atom, 3]

output = {
    "si_trunk": si_trunk_cached,
    "zij_trunk": zij_modified,
    "atom_positions_predicted": atom_positions,
}

with torch.no_grad(), torch.amp.autocast("cuda"):
    aux_output = model.aux_heads(
        batch=cached_batch,
        si_input=si_input_cached,
        output=output,
        use_zij_trunk_embedding=True,
    )
    output.update(aux_output)

confidence = get_confidence_scores(cached_batch, output, config)
# confidence: {pLDDT: [K, N], pTM: [K], ...}
```

## 5. Adaptive Fallback Stratejisi

### 5.1 Fallback Kademesi

Her step sonrası confidence check:

```
Confidence OK (ranking_score >= cutoff)?
  ├── EVET → kabul et, sonraki step'e geç
  └── HAYIR → Fallback Level 1: df'yi düşür
        ├── Confidence OK? → kabul et
        └── HAYIR → Fallback Level 2: mode subset'i küçült
              ├── Confidence OK? → kabul et
              └── HAYIR → Fallback Level 3: z_mixing alpha'yı düşür
                    ├── Confidence OK? → kabul et
                    └── HAYIR → en iyi denemeyi kabul et (forced accept)
```

### 5.2 Fallback Parametreleri

```python
# ModeDriveConfig'e eklenecek yeni alanlar:

# Confidence cutoffs
confidence_ptm_cutoff: float = 0.5       # pTM minimum
confidence_plddt_cutoff: float = 0.6     # mean pLDDT minimum
confidence_ranking_cutoff: float = 0.5   # ranking score minimum

# Multi-sample
num_diffusion_samples: int = 5           # K: samples per step

# Fallback parameters
enable_confidence_fallback: bool = True
fallback_df_factor: float = 0.5          # df *= 0.5 at level 1
fallback_max_combo_size: int = 1         # reduce to single mode at level 2
fallback_alpha_factor: float = 0.5       # alpha *= 0.5 at level 3
max_fallback_retries: int = 3            # max retries per step
```

### 5.3 Fallback Algoritması

```python
def _step_with_fallback(self, step_idx, coords, zij_trunk, ...):
    """Single step with confidence-guided fallback."""

    # Original parametreler
    orig_df = self.config.df
    orig_max_combo = self.config.max_combo_size
    orig_alpha = self.config.z_mixing_alpha

    best_result = None
    best_ranking = -1.0

    for level in range(self.config.max_fallback_retries + 1):
        if level == 1:
            # Fallback L1: df düşür
            current_df = orig_df * self.config.fallback_df_factor
        elif level == 2:
            # Fallback L2: mode subset küçült
            current_max_combo = self.config.fallback_max_combo_size
        elif level == 3:
            # Fallback L3: z_mixing alpha düşür
            current_alpha = orig_alpha * self.config.fallback_alpha_factor

        # Pipeline step çalıştır
        result = self._run_single_attempt(
            coords, zij_trunk,
            df=current_df,
            max_combo_size=current_max_combo,
            alpha=current_alpha,
        )

        # Confidence check
        if result.ranking_score >= self.config.confidence_ranking_cutoff:
            return result  # Kabul

        if result.ranking_score > best_ranking:
            best_result = result
            best_ranking = result.ranking_score

    # Forced accept: en iyi deneme
    return best_result
```

### 5.4 Fallback Seviyeleri Detay

| Level | Aksiyon | Neden | Etki |
|-------|---------|-------|------|
| 0 | Normal çalış | - | Tam deplasman |
| 1 | df *= 0.5 | Çok büyük deplasman → clash | Daha küçük hareket |
| 2 | max_combo_size = 1 | Çok fazla mod karışımı → gürültü | Tek mod, temiz hareket |
| 3 | alpha *= 0.5 | z_pseudo çok dominant → trunk'tan uzak | Trunk'a yakın kal |
| forced | En iyi denemeyi al | Hiçbiri geçemedi | Minimum hasar |

## 6. StepResult Güncellemesi

```python
@dataclass
class StepResult:
    combo: ModeCombo
    displaced_ca: torch.Tensor       # [N, 3]
    new_ca: torch.Tensor             # [N, 3]  (best sample)
    z_modified: torch.Tensor         # [N, N, 128]
    contact_map: torch.Tensor        # [N, N]
    rmsd: float
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    b_factors: torch.Tensor
    df_used: float = 0.0

    # YENİ: Confidence metrikleri
    plddt: torch.Tensor | None = None       # [N] best sample pLDDT
    ptm: float | None = None                # best sample pTM
    ranking_score: float | None = None      # best sample ranking
    all_ptm: torch.Tensor | None = None     # [K] tüm samples
    all_ranking: torch.Tensor | None = None # [K] tüm samples
    fallback_level: int = 0                 # 0=normal, 1-3=fallback
    num_samples: int = 1                    # K
```

## 7. Dosya Değişiklikleri

### 7.1 `src/of3_diffusion.py`

| Değişiklik | Detay |
|-----------|-------|
| `DiffusionResult` dataclass | Yeni return type |
| `load_of3_diffusion()` | `num_samples` param, aux_heads çağrısı |
| `diffusion_fn()` | Multi-sample, confidence hesaplama, ranking |
| `_extract_ca()` | [K, N_atom, 3] → [K, N, 3] multi-sample desteği |

### 7.2 `src/mode_drive.py`

| Değişiklik | Detay |
|-----------|-------|
| `ModeDriveConfig` | 8 yeni confidence/fallback alanı |
| `StepResult` | 7 yeni confidence alanı |
| `_step_with_fallback()` | Yeni method: fallback mantığı |
| `step()` | Confidence check ve fallback entegrasyonu |
| `run()` | Fallback istatistikleri loglama |

### 7.3 `notebooks/test_mode_drive.ipynb`

| Değişiklik | Detay |
|-----------|-------|
| Config cell | Confidence cutoff'lar ve num_samples ekleme |
| `load_of3_diffusion()` | `num_samples=5` parametresi |
| Visualization | pTM/pLDDT trajectory plot'ları |

## 8. Notebook Config Değişiklikleri

```python
# ══════════════════════════════════════════════════════════════
# Config — her deneme için değiştir, setup tekrar çalışmasın
# ══════════════════════════════════════════════════════════════

# ... mevcut ANM/pipeline parametreleri ...

# ── Confidence & Multi-Sample ──
NUM_DIFFUSION_SAMPLES = 5                # K: diffusion sample sayısı
CONFIDENCE_PTM_CUTOFF = 0.5              # pTM minimum eşik
CONFIDENCE_PLDDT_CUTOFF = 0.6            # mean pLDDT minimum eşik
CONFIDENCE_RANKING_CUTOFF = 0.5          # ranking score minimum eşik

# ── Adaptive Fallback ──
ENABLE_FALLBACK = True                   # confidence-guided fallback
FALLBACK_DF_FACTOR = 0.5                 # Level 1: df'yi bu kadar küçült
FALLBACK_MAX_COMBO_SIZE = 1              # Level 2: tek mod'a düşür
FALLBACK_ALPHA_FACTOR = 0.5              # Level 3: alpha'yı bu kadar küçült
MAX_FALLBACK_RETRIES = 3                 # step başına max retry
```

## 9. Yeni Visualization Cell'leri

### 9.1 Confidence Trajectory

```python
# Plot: pTM ve mean pLDDT step bazında
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

steps = range(1, len(result.step_results) + 1)
ptms = [r.ptm for r in result.step_results]
plddts = [r.plddt.mean().item() for r in result.step_results]
fallbacks = [r.fallback_level for r in result.step_results]

ax1.plot(steps, ptms, 'o-', label='pTM')
ax1.axhline(CONFIDENCE_PTM_CUTOFF, ls='--', c='r', label='cutoff')
ax1.set_ylabel('pTM'); ax1.legend()

ax2.plot(steps, plddts, 's-', label='mean pLDDT')
ax2.axhline(CONFIDENCE_PLDDT_CUTOFF, ls='--', c='r', label='cutoff')
ax2.set_ylabel('mean pLDDT'); ax2.legend()

# Fallback level'ları renkle göster
for ax in [ax1, ax2]:
    for s, fl in zip(steps, fallbacks):
        if fl > 0:
            ax.axvspan(s-0.4, s+0.4, alpha=0.2, color='orange')
```

### 9.2 Sample Distribution

```python
# Plot: her step'teki K sample'ın ranking dağılımı
fig, ax = plt.subplots(figsize=(12, 4))
for i, r in enumerate(result.step_results):
    if r.all_ranking is not None:
        x = [i+1] * len(r.all_ranking)
        ax.scatter(x, r.all_ranking.numpy(), alpha=0.5, s=30)
        ax.scatter(i+1, r.ranking_score, c='red', s=80, zorder=5)  # best
ax.axhline(CONFIDENCE_RANKING_CUTOFF, ls='--', c='gray')
ax.set_xlabel('Step'); ax.set_ylabel('Ranking Score')
```

## 10. İmplementasyon Sırası

### Phase 1: Multi-Sample Diffusion (of3_diffusion.py)
- [ ] `DiffusionResult` dataclass tanımla
- [ ] `load_of3_diffusion()` → `num_samples` parametresi ekle
- [ ] `diffusion_fn()` → K sample üret
- [ ] `aux_heads` çağrısı ile confidence hesapla
- [ ] `get_confidence_scores` entegre et
- [ ] Ranking score hesapla ve best sample seç
- [ ] `_extract_ca()` → multi-sample desteği [K, N_atom, 3]

### Phase 2: Adaptive Fallback (mode_drive.py)
- [ ] `ModeDriveConfig` → 8 yeni alan ekle
- [ ] `StepResult` → confidence alanları ekle
- [ ] `_step_with_fallback()` methodu yaz
- [ ] `step()` → fallback entegrasyonu
- [ ] `run()` → fallback istatistikleri logla
- [ ] Backward compat: `diffusion_fn` eski format (sadece coords) döndürürse de çalışsın

### Phase 3: Notebook Güncellemesi
- [ ] Config cell → confidence/fallback parametreleri
- [ ] `load_of3_diffusion(num_samples=5)` çağrısı
- [ ] Confidence trajectory visualization
- [ ] Sample distribution visualization
- [ ] Summary dashboard'a confidence ekle

## 11. Edge Case'ler

### 11.1 GPU Memory
K=5 sample → ~5x diffusion memory. Eğer OOM:
- `num_samples=3` veya `num_samples=1` ile fallback
- aux_heads'ı ayrı ayrı per-sample çalıştır (batch loop)

### 11.2 Eski diffusion_fn Uyumluluğu
Pipeline, eski `diffusion_fn` (sadece coords dönen) ile de çalışmalı:
```python
result = diffusion_fn(z_mod)
if isinstance(result, DiffusionResult):
    ca = result.best_ca
    ptm = result.ptm
else:
    ca = result  # eski format
    ptm = None   # confidence yok → fallback devre dışı
```

### 11.3 Confidence Yok = Fallback Kapalı
`enable_confidence_fallback=True` ama `diffusion_fn` confidence dönmüyorsa → fallback otomatik devre dışı, uyarı logla.

## 12. Beklenen Etki

| Metrik | Şu An | Beklenen |
|--------|-------|----------|
| Yapı kalitesi | Bilinmiyor (tek sample, kontrol yok) | pTM > 0.5 garanti |
| Sample varyansı | Tek sample, şansla iyi/kötü | 5 sample'dan en iyi |
| Fallback oranı | - | Step'lerin ~%20-30'unda beklenir |
| Hesaplama maliyeti | 1x diffusion/step | ~5x diffusion + aux_heads |
| Toplam süre | ~30s/step | ~2-3 dk/step (GPU'ya bağlı) |

## 13. Gelecek İyileştirmeler

1. **PAE-guided mode selection**: PAE matrisinden hangi residue çiftlerinin belirsiz olduğunu öğren → o bölgeye odaklan
2. **Confidence-weighted alpha**: düşük pLDDT bölgelerinde alpha azalt, yüksek bölgelerde koru
3. **Adaptive K**: İlk step'te K=5, sonraki step'lerde K=3 (confidence stabil ise)
4. **Early stopping**: N ardışık step pTM > 0.7 ise dur
