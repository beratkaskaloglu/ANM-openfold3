# Confidence V2: Genişletilmiş Metrik Entegrasyonu

*Tarih: 2026-04-19*
*Durum: PLAN — henüz implementasyon yok*

## Motivasyon

Pipeline çıktısında gözlemlenen sorunlar:
1. pLDDT ayırt edici değil (88-91 arası, RMSD=27Å'da bile 89)
2. Step 3+ hep L9 forced-accept'e düşüyor → cascade failure
3. L9 en yüksek ranking'i seçiyor ama fiziksel olarak anlamsız yapıları kabul edebiliyor (Rg patlamış)
4. rejected=True sonrası aynı MD parametreleriyle tekrar çalışınca aynı sonuç geliyor (stall loop)

## Hedef

OF3'ün zaten ürettiği ama kullanmadığımız metrikleri pipeline'a entegre etmek. **Minimal kod değişikliği**, mevcut mimariyle uyumlu.

---

## Faz 1: DiffusionResult Genişletme (of3_diffusion.py)

### 1.1 Yeni alanlar

```python
@dataclass
class DiffusionResult:
    # Mevcut
    all_ca: torch.Tensor          # [K, N, 3]
    best_ca: torch.Tensor         # [N, 3]
    best_idx: int
    plddt: torch.Tensor | None    # [K, N]
    ptm: torch.Tensor | None      # [K]
    ranking: torch.Tensor | None  # [K]

    # YENİ — Faz 1
    pae: torch.Tensor | None = None           # [K, N, N] veya [N, N] (sample 0)
    contact_probs: torch.Tensor | None = None  # [N, N] predicted contact probs
    has_clash: bool | None = None              # OF3 clash detection
    mean_pae: float | None = None              # mean PAE (best sample)

    # YENİ — Faz 2
    sample_rmsd: torch.Tensor | None = None    # [K*(K-1)/2] pairwise inter-sample RMSD
    sample_rmsf: torch.Tensor | None = None    # [N] per-residue RMSF across K samples
    consensus_score: float | None = None       # 1/(1 + mean_inter_sample_rmsd)
```

### 1.2 _compute_confidence() değişiklikleri

```python
def _compute_confidence(atom_positions, zij_modified) -> dict | None:
    # ... mevcut aux_heads + get_confidence_scores ...

    # YENİ: PAE çıkar
    pae = confidence.get("pae")          # [1, K, N, N] veya [1, N, N]
    contact_probs = confidence.get("contact_probs")  # [1, N, N]
    has_clash = confidence.get("has_clash")

    return confidence  # zaten dict — ek anahtarlar otomatik gelir
```

### 1.3 diffusion_fn() değişiklikleri

```python
# Mevcut confidence dict'ten yeni alanları çıkar
pae_raw = confidence.get("pae")
contact_probs_raw = confidence.get("contact_probs")
has_clash_raw = confidence.get("has_clash")

# PAE: squeeze batch dim, mean hesapla
if pae_raw is not None:
    pae = pae_raw.squeeze(0)  # [K, N, N] veya [N, N]
    mean_pae = float(pae[best_idx].mean().item()) if pae.dim() == 3 else float(pae.mean().item())
else:
    pae = None
    mean_pae = None

# Contact probs
if contact_probs_raw is not None:
    contact_probs = contact_probs_raw.squeeze(0)  # [N, N]
else:
    contact_probs = None

# DiffusionResult'a ekle
return DiffusionResult(
    ...,
    pae=pae,
    contact_probs=contact_probs,
    has_clash=bool(has_clash_raw) if has_clash_raw is not None else None,
    mean_pae=mean_pae,
)
```

**Dosya:** `src/of3_diffusion.py`
**Tahmini değişiklik:** ~30 satır ekleme

---

## Faz 2: Sample Tutarlılığı (K>1 durumunda)

### 2.1 Inter-sample RMSD hesaplama

```python
# diffusion_fn() içinde, K>1 ise:
if K > 1:
    from .mode_drive_utils import compute_rmsd as _rmsd

    pairwise = []
    rmsf_coords = all_ca.float()  # [K, N, 3]

    for i in range(K):
        for j in range(i+1, K):
            pairwise.append(_rmsd(all_ca[i], all_ca[j]))

    sample_rmsd = torch.tensor(pairwise)
    mean_inter = sample_rmsd.mean().item()
    consensus_score = 1.0 / (1.0 + mean_inter)

    # Per-residue RMSF across samples
    mean_pos = rmsf_coords.mean(dim=0)  # [N, 3]
    deviations = rmsf_coords - mean_pos.unsqueeze(0)  # [K, N, 3]
    msf = (deviations ** 2).sum(dim=-1).mean(dim=0)  # [N]
    sample_rmsf = msf.sqrt()
```

**Dosya:** `src/of3_diffusion.py`
**Tahmini değişiklik:** ~20 satır ekleme
**Not:** K=1 (default) durumunda bu metrikler None kalır, overhead yok.

---

## Faz 3: StepResult ve _confidence_ok() Genişletme (mode_drive.py + mode_drive_config.py)

### 3.1 StepResult yeni alanlar

```python
@dataclass
class StepResult:
    # ... mevcut ...

    # YENİ confidence alanları
    mean_pae: float | None = None           # ortalama PAE (düşük = iyi)
    has_clash: bool | None = None           # clash var mı
    consensus_score: float | None = None    # inter-sample uyum (yüksek = iyi)
    contact_consistency: float | None = None  # ANM vs OF3 contact uyumu
    rg_ratio: float | None = None           # Rg_obs / Rg_expected
```

### 3.2 _downstream_from_displaced() değişiklikleri

```python
# Mevcut diff_result çıkışına ek:
mean_pae_out = None
has_clash_out = None
consensus_out = None
contact_consistency_out = None
rg_ratio_out = None

if hasattr(diff_result, "mean_pae"):
    mean_pae_out = diff_result.mean_pae
    has_clash_out = diff_result.has_clash
    consensus_out = diff_result.consensus_score

# Contact consistency: ANM-displaced vs OF3-predicted
if diff_result.contact_probs is not None:
    input_contact = contact  # coords_to_contact'tan geldi
    pred_contact = diff_result.contact_probs
    # Overlap: ne kadar örtüşüyor
    mask = input_contact > 0.5
    if mask.any():
        contact_consistency_out = float(pred_contact[mask].mean().item())

# Rg ratio
N = new_ca.shape[0]
rg_obs = float(((new_ca - new_ca.mean(0))**2).sum(1).mean().sqrt().item())
rg_exp = 2.2 * (N ** 0.38)
rg_ratio_out = rg_obs / rg_exp
```

### 3.3 Yeni config alanları

```python
# mode_drive_config.py — Confidence V2
confidence_mean_pae_cutoff: float | None = None       # max mean PAE (None=disabled)
confidence_consensus_cutoff: float | None = None      # min consensus score (None=disabled)
confidence_rg_max: float = 2.5                        # max Rg ratio (>2.5 = yapı patlamış)
confidence_rg_min: float = 0.3                        # min Rg ratio (<0.3 = aşırı sıkışmış)
confidence_clash_reject: bool = True                  # has_clash=True → reject
confidence_contact_consistency_cutoff: float | None = None  # min contact overlap

# Warmup: ilk N step'te cutoff'ları gevşet
confidence_warmup_steps: int = 0                      # 0=disabled
confidence_warmup_ptm_cutoff: float = 0.35            # warmup sırasında pTM cutoff
confidence_warmup_ranking_cutoff: float = 0.40        # warmup sırasında ranking cutoff

# Stall prevention
max_consecutive_rejected: int = 0                     # 0=unlimited, >0 = N ardışık rejected sonra dur
```

### 3.4 _confidence_ok() genişletme

```python
def _confidence_ok(self, result: StepResult, step_idx: int = 0) -> bool:
    cfg = self.config

    # Mevcut None check...

    # Step-adaptive cutoff (warmup)
    in_warmup = cfg.confidence_warmup_steps > 0 and step_idx < cfg.confidence_warmup_steps
    ptm_cut = cfg.confidence_warmup_ptm_cutoff if in_warmup else cfg.confidence_ptm_cutoff
    rank_cut = cfg.confidence_warmup_ranking_cutoff if in_warmup else cfg.confidence_ranking_cutoff

    # Fiziksel filtreler (her zaman aktif)
    if result.rg_ratio is not None:
        if result.rg_ratio > cfg.confidence_rg_max or result.rg_ratio < cfg.confidence_rg_min:
            return False

    if cfg.confidence_clash_reject and result.has_clash:
        return False

    # Mevcut pTM/pLDDT/ranking kontrolü (warmup-adjusted cutoff'larla)
    if result.ptm is not None and result.ptm < ptm_cut:
        return False
    if result.plddt is not None:
        mean_plddt = result.plddt.mean().item()
        if mean_plddt < cfg.confidence_plddt_cutoff:
            return False
    if result.ranking_score is not None and result.ranking_score < rank_cut:
        return False

    # V2 metrikler (None=disabled)
    if cfg.confidence_mean_pae_cutoff is not None and result.mean_pae is not None:
        if result.mean_pae > cfg.confidence_mean_pae_cutoff:
            return False
    if cfg.confidence_consensus_cutoff is not None and result.consensus_score is not None:
        if result.consensus_score < cfg.confidence_consensus_cutoff:
            return False
    if cfg.confidence_contact_consistency_cutoff is not None and result.contact_consistency is not None:
        if result.contact_consistency < cfg.confidence_contact_consistency_cutoff:
            return False

    return True
```

**Dosya:** `src/mode_drive.py`, `src/mode_drive_config.py`
**Tahmini değişiklik:** ~60 satır

---

## Faz 4: L9 İyileştirme ve Stall Prevention

### 4.1 L9'da Rg filtresi

```python
# _track() içinde — L9 candidate'i olarak sadece fiziksel olarak anlamlı yapıları kabul et
def _track(result, *, level=0, desc=""):
    nonlocal best_result, best_ranking

    # Rg filtresi: patlamış yapıları L9 havuzundan bile çıkar
    if result.rg_ratio is not None and result.rg_ratio > cfg.confidence_rg_max:
        # L9 candidate'i olarak ekleme — bu yapı fiziksel olarak anlamsız
        if cfg.autostop_verbose:
            print(f"      [FB L{level}] SKIP  Rg={result.rg_ratio:.1f} > {cfg.confidence_rg_max}")
        return False

    r_score = result.ranking_score if result.ranking_score is not None else 0.0
    if r_score > best_ranking:
        best_ranking = r_score
        best_result = result
    return self._confidence_ok(result, step_idx=step_idx)
```

### 4.2 Max consecutive rejected

```python
# run() içinde:
consecutive_rejected = 0

for step_idx in range(cfg.n_steps):
    step_result = ...

    if step_result.rejected:
        consecutive_rejected += 1
        if cfg.max_consecutive_rejected > 0 and consecutive_rejected >= cfg.max_consecutive_rejected:
            if verbose:
                print(f"  STOP: {consecutive_rejected} consecutive rejected steps — pipeline stalled")
            break
    else:
        consecutive_rejected = 0
    ...
```

### 4.3 Rejected sonrası parametre mutasyonu

```python
# run() içinde, rejected sonrası:
if step_result.rejected:
    consecutive_rejected += 1
    # Sonraki step için alpha'yı hafifçe azalt (stall'dan çıkış)
    cfg.z_mixing_alpha = max(0.05, cfg.z_mixing_alpha * 0.85)
else:
    consecutive_rejected = 0
    # Başarılı step — alpha'yı geri getir
    cfg.z_mixing_alpha = orig_alpha
```

**Dosya:** `src/mode_drive.py`
**Tahmini değişiklik:** ~25 satır

---

## Faz 5: Verbose Çıktı Zenginleştirme

### 5.1 Yeni sütunlar

```
  Step  RMSD_init     df     a    Combo   RMSD_tgt  TM_tgt   pTM  pLDDT  mPAE   Rg  Cons   FB
```

Yeni sütunlar:
- `mPAE` — mean PAE (düşük = iyi)
- `Rg` — Rg ratio (1.0 civarı = normal)
- `Cons` — consensus score (yüksek = iyi, sadece K>1)

### 5.2 Fallback verbose'da yeni metrikler

```
[FB L4] FAIL  a x0.25->0.25  pk=420 tk=44  pTM=0.342  pLDDT=85.9  rank=0.446
              mPAE=12.3  Rg=1.8  contact_cons=0.62  RMSD_init=3.06A
```

---

## Grid Search Planı

İlk etapta şu kombinasyonları test etmeyi planlıyoruz:

### Deney 1: Warmup etkisi
```python
configs = [
    {"confidence_warmup_steps": 0},   # baseline
    {"confidence_warmup_steps": 3, "confidence_warmup_ptm_cutoff": 0.35},
    {"confidence_warmup_steps": 5, "confidence_warmup_ptm_cutoff": 0.30},
    {"confidence_warmup_steps": 10, "confidence_warmup_ptm_cutoff": 0.25},
]
```

### Deney 2: Rg filtresi
```python
configs = [
    {"confidence_rg_max": 999},   # disabled (baseline)
    {"confidence_rg_max": 3.0},
    {"confidence_rg_max": 2.5},
    {"confidence_rg_max": 2.0},
]
```

### Deney 3: Max consecutive rejected
```python
configs = [
    {"max_consecutive_rejected": 0},   # unlimited (baseline)
    {"max_consecutive_rejected": 3},
    {"max_consecutive_rejected": 5},
]
```

### Deney 4: Alpha decay on rejection
```python
# Rejected sonrası alpha *= 0.85 vs sabit
```

### Deney 5: PAE cutoff (Faz 1 tamamlandıktan sonra)
```python
configs = [
    {"confidence_mean_pae_cutoff": None},  # disabled
    {"confidence_mean_pae_cutoff": 15.0},
    {"confidence_mean_pae_cutoff": 10.0},
    {"confidence_mean_pae_cutoff": 8.0},
]
```

### Deney 6: K=3 sample tutarlılığı (Faz 2 tamamlandıktan sonra)
```python
configs = [
    {"num_diffusion_samples": 1},   # baseline
    {"num_diffusion_samples": 3, "confidence_consensus_cutoff": 0.3},
    {"num_diffusion_samples": 3, "confidence_consensus_cutoff": 0.5},
]
```

---

## Değişecek Dosyalar (Özet)

| Dosya | Faz | Değişiklik |
|-------|-----|-----------|
| `src/of3_diffusion.py` | 1, 2 | DiffusionResult genişletme, PAE/contact_probs/consensus çıkarma |
| `src/mode_drive_config.py` | 3, 4 | Yeni config alanları (V2 cutoff'lar, warmup, stall prevention) |
| `src/mode_drive.py` | 3, 4, 5 | StepResult genişletme, _confidence_ok() V2, Rg filtresi, stall prevention |
| `tests/test_mode_drive_coverage.py` | 3 | Yeni confidence metriklerine testler |
| `tests/test_of3_diffusion.py` | 1, 2 | DiffusionResult yeni alanlarına testler (mock) |

## Uygulama Sırası

1. **Faz 3+4 önce** — Rg filtresi, warmup, stall prevention (OF3 olmadan test edilebilir, hemen etkili)
2. **Faz 1** — PAE/contact_probs çıkarma (OF3 gerekli, Colab'da test)
3. **Faz 2** — Sample tutarlılığı (K>1 gerekli, compute cost artar)
4. **Faz 5** — Verbose zenginleştirme (Faz 1-4 tamamlandıktan sonra)
5. **Grid search** — Her faz sonrası ilgili deneyleri çalıştır

---

## Referanslar

- [[research/confidence_metrics_analysis]] — Detaylı metrik analizi ve literatür
- [[research/alpha_scheduling_strategies]] — Alpha zamanlama yaklaşımları
- [[architecture/13-confidence-guided-pipeline]] — Mevcut confidence pipeline mimarisi
- AF3 summary_confidences JSON yapısı — pae, contact_probs, has_clash, ranking_score
- Thacker (2026) — pLDDT FalseVerify oranı
- Oda et al. (2024) — İteratif refinement'ta yanıltıcı confidence
