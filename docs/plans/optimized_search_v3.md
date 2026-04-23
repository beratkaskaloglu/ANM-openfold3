# Optimized Search V3 - Implementation Plan

## Problem Analizi

Grid Search V2'den ogrendiklerimiz (N=214, 1AKE->4AKE):

### Temel Sorunlar
1. **pTM=0.6 cutoff N=214 icin ulasilamaz** — tipik L4 pTM 0.30-0.59 arasi
2. **Alpha death spiral** — L9 FORCE -> alpha=0.05 -> RMSD_init 2-5A -> rastgele osilasyon
3. **30 step'in cogu bos geciyor** — 2-6/30 accept, geri kalan GPU bossa calisiyor
4. **Tek metrik (pTM) gate cok kati** — baska iyi metrikler (mean_pae, ranking, Rg) var ama kullanilmiyor

### V2 Sonuclari Ozet
| Parametre | En iyi deger | Etki |
|-----------|-------------|------|
| warmup | 0 | warmup>=5 felaket |
| rg_max | 2.5 | minimal etki |
| stall | 8 | GPU tasarrufu |
| alpha_decay | 0.8 | **en buyuk etki** — 6/30 accept |

---

## V3 Stratejisi: Composite Confidence Score

### Fikir
Tek tek metrik cutoff'lari yerine **agirlikli composite score** kullan.
Her metrik 0-1 arasi normalize edilir, sonra agirlikli toplam alinir.
Tek bir threshold'la accept/reject karari verilir.

### Avantajlar
- pTM tek basina zayif ama diger metriklerle kombine guclenir
- Bir metrik dusuk olsa bile toplam yeterli olabilir (OR-like behavior)
- Threshold ayarlamak daha kolay (tek sayi)

### Composite Score Formulasyonu

```
composite = w_ptm * norm_ptm
          + w_plddt * norm_plddt
          + w_pae * norm_pae
          + w_rg * norm_rg
          + w_contact * norm_contact_recon

where:
  norm_ptm    = clamp(ptm / 0.8, 0, 1)           # 0.8 = ideal pTM
  norm_plddt  = clamp((plddt - 50) / 40, 0, 1)   # 50-90 range -> 0-1
  norm_pae    = clamp(1 - pae/25, 0, 1)           # 0=25A(kotu), 1=0A(iyi)
  norm_rg     = 1 - clamp(|rg_ratio - 1| / 1.5, 0, 1)  # 1.0=ideal, uzaklastikca duser
  norm_contact= clamp((cr + 0.1) / 0.8, 0, 1)    # -0.1 to 0.7 range
```

### Agirlik Seti Aday Gridi

| Set | w_ptm | w_plddt | w_pae | w_rg | w_contact | threshold |
|-----|-------|---------|-------|------|-----------|-----------|
| A: pTM-heavy | 0.40 | 0.20 | 0.25 | 0.10 | 0.05 | 0.50 |
| B: PAE-heavy | 0.20 | 0.15 | 0.40 | 0.15 | 0.10 | 0.45 |
| C: balanced | 0.25 | 0.20 | 0.25 | 0.15 | 0.15 | 0.45 |
| D: physical | 0.15 | 0.15 | 0.30 | 0.25 | 0.15 | 0.40 |

---

## Notebook Yapisi

### Cell 1-3: Setup + OF3 (mevcut, degismiyor)
Ayni Environment/Config/OF3 loading hucreleri.

### Cell 4: Composite Score fonksiyonu (YENi)
- `compute_composite_score(step_result, weights)` — StepResult'tan composite score hesapla
- `CompositeWeights` dataclass — agirlik seti
- Normalizasyon fonksiyonlari

### Cell 5: Optimized Pipeline Wrapper (YENi)
Mevcut `ModeDrivePipeline.run()` yerine ozel bir runner:

```python
def run_optimized(
    initial_ca, zij_trunk, target_ca,
    weights: CompositeWeights,
    composite_threshold: float,
    alpha_decay: float = 0.8,
    max_stall: int = 8,
    n_steps: int = 20,         # 30 yerine 20 — stall ile otomatik duracak
    alpha_init: float = 0.7,
) -> dict:
```

**Kilit farklar:**
1. `_confidence_check` override edilmez — composite score dogrudan kullanilir
2. Pipeline'in kendi `run()` metodu yerine custom loop
3. Her step'te composite score hesaplanir ve loglanir
4. Accept/reject karari composite score'a gore verilir

### Cell 6: Agirlik Gridi Taramasi
4 agirlik seti x 3 threshold = 12 run (vs V2'nin 35+ run'i)

### Cell 7: Alpha Schedule Karsilastirmasi
En iyi agirlik seti ile:
- Fixed alpha (no decay)
- Linear decay (0.7 -> 0.1 over rejected steps)
- Geometric decay (0.8x per reject)
- Cosine annealing

### Cell 8: Final Combined Run
En iyi kombinasyonla 30 step kosu.

### Cell 9: Sonuclari Kaydet + Grafikler

---

## Hiz Optimizasyonlari

### Neden V2 yavas?
1. Her run 30 step x ~30s OF3 call = **~15 dakika**
2. 35+ konfigürasyon = **~9 saat**
3. Cogu step rejected — bos OF3 call

### V3 Hizlandirma
1. **n_steps=20** (30'dan dusur) — son 10 step zaten stall
2. **max_stall=8** — default acik, gereksiz step'leri kes
3. **alpha_decay=0.8** — default acik, accept oranini artir
4. **12 run** (35+ yerine) — composite score grid daha kucuk
5. **Erken cikis** — eger ilk 5 step'te 0 accept ise skip (composite < 0.2)

### Tahmini sure
- 12 run x 15 dakika = ~3 saat (V2'nin 9 saatinden 3x daha hizli)
- Stall + erken cikis ile pratikte ~2 saat

---

## Uygulama Adimi (step-by-step)

### Adim 1: Composite score modulu
`src/composite_confidence.py` — kucuk, bagimsiz modül:
- `CompositeWeights` dataclass
- `normalize_metrics(step_result) -> dict`
- `compute_composite(normalized, weights) -> float`

### Adim 2: Notebook
`notebooks/optimized_search_v3.ipynb`:
- Setup (mevcut kopyala)
- Composite score + custom runner
- Grid taramasi
- Analiz + grafikler

### Adim 3: Sonuc karsilastirmasi
V2 baseline ile V3 composite'in karsilastirma tablosu:
- Accept orani
- Final RMSD
- Final TM-score
- Wall time

---

## Riskler ve Azaltma

| Risk | Azaltma |
|------|---------|
| Composite score cok gevsek -> kotu yapilar accept | threshold'u 0.5'ten basla, asagi dusur |
| Composite score cok siki -> V2 gibi hep reject | agirlik setleri arasi karsilastirma |
| OF3 call suresi degisken | timing logla, medyan raporla |
| N=214'e ozgu sonuclar | Farkli protein boyutlariyla test (gelecek) |

---

## Basari Kriterleri
- Accept orani > 20% (6+/30 step) vs V2 baseline 7% (2/30)
- TM-score vs target > 0.60 (V2 baseline ~0.65-0.73)
- RMSD < 3.5A (V2 baseline ~3-4A)
- Wall time < 3 saat (V2 ~9 saat)
