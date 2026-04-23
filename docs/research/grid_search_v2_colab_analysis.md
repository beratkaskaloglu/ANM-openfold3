# Grid Search V2 - Colab Sonuclari Analizi (N=214)

## Genel Bilgiler
- **Protein boyutu**: N=214 residue
- **n_steps**: 30
- **Strateji**: autostop
- **Fallback levels**: (0, 1, 4, 9) — L0 baseline, L1 back-off fractions, L4 alpha decay, L9 force-accept
- **Baseline pTM cutoff**: 0.6

---

## Experiment 1: Warmup

| Config | warmup_steps | ptm_cutoff | Accept | Steps | RMSD (A) | TM | Notlar |
|--------|-------------|------------|--------|-------|----------|-----|--------|
| exp1[0] | 0 | - | 2/30 | 30 | 3.08 | 0.675 | Baseline, guvenli |
| exp1[1] | 3 | 0.35 | 3/30 | 30 | 4.33 | 0.721 | Hafif iyilesme ama step 10&29'da sigramalar |
| exp1[2] | 5 | 0.30 | 5/30 | 30 | **38.83** | 0.036 | FELAKET: Rg patlamasi |
| exp1[3] | 8 | 0.25 | 4/30 | 30 | **80.97** | 0.006 | Tamamen basarisiz |
| exp1[4] | 10 | 0.20 | 5/30 | 30 | **79.20** | 0.009 | Tamamen basarisiz |

### Analiz
- **warmup=0** veya **warmup=3** guvenli secenekler
- **warmup >= 5** felaket: Dusuk warmup cutoff kotu yapilari kabul ediyor (Rg>2.5), sonra tum denemeler SKIP yiyor, L9 FORCE'a dusuyor ve baseline alpha=0.70 ile yapi surekli sapiyor
- **Sonuc**: `warmup_steps=0` (degistirmemeye gerek yok)

---

## Experiment 2: Rg Filter

| Config | rg_max | Accept | Steps | RMSD (A) | TM | Notlar |
|--------|--------|--------|-------|----------|-----|--------|
| exp2[0] | 999 (off) | 2/30 | 30 | 3.13 | 0.656 | Rg filtresi yok |
| exp2[1] | 3.0 | 2/30 | 30 | 3.93 | 0.718 | |
| exp2[2] | 2.5 | 3/30 | 30 | **2.79** | 0.648 | En iyi exp2 |
| exp2[3] | 2.0 | 2/30 | 30 | 4.17 | 0.719 | Daha fazla SKIP |
| exp2[4] | 1.8 | 2/30 | 30 | 4.17 | 0.738 | |

### Analiz
- Rg filtresi **minimal etki** yapiyor cunku L4 yapilari zaten kompakt (Rg ~1.0-1.3)
- L0/L1 seviyelerinde SKIP artiyor ama final sonucu cok degistirmiyor
- **Sonuc**: `rg_max=2.5` (mevcut default, uygun)

---

## Experiment 3: Stall Detection (max_consecutive_rejected)

| Config | stall | Accept | Steps | RMSD (A) | TM | Notlar |
|--------|-------|--------|-------|----------|-----|--------|
| exp3[0] | 0 (off) | 2/30 | 30 | 4.70 | 0.737 | Hic stall detection yok |
| exp3[1] | 3 | 2/5 | 5 | 5.04 | 0.760 | Cok erken duruyor |
| exp3[2] | 5 | 2/7 | 7 | 2.68 | 0.658 | |
| exp3[3] | 8 | 3/11 | 11 | **2.48** | 0.600 | Iyi denge |
| exp3[4] | 10 | 2/12 | 12 | 3.28 | 0.646 | |

### Analiz
- **stall=3** cok agresif, pipeline 5 adimda bitiyor
- **stall=8** en iyi denge: 11 adimda 3 kabul, RMSD=2.48A
- stall=0 (30 adim) gereksiz GPU hesaplama yapiyor, acceptance artmadan
- **Sonuc**: `max_consecutive_rejected=8` onerilen

---

## Experiment 4: Alpha Decay (rejected_alpha_decay)

| Config | decay | Accept | Steps | RMSD (A) | TM | Notlar |
|--------|-------|--------|-------|----------|-----|--------|
| exp4[0] | 1.0 | 2/30 | 30 | 3.33 | 0.704 | Decay yok (baseline) |
| exp4[1] | 0.9 | 5/30 | 30 | 3.78 | 0.683 | Iyi acceptance |
| exp4[2] | 0.8 | **6/30** | 30 | **2.68** | 0.618 | **EN IYI acceptance** |
| exp4[3] | 0.7 | 4/30 | 30 | 3.38 | 0.695 | |
| exp4[4] | 0.5 | ~6/30 | 30 | ~2.4 | ~0.6 | Cikti kesildi |

### Analiz
- **decay=0.8** en yuksek acceptance orani (6/30) ve iyi RMSD (2.68A)
- Mekanizma: Her reject'te alpha kucuyor (0.70 -> 0.56 -> 0.45 -> ...), OF3 diffusion daha kucuk perturbasyonlar gordugunce pTM 0.6 esigine yaklasiyor
- Cok hizli decay (0.5) alpha'yi near-zero'ya dusuruyor -> RMSD_init cok kucuk, yapiyi "freeze" ediyor
- **Sonuc**: `rejected_alpha_decay=0.8` onerilen

---

## Temel Bulgular

### 1. pTM cutoff 0.6, N=214 icin zor
Bu protein icin tipik L4 pTM degerleri 0.30-0.59 arasinda. Bu durum cascade tetikliyor:
- L0 FAIL -> L1 FAIL -> L4 FAIL -> L9 FORCE
- L9 FORCE'ta alpha 0.05'e sabitlenmis, RMSD_init 2-5A arasi, yapiyi rastgele saliniyor

### 2. Alpha death spiral
L9 FORCE -> alpha stuck at 0.05 -> RMSD_init 2-5A -> yapilar random oscillation yapiyor -> accept edilemiyor -> tekrar L9 FORCE

### 3. Alpha decay bu donguyu kiriyor
`rejected_alpha_decay=0.8` ile her reject'te baseline alpha kuculuyor:
0.70 -> 0.56 -> 0.45 -> 0.36 -> ...
Kucuk alpha -> kucuk perturbasyonlar -> OF3 daha yuksek pTM uretebiliyor -> accept sansi artiyor

### 4. Stall detection gereksiz hesaplamayi onluyor
8 ardisik reject'ten sonra pipeline durmasi, acceptance artmadan GPU bosu bosuna calismasini engelliyor.

---

## Onerilen exp9_combined Parametreleri

```python
{
    "confidence_warmup_steps": 0,      # warmup gereksiz
    "confidence_rg_max": 2.5,          # default uygun
    "max_consecutive_rejected": 8,     # stall detection
    "rejected_alpha_decay": 0.8,       # en iyi acceptance
}
```

### Beklenen iyilesme
- Baseline: 2/30 accepted, RMSD ~3-4A
- Combined: 4-8/30 accepted, RMSD ~2-3A (tahmini)

---

## Eksik Deneyler (beklemede)
- [ ] exp5_pae_cutoff
- [ ] exp6_consensus
- [ ] exp7_contact_recon
- [ ] exp8_contact_of3

Bu deneyler tamamlandiktan sonra kapsamli cross-experiment analiz yapilacak ve exp9_combined icin nihai parametreler belirlenecek.
