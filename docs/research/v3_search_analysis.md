# V3 Composite Confidence Search - Detayli Analiz

**Tarih:** 2026-04-24
**Protein:** 1AKE -> 4AKE (N=214, Chain A)
**Baseline RMSD:** 7.14 A, TM=0.45

---

## Ozet Tablo

| Faz                     | En Iyi Config              | Kabul/Toplam | best_TM    | best_RMSD  | Wall  |
| ----------------------- | -------------------------- | ------------ | ---------- | ---------- | ----- |
| Phase 1 (Weight Grid)   | B_pae_heavy_t0.50          | 3/11         | 0.7100     | 3.95 A     | 31 dk |
| Phase 1 (En cok kabul)  | A_ptm_heavy_t0.50          | 6/20         | 0.6525     | 5.02 A     | 60 dk |
| Phase 2 (Alpha)         | alpha_conservative (a=0.3) | 8/20         | 0.7339     | 3.78 A     | 47 dk |
| Phase 3 (pTM cutoff)    | fb_ptm050                  | 14/20        | **0.8329** | **2.62 A** | 50 dk |
| Phase 4 (Final 30-step) | FINAL_30step               | 8/30         | 0.8163     | 3.13 A     | 84 dk |
| V2 Baseline             | pTM=0.6, no decay          | 4/20         | 0.7751     | 3.35 A     | 53 dk |

**V3 en iyi vs V2 baseline:** TM +0.058 (0.833 vs 0.775), RMSD -0.73 A (2.62 vs 3.35)

---

## Phase 1: Weight Grid (4 preset x 3 threshold = 12 run)

### Weight Presets
| Preset | w_ptm | w_plddt | w_pae | w_rg | w_contact |
|--------|-------|---------|-------|------|-----------|
| A_ptm_heavy | 0.35 | 0.15 | 0.20 | 0.20 | 0.10 |
| B_pae_heavy | 0.15 | 0.10 | 0.35 | 0.25 | 0.15 |
| C_balanced | 0.20 | 0.15 | 0.25 | 0.25 | 0.15 |
| D_physical | 0.10 | 0.10 | 0.25 | 0.35 | 0.20 |

### Sonuclar
| Config | Kabul/Toplam | Accept% | best_TM | best_RMSD | last_TM | Wall |
|--------|-------------|---------|---------|-----------|---------|------|
| A_ptm_heavy_t0.45 | 6/20 | 30% | 0.4494 | 8.06 | 0.063 | 64 dk |
| **A_ptm_heavy_t0.50** | **6/20** | **30%** | **0.6525** | **5.02** | 0.020 | 61 dk |
| A_ptm_heavy_t0.55 | 4/20 | 20% | 0.5829 | 5.81 | 0.037 | 59 dk |
| B_pae_heavy_t0.45 | 4/12 | 33% | 0.5733 | 5.47 | 0.031 | 38 dk |
| **B_pae_heavy_t0.50** | **3/11** | **27%** | **0.7100** | **3.95** | 0.036 | 31 dk |
| B_pae_heavy_t0.55 | 4/20 | 20% | 0.5881 | 5.82 | 0.056 | 56 dk |
| C_balanced_t0.45 | 2/10 | 20% | 0.5996 | 5.75 | 0.060 | 30 dk |
| C_balanced_t0.50 | 4/14 | 29% | 0.5511 | 6.16 | 0.051 | 40 dk |
| C_balanced_t0.55 | 3/12 | 25% | 0.5725 | 5.74 | 0.145 | 31 dk |
| D_physical_t0.45 | 4/13 | 31% | 0.5988 | 5.37 | 0.030 | 35 dk |
| D_physical_t0.50 | 3/17 | 18% | 0.5605 | 5.82 | 0.049 | 49 dk |
| D_physical_t0.55 | 3/18 | 17% | 0.6249 | 4.73 | 0.073 | 46 dk |

### Phase 1 Yorumlari
- **Threshold=0.50** genel olarak en iyi sonuclari veriyor (0.45 cok gevsek, 0.55 cok siki)
- **B_pae_heavy** en yuksek TM'ye ulasti (0.71) ama az adimda stall etti
- **A_ptm_heavy** en cok kabul etti (6/20) ama step 1 disindaki kabuller kotu
- **TUM konfigurasyonlarda last_TM < 0.15** — yapi sonunda her zaman bozuluyor
- Notebook auto-select: A_ptm_heavy_t0.50 (kabul sayisi primary, TM secondary)

---

## Phase 2: Alpha Schedule (6 strateji)

Phase 1 en iyisi (A_ptm_heavy, threshold=0.50) ile test edildi.

| Config | Alpha | Decay | Kabul/Toplam | Accept% | best_TM | Wall |
|--------|-------|-------|-------------|---------|---------|------|
| alpha_no_decay | 0.7 | 1.0 | 3/17 | 18% | 0.6382 | 52 dk |
| alpha_slow_decay | 0.7 | 0.9 | 2/10 | 20% | 0.5122 | 30 dk |
| alpha_medium_decay | 0.7 | 0.8 | 7/20 | 35% | 0.6549 | 60 dk |
| alpha_fast_decay | 0.7 | 0.7 | 6/20 | 30% | 0.6405 | 60 dk |
| alpha_low_start | 0.5 | 0.85 | 6/20 | 30% | 0.5981 | 60 dk |
| **alpha_conservative** | **0.3** | **1.0** | **8/20** | **40%** | **0.7339** | **47 dk** |

### Phase 2 Yorumlari
- **alpha_conservative (a=0.3, decay yok) acik ara kazanan**
- Dusuk alpha = kucuk yapisal deplasman = OF3 daha iyi katlar
- alpha_conservative ilk 3 adimda artan TM gosterdi: 0.55 -> 0.63 -> 0.73
- Bu diger hicbir konfigurasyonda gorulmedi — genelde step 1 iyi, step 2 felaket
- Decay stratejileri (0.7-0.9) alpha=0.3 sabitin gerisinde kaldi

**Kritik bulgu:** Buyuk adim atip sonra kucultmek (decay) yerine, **bastan kucuk adim atmak** cok daha etkili. Cunku buyuk ilk adim yapiyi bozuyor ve decay bunu kurtaramiyor.

---

## Phase 3: Internal pTM Cutoff (4 seviye)

Phase 2 en iyisi (alpha=0.3, decay=1.0) ile test edildi.

| Config | pTM Cutoff | Ranking Cutoff | Kabul/Toplam | Accept% | best_TM | Wall |
|--------|-----------|----------------|-------------|---------|---------|------|
| fb_ptm010 | 0.10 | 0.05 | 7/17 | 41% | 0.7753 | 42 dk |
| fb_ptm020 | 0.20 | 0.10 | 8/20 | 40% | 0.6907 | 48 dk |
| fb_ptm035 | 0.35 | 0.20 | 11/20 | 55% | 0.7685 | 52 dk |
| **fb_ptm050** | **0.50** | **0.30** | **14/20** | **70%** | **0.8329** | **50 dk** |

### fb_ptm050 Trajektori (TUM deneyler icinde en iyi)
| Step | Kabul  | TM_tgt    | RMSD_tgt | comp_score | pTM   | Rg   |
| ---- | ------ | --------- | -------- | ---------- | ----- | ---- |
| 1    | ACCEPT | 0.573     | 6.74     | 0.954      | 0.810 | 0.98 |
| 2    | ACCEPT | 0.655     | 5.37     | 0.834      | 0.617 | 1.02 |
| 3    | ACCEPT | 0.662     | 5.03     | 0.766      | 0.517 | 1.04 |
| 4    | ACCEPT | 0.794     | 3.30     | 0.747      | 0.495 | 1.12 |
| 5    | ACCEPT | **0.833** | **2.62** | 0.721      | 0.460 | 1.18 |
| 6    | ACCEPT | 0.758     | 3.75     | 0.717      | 0.448 | 1.10 |
| 7    | ACCEPT | 0.814     | 3.01     | 0.711      | 0.440 | 1.14 |
| 8    | ACCEPT | 0.575     | 5.71     | 0.668      | 0.398 | 1.34 |
| 9    | ACCEPT | 0.823     | 2.71     | 0.640      | 0.342 | 1.20 |
| 10   | ACCEPT | 0.596     | 5.49     | 0.648      | 0.352 | 1.21 |
| 11   | ACCEPT | 0.361     | 9.36     | 0.575      | 0.309 | 1.56 |
| 12   | ACCEPT | 0.231     | 12.76    | 0.571      | 0.322 | 1.61 |
| ...  | reject | ...       | ...      | ...        | ...   | ...  |
| 17   | ACCEPT | 0.117     | 20.94    | 0.521      | 0.384 | 1.90 |
| 20   | ACCEPT | 0.150     | 18.02    | 0.552      | 0.397 | 1.85 |

### Phase 3 Yorumlari
- **pTM cutoff=0.50 dramatik fark yaratti** — %70 kabul orani, TM=0.83
- Yuksek internal pTM cutoff, fallback ladder'i daha siki calistirdi
- OF3 daha yuksek confidence yapilar uretmeye zorlandigi icin yapisal tutarlilik artti
- **Ilk 10 adim boyunca TM>0.5 korundu** — bu benzersiz bir basari
- Drift step 11'den sonra basladi (Rg artisi: 1.56 -> 1.61 -> 1.90)
- Step 5 ve 9'da TM>0.82 — pipeline target'a cok yaklasti

**Neden pTM cutoff bu kadar etkili?** Dusuk cutoff'ta (0.10-0.20) pipeline her OF3 ciktisini kabul eder — kotu yapilar da dahil. Yuksek cutoff (0.50) sadece OF3'un kendinin "emin" oldugu yapilari gecirir. Bu, composite score'dan bagimsiz bir kalite filtresi.

---

## Phase 4: Final 30-Step Run

En iyi Phase 1-3 parametreleri:
- Weights: A_ptm_heavy (w_ptm=0.35, w_plddt=0.15, w_pae=0.20, w_rg=0.20, w_contact=0.10)
- Threshold: 0.50
- Alpha: 0.30, Decay: 1.0
- Max stall: 12

| Metrik | Deger |
|--------|-------|
| Kabul/Toplam | 8/30 (27%) |
| best_TM | 0.8163 |
| best_RMSD | 3.13 A |
| last_TM | 0.028 |
| last_RMSD | 42.35 A |
| Wall time | 84 dk |

### Final Trajektori (kabul edilen adimlar)
| Step | TM_tgt | RMSD_tgt | rmsd_init | comp | FB |
|------|--------|----------|-----------|------|----|
| 1 | 0.564 | 6.93 | 1.63 | 0.954 | 0 |
| 2 | 0.781 | 3.35 | 4.56 | 0.826 | 0 |
| 3 | **0.816** | **3.13** | 8.94 | 0.647 | 0 |
| 4 | 0.351 | 9.07 | 12.50 | 0.564 | 0 |
| 5 | 0.122 | 17.00 | 18.35 | 0.506 | 1 |
| 9 | 0.118 | 19.19 | 20.29 | 0.529 | 4 |
| 12 | 0.092 | 20.53 | 24.14 | 0.502 | 4 |
| 20 | 0.053 | 29.96 | 30.73 | 0.590 | 0 |

**Phase 4'un Phase 3'ten kotu olmasi:** Phase 4, Phase 3'un ptm_cutoff=0.50'yi kullanmadi — Phase 1'in dusuk ptm_cutoff=0.15 ayarini tasiyor. Bu yuzden fb_ptm050'deki kalite korunamadi.

---

## Yapisal Drift Analizi

### Evrensel Sorun
Tum konfigurasyonlarda ayni pattern:
1. Step 1: iyi (TM 0.45-0.57, Rg ~1.0)
2. Step 2-3: degrade baslar veya devam eder (config'e bagli)
3. Step 5+: yapi acilir (Rg>1.5), TM<0.2
4. Step 10+: tamamen bozuk (Rg>2.0, TM<0.1)

### Istisna: fb_ptm050
Tek istisnai performans: 10 adim boyunca TM>0.5. Sebebi:
- Alpha=0.3 (kucuk adim)
- pTM cutoff=0.50 (OF3 kendi confidence filtresi)
- Bu iki filtre birlikte yapisal tutarliligi korudu

### RMSD_init vs TM_tgt Korelasyonu

| rmsd_init | Ornek | Ort. TM | Yorum |
|-----------|-------|---------|-------|
| 0-5 A | 5 | 0.62 | Iyi — yapi korunmus |
| 5-10 A | 17 | 0.62 | Iyi — OF3 hala katlayabiliyor |
| 10-15 A | 12 | 0.27 | Tehlike bolgesI |
| 15-20 A | 19 | 0.16 | Kotu — kurtarma zor |
| 20-30 A | 31 | 0.09 | Felaket — yapi kayip |
| 30+ A | 2 | 0.05 | Tamamen bozuk |

**Kritik esik: rmsd_init ~10 A.** Bu sinirin otesinde OF3 yapiyi dogru katlamakta basarisiz.

---

## Oneriler

### 1. RMSD_init Hard Cutoff (YUKSEK ONCELIK)
`rmsd_init > 10.0 A` olan adimlari hard-reject et. Bu tek basina drift'i onleyebilir.
Rg cutoff gibi composite score'dan bagimsiz calisir.

### 2. Internal pTM Cutoff = 0.50 (YUKSEK ONCELIK)
Phase 3 sonuclari acik: pipeline'in kendi confidence filtresi composite score'dan daha etkili.
`confidence_ptm_cutoff=0.50` sabit kullanilmali.

### 3. Alpha = 0.3 Sabit (ORTA ONCELIK)
Decay stratejileri gereksiz. Sabit kucuk alpha en iyi sonucu veriyor.

### 4. Best-So-Far Tracking (GELECEK)
Her step'te "en iyi TM'ye sahip yapi"yi kaydet. Eger mevcut TM, best'ten %50+ dusukse, best'e geri don.

### 5. Adaptive Stopping (GELECEK)
Step 5-10 arasinda TM zirveye ulasiyor. Eger 3 ardisik step'te TM dusuyorsa, pipeline'i durdur ve en iyi yapiyi dondur.

---

## V2 vs V3 Karsilastirmasi

| Metrik | V2 Baseline | V3 Best (fb_ptm050) | Iyilesme |
|--------|------------|---------------------|----------|
| Accept rate | 20% (4/20) | 70% (14/20) | +50pp |
| best_TM | 0.7751 | 0.8329 | +0.058 |
| best_RMSD | 3.35 A | 2.62 A | -0.73 A |
| Wall time | 53 dk | 50 dk | -3 dk |
| Drift onset | Step 2 | Step 11 | +9 adim |

V3'un en buyuk kazanimi TM/RMSD iyilesmesi degil, **drift'i 9 adim geciktirmesi**. Bu, pipeline'in daha uzun sureli kararli calismasini sagladi.

---

## Sonraki Adimlar
1. `rmsd_init` hard cutoff implementasyonu
2. `confidence_ptm_cutoff=0.50` default yapma
3. Best-so-far rollback mekanizmasi
4. Farkli protein boyutlariyla test (N=100, N=500)
5. Physical weight preset'i (D_physical) ile alpha=0.3 + ptm_cutoff=0.50 kombinasyonu
