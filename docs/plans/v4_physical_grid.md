# V4 Search Plan — Physical Weights + Alpha=0.7 with Safeguards

**Tarih:** 2026-04-25
**Amac:** Yeni drift korumalari (rmsd_init cutoff, best-so-far rollback, adaptive stop) ile alpha=0.7 + decay stratejilerini yeniden test et. Physical weight presetlerini de dahil et.

---

## Motivasyon

V3'te alpha=0.3 kazandi CUNKU drift korumalari yoktu. Simdi 3 yeni mekanizma eklendi:
1. **rmsd_init hard cutoff (10A)** — buyuk deplasman yapan adimlari hard-reject
2. **Best-so-far rollback** — TM %40+ duserse en iyi yapiya geri don
3. **Adaptive early stopping** — 3 ardisik accepted step'te TM dusuyorsa dur

Bu korumalarla alpha=0.7 tekrar denenebilir:
- Buyuk adim = hedef yapiya daha hizli yaklasma potansiyeli
- rmsd_init>10A cutoff buyuk adimlarin kotu olanlarini filtreler
- Rollback, drift olursa otomatik kurtarma saglar
- Adaptive stop, gereksiz step'leri onler

## V3'ten Ogrenilen Sabit Parametreler

| Parametre | Deger | Kaynak |
|-----------|-------|--------|
| confidence_ptm_cutoff | 0.50 | Phase 3 kazanani |
| confidence_rmsd_init_max | 10.0 | rmsd_init vs TM korelasyonu |
| enable_best_rollback | True | Yeni |
| best_rollback_tm_drop | 0.40 | Yeni |
| enable_adaptive_stop | True | Yeni |
| adaptive_stop_window | 3 | Yeni |

---

## Phase A: Alpha=0.7 + Decay (4 preset x 3 threshold x 3 decay = 36 run)

### Mantik
Alpha=0.7 ile buyuk adim at, decay ile kucult. Yeni korumalar drift'i yakalayacak.

### Grid
| Parametre | Degerler |
|-----------|----------|
| Weight presets | A_ptm_heavy, B_pae_heavy, C_balanced, D_physical |
| Thresholds | 0.45, 0.50, 0.55 |
| Alpha | 0.7 (sabit) |
| Alpha decay | 0.8, 0.9, 1.0 |
| n_steps | 20 |

### Karsilastirma hedefi
V3 best (alpha=0.3, A_ptm_heavy, ptm_cutoff=0.50): TM=0.833, RMSD=2.62A

---

## Phase B: Physical Presets + Alpha=0.7 (4 preset x 4 threshold x 2 decay = 32 run)

### Yeni Physical Presetler
| Preset | w_ptm | w_plddt | w_pae | w_rg | w_cr | rg_norm |
|--------|-------|---------|-------|------|------|---------|
| D_physical | 0.10 | 0.10 | 0.25 | 0.35 | 0.20 | quadratic |
| E_physical_strict | 0.05 | 0.10 | 0.25 | 0.40 | 0.20 | strict |
| F_physical_balanced | 0.10 | 0.10 | 0.25 | 0.30 | 0.25 | strict |
| G_rg_dominant | 0.05 | 0.05 | 0.20 | 0.50 | 0.20 | strict |

### Grid
| Parametre | Degerler |
|-----------|----------|
| Weight presets | D, E, F, G |
| Thresholds | 0.40, 0.45, 0.50, 0.55 |
| Alpha | 0.7 |
| Alpha decay | 0.8, 0.9 |
| n_steps | 20 |

---

## Phase C: Final 30-step (en iyi A+B konfigurasyonu)

Phase A ve B'nin en iyileri ile 30 step kosu.

---

## Tahmini Sure

- Phase A: 36 run x ~3 dk (korumalarla daha kisa) = ~2 saat
- Phase B: 32 run x ~3 dk = ~1.5 saat
- Phase C: 1 run x ~5 dk = ~5 dk
- **Toplam:** ~3.5 saat

---

## Basari Kriterleri

| Metrik | V3 Best (alpha=0.3) | V4 Hedef (alpha=0.7) |
|--------|---------------------|----------------------|
| best_TM | 0.833 | >= 0.80 |
| Accept rate | 70% | >= 30% |
| Drift onset | Step 11 | >= Step 8 (rollback ile kurtarma) |
| best_RMSD | 2.62 A | <= 3.5 A |
| Rollback count | N/A | Observe |
