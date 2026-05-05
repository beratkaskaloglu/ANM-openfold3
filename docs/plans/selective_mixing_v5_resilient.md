# V5 Selective Mixing — Resilient Colab Notebook Plan

**Tarih:** 2026-05-05
**Durum:** Aktif
**Ilgili:** `notebooks/selective_mixing_v5.ipynb`, `src/selective_mixing.py`

---

## Problem

Colab sik sik kesiliyor (runtime disconnect, timeout, GPU preemption).
Mevcut V5 notebook monolitik: kesintide tum sonuclar kaybolur.

## Cozum: Phase-by-Phase Drive Checkpointing

Her fazin sonunda sonuclar Drive'a JSON olarak kaydedilir.
Notebook tekrar calistirildiginda onceki fazlarin sonuclarini Drive'dan okur, atlayip kaldigi yerden devam eder.

## Mimari

```
Drive: /MyDrive/ANM-openfold3/search_v5/
├── checkpoint_phase_a.json    # Phase A sonuclari
├── checkpoint_phase_b.json    # Phase B sonuclari
├── checkpoint_phase_c.json    # Phase C sonuclari
├── checkpoint_phase_d.json    # Phase D sonuclari
└── summary_final.json         # Tum fazlarin ozet tablosu
```

### Her Checkpoint Icerigi

```json
{
  "phase": "phase_b",
  "completed_at": "2026-05-05T14:30:00",
  "results": [...],
  "best_result": {...},
  "derived_params": {"BEST_CUTOFF": 0.05}
}
```

### Resume Mantigi

```python
def load_or_run_phase(phase_name, run_fn):
    ckpt_path = f'{DRIVE_SAVE_DIR}/checkpoint_{phase_name}.json'
    if os.path.exists(ckpt_path):
        print(f'[RESUME] {phase_name} Drive\'dan yukleniyor...')
        return json.load(open(ckpt_path))
    results = run_fn()
    save_checkpoint(phase_name, results)
    return results
```

### Faz Bagimliliklari

```
Phase A (baseline) → bagimsiz
Phase B (cutoff sweep) → bagimsiz
Phase C (alpha sweep) → Phase B'den BEST_CUTOFF (veya her ikisi: 0.05, 0.10)
Phase D (mapping/weight) → Phase B+C'den en iyi parametreler
Analysis → Tum fazlar
```

## Notebook Cell Yapisi

| # | Cell | Kesilebilir? | Kayit |
|---|------|-------------|-------|
| 1 | Environment setup | Hayir (tek sefer) | - |
| 2 | Config | Hayir | - |
| 3 | Converter + PDB + OF3 | Hayir (tek sefer) | - |
| 4 | Runner fonksiyonlari | Hayir | - |
| 5 | Phase A (2 run) | Evet → Drive | checkpoint_phase_a.json |
| 6 | Phase B (5 run) | Evet → Drive | checkpoint_phase_b.json |
| 7 | Phase C (12 run) | Evet → Drive | checkpoint_phase_c.json |
| 8 | Phase D (9 run) | Evet → Drive | checkpoint_phase_d.json |
| 9 | Analysis + ozet | - | summary_final.json |

## Onemli Detaylar

1. **Her run bittikten sonra flush**: Sadece faz sonu degil, her run sonunda da partial save yapilir (run icinde bile kesilirse son completed run'a kadar korunur)
2. **Tensor serialize**: `alpha_mask_snapshot` numpy array olarak kaydedilir, torch tensor degil
3. **V5 notebook'tan fark**: Ayni runner, ayni config — sadece checkpoint + resume logic eklendi
4. **Phase C dual cutoff**: Hem 0.05 hem 0.10 ile 12 run (onceki notebook'taki son commit ile uyumlu)

## Kullanim

```
1. Notebook'u Colab'da ac
2. Tum cell'leri sirasi ile calistir
3. Kesilirse: Runtime > Run all (veya kesilen cell'den itibaren)
4. Drive'daki checkpoint'lar otomatik yuklenir, tamamlanan fazlar atlanir
```
