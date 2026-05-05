# Agent: V5 Resilient Notebook Builder

## Gorev

Selective mixing V5 grid search icin Colab'da kesintiye dayanikli (resilient) notebook olustur.

## Baglam

- **Proje:** ANM-openfold3 — GNM/ANM mode-drive ile OF3 diffusion'i yonlendirme
- **Onceki notebook:** `notebooks/selective_mixing_v5.ipynb` (tum fazlar tek notebook, checkpoint yok)
- **src:** `src/selective_mixing.py`, `src/mode_drive.py`, `src/composite_confidence.py`
- **Plan:** `docs/plans/selective_mixing_v5_resilient.md`

## Kullanilan Skill & Plugin'ler

| Skill/Plugin | Amac |
|---|---|
| `python-patterns` | Python idioms, dataclass config |
| `python-testing` | Test coverage (src icin) |
| `pytorch-patterns` | Tensor ops, no_grad, GPU memory |
| `deep-research` | Literature/method search |
| `search-first` | Onceki implementasyonlari kontrol |

## Kurallar

1. **Drive checkpointing zorunlu**: Her faz sonunda `checkpoint_phase_X.json` kaydet
2. **Resume-first**: Her faz basinda checkpoint var mi kontrol et, varsa yukle ve atla
3. **Partial save**: Her individual run sonunda da partial checkpoint yaz (faz icinde kesilirse)
4. **Tensor yok JSON'da**: numpy array → `.tolist()`, torch tensor → `.numpy().tolist()`
5. **V5 ile tutarli**: Ayni runner, ayni config parametreleri — sadece I/O wrapper farkli
6. **Turk dili**: Markdown cell'ler Turkce
7. **Heatmap inline**: Selective run'larda alpha_mask heatmap goster (max 6 per run)
8. **Summary always**: Analysis cell tum fazlari birlestirip ozet tablo + grafik cikarir

## Dosya Yapisi

```
notebooks/
├── selective_mixing_v5.ipynb          # Orijinal (degistirilmez)
└── selective_mixing_v5_resilient.ipynb # Yeni: checkpoint + resume
```

## Faz Detaylari

### Phase A: Uniform vs Selective (2 run)
- A1: selective_mixing=False (V4 baseline)
- A2: selective_mixing=True (default params)
- Derived: delta_TM

### Phase B: change_cutoff sweep (5 run)
- cutoff: [0.05, 0.1, 0.15, 0.2, 0.3]
- Derived: BEST_CUTOFF

### Phase C: alpha_base x alpha_max (12 run)
- cutoff: [0.05, 0.10] x base: [0.0, 0.05] x max: [0.5, 0.7, 1.0]
- Derived: BEST_ALPHA_BASE, BEST_ALPHA_MAX, BEST_CUTOFF (final)

### Phase D: mapping & weight (9 run)
- mapping: [linear, sigmoid, step] x w: [(0.3,0.7), (0.5,0.5), (0.7,0.3)]
- Derived: BEST_MAPPING, BEST_W_C, BEST_W_D

### Analysis
- Genel sonuc tablosu (tum fazlar, sorted by TM)
- TM-score bar chart
- Step trajectory (best run)
- V4 vs V5 karsilastirmasi
- Final summary → Drive

## Checkpoint Format

```python
{
    "phase": "phase_b",
    "completed_at": "ISO timestamp",
    "n_runs": 5,
    "results": [
        {
            "label": "B_cutoff_0.05",
            "selective_mixing": True,
            "selective_params": {...},
            "best_tm": 0.85,
            "best_rmsd": 2.1,
            "accepted": 8,
            "total_steps": 20,
            ...
            "step_metrics": [...]  # her step detayi
        }
    ],
    "derived_params": {"BEST_CUTOFF": 0.05}
}
```

## Test Kontrol

Notebook'u lokal olarak import hatasi olmadan yuklendigini dogrula:
```bash
python -c "import json; json.load(open('notebooks/selective_mixing_v5_resilient.ipynb'))"
```
