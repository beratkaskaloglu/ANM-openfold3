# 12 — MSA-Enabled Training Plan

> Mevcut MSA-siz shard'lari silip, ColabFold MSA ile yeni shard uretimi ve bastan egitim plani.

## Motivasyon

### Problem
Mevcut 300 protein shard'lari `use_msa_server=False` ile uretildi (`scripts/extract_pairs.py` satir 144). Test notebook'ta OF3 diffusion `use_msa_server=True` ile calistirilinca:

- **MSA-siz zij_trunk**: std ~75, dar dagilim
- **MSA-li zij_trunk**: std ~104, genis dagilim
- Contact head MSA-siz veriyle egitildi → MSA-li veriyle test edilince **B-factor r = -0.206** (negatif korelasyon)

### Cozum
1. Mevcut shard'lari sil
2. `use_msa_server=True` ile yeni shard uret
3. Contact head'i bastan egit (checkpoint'tan resume yok)

## Degisiklikler

### `scripts/extract_pairs.py`
```diff
- use_msa_server=False,
+ use_msa_server=use_msa_server,

+ parser.add_argument('--use-msa-server', action='store_true', default=False)
```

### `notebooks/train_2000.ipynb` — Cell 6 (Shard Generation + Training)
- `USE_MSA_SERVER = True` config parametresi eklendi
- `extract_pairs.py` cagrisi `--use-msa-server` flag'i ile
- Checkpoint resume devre disi (eski MSA-siz checkpoint'lar skip)
- `SHARD_DIR_DRIVE` yeni path: `shards_msa/` (eski shard'larla karismamasi icin)
- Eger eski shard'lar varsa kullaniciya uyari gosterir

## Egitim Konfigurasyonu

### Phase 1: Full Training (contact + GNM + recon)
| Parametre | Deger | Aciklama |
|-----------|-------|----------|
| ALPHA | 1.0 | Contact loss (focal) |
| BETA | 0.3 | GNM eigenvalue loss |
| GAMMA | 0.05 | Reconstruction loss |
| FT_LR | 3e-4 | OneCycleLR max_lr |
| FT_EPOCHS | 1000 | Maks epoch |
| FT_PATIENCE | 80 | Early stopping |
| FT_GRAD_CLIP | 1.0 | Gradient clipping |
| FT_GRAD_ACCUM | 4 | Gradient accumulation |
| N_MODES | 20 | GNM decomposition mod sayisi |
| SEQ_SEP_MIN | 6 | Minimum sequence separation |
| BOTTLENECK_DIM | 64 | Contact head bottleneck |
| C_Z | 128 | Pair representation boyutu |

### Neden ALPHA=1.0, BETA=0.3?
Onceki egitim ALPHA=0 (sadece GNM fine-tuning) kullaniyordu. Bu, contact map kalitesini ihmal ediyordu. MSA-li veriyle bastan egitimde her uc loss birlestirilerek:
- Contact loss: dogrudan temas tahmini kalitesi
- GNM loss: fiziksel anlam (B-faktor korelasyonu)
- Recon loss: z_ij geri-donus kalitesi (pipeline icin kritik)

### Phase 2 (opsiyonel): GNM-only Fine-Tuning
Eger Phase 1 sonunda B-factor r < 0.85 ise:
```
ALPHA = 0.0, BETA = 1.0, GAMMA = 0.0
FT_LR = 1e-4, FT_EPOCHS = 200
```

## Shard Uretim Akisi

```
PDB ID + Sequence
    → write_query_json()
    → InferenceExperimentRunner(use_msa_server=True)
        → data_module.prepare_data()  ← ColabFold MSA tetiklenir
        → data_module.setup()
        → runner.run()
    → latent_output.pt → pair_repr [N, N, 128]
    → download PDB → CA coords [N, 3]
    → pack into shard_XXXX.npz
```

**Onemli**: `prepare_data()` ColabFold MSA server'i tetikler. Bu otomatik olarak `runner.run()` icinde cagrilir cunku Lightning trainer kullanilir.

## Hedef Metrikler

| Metrik | Hedef | Onceki (MSA-siz) |
|--------|-------|------------------|
| Val adj accuracy | >= 0.85 | ~0.82 |
| Val B-factor r | >= 0.85 | 0.343 (MSA-siz test) |
| Test B-factor r | >= 0.80 | -0.206 (MSA-li test) |

## Zaman Tahmini

- Shard uretimi: ~2-3 saat (300 protein, MSA ile daha yavas)
- Egitim: ~30-60 dk (1000 epoch, A100)
- Toplam: ~3-4 saat

## Kontrol Listesi

- [ ] Drive'daki eski MSA-siz shard'lar yedeklendi/silindi
- [ ] `extract_pairs.py` MSA flag eklendi
- [ ] `train_2000.ipynb` guncellendi
- [ ] Colab'da GPU runtime secildi (A100 tercih)
- [ ] Drive mount calisir durumda
- [ ] Shard uretimi tamamlandi (300 protein)
- [ ] Egitim tamamlandi
- [ ] Test B-factor r >= 0.80 dogrulandi
- [ ] `test_mode_drive.ipynb` yeni checkpoint ile test edildi
