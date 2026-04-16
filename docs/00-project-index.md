# ANM-OpenFold3 Project

## Overview
Anisotropic Network Model (ANM) tabanli protein dinamik analizi ile OpenFold3 yapi tahmini entegrasyonu.

## Navigation

### Architecture
- [[architecture/01-openfold3-inference-pipeline]] - Inference pipeline mimarisi
- [[architecture/02-model-architecture]] - Model katmanlari ve moduller
- [[architecture/03-data-flow]] - Veri akis diyagrami
- [[architecture/04-mlx-fork]] - Apple Silicon MLX fork (macOS inference)
- [[architecture/05-gnm-contact-learner]] - GNM-Contact Learner: pair_repr → learned connectivity matrix
- [[architecture/06-gnm-math-detail]] - GNM matematigi: distance→probability, Kirchhoff, eigendecomposition
- [[architecture/07-implementation-review]] - Kod incelemesi, matematik dogrulamasi
- [[architecture/07a-training-plan]] - Training plan: GNM-Contact Learner egitim stratejisi
- [[architecture/08-anm-theory]] - ANM teorisi: Hessian, 3D modlar, collectivity, GNM karsilastirmasi
- [[architecture/09-anm-mode-drive]] - ANM Mode-Drive Pipeline: collectivity-ranked iteratif konformasyonel kesif
- [[architecture/10-iterative-refinement]] - df eskalasyonu, iterasyon dinamikleri, failure modlari
- [[architecture/11-pipeline-mathematics]] - End-to-end matematiksel referans: Hessian, eigen, collectivity, displacement, contact, z-blending, Kabsch, RMSD, TM-score
- [[architecture/12-msa-training-plan]] - MSA-enabled training plan: shard yenileme, loss weights, hedef metrikler
- [[architecture/13-confidence-guided-pipeline]] - Confidence-guided adaptive pipeline: pLDDT/pTM, multi-sample diffusion, fallback stratejisi

### Modules
- [[modules/anm-mode-drive]] - ANM Mode-Drive modul referansi ve API
- [[modules/autostop-adapter]] - Autostop adapter: IW-ENM MD + early-stop monitor bridge
- [[modules/input-embedder]] - Input Embedder detaylari
- [[modules/msa-module]] - MSA Module detaylari
- [[modules/pairformer]] - PairFormer detaylari
- [[modules/diffusion-module]] - Diffusion Module detaylari
- [[modules/prediction-heads]] - Prediction Heads detaylari

### Setup
- [[setup/conda-setup]] - Conda ortam kurulumu
- [[setup/installation]] - OpenFold3 kurulum rehberi

### Plans
- [[IMPLEMENTATION_PLAN]] - Scale-up plan: 2000 PDB fine-tuning

## Project Structure
```
ANM-openfold3/
├── openfold3-repo/          # OpenFold3 orijinal kaynak (CUDA/Linux)
├── openfold3-mlx/           # OpenFold3 MLX fork (macOS Apple Silicon)
├── docs/                    # Obsidian vault - dokumantasyon
│   ├── architecture/        # Mimari diyagramlar ve aciklamalar
│   ├── modules/             # Modul detay sayfalari
│   ├── setup/               # Kurulum ve konfigurasyon
│   └── _archive/            # Eski/kullanilmayan dokumantasyon
├── src/                     # ANM entegrasyon kodu
│   ├── anm.py               # ANM Hessian, eigendecomp, displacement, collectivity
│   ├── contact_head.py       # ContactProjectionHead (z ↔ C)
│   ├── converter.py          # PairContactConverter wrapper
│   ├── coords_to_contact.py  # Koordinat → soft contact map
│   ├── data.py               # Dataset ve data loading
│   ├── ground_truth.py       # Ground truth contact map uretimi
│   ├── inverse.py            # Inverse path: C → z_ij
│   ├── kirchhoff.py          # GNM Kirchhoff ve eigendecomposition
│   ├── losses.py             # Training loss functions
│   ├── mode_combinator.py    # Collectivity-ranked mod kombinasyonlari
│   ├── mode_drive.py         # Pipeline orchestrator (df eskalasyonlu)
│   ├── of3_diffusion.py      # OF3 diffusion wrapper (trunk once, diffusion per step)
│   ├── model.py              # Model tanimlari
│   └── train.py              # Training loop
├── tests/                   # Test suite
│   ├── test_ground_truth.py  # Ground truth testleri
│   ├── test_kirchhoff.py     # Kirchhoff testleri
│   ├── test_contact_head.py  # Contact head testleri
│   ├── test_losses.py        # Loss function testleri
│   ├── test_inverse.py       # Inverse path testleri
│   └── test_model.py         # Model testleri
├── scripts/                 # Yardimci scriptler
│   ├── fetch_pdb_list.py     # PISCES-based PDB listesi olusturma
│   ├── extract_pairs.py      # PDB'den pair representation cikarma
│   ├── pack_shards.py        # Shard paketleme (buyuk dataset)
│   └── train_large.py        # Scale-up training scripti
└── notebooks/               # Jupyter notebooks
    ├── full_pipeline.ipynb    # Tam pipeline notebook
    ├── train_2000.ipynb       # 2000 PDB training notebook
    └── test_mode_drive.ipynb  # ANM Mode-Drive test notebook
```

## Temel Kavramlar

| Kavram | Aciklama |
|--------|----------|
| Collectivity | κ = (1/N)·exp(-Σ u²·ln(u²)) — mod ne kadar kollektif |
| Global df | Tek deplasman parametresi (Å), df_min → df_max eskalasyon |
| RMSD (initial'den) | Yuksek = daha fazla kesif = iyi |
| n_steps | Sabit adim sayisi, erken durma yok |
| z_ij evolution | Her adimda hem coords hem z_ij guncellenir |

## Tags
#bioinformatics #protein-structure #openfold3 #anm #collectivity
