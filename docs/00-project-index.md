# ANM-OpenFold3 Project

## Overview
Anisotropic Network Model (ANM) tabanli protein dinamik analizi ile OpenFold3 yapi tahmini entegrasyonu.

## Navigation
- [[architecture/01-openfold3-inference-pipeline]] - Inference pipeline mimarisi
- [[architecture/02-model-architecture]] - Model katmanlari ve moduller
- [[architecture/03-data-flow]] - Veri akis diyagrami
- [[architecture/04-mlx-fork]] - Apple Silicon MLX fork (macOS inference)
- [[architecture/05-gnm-contact-learner]] - GNM-Contact Learner: pair_repr → learned connectivity matrix
- [[architecture/06-gnm-math-detail]] - GNM matematigi: distance→probability, Kirchhoff, eigendecomposition
- [[architecture/08-anm-theory]] - ANM teorisi: Hessian, 3D modlar, collectivity, GNM karsilastirmasi
- [[architecture/09-anm-mode-drive]] - ANM Mode-Drive Pipeline: collectivity-ranked iteratif konformasyonel kesif
- [[architecture/10-iterative-refinement]] - df eskalasyonu, iterasyon dinamikleri, failure modlari
- [[modules/anm-mode-drive]] - ANM Mode-Drive modul referansi ve API
- [[modules/input-embedder]] - Input Embedder detaylari
- [[modules/msa-module]] - MSA Module detaylari
- [[modules/pairformer]] - PairFormer detaylari
- [[modules/diffusion-module]] - Diffusion Module detaylari
- [[modules/prediction-heads]] - Prediction Heads detaylari
- [[setup/conda-setup]] - Conda ortam kurulumu
- [[setup/installation]] - OpenFold3 kurulum rehberi

## Project Structure
```
ANM-openfold3/
├── openfold3-repo/          # OpenFold3 orijinal kaynak (CUDA/Linux)
├── openfold3-mlx/           # OpenFold3 MLX fork (macOS Apple Silicon)
├── docs/                    # Obsidian vault - dokumantasyon
│   ├── architecture/        # Mimari diyagramlar ve aciklamalar
│   ├── modules/             # Modul detay sayfalari
│   └── setup/               # Kurulum ve konfigurasyon
├── src/                     # ANM entegrasyon kodu
│   ├── anm.py               # ANM Hessian, eigendecomp, displacement, collectivity
│   ├── contact_head.py       # ContactProjectionHead (z ↔ C)
│   ├── converter.py          # PairContactConverter wrapper
│   ├── coords_to_contact.py  # Koordinat → soft contact map
│   ├── mode_combinator.py    # Collectivity-ranked mod kombinasyonlari
│   ├── mode_drive.py         # Pipeline orchestrator (df eskalasyonlu)
│   ├── kirchhoff.py          # GNM Kirchhoff ve eigendecomposition
│   └── losses.py             # Training loss functions
└── notebooks/               # Jupyter notebooks
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
