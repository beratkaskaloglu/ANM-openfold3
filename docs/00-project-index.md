# ANM-OpenFold3 Project

## Overview
Anisotropic Network Model (ANM) tabanlı protein dinamik analizi ile OpenFold3 yapı tahmini entegrasyonu.

## Navigation
- [[architecture/01-openfold3-inference-pipeline]] - Inference pipeline mimarisi
- [[architecture/02-model-architecture]] - Model katmanları ve modüller
- [[architecture/03-data-flow]] - Veri akış diyagramı
- [[architecture/04-mlx-fork]] - Apple Silicon MLX fork (macOS inference)
- [[architecture/05-gnm-contact-learner]] - GNM-Contact Learner: pair_repr → learned connectivity matrix
- [[architecture/06-gnm-math-detail]] - GNM matematiği: distance→probability, Kirchhoff, eigendecomposition
- [[modules/input-embedder]] - Input Embedder detayları
- [[modules/msa-module]] - MSA Module detayları
- [[modules/pairformer]] - PairFormer detayları
- [[modules/diffusion-module]] - Diffusion Module detayları
- [[modules/prediction-heads]] - Prediction Heads detayları
- [[setup/conda-setup]] - Conda ortam kurulumu
- [[setup/installation]] - OpenFold3 kurulum rehberi

## Project Structure
```
ANM-openfold3/
├── openfold3-repo/          # OpenFold3 orijinal kaynak (CUDA/Linux)
├── openfold3-mlx/           # OpenFold3 MLX fork (macOS Apple Silicon)
├── docs/                    # Obsidian vault - dokümantasyon
│   ├── architecture/        # Mimari diyagramlar ve açıklamalar
│   ├── modules/             # Modül detay sayfaları
│   └── setup/               # Kurulum ve konfigürasyon
├── src/                     # Bizim ANM entegrasyon kodumuz (ileride)
└── notebooks/               # Jupyter notebooks (ileride)
```

## Tags
#bioinformatics #protein-structure #openfold3 #anm
