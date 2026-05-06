# ANM-OpenFold3 Dokümantasyon

> Protein konformasyonel dinamiklerini ANM (Anisotropic Network Model) titreşim modları ile OpenFold3 diffusion modeline besleyerek hedef-bağımsız yapısal keşif yapan pipeline.

---

## Hızlı Başlangıç

```bash
# Kurulum
conda activate openfold3
pip install -e .

# Temel kullanım (notebook)
jupyter notebook notebooks/mode_drive_explorer.ipynb

# Grid search (Colab A100 önerilir)
jupyter notebook notebooks/grid_search_v6.ipynb

# Benchmark (9 protein, open↔closed)
jupyter notebook notebooks/benchmark_open_closed.ipynb
```

---

## Dokümantasyon Haritası

### Mimari & Teknik

| Doküman | Açıklama |
|---------|----------|
| [[architecture/pipeline-deep-dive]] | Pipeline'ın tam teknik açıklaması — tensör boyutları, device akışı, tüm mekanizmalar |
| [[architecture/configuration-reference]] | Tüm `ModeDriveConfig` parametreleri tablo halinde |
| [[architecture/14-selective-mixing-pipeline]] | Selective mixing (per-pair adaptive alpha) tasarım dokümanı |
| [[architecture/13-confidence-guided-pipeline]] | Confidence gating V1/V2 tasarımı |
| [[architecture/08-anm-theory]] | ANM/GNM matematiksel temeli |
| [[architecture/09-anm-mode-drive]] | Mode-Drive pipeline ilk tasarım |
| [[architecture/11-pipeline-mathematics]] | Pipeline matematiği detaylı |

### Araştırma

| Doküman | Açıklama |
|---------|----------|
| [[research/research-journey]] | Kronolojik araştırma özeti (V1→V5, benchmark, bulgular) |
| [[research/iw-enm-and-converter]] | IW-ENM elastic MD + PairContactConverter encoder-decoder eğitimi |
| [[research/confidence_metrics_analysis]] | Confidence metrik analizi (708 step korelasyon) |
| [[research/v3_search_analysis]] | V3 grid search sonuçları |
| [[research/grid_search_v2_colab_analysis]] | V2 grid search Colab analizi |
| [[research/benchmark_issues_analysis]] | Benchmark sorunları ve kök neden analizi |
| [[research/benchmark_fix_plan]] | Benchmark fix implementasyon planı |
| [[research/alpha_scheduling_strategies]] | Alpha scheduling stratejileri |
| [[research/enm_early_stopping_criteria]] | ENM early stopping kriterleri |

### OpenFold3 Mimarisi

| Doküman | Açıklama |
|---------|----------|
| [[architecture/01-openfold3-inference-pipeline]] | OF3 inference pipeline genel bakış |
| [[architecture/02-model-architecture]] | OF3 model mimarisi |
| [[architecture/03-data-flow]] | OF3 veri akışı |
| [[modules/input-embedder]] | Input embedder modülü |
| [[modules/msa-module]] | MSA modülü |
| [[modules/pairformer]] | Pairformer modülü |
| [[modules/diffusion-module]] | Diffusion modülü |
| [[modules/prediction-heads]] | Prediction heads |

### Planlar

| Doküman | Açıklama |
|---------|----------|
| [[plans/selective_mixing_v5_resilient]] | V5 resilient pipeline planı |
| [[plans/confidence_v2_implementation]] | Confidence V2 implementasyon planı |
| [[plans/autostop_integration]] | Autostop entegrasyon planı |
| [[plans/optimized_search_v3]] | V3 optimized search planı |
| [[plans/v4_physical_grid]] | V4 fiziksel grid planı |

### Kurulum

| Doküman | Açıklama |
|---------|----------|
| [[setup/installation]] | Kurulum rehberi |
| [[setup/conda-setup]] | Conda ortam kurulumu |

---

## Kaynak Dosyaları

```
src/
├── mode_drive.py          # Ana pipeline (ModeDrivePipeline sınıfı)
├── mode_drive_config.py   # Konfigürasyon dataclass
├── mode_drive_utils.py    # Kabsch, RMSD, TM-score, MDS
├── selective_mixing.py    # Per-pair adaptive alpha mask
├── of3_diffusion.py       # OF3 model yükleme, trunk, diffusion
├── converter.py           # PairContactConverter (encoder-decoder)
├── coords_to_contact.py   # Sigmoid soft contact hesabı
├── train.py               # Converter eğitim scripti
├── anm.py                 # ANM Hessian, mod hesabı, B-factor
├── iw_enm/               # IW-ENM elastic MD simülasyon paketi
│   ├── simulation.py      # Ana simülasyon döngüsü
│   ├── integrator.py      # Velocity Verlet integrator
│   ├── network.py         # Etkileşim ağı (spring constants)
│   ├── turnpoint.py       # Konformasyon dönüm noktası tespiti
│   └── finetune/          # Surrogate MLP optimizasyon
└── __init__.py
```

---

## Pipeline Özeti

```
PDB → CA coords [N,3]
  → ANM Hessian → eigenmodes [N,3] × K
    → displacement (df × mode) → displaced coords [N,3]
      → sigmoid contact [N,N]
        → PairContactConverter → z_pseudo [N,N,128]
          → selective blend (alpha_mask ⊙ z_pseudo + (1-alpha_mask) ⊙ zij_trunk)
            → OF3 Diffusion → new coords [N,3] + confidence
              → confidence gating → accept/reject
                → next step (veya fallback)
```

---

## Temel Kavramlar

| Kavram | Açıklama |
|--------|----------|
| **ANM** | Anisotropic Network Model — protein CA atomları arası elastik ağ, düşük frekanslı titreşim modları |
| **Mode-Drive** | ANM modlarını kullanarak yapıyı iteratif olarak hareket ettirme |
| **Selective Mixing** | Sadece hareket eden bölgelerde z-pair değiştirme (alpha_mask) |
| **Confidence Gating** | OF3'ün ürettiği yapının kalitesini pTM/pLDDT/ranking ile değerlendirme |
| **Fallback** | Düşük confidence'ta df/alpha küçültüp tekrar deneme |
| **PairContactConverter** | OF3 trunk z-pair [N,N,128] ↔ contact map [N,N] dönüşümü |
| **IW-ENM** | Importance-weighted elastic network MD simülasyonu |

---

## Notebooklar

| Notebook | Açıklama |
|----------|----------|
| `mode_drive_explorer.ipynb` | Tek protein interaktif keşif |
| `grid_search_v6.ipynb` | Parametre grid search (Colab) |
| `benchmark_open_closed.ipynb` | 9 protein open↔closed benchmark |
| `selective_mixing_v5_resilient.ipynb` | V5 resilient pipeline demo |
| `train_2000.ipynb` | Converter eğitim notebook |
| `test_mode_drive.ipynb` | Pipeline test/debug |
