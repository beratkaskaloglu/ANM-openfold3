# GNM-Contact Learner: Kod İncelemesi, Matematik Doğrulaması ve Eğitim Planı

> Bu doküman mevcut implementasyonu modül modül inceler, matematiksel
> fonksiyonları formülleriyle doğrular, ve eğitimin nerede/nasıl
> yapılacağını planlar.

---

## 1. Modül Haritası ve Veri Akışı

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
│                                                                     │
│  PDB file                                                           │
│    │                                                                │
│    ├──→ data.py ──→ coords_ca [N,3]                                 │
│    │                    │                                            │
│    │                    ▼                                            │
│    │              ground_truth.py                                    │
│    │                    │                                            │
│    │                    ▼                                            │
│    │              C_gt [N,N]  ◄── sigmoid(-(d-10)/1.5)              │
│    │                                                                │
│    └──→ OpenFold3 trunk (frozen)                                    │
│              │                                                      │
│              ▼                                                      │
│         z [B,N,N,128]  (pair representation)                        │
│              │                                                      │
│              ▼                                                      │
│         contact_head.py (trainable, ~8K params)                     │
│              │                                                      │
│              ├──→ FORWARD:  z → W_enc → h[32] → v·h → σ → C_pred   │
│              │                                                      │
│              └──→ RECON:    z → W_enc → h[32] → W_dec → z_recon     │
│                                                                     │
│         losses.py                                                   │
│              │                                                      │
│              ├── L_contact = BCE(C_pred, C_gt)  (|i-j|≥6 only)     │
│              │                                                      │
│              ├── L_gnm = kirchhoff → eigh → compare spectra         │
│              │     ├── L_eigenvalue: MSE(norm 1/λ)                  │
│              │     ├── L_bfactor:   MSE(norm B)                     │
│              │     └── L_eigvec:    1-|cos(V_pred, V_gt)|           │
│              │                                                      │
│              └── L_recon = MSE(z_sym, z_recon)                      │
│                                                                     │
│         L_total = α·L_contact + β·L_gnm + γ·L_recon                │
│                                                                     │
│         train.py: AdamW → clip_grad(1.0) → CosineAnnealing         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INVERSE PIPELINE                             │
│                        (eğitim sonrası)                             │
│                                                                     │
│  coords [N,3] → dist → sigmoid → C [N,N]                           │
│                                      │                              │
│                                      ▼                              │
│                               inverse.py                            │
│                     logit(C) → ×v_norm → W_dec → pseudo_z [N,N,128]│
│                                                                     │
│  Sonuç: Herhangi bir protein koordinatından,                        │
│  OpenFold3 pair representation uzayında bir tensor.                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Modül-Modül Matematik Doğrulaması

### 2.1 ground_truth.py — Soft Contact Map

**Formül:**
```
d_ij = ||r_i - r_j||₂          (Euclidean distance, Å)

C_gt[i,j] = σ(-(d_ij - r_cut) / τ)
           = 1 / (1 + exp((d_ij - r_cut) / τ))

C_gt[i,i] = 0                   (self-contact tanımsız)
```

**Sayısal kontrol (r_cut=10.0, τ=1.5):**

| d_ij (Å) | (d-10)/1.5 | exp(...)  | C_gt      | Yorum            |
|-----------|-----------|-----------|-----------|------------------|
| 3.8       | -4.13     | 0.016     | **0.984** | Yakın komşu      |
| 7.0       | -2.00     | 0.135     | **0.881** | Kısa-menzil      |
| 10.0      | 0.00      | 1.000     | **0.500** | Cutoff noktası   |
| 15.0      | +3.33     | 28.03     | **0.034** | Uzak              |
| 30.0      | +13.33    | 6.2×10⁵  | **~0**    | Çok uzak          |

**Test durumu:** 46 testten 7'si bu modülü doğruluyor.
- Diagonal = 0 ✓
- Symmetry ✓
- [0,1] range ✓
- d=10Å → C ≈ 0.5 ✓
- d < r_cut → C > 0.5 ✓
- d ≫ r_cut → C ≈ 0 ✓

**Durum: DOĞRU ✓**

---

### 2.2 kirchhoff.py — Kirchhoff Matrix

**Formül:**
```
Γ[i,j] = -C[i,j]             (i ≠ j, off-diagonal)
Γ[i,i] = Σ_j C[i,j]          (diagonal = coordination number)
Γ += ε·I                      (ε = 1e-6, numerical stability)
```

**Özellikler (doğrulanmış):**
- `Γ` simetriktir (C simetrik → Γ simetrik) ✓
- Satır toplamları = ε (eps=0 iken tam 0) ✓
- Pozitif yarı-belirli (PSD): tüm eigenvalues ≥ 0 ✓
- `Γ` Graph Laplacian'dır: L = D - A, burada D=diag(row_sums), A=C ✓

**gnm_decompose:**
```
eigh(Γ) → λ₀ ≤ λ₁ ≤ ... ≤ λ_{N-1}

λ₀ ≈ ε  (trivial mode, skip)
λ₁ ... λ_k  (non-trivial modes, kullanılır)

B_i = Σ_{k=1}^{n_modes} V_ik² / λ_k    (B-factor, flexibility)
```

**Gradient akışı (kritik):**
```
L_total → backward → eigh gradient → Kirchhoff gradient → C_pred gradient → W_enc, v gradient

eigh backward: ∂L/∂Γ = V · (F ⊙ (Vᵀ · ∂L/∂V)) · Vᵀ + diag(∂L/∂λ)
burada F[i,j] = 1/(λ_j - λ_i)  (i≠j)

Tehlike: λ_i ≈ λ_j → F[i,j] → ∞ → NaN
Çözüm: ε·I regularizasyonu eigenvalue'ları ayırır
```

**Test durumu:** 5 test
- Eigenvalues > 0 ✓
- B-factors > 0 ✓
- n_modes > N-1 → otomatik clamp ✓
- Gradient C'ye akar, NaN yok ✓

**Durum: DOĞRU ✓**

---

### 2.3 contact_head.py — Invertible Bottleneck

**Forward path (training):**
```
z_sym = 0.5·(z + zᵀ)                    [B,N,N,128]  symmetrize
h = z_sym @ W_enc                         [B,N,N,32]   encode
logits = Σ_k (h_k · v_k)                  [B,N,N]      dot product
logits = 0.5·(logits + logitsᵀ)           [B,N,N]      symmetrize
logits[i,i] = 0                            [B,N,N]      mask diagonal
C_pred = sigmoid(logits) · diag_mask       [B,N,N]      probability
```

**Inverse path (inference sonrası):**
```
logit_ij = log(C_ij / (1 - C_ij))         [N,N]        sigmoid⁻¹
v_norm = v / ||v||                          [32]         normalize
h_approx = logit_ij · v_norm               [N,N,32]     rank-1 broadcast
pseudo_z = h_approx @ W_dec                [N,N,128]    decode
```

**Parametreler:**
```
W_enc:  128 × 32 = 4,096
v:      32            = 32
W_dec:  32 × 128  = 4,096
Toplam:             8,224
```

**Bilinen sınırlama:** Inverse path rank-1 yaklaşımdır.
`h_approx = logit · v_norm` her (i,j) çifti için h vektörünü v
yönünde zorlar. Bu, orijinal h'nin v yönündeki bileşenini kurtarır
ama diğer 31 boyutu kaybeder. Bu nedenle L_recon loss'u önemli —
W_enc·W_dec çarpımını yaklaşık identity yapmaya zorlar.

**Test durumu:** 11 test
- Shape, symmetry, diagonal=0, [0,1] range ✓
- Gradient akışı ✓
- Inverse shape, symmetry, no_grad ✓
- Param count = 8224 ✓

**Durum: DOĞRU ✓**

---

### 2.4 losses.py — Composite Loss

**L_contact: Weighted Binary Cross-Entropy**
```
mask = {(i,j) : |i-j| ≥ 6}           sequence separation filter
L_contact = -1/|mask| · Σ_{(i,j)∈mask} [
    C_gt · log(C_pred) + (1-C_gt) · log(1-C_pred)
]
```
Neden |i-j| ≥ 6? Ardışık residue'ler (i±1, i±2, ...) her zaman yakın,
trivial bilgi. Long-range contacts bilgilendirici.

**L_gnm: Physics-Informed Loss (3 bileşen)**

```
[1] L_eigenvalue:
    inv_pred = 1/λ_pred,  inv_gt = 1/λ_gt       (her biri [n_modes])
    norm_pred = inv_pred / Σ(inv_pred)             normalize
    norm_gt   = inv_gt   / Σ(inv_gt)              normalize
    L_eig = MSE(norm_pred, norm_gt)

[2] L_bfactor:
    B_pred = Σ_k V_pred_ik² / λ_pred_k            [N]
    B_gt   = Σ_k V_gt_ik²   / λ_gt_k              [N]
    B_pred_norm = B_pred / max(B_pred)              normalize
    B_gt_norm   = B_gt   / max(B_gt)               normalize
    L_bf = MSE(B_pred_norm, B_gt_norm)

[3] L_eigvec:
    cos_k = |cos(V_pred_k, V_gt_k)|               phase-invariant
    L_vec = mean(1 - cos_k)                        over k modes

L_gnm = 1.0·L_eig + 1.0·L_bf + 0.5·L_vec
```

**L_recon: Autoencoder Reconstruction**
```
z_sym = 0.5·(z + zᵀ)                               [B,N,N,128]
z_recon = W_dec(W_enc(z_sym))                        [B,N,N,128]
L_recon = MSE(z_sym, z_recon)
```
Bu loss, W_enc·W_dec'in subspace projection olarak tutarlı kalmasını
sağlar. Inverse path'in kalitesini doğrudan etkiler.

**Total:**
```
L_total = α·L_contact + β·L_gnm + γ·L_recon

Defaults: α=1.0, β=0.5, γ=0.1
```

**Test durumu:** 14 test
- Identical inputs → loss ≈ 0 ✓
- Gradient eigh'den akar ✓
- beta=0 → GNM ignore ✓
- Reconstruction loss doğru ✓

**Durum: DOĞRU ✓**

---

## 3. Test Özeti

```
tests/test_ground_truth.py    7 tests  ✓  contact map doğruluğu
tests/test_kirchhoff.py       5 tests  ✓  Kirchhoff + GNM + gradient
tests/test_contact_head.py   11 tests  ✓  forward + inverse + bottleneck
tests/test_losses.py         14 tests  ✓  BCE + GNM + recon + total
tests/test_inverse.py         5 tests  ✓  coords → pseudo pair repr
─────────────────────────────────────────
TOTAL                        46 tests  ALL PASS
```

---

## 4. Eğitim: Nerede ve Nasıl?

### 4.1 Opsiyonlar

| Platform | GPU | VRAM | Maliyet | Uygunluk |
|----------|-----|------|---------|----------|
| **Lokal (M-series Mac)** | Apple Silicon MPS | 16-96 GB shared | $0 | Prototyping, küçük dataset |
| **Google Colab Free** | T4 | 15 GB | $0 | Deneme, tek PDB |
| **Google Colab Pro** | A100/L4 | 40-80 GB | ~$10/ay | Orta dataset |
| **Lambda/Vast.ai** | A100 80GB | 80 GB | ~$1-2/saat | Tam eğitim |
| **Üniversite HPC** | Varies | Varies | $0 (akademik) | En iyi seçenek |

### 4.2 Memory Analizi

```
Protein boyutu N=200 residue için:

pair_repr z:           B × 200 × 200 × 128 × 4 bytes = ~20 MB/sample
C_pred, C_gt:          200 × 200 × 4 bytes = ~0.16 MB
Kirchhoff Γ:           200 × 200 × 4 bytes = ~0.16 MB
eigh workspace:        ~N² × 8 bytes = ~0.3 MB
ContactHead params:    8,224 × 4 bytes = ~33 KB

OpenFold3 trunk:       ~600M params = ~2.4 GB (frozen, inference only)
OpenFold3 inference:   peak ~8-16 GB (depends on N)
```

**Sonuç:** OpenFold3 trunk GPU'da en az 16 GB VRAM gerektirir.
Ama pair repr'ı önceden cache'leyebiliriz (Strateji B).

### 4.3 Eğitim Stratejileri

#### Strateji A: End-to-End (OpenFold3 + ContactHead)

```
Her batch'te:
  1. OpenFold3 trunk forward (no_grad, ~8 GB VRAM, ~30 sec/sample)
  2. ContactHead forward + loss + backward (~100 MB, ~50 ms)

Gereksinimler:
  - GPU: A100 40GB+ önerilir
  - OpenFold3 checkpoint gerekli
  - Çok yavaş: trunk inference her batch'te tekrar

Kullanım durumu: Final eğitim, az sample
```

#### Strateji B: Pair Repr Caching (Önerilen başlangıç)

```
Aşama 1 — Cache oluştur (bir kez):
  for pdb in dataset:
      z = openfold3.run_trunk(pdb)         # GPU'da
      torch.save(z, f"cache/{pdb_id}.pt")  # disk'e

Aşama 2 — ContactHead eğitimi (hızlı, CPU/MPS yeterli):
  for batch in loader:
      z = load_cached(batch)      # disk'ten
      C_pred = contact_head(z)    # çok hızlı, ~8K param
      loss = total_loss(C_pred, C_gt, z_sym, z_recon)
      loss.backward()             # sadece 8K param'a gradient

Cache boyutları:
  N=200: ~20 MB/sample
  1000 sample: ~20 GB disk
  10000 sample: ~200 GB disk
```

**Strateji B avantajları:**
- OpenFold3 trunk'ı sadece 1 kez çalıştırırsın
- Eğitim laptop'ta bile yapılabilir (MPS/CPU)
- Hızlı iterasyon (epoch ~saniyeler)
- Hyperparameter search kolaylaşır

### 4.4 Önerilen Eğitim Planı

```
Aşama 0: Sanity Check (bugün, lokal)
─────────────────────────────────────
  - Sentetik veri ile overfit testi
  - 1 protein, 50 epoch, loss → 0 olmalı
  - Gradient akışı doğrulama
  - MPS veya CPU yeterli

Aşama 1: Küçük Dataset (lokal veya Colab)
──────────────────────────────────────────
  - 10-50 PDB structure
  - Pair repr cache oluştur (OpenFold3 gerekli, GPU)
  - ContactHead eğitimi (cached, CPU/MPS)
  - Hyperparameter tuning: α, β, γ, k, n_modes
  - Hedef: B-factor Pearson r > 0.5

Aşama 2: Orta Dataset (Colab Pro veya HPC)
───────────────────────────────────────────
  - 500-1000 PDB (resolution < 2.5Å, X-ray)
  - Pair repr cache (~10-20 GB)
  - Daha uzun eğitim: 100 epoch
  - WandB logging aktif
  - Hedef: B-factor r > 0.7, adj accuracy > 0.8

Aşama 3: Tam Dataset (HPC veya Cloud)
──────────────────────────────────────
  - 5000-10000 PDB
  - Time-split test set (post-2022)
  - Ablasyon çalışmaları
  - Hedef: Yayın kalitesi sonuçlar
```

### 4.5 Aşama 0 — Sentetik Overfit Test (Hemen Çalıştırılabilir)

```python
# Bu test OpenFold3 gerektirmez — sentetik pair repr ile çalışır

import torch
from src.contact_head import ContactProjectionHead
from src.ground_truth import compute_gt_probability_matrix
from src.losses import total_loss

# Sentetik protein: 30 residue, düz çizgi
coords = torch.zeros(30, 3)
coords[:, 0] = torch.arange(30, dtype=torch.float32) * 3.8

# Ground truth
C_gt = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)

# Sentetik pair repr (random, ama sabit)
torch.manual_seed(42)
z_fake = torch.randn(1, 30, 30, 128)

# Model (sadece ContactHead, OpenFold3 yok)
head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

# Overfit: 50 epoch, tek sample
for epoch in range(50):
    C_pred = head(z_fake)
    z_sym = 0.5 * (z_fake + z_fake.transpose(1, 2))
    h = head.w_enc(z_sym)
    z_recon = head.w_dec(h)

    loss, details = total_loss(
        C_pred.squeeze(0), C_gt,
        z_original=z_sym, z_reconstructed=z_recon,
        alpha=1.0, beta=0.5, gamma=0.1, n_modes=10
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f} "
              f"L_contact={details['L_contact']:.4f} "
              f"L_gnm={details['L_gnm']:.4f} "
              f"L_recon={details['L_recon']:.4f}")

# Beklenen: loss monoton azalır, NaN yok
```

---

## 5. Kritik Kontrol Noktaları

| # | Kontrol | Durum | Not |
|---|---------|-------|-----|
| 1 | Ground truth sigmoid doğru | ✓ | d=10→0.5, d<10→>0.5, d>10→<0.5 |
| 2 | Kirchhoff satır toplamı = 0 | ✓ | eps=0 ile test edildi |
| 3 | Kirchhoff PSD | ✓ | Tüm eigenvalues ≥ 0 |
| 4 | eigh gradient akar | ✓ | C.grad is not None, no NaN |
| 5 | eps regularizasyon | ✓ | Degenerate eigenvalue koruması |
| 6 | Trivial mode skip | ✓ | index 0 atlanıyor |
| 7 | Eigenvector sign ambiguity | ✓ | abs(cosine_similarity) |
| 8 | Eigenvalue normalizasyon | ✓ | sum-normalize inverse eigenvalues |
| 9 | B-factor normalizasyon | ✓ | max-normalize |
| 10 | Sequence separation | ✓ | \|i-j\| ≥ 6 filter |
| 11 | ContactHead symmetry | ✓ | Input + output symmetrize |
| 12 | ContactHead diagonal = 0 | ✓ | Mask + re-zero after sigmoid |
| 13 | No NumPy in forward | ✓ | Pure PyTorch, gradient korunur |
| 14 | Inverse path çalışır | ✓ | Shape, symmetry, no_grad |
| 15 | Reconstruction loss | ✓ | MSE(z, W_dec(W_enc(z))) |

---

## 6. Hyperparameter Rehberi

| Parametre | Default | Açıklama | Tuning notu |
|-----------|---------|----------|-------------|
| `c_z` | 128 | Pair repr boyutu (OF3 sabit) | Değiştirme |
| `bottleneck_dim` | 32 | Darboğaz boyutu | 16-64 dene |
| `r_cut` | 10.0 Å | GNM cutoff | 7-15 Å, standart=10 |
| `tau` | 1.5 | Sigmoid keskinliği | 0.5-3.0 |
| `n_modes` | 20 | GNM mod sayısı | 5-50, N'ye bağlı |
| `alpha` | 1.0 | Contact loss ağırlığı | Sabit tut |
| `beta` | 0.5 | GNM loss ağırlığı | 0.1-2.0 |
| `gamma` | 0.1 | Recon loss ağırlığı | 0.01-0.5 |
| `seq_sep_min` | 6 | Min sequence separation | 4-12 |
| `lr` | 1e-4 | Learning rate | 1e-5 - 1e-3 |
| `weight_decay` | 1e-2 | AdamW regularization | 1e-3 - 1e-1 |
| `epochs` | 100 | Epoch sayısı | Dataset boyutuna göre |

---

## 7. Sonraki Adımlar

```
[  ] Aşama 0: Sentetik overfit test çalıştır (yukarıdaki kod)
[  ] OpenFold3 checkpoint edinme (model weights)
[  ] 10 PDB download + Cα extraction test
[  ] Pair repr cache pipeline oluştur
[  ] Aşama 1: Küçük dataset eğitimi
[  ] WandB entegrasyonu test
[  ] Ablasyon: L_contact only vs L_contact + L_gnm
[  ] Aşama 2: Orta dataset + sonuç analizi
```

---

#gnm #training #architecture #review #mathematics
