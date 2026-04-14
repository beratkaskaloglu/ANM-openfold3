# Training of Trunk Vector: GNM-Contact Learner Pipeline

## Overview

We train a lightweight **ContactProjectionHead** (8.2K parameters) on top of OpenFold3's frozen pair representation $z_{ij} \in \mathbb{R}^{N \times N \times 128}$ to predict protein contact maps. The model is supervised by a physics-informed loss combining **focal contact loss** and **Gaussian Network Model (GNM)** eigendecomposition objectives.

```
┌──────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  PDB Sequence ──→ OpenFold3 (frozen) ──→ z_ij [N,N,128] │
│                                              │            │
│                                              ▼            │
│                                   ContactProjectionHead   │
│                                   (trainable, 8.2K)      │
│                                              │            │
│                                              ▼            │
│                                     C_pred [N,N] ∈[0,1]  │
│                                              │            │
│                              ┌───────────────┼───────┐   │
│                              ▼               ▼       ▼   │
│                         L_contact       L_gnm    L_recon │
│                              └───────────────┼───────┘   │
│                                              ▼            │
│                              L_total = α·Lc + β·Lg + γ·Lr│
└──────────────────────────────────────────────────────────┘
```

---

## 1. Input: OpenFold3 Pair Representation

OpenFold3'un trunk modülü, verilen protein sekansı için bir **pair representation** üretir:

$$z_{ij} \in \mathbb{R}^{N \times N \times c_z}, \quad c_z = 128$$

Bu tensor, rezidü çifti $(i, j)$ arasındaki yapısal ve evrimsel bilgiyi kodlar. Trunk ağırlıkları **frozen** — sadece downstream head eğitilir.

### Extraction

```python
# OpenFold3 inference
runner = InferenceExperimentRunner(config, ...)
runner.setup()
runner.run(query_set)

# Latent output'tan pair repr
latent = torch.load("*_latent_output.pt")
z_ij = latent["pair_repr"]  # [1, N, N, 128]
```

---

## 2. Ground Truth: Soft Contact Map

Cα koordinatlarından sigmoid-temelli yumuşak contact map hesaplanır:

$$C_{gt}(i,j) = \sigma\!\left(\frac{-(d_{ij} - r_{cut})}{\tau}\right)$$

burada:
- $d_{ij} = \|x_i^{C\alpha} - x_j^{C\alpha}\|_2$ — Cα mesafesi (Å)
- $r_{cut} = 8.0$ Å — cutoff merkezi ($d = r_{cut}$ → $C = 0.5$)
- $\tau = 1.0$ — sıcaklık (keskinlik kontrolü)
- $\sigma(x) = 1/(1 + e^{-x})$ — sigmoid fonksiyonu
- $C_{gt}(i,i) = 0$ — diagonal sıfır

### Sigmoid Profile

| $d_{ij}$ (Å) | $\tau = 0.5$ | $\tau = 1.0$ | $\tau = 1.5$ |
|:---:|:---:|:---:|:---:|
| 5.0 | 0.997 | 0.953 | 0.881 |
| 8.0 | 0.500 | 0.500 | 0.500 |
| 11.0 | 0.003 | 0.047 | 0.119 |

```python
def compute_gt_probability_matrix(coords_ca, r_cut=8.0, tau=1.0):
    dist = torch.cdist(coords_ca, coords_ca)  # [N, N]
    c_gt = torch.sigmoid(-(dist - r_cut) / tau)
    c_gt.fill_diagonal_(0.0)
    return c_gt
```

---

## 3. Model: ContactProjectionHead

Minimal invertible bottleneck:

```
┌─────────────────────────────────────────────────┐
│           ContactProjectionHead                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  ENCODER (Forward Path):                        │
│                                                  │
│    z_sym = 0.5·(z + zᵀ)        [N, N, 128]    │
│         │                                        │
│         ▼                                        │
│    h = W_enc(z_sym)             [N, N, 64]     │
│         │       W_enc: Linear(128, 64)         │
│         ▼                                        │
│    logits = h · v               [N, N]          │
│         │       v: Parameter(64)               │
│         ▼                                        │
│    logits = 0.5·(logits + logitsᵀ)            │
│    logits[i,i] = 0                              │
│         │                                        │
│         ▼                                        │
│    c_pred = σ(logits)           [N, N] ∈[0,1]  │
│                                                  │
│  DECODER (Inverse Path):                        │
│                                                  │
│    logit = log(c / (1-c))       [N, N]          │
│         │                                        │
│         ▼                                        │
│    h_approx = logit · v̂ᵀ       [N, N, 64]     │
│         │                                        │
│         ▼                                        │
│    pseudo_z = W_dec(h_approx)   [N, N, 128]    │
│                W_dec: Linear(64, 128)           │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Parameters

| Layer | Shape | #Params |
|:---:|:---:|:---:|
| $W_{enc}$ | $128 \times 64$ | 8,192 |
| $v$ (contact vector) | $64$ | 64 |
| $W_{dec}$ | $64 \times 128$ | 8,192 |
| **Total** | | **~16,448** |

### Key Properties
- **Symmetric output**: $C_{pred} = C_{pred}^T$ (enforced)
- **Zero diagonal**: $C_{pred}(i,i) = 0$ (self-contact undefined)
- **Bounded**: $C_{pred} \in [0, 1]$ (sigmoid)
- **Invertible**: decoder path allows $C \to \hat{z}$ reconstruction

---

## 4. Loss Functions

### 4.1 Total Loss

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{contact} + \beta \cdot \mathcal{L}_{gnm} + \gamma \cdot \mathcal{L}_{recon}$$

| Weight | Default | Purpose |
|:---:|:---:|:---|
| $\alpha$ | 1.0 | Contact map accuracy |
| $\beta$ | 0.3 | Physics-informed dynamics |
| $\gamma$ | 0.05 | Bottleneck regularization |

---

### 4.2 Contact Loss (Focal)

Standard BCE, contact map'lerde pozitif sınıf azınlıkta (~10%) olduğu için **focal loss** ile ağırlıklandırılır:

$$\mathcal{L}_{focal} = -\frac{1}{|\mathcal{M}|} \sum_{(i,j) \in \mathcal{M}} \alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

burada:
- $\mathcal{M} = \{(i,j) : |i-j| \geq 6\}$ — sequence separation filtresi
- $p_t = C_{pred}$ if $C_{gt} = 1$, else $1 - C_{pred}$
- $\alpha_t = 0.75$ if $C_{gt} = 1$, else $0.25$
- $\gamma_{focal} = 2.0$ — focusing parameter

**Sequence separation**: Yakın komşular ($|i-j| < 6$) trivially contact'ta olduğu için loss'a dahil edilmez.

```python
# Pseudocode
mask = |i - j| >= 6
bce = -[y·log(p) + (1-y)·log(1-p)]
focal_weight = α_t · (1 - p_t)^γ
L_contact = mean(focal_weight · bce)[mask]
```

---

### 4.3 GNM Loss (Physics-Informed)

Predicted ve ground-truth contact map'lerden **Kirchhoff matrisi** oluşturulur ve GNM eigendecomposition karşılaştırılır.

#### Step 1: Kirchhoff Matrix

$$\Gamma_{ij} = \begin{cases} -C_{ij} & i \neq j \\ \sum_k C_{ik} & i = j \end{cases} + \varepsilon I$$

- Simetrik, pozitif yarı-tanımlı
- İlk eigenvalue $\lambda_0 \approx 0$ (translational mode)
- $\varepsilon = 10^{-6}$ (numerical stability)

#### Step 2: Eigendecomposition

$$\Gamma = V \Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_N)$$

İlk 20 non-trivial mode ($\lambda_1 ... \lambda_{20}$, $\lambda_0$ skip edilir):

#### Step 3: B-factors

$$B_i = \sum_{k=1}^{K} \frac{V_{ik}^2}{\lambda_k}$$

Rezidü fleksibilitesinin ölçüsü — deneysel B-factor'larla koreleli.

#### GNM Loss Components

$$\mathcal{L}_{gnm} = w_{eig} \cdot L_{eig} + w_{bf} \cdot L_{bf} + w_{vec} \cdot L_{vec}$$

| Component | Formula | Weight |
|:---|:---|:---:|
| Eigenvalue | $L_{eig} = \text{MSE}\!\left(\frac{1/\lambda^{pred}}{\sum 1/\lambda^{pred}}, \frac{1/\lambda^{gt}}{\sum 1/\lambda^{gt}}\right)$ | 1.0 |
| B-factor | $L_{bf} = \text{MSE}\!\left(\frac{B^{pred}}{B^{pred}_{max}}, \frac{B^{gt}}{B^{gt}_{max}}\right)$ | 1.0 |
| Eigenvector | $L_{vec} = \text{mean}\!\left(1 - |\cos(V^{pred}_k, V^{gt}_k)|\right)$ | 0.5 |

> **Note:** Eigenvector loss'ta $|\cos|$ kullanılır çünkü eigenvector'lar $\pm$ sign ambiguity'ye sahip.

```python
# CPU'da çalışır (CUSOLVER hataları için)
vals, vecs = torch.linalg.eigh(gamma.cpu())

# İlk 20 non-trivial mode
eig_vals = vals[1:21]      # skip λ₀ ≈ 0
eig_vecs = vecs[:, 1:21]   # [N, 20]

# B-factors
b_factors = (eig_vecs ** 2) @ (1.0 / eig_vals)  # [N]
```

---

### 4.4 Reconstruction Loss

Bottleneck'in bilgi kaybını minimize eder:

$$\mathcal{L}_{recon} = \text{MSE}(z_{sym}, W_{dec}(W_{enc}(z_{sym})))$$

---

## 5. Optimization

### AdamW

$$\theta_{t+1} = \theta_t - \eta_t \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \cdot \theta_t\right)$$

| Parameter | Value |
|:---|:---:|
| Learning rate ($\eta$) | $3 \times 10^{-4}$ |
| Weight decay ($\lambda$) | $10^{-2}$ |
| Gradient clip | 1.0 (max norm) |
| Gradient accumulation | 4 steps |

### OneCycleLR Schedule

```
    η
    ↑  max_lr = 3e-4
    │     /\
    │    /  \
    │   /    \_______ cosine decay
    │  / warm \
    │ /  up    \
    ├──────────────────→ epoch
    │ 5%        95%
    │
    └─ initial_lr = max_lr/10 = 3e-5
       final_lr = max_lr/1000 = 3e-7
```

---

## 6. Data Pipeline

### 6.1 Extraction Phase

```
┌─────────────────────────────────────────────────────────┐
│  For each protein in pdb_2000.json:                      │
│                                                          │
│  1. Write query JSON (sequence → OpenFold3 format)      │
│  2. OpenFold3 inference → pair_repr [1, N, N, 128]     │
│  3. Download PDB → extract Cα coords [N, 3]            │
│  4. Save to .pt files                                    │
│                                                          │
│  Every 10 proteins:                                      │
│  5. Pack .pt → .npz shard                               │
│  6. Delete .pt files (disk savings)                      │
│  7. Write .ok marker (resume-safe)                       │
│  8. Train 30 epochs on accumulated data                  │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Shard Format (.npz)

```python
{
    "pdb_ids":      np.array(["3NIR", "5D8V", ...]),  # [K]
    "pair_repr_0":  np.float32 [N₀, N₀, 128],
    "coords_ca_0":  np.float32 [N₀, 3],
    "pair_repr_1":  np.float32 [N₁, N₁, 128],
    "coords_ca_1":  np.float32 [N₁, 3],
    ...
}
```

### 6.3 Dataset Loading

```python
class ShardedPairReprDataset(Dataset):
    """
    Lazy-loads one shard at a time.
    Variable protein sizes handled per-protein (no padding).
    Ground truth computed on-the-fly from coords.
    """
    def __getitem__(self, idx):
        pair_repr = shard[f"pair_repr_{i}"]  # [N, N, 128]
        coords_ca = shard[f"coords_ca_{i}"]  # [N, 3]
        c_gt = sigmoid_contact(coords_ca, r_cut=8.0, tau=1.0)
        return {"pair_repr": pair_repr, "c_gt": c_gt}
```

---

## 7. Training Configuration

### Incremental Training Loop

Her 10 protein'den sonra model güncellenir — **online/incremental learning**:

```
Session 1:  [Batch 0] → Train 30 ep → [Batch 1] → Train 30 ep → ...
Session 2:  Resume from checkpoint → [Batch K] → Train 30 ep → ...
```

| Parameter | Value | Notes |
|:---|:---:|:---|
| Batch size (proteins) | 10 | Per inference round |
| Train epochs/batch | 30 | Short bursts |
| Total proteins | 2000 | ~200 batches |
| Val fraction | 10% | Random split |
| Test fraction | 10% | Held out |
| Early stopping patience | 999 | Disabled in incremental mode |

### Resume Strategy

```
Google Drive (persistent):
  ├── checkpoints/latest.pt    ← model + optimizer state
  └── progress/shard_XXXX.ok   ← which batches are done

Local disk (ephemeral):
  ├── /content/shards/         ← current .npz files
  ├── /content/pair_reprs/     ← temp .pt files
  └── /content/coords/         ← temp .pt files
```

---

## 8. Evaluation Metrics

| Metric | Formula | Target |
|:---|:---|:---:|
| **Adjacency Accuracy** | $\frac{1}{N^2}\sum_{ij} \mathbb{1}[\hat{C}_{ij} > 0.5] == \mathbb{1}[C^{gt}_{ij} > 0.5]$ | > 0.85 |
| **B-factor Pearson r** | $\text{corr}(B^{pred}, B^{gt})$ | > 0.7 |
| **Contact Loss** | Focal BCE | < 0.2 |
| **GNM Loss** | Kirchhoff spectral distance | < 0.1 |

---

## 9. Inverse Path: Generating Pair Representations

Trained model'in **decoder** yolu, contact map'ten pseudo pair representation üretebilir:

$$C_{ij} \xrightarrow{\text{logit}} \xrightarrow{\times \hat{v}^T} h \xrightarrow{W_{dec}} \hat{z}_{ij}$$

Bu, **herhangi bir proteinin** koordinatlarından (PDB dosyasından) pair representation üretmeyi sağlar — OpenFold3 inference'a gerek kalmadan:

```python
# Kullanım
coords_ca = extract_ca_coords("protein.pdb")  # [N, 3]
c_gt = compute_gt_probability_matrix(coords_ca, r_cut=8.0)
pseudo_z = trained_head.inverse(c_gt)  # [N, N, 128]
```

---

## 10. Computational Budget

| Phase | Time/protein | Hardware |
|:---|:---:|:---:|
| OpenFold3 inference | ~100s | A100 40GB |
| Coord download | ~2s | Network |
| Training (30 ep) | ~6s | A100 |
| **Total per batch (10)** | **~17 min** | |
| **2000 proteins** | **~56 hours** | ~5 sessions |

---

## References

1. Bahar, I., Atilgan, A.R., Erman, B. (1997). Direct evaluation of thermal fluctuations in proteins using a single-parameter harmonic potential. *Folding and Design*, 2(3), 173-181.
2. OpenFold3: Abramson et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3.
3. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
