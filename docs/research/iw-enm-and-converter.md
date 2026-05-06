# IW-ENM ve PairContactConverter Teknik Dokumantasyonu

> Bu dokuman, IW-ENM (Interaction-Weighted Elastic Network Model) MD simulasyon sistemini ve PairContactConverter encoder-decoder mimarisini detayli olarak aciklar.

---

## 1. IW-ENM (Interaction-Weighted Elastic Network Model)

### 1.1 Ne: Elastic Network Uzerinde MD Simulasyonu

IW-ENM, klasik ANM/GNM modellerinin otesinde, **protein yapisinin tam bir Velocity Verlet MD simulasyonunu** elastic network uzerinde calistirir. Her adimda:

1. Network yeniden insa edilir (dinamik topology)
2. Kuvvetler hesaplanir
3. Pozisyonlar ve hizlar guncellenir

Bu, statik normal mode analizinden farkli olarak **anharmonik** ve **nonlinear** hareket orneklemesi saglar.

### 1.2 Neden: ANM'den Daha Gercekci Dinamik Ornekleme

- **ANM/GNM**: Tek bir Hessian'a dayali harmonik modlar. Kucuk deplasman varsayimi.
- **IW-ENM**: Her adimda network rebuild → topology degisir, spring'ler kopsup yenileri olusabilir. Bu, konformasyon degisimlerini (domain motion, hinge bending) yakalayabilir.

Ozellikle **open/closed** gecisleri ve **turning point** tespiti icin idealdir.

### 1.3 Mimari

#### Network Yapisi (`network.py`)

```python
class InteractionWeightedENM:
    """Sidechain interaction-weighted elastic network model."""

    def __init__(self, R_bb=10.0, R_sc=6.0, K_0=1.0, d_0=3.8, n_ref=7.0,
                 use_native_distances=True, use_atomic_packing=False):
```

**Spring constant formulu:**

```
K_ij = K_0 * phi_ij * w_ij * mask_ij
```

Burada:
- `phi_ij = sqrt(n_i * n_j) / n_ref` — sidechain interaction count'a dayali importance weighting
- `w_ij = (V_i + V_j) / (2 * V_ref)` — aminoasit hacim agirligi
- `mask_ij = 1 if d_ij < R_bb else 0` — cutoff filtresi

**Kuvvet hesabi:**

```python
def compute_forces(self, coords, neighbors, K_matrix):
    """F_i = sum_j K_ij * (1 - d0_ij / d_ij) * (r_j - r_i)"""
    dr = coords[None, :, :] - coords[:, None, :]  # (N, N, 3)
    dist = np.linalg.norm(dr, axis=2)              # (N, N)
    d0 = self._get_d0_matrix(dist)                 # native distances
    scalar = K_matrix * (1.0 - d0 / dist)          # (N, N)
    forces = np.sum(scalar[:, :, None] * dr, axis=1)  # (N, 3)
```

**Enerji:**

```
U = 0.5 * sum_{i<j} K_ij * (d_ij - d0_ij)^2
```

#### Hessian ve Equilibrium Distance

Klasik ANM'de Hessian matris analitik olarak cikarilir. IW-ENM'de ise:
- Equilibrium uzakliklari **native CA-CA distances** olarak set edilir: `self._eq_distances = cdist(coords_ca, coords_ca)`
- Her adimda yeni `dist_matrix` hesaplanir ve kuvvetler `(1 - d0/d)` terimi uzerinden turetilir

#### Integrator (`integrator.py`)

**Velocity Verlet** schemasi, damping destegi ile:

```python
class VelocityVerletIntegrator:
    def step(self, coords, velocities, coords_sc, enm, res_names, ...):
        # 1. Mevcut network ve kuvvetler
        forces = enm.compute_forces(coords, neighbors, K_matrix)
        if self.damping > 0:
            forces -= self.damping * velocities

        # 2. Half-step velocity
        v_half = velocities + (dt / (2*m)) * forces

        # 3. Full-step position
        r_new = coords + dt * v_half

        # 4. Sidechain + tum atomlar rigid body olarak hareket
        delta = r_new - coords
        coords_sc_new = coords_sc + delta
        atom_coords_new = atom_coords + delta[atom_res_idx]

        # 5. YENI POZISYONLARDA network rebuild
        forces_new = enm.compute_forces(r_new, neighbors_new, K_new)

        # 6. Full-step velocity
        v_new = v_half + (dt / (2*m)) * forces_new
```

**Velocity initialization modlari:**
- `random`: Rastgele yonler, uniform buyukluk
- `breathing`: Center of mass'tan disa dogru (simetrik genisleme)
- `zero`: Sifir hiz (relaxation only)

#### Importance Weighting

Her residue icin **sidechain interaction count** `n_i` hesaplanir:

```python
def count_interactions(self, coords_sc):
    """CB-CB mesafesi < R_sc olan komsu sayisi."""
    dist_matrix = cdist(coords_sc, coords_sc)
    contact_matrix = (dist_matrix < self.R_sc) & (dist_matrix > 0.0)
    return contact_matrix.sum(axis=1)  # (N,)
```

Alternatif olarak **atomic packing** modu: tum heavy atom'lar uzerinden inter-residue contact sayisi (cKDTree ile).

Spring constant'lar `phi = sqrt(n_i * n_j) / n_ref` ile agirliklandirilir. Boylece **gommulu residue'ler** (core) daha guclu spring'lere, **surface residue'ler** daha zayif spring'lere sahip olur.

### 1.4 Config Parametreleri (`config.py`)

| Parametre | Default | Aciklama |
|-----------|---------|----------|
| `R_bb` | 10.0 A | Backbone (CA-CA) cutoff. Bu mesafe icindeki ciftler spring ile bagli |
| `R_sc` | 6.0 A | Sidechain (CB-CB) cutoff. Interaction count icin |
| `K_0` | 1.0 | Base spring constant. Tum K_ij'ler bununla carpilir |
| `d_0` | 3.8 A | Default equilibrium distance (native distances kullanilmazsa) |
| `n_ref` | 7.0 | Referans interaction sayisi. phi normalizasyonu icin |
| `dt` | 0.01 | Zaman adimi. Kucuk = karali, buyuk = hizli ama patlar |
| `mass` | 1.0 | Uniform atom kutlesi |
| `n_steps` | 10000 | Toplam simulasyon adimi |
| `save_every` | 100 | Her N adimda frame kaydet |
| `damping` | 0.1 | Viscous damping katsayisi. 0 = NVE, >0 = overdamped |
| `v_mode` | "random" | Baslangic hiz modu: random/breathing/zero |
| `v_magnitude` | 1.0 | Hiz buyuklugu |
| `crash_threshold` | 0.5 A | Bu mesafenin altinda atomic clash sayilir |

### 1.5 Turnpoint Detection (`turnpoint.py`)

**Konsept:** IW-ENM trajectory'lerinde `E_total` ve `spring_count` hedef konformasyona yaklasirken AZALIR. Tersine dondukleri an, yapi bozulmaya baslar.

```python
def find_turning_point(values, smooth_w=11, warmup_skip=0.30, min_len=10):
    """
    1. Ilk %30'u skip et (kinetic→potential relaxation transient)
    2. Moving-average smooth (w=11)
    3. Post-warmup bolgenin global minimum'unu bul
    """
    s = _smooth(values, smooth_w)
    skip = max(1, int(n * warmup_skip))
    idx_rel = int(np.argmin(s[skip:]))
    return skip + idx_rel
```

**Best frame secimi:**

```python
def select_best_frame(simulation, back_off=1, weight_crash=0.30, ...):
    # E_total ve spring_count icin ayri turning point bul
    idx_e = find_turning_point(e_tot, ...)
    idx_s = find_turning_point(n_spr, ...)
    # Erken olan tercih edilir
    idx_turn = min(idx_e, idx_s)
    # Composite score: RMSD + crash penalty
    score = rmsd_target + weight_crash * log(1 + crashes_until)
```

### 1.6 Finetune / Surrogate Optimizasyonu

#### Composite Loss (`finetune/loss.py`)

```python
loss = -delta_rmsd                           # RMSD iyilesmesi (buyuk = iyi)
     + alpha * log(1 + max(0, crashes))      # Crash penalty
     + beta  * max(0, e_drift)               # Enerji drift'i (kararsizlik)
     + gamma * log(1 + plateau_std)          # Plateau noise
```

Default agirliklar: `alpha=0.30, beta=0.01, gamma=0.10`

#### Surrogate MLP (`finetune/surrogate.py`)

Grid search sonuclarindan bir **surrogate model** egitilir:

```python
class SurrogateMLP(nn.Module):  # MLX backend
    """D → 64 → 64 → 1, GELU activations."""
    def __init__(self, in_dim=4, hidden=64):
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
```

**Workflow:**
1. Grid search ile `(K_0, dt, damping, R_bb, ...)` → `composite_loss` veri toplama
2. Input normalize: `[0, 1]^D`
3. Output normalize: `(y - mean) / std`
4. MLP egitimi (600 epoch, Adam, lr=3e-3)
5. **Gradient descent through surrogate** → optimal parametreler bulma:

```python
def optimize_params(model, stats, n_starts=128, steps=500, lr=0.02):
    """Random starts → gradient descent → clip to [0,1] → decode back."""
    for _ in range(steps):
        g = grad_fn(p)
        p = p - lr * g
        p = mx.clip(p, 0.0, 1.0)
```

---

## 2. PairContactConverter (Encoder-Decoder)

### 2.1 Mimari Genel Bakis

```
z_pair [B, N, N, 128] ──→ ContactProjectionHead ──→ C [B, N, N]
                                    │
                              (inverse)
                                    │
C [N, N] ──→ logit ──→ w_inv MLP ──→ pseudo_z [N, N, 128]
```

**Toplam parametre sayisi:** 24,896

### 2.2 Encoder: z_pair → contact

```python
class ContactProjectionHead(nn.Module):
    def __init__(self, c_z=128, bottleneck_dim=64):
        # Encoder path
        self.w_enc = nn.Linear(c_z, bottleneck_dim, bias=False)  # [128 → 64]
        self.v = nn.Parameter(torch.randn(bottleneck_dim))       # [64]
```

**Forward path:**

```python
def forward(self, z):  # z: [B, N, N, 128]
    z_sym = 0.5 * (z + z.transpose(1, 2))    # Symmetrise
    h = self.w_enc(z_sym)                      # [B, N, N, 64]
    logits = (h * self.v).sum(dim=-1)          # [B, N, N] dot product
    logits = 0.5 * (logits + logits.T)         # Symmetrise logits
    logits = logits * diag_mask                 # Zero diagonal
    c_pred = torch.sigmoid(logits)             # [B, N, N] ∈ [0, 1]
    return c_pred
```

**Tensor shape akisi:**
```
[B, N, N, 128] → w_enc → [B, N, N, 64] → dot(v) → [B, N, N] → sigmoid → [B, N, N]
```

### 2.3 Decoder (Inverse): contact → z_pair

```python
# Learned inverse MLP
self.w_inv = nn.Sequential(
    nn.Linear(1, bottleneck_dim),    # [1 → 64]
    nn.GELU(),
    nn.Linear(bottleneck_dim, c_z),  # [64 → 128]
)

def inverse(self, c):  # c: [N, N] ∈ (0, 1)
    c_clamped = c.clamp(1e-6, 1.0 - 1e-6)
    logit = torch.log(c_clamped / (1.0 - c_clamped))  # [N, N]
    pseudo_z = self.w_inv(logit.unsqueeze(-1))         # [N, N, 1] → [N, N, 128]
    return pseudo_z
```

**Tensor shape akisi:**
```
[N, N] → logit → [N, N] → unsqueeze → [N, N, 1] → MLP → [N, N, 128]
```

### 2.4 Loss Fonksiyonu (`losses.py`)

Toplam loss **uc bilesenden** olusur:

```
L_total = alpha * L_contact + beta * L_gnm + gamma * L_recon
```

#### L_contact: Focal Loss (veya BCE)

```python
def focal_loss(c_pred, c_gt, seq_sep_min=6, focal_gamma=2.0, focal_alpha=0.75):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    # Sadece |i-j| >= 6 olan ciftler kullanilir
    # focal_gamma=2.0 → zor orneklere odaklan
    # focal_alpha=0.75 → positive class agirlik (contact map ~%10 positive)
```

#### L_gnm: Physics-Informed GNM Loss

```python
def gnm_loss(c_pred, c_gt, n_modes=20):
    # 1. soft_kirchhoff(C) → Kirchhoff matrix Gamma
    # 2. eigh(Gamma) → eigenvalues, eigenvectors, B-factors
    # 3. Compare:
    #    L_eigenvalue = MSE(1/lambda_pred_norm, 1/lambda_gt_norm)
    #    L_bfactor = MSE(B_pred_norm, B_gt_norm)
    #    L_eigvec = (1 - |cos(v_pred, v_gt)|).mean()
    # 4. Gradient: CPU eigh → normalize grad → surrogate loss on GPU
```

#### L_recon: Reconstruction Loss

```python
def reconstruction_loss(z_original, z_reconstructed):
    """Normalised MSE: divide by z_original.std() to keep O(1)."""
    scale = z_original.std().clamp(min=1e-4)
    return F.mse_loss(z_reconstructed / scale, z_original / scale)
```

### 2.5 Training Pipeline

#### `train.py` — Temel egitim dongusu

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-2)
- **Scheduler:** CosineAnnealingLR
- **Grad clipping:** max_norm=1.0
- **Validation metrikler:** adj_accuracy, B-factor Pearson r

#### `scripts/train_large.py` — Production-scale egitim

| Ozellik | Deger |
|---------|-------|
| Optimizer | AdamW, lr=3e-4 |
| Scheduler | OneCycleLR (warmup %5, cosine decay) |
| Gradient accumulation | 4 step |
| Focal loss | gamma=2.0, alpha=0.75 |
| Loss weights | alpha=1.0, beta=0.3, gamma=0.05 |
| Early stopping | patience=50 |
| Epochs | 500 (max) |

#### Notebook (`train_2000.ipynb`) — Colab A100 Egitimi

Gercek production egitimi:
- **Dataset:** 293 protein (MSA-enabled OF3 inference)
- **Shard format:** `.npz`, her shard 10 protein
- **Loss weights:** alpha=0.2, beta=0.5, gamma=2.0
- **Grad accum:** 4
- **OneCycleLR:** lr=3e-4, div_factor=1.0

### 2.6 Dataset: OF3 Trunk Output'larindan Veri Uretimi

```
PDB (RCSB) → OpenFold3 inference (MSA-enabled) → z_pair [N, N, 128] (trunk output)
                                                → coords_ca [N, 3]

coords_ca → compute_gt_probability_matrix() → c_gt [N, N] (sigmoid soft contact)
```

**Shard formati (`ShardedPairReprDataset`):**

```python
# shard_XXXX.npz icerigi:
#   pdb_ids:       list of str
#   pair_repr_0:   [N0, N0, 128] float16/32
#   coords_ca_0:   [N0, 3] float32
#   pair_repr_1:   [N1, N1, 128]
#   coords_ca_1:   [N1, 3]
#   ...
```

Ground truth `c_gt`, training sirasinda **on-the-fly** hesaplanir:
```python
c_gt = compute_gt_probability_matrix(coords_ca, r_cut=8.0, tau=1.0)
```

### 2.7 Egitim Sonuclari

Notebook'tan (293 protein, MSA-enabled, A100):

| Metrik | Deger | Hedef |
|--------|-------|-------|
| Test Loss | 1.9183 | — |
| Adjacency Accuracy | 0.9768 | >= 0.85 PASS |
| B-factor Pearson r | 0.8143 | >= 0.80 PASS |
| Best Epoch | 2 (early converge) | — |
| Training Time | 7.6 min (82 epoch) | — |
| Parameters | 24,896 | — |

**Adenylate Kinase benchmark (1AKE open / 4AKE closed):**
- Inverse path (coords → GT contact → z_pseudo → head → c_pred) basarili calisiyor
- Hem open hem closed konformasyon icin B-factor profilleri yuksek korelasyonla uretiliyor

---

## 3. coords_to_contact

### 3.1 Sigmoid Soft Contact Formulu

```python
def coords_to_contact(coords, r_cut=10.0, tau=1.5):
    """
    C_ij = sigmoid(-(d_ij - r_cut) / tau)
    C_ii = 0
    """
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    c = torch.sigmoid(-(dist - r_cut) / tau)
    c.fill_diagonal_(0.0)
    return c
```

### 3.2 Parametreler

| Parametre | Deger | Aciklama |
|-----------|-------|----------|
| `r_cut` | 10.0 A (veya 8.0) | Sigmoid merkezi. d=r_cut'ta C=0.5 |
| `tau` | 1.5 (veya 1.0) | Sicaklik. Kucuk → keskin gecis (binary'ye yakin). Buyuk → yumusak |

### 3.3 Neden Soft Contact (Differentiable)

Hard contact (`C = 1 if d < r_cut else 0`) kullanilmaz cunku:

1. **Gradient yok:** Step fonksiyonunun turevi sifir. Backprop calismaz.
2. **Kirchhoff eigendecomposition:** `soft_kirchhoff(C)` uzerinden GNM loss hesaplanir. C'nin surekli ve turevlenebilir olmasi gerekir.
3. **Mode-drive pipeline:** Deplase koordinatlar → yeni contact → yeni z_pair zincirinde gradient akisi gereklidir.

Sigmoid formulu sayesinde:
```
dC/dd = C * (1 - C) * (1/tau)
```

Bu, `r_cut` civarinda en buyuk gradient'i verir ve training sirasinda network'un **boundary contacts** (d ~ r_cut) uzerinde en cok ogrendigi bolgeyi tanimlar.

---

## Dosya Haritasi

| Dosya | Rol |
|-------|-----|
| `src/iw_enm/config.py` | Simulasyon parametreleri |
| `src/iw_enm/structure.py` | PDB/CIF parsing, CA/CB/atom extraction |
| `src/iw_enm/network.py` | Spring network build + force/energy hesabi |
| `src/iw_enm/integrator.py` | Velocity Verlet + damping |
| `src/iw_enm/simulation.py` | Ana simulasyon dongusu + crash tracking |
| `src/iw_enm/turnpoint.py` | Konformasyon degisim noktasi tespiti |
| `src/iw_enm/analysis.py` | RMSD, RMSF, TM-score, Kabsch alignment |
| `src/iw_enm/finetune/loss.py` | Composite scoring (delta_rmsd + penalties) |
| `src/iw_enm/finetune/surrogate.py` | MLX surrogate MLP + gradient optimization |
| `src/iw_enm/finetune/runner.py` | One-shot simulation builder |
| `src/contact_head.py` | ContactProjectionHead encoder-decoder |
| `src/converter.py` | PairContactConverter wrapper + GNM analysis |
| `src/losses.py` | Focal + GNM + Reconstruction loss |
| `src/train.py` | Basic training loop |
| `scripts/train_large.py` | Production training with OneCycleLR |
| `src/coords_to_contact.py` | Differentiable soft contact map |
| `src/ground_truth.py` | GT contact map from CA coordinates |
| `src/data.py` | ShardedPairReprDataset |
