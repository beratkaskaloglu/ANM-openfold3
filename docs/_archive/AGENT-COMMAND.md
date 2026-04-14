# Implementation Task: GNM-Contact Learner Module (Invertible Bottleneck)

## Context
- Proje dizini: `/Users/berat/Projects/ANM-openfold3`
- Conda env: `ANM-openfold` (Python 3.11, PyTorch, ProDy, BioPython)
- OpenFold3 repo: `/Users/berat/Projects/ANM-openfold3/openfold3-repo`

**Mimari dokümanları OKU (formüller, sayısal örnekler, gradient akışı burada):**
- `/Users/berat/Projects/ANM-openfold3/docs/architecture/05-gnm-contact-learner.md`
- `/Users/berat/Projects/ANM-openfold3/docs/architecture/06-gnm-math-detail.md`

---

## Genel Mimari

```
TRAIN (Forward):
pair_repr [B,N,N,128] → W_enc [128×32] → h [B,N,N,32] → dot(v) [32→1] → sigmoid → C_pred [B,N,N]

INVERSE (Sonra kullanım):
coords [N,3] → dist → sigmoid_cutoff → C [N,N] → logit(C) → ×v^T → [N,N,32] → W_dec [32×128] → pseudo_pair [N,N,128]
```

Öğrenilen parametreler:
- `W_enc`: Linear(128, 32) — 4,096 params
- `v`: Linear(32, 1, bias=False) — 32 params
- `W_dec`: Linear(32, 128) — 4,096 params
- **Toplam: ~8,224 parametre**

---

## Dosya Yapısı

`/Users/berat/Projects/ANM-openfold3/src/` altında oluştur:

```
src/
├── __init__.py
├── ground_truth.py      # PDB coords → soft contact map
├── kirchhoff.py         # Differentiable Kirchhoff + GNM decompose
├── contact_head.py      # Invertible bottleneck projection head
├── losses.py            # Contact loss + GNM loss
├── model.py             # GNMContactLearner (frozen OF3 + trainable head)
├── inverse.py           # Inverse path: coords → pseudo pair repr
├── data.py              # Dataset, PDB loading, batching
└── train.py             # Training loop
tests/
├── test_ground_truth.py
├── test_kirchhoff.py
├── test_contact_head.py
├── test_losses.py
└── test_inverse.py
```

---

## Modül Detayları

### 1. `src/ground_truth.py`

```python
def compute_gt_probability_matrix(coords_ca: torch.Tensor, r_cut: float = 10.0, tau: float = 1.5) -> torch.Tensor:
    """
    PDB Cα koordinatlarından soft probability contact map.

    Args:
        coords_ca: [N, 3] Cα positions (Ångström)
        r_cut: 10.0 (GNM standard cutoff, sigmoid center)
        tau: 1.5 (sigmoid temperature)

    Returns:
        C_gt: [N, N] values in [0, 1], symmetric, diagonal=0

    Formula:
        C_gt[i,j] = sigmoid(-(d_ij - r_cut) / tau)
                   = 1 / (1 + exp((d_ij - r_cut) / tau))
        C_gt[i,i] = 0
    """
```

### 2. `src/kirchhoff.py`

```python
def soft_kirchhoff(C: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Probability matrix → Kirchhoff matrix (differentiable).

    Args:
        C: [N, N] contact probabilities, diagonal=0, symmetric
        eps: regularization (trivial mode'u shift eder, NaN önler)

    Returns:
        Gamma: [N, N] Kirchhoff matrix

    Formula:
        Γ[i,j] = -C[i,j]           (i ≠ j)
        Γ[i,i] = Σ_k C[i,k]       (coordination number)
        Γ += ε·I                    (regularization)
    """


def gnm_decompose(Gamma: torch.Tensor, n_modes: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kirchhoff → GNM eigendecomposition (differentiable via torch.linalg.eigh).

    Args:
        Gamma: [N, N] Kirchhoff (symmetric PSD)
        n_modes: number of non-trivial modes

    Returns:
        eigenvalues: [n_modes] (ascending, trivial mode skipped)
        eigenvectors: [N, n_modes]
        b_factors: [N] per-residue flexibility

    Formula:
        eigh(Γ) → λ_0 ≤ λ_1 ≤ ... ≤ λ_{N-1}
        Skip λ_0 ≈ 0 (trivial translation mode)
        B_i = Σ_{k=1}^{n_modes} V_ik² / λ_k
    """
```

### 3. `src/contact_head.py`

```python
class ContactProjectionHead(nn.Module):
    """
    Invertible bottleneck: pair_repr → contact probability.

    Forward:  z [B,N,N,128] → W_enc → h [B,N,N,32] → v·h → sigmoid → C [B,N,N]
    Inverse:  C [N,N] → logit → ×v^T → [N,N,32] → W_dec → pseudo_z [N,N,128]

    Learnable params:
        W_enc: Linear(128, 32)        # encoder
        v: Linear(32, 1, bias=False)  # contact direction vector
        W_dec: Linear(32, 128)        # decoder (inverse path)
    """

    def __init__(self, c_z: int = 128, bottleneck_dim: int = 32):
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, N, N, c_z]
        Returns: C_pred [B, N, N] in [0,1], symmetric, diagonal=0

        Steps:
            1. z_sym = 0.5 * (z + z.transpose(1,2))
            2. h = z_sym @ W_enc                    # [B,N,N,32]
            3. logits = (h @ v).squeeze(-1)         # [B,N,N]
            4. logits = 0.5*(logits + logits.T)     # enforce symmetry
            5. logits[diagonal] = -inf (veya mask)  # diagonal=0 after sigmoid
            6. C_pred = sigmoid(logits)
        """

    def inverse(self, C: torch.Tensor) -> torch.Tensor:
        """
        C: [N, N] contact probability (from coordinates)
        Returns: pseudo_pair_repr [N, N, 128]

        Steps:
            1. logits = logit(C) = log(C / (1-C))   # inverse sigmoid
            2. h_approx = logits.unsqueeze(-1) * v.T # [N,N,32] broadcast
            3. pseudo_z = h_approx @ W_dec           # [N,N,128]
        """
```

### 4. `src/losses.py`

```python
def contact_loss(C_pred: torch.Tensor, C_gt: torch.Tensor, seq_sep_min: int = 6) -> torch.Tensor:
    """
    Weighted BCE with sequence separation filter.

    Sadece |i-j| >= seq_sep_min olan çiftleri hesaba kat.
    (Yakın komşular trivial, long-range contacts önemli)
    """


def gnm_loss(C_pred: torch.Tensor, C_gt: torch.Tensor, n_modes: int = 20) -> tuple[torch.Tensor, dict]:
    """
    Physics-informed GNM loss. Her iki C'den Kirchhoff → eigh → compare.

    Components:
        L_eigenvalue: MSE(normalized 1/λ_pred, normalized 1/λ_gt)
        L_bfactor:    MSE(normalized B_pred, normalized B_gt)
        L_eigvec:     mean(1 - |cos(V_pred_k, V_gt_k)|) for k modes

    Returns:
        loss_total: scalar
        details: dict with component values
    """


def reconstruction_loss(z_original: torch.Tensor, z_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Autoencoder reconstruction loss (optional, stabilizes inverse path).

    L_recon = MSE(z_original, Decode(Encode(z_original)))
    """


def total_loss(
    C_pred: torch.Tensor,
    C_gt: torch.Tensor,
    z_original: torch.Tensor = None,
    z_reconstructed: torch.Tensor = None,
    alpha: float = 1.0,      # contact weight
    beta: float = 0.5,       # gnm weight
    gamma: float = 0.1,      # reconstruction weight (optional)
) -> tuple[torch.Tensor, dict]:
    """
    L_total = α·L_contact + β·L_gnm + γ·L_recon
    """
```

### 5. `src/model.py`

```python
class GNMContactLearner(nn.Module):
    """
    Full model: OpenFold3 (frozen) + ContactProjectionHead (trainable).

    Forward returns:
        C_pred: [B, N, N] contact probabilities
        z: [B, N, N, 128] pair representation (detached, for analysis)
        z_recon: [B, N, N, 128] reconstructed pair repr (for L_recon)
    """

    def __init__(self, openfold_model, c_z: int = 128, bottleneck_dim: int = 32):
        # Freeze openfold
        # Create ContactProjectionHead
        ...

    def forward(self, batch: dict) -> dict:
        """
        Returns dict with keys: 'C_pred', 'pair_repr', 'pair_repr_recon'
        """
        # 1. Extract pair repr from trunk (no_grad)
        with torch.no_grad():
            s_input, s, z = self.openfold.run_trunk(batch)

        # 2. Contact prediction (forward path)
        C_pred = self.contact_head(z)

        # 3. Reconstruction (encode then decode, for L_recon)
        z_sym = 0.5 * (z + z.transpose(1,2))
        h = z_sym @ self.contact_head.W_enc.weight.T
        z_recon = h @ self.contact_head.W_dec.weight

        return {'C_pred': C_pred, 'pair_repr': z.detach(), 'pair_repr_recon': z_recon}
```

### 6. `src/inverse.py`

```python
class PairReprFromCoords:
    """
    Inverse path: coordinates → pseudo pair representation.

    Kullanım: Eğitim sonrası, herhangi bir protein koordinatından
    OpenFold3 pair representation uzayında bir tensor üretir.

    Bu tensor downstream ANM/TE analizlerinde kullanılabilir.
    """

    def __init__(self, trained_contact_head: ContactProjectionHead):
        self.head = trained_contact_head
        self.head.eval()

    def __call__(self, coords_ca: torch.Tensor, r_cut: float = 10.0, tau: float = 1.5) -> torch.Tensor:
        """
        coords_ca: [N, 3]
        Returns: pseudo_pair_repr [N, N, 128]

        Pipeline:
            1. dist = cdist(coords, coords)
            2. C = sigmoid(-(dist - r_cut) / tau)
            3. pseudo_pair = self.head.inverse(C)
        """

    def from_pdb(self, pdb_path: str, chain_id: str = 'A') -> torch.Tensor:
        """PDB dosyasından direkt pseudo pair repr üret."""
```

### 7. `src/data.py`

```python
class ProteinContactDataset(torch.utils.data.Dataset):
    """
    Dataset: PDB structures → (openfold3_features, coords_ca)

    Her sample:
        - openfold3 input features (sequence, MSA, etc.)
        - Cα coordinates [N, 3] (ground truth)
    """

def load_ca_coords(pdb_path: str, chain_id: str = 'A') -> torch.Tensor:
    """PDB/mmCIF → Cα coordinates [N, 3]"""

def create_dataloader(pdb_list: list[str], batch_size: int = 1, ...) -> DataLoader:
    """Batch size=1 (protein boyutları farklı, padding gerekir)"""
```

### 8. `src/train.py`

```python
def train(
    model: GNMContactLearner,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    device: str = 'mps',  # Apple Silicon
):
    """
    Training loop.

    - Optimizer: AdamW (weight_decay=1e-2)
    - Scheduler: CosineAnnealingLR
    - Gradient clipping: max_norm=1.0
    - Only contact_head parameters are optimized
    - Log: L_contact, L_gnm (L_eig, L_bf, L_vec), L_recon
    """
```

---

## Tests

### `tests/test_ground_truth.py`
- Bilinen 5 residue koordinatları → expected C_gt values kontrol
- Diagonal = 0
- Symmetric: C_gt == C_gt.T
- Range: tüm değerler [0, 1]
- d=10Å → C ≈ 0.5

### `tests/test_kirchhoff.py`
- Row sums ≈ 0 (eps hariç)
- Symmetric: Γ == Γ.T
- All eigenvalues >= 0
- Smallest eigenvalue ≈ eps
- **Gradient flows**: `C.requires_grad_(True)` → loss.backward() → `C.grad is not None`

### `tests/test_contact_head.py`
- Output shape: [B, N, N]
- Symmetric: C_pred[i,j] == C_pred[j,i]
- Range: [0, 1]
- Diagonal = 0
- Inverse shape: [N, N, 128]
- Roundtrip: encode → decode yaklaşık orijinale yakın (direction korunuyor)

### `tests/test_losses.py`
- C_pred == C_gt → L_contact ≈ 0
- Aynı Kirchhoff → L_gnm ≈ 0
- Gradient flows through eigh (no NaN)
- Loss decreases during optimization (3-step overfit test)

### `tests/test_inverse.py`
- coords → C → inverse → pseudo_pair shape = [N, N, 128]
- Farklı coords → farklı pseudo_pair
- Yakın residue'lar → yüksek norm, uzak → düşük norm

---

## Teknik Uyarılar

1. **torch.linalg.eigh**: Γ'ya mutlaka `eps*I` ekle, yoksa degenerate eigenvalue'larda NaN gradient
2. **Eigenvector sign ambiguity**: Loss'ta `abs(cosine_similarity)` kullan
3. **Skip trivial mode**: eigenvalues[0] ≈ 0, index 0'ı atla
4. **Forward pass'ta NumPy KULLANMA**: Gradient kırılır, her şey torch tensor
5. **Diagonal masking**: sigmoid'dan sonra `fill_diagonal_(0)` (gradient-friendly)
6. **Sequence separation**: |i-j| < 6 olan çiftler trivial, loss'a katma
7. **Normalization**: B-factors ve eigenvalues'ı compare etmeden önce normalize et (max veya sum ile)
8. **Device**: macOS'ta `device='mps'` (Apple Silicon GPU)

---

## Çalıştırma

```bash
conda activate ANM-openfold
cd /Users/berat/Projects/ANM-openfold3

# Tests
python -m pytest tests/ -v

# Training (sonra, data hazır olunca)
python -m src.train --epochs 100 --lr 1e-4 --device mps
```
