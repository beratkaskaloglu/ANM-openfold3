# ANM Mode-Drive Pipeline — Mathematical Operations

End-to-end mathematical reference for every operation in the pipeline, ordered by execution flow.

```
coords → Hessian → eigen → collectivity rank → displace → contact → z_pseudo → blend z → [diffusion] → new coords → repeat
```

---

## 1. ANM Hessian Construction (`anm.build_hessian`)

**Input:** CA coordinates $\mathbf{r}_i \in \mathbb{R}^3$, $i = 1 \ldots N$

**Pairwise distance:**

$$d_{ij} = \|\mathbf{r}_j - \mathbf{r}_i\|$$

**Soft contact weight (sigmoid cutoff):**

$$w_{ij} = \sigma\!\left(-\frac{d_{ij} - r_\text{cut}}{\tau}\right) = \frac{1}{1 + \exp\!\left(\frac{d_{ij} - r_\text{cut}}{\tau}\right)}$$

Default: $r_\text{cut} = 15\,\text{A}$, $\tau = 1.0$

**Unit direction vector:**

$$\hat{e}_{ij} = \frac{\mathbf{r}_j - \mathbf{r}_i}{d_{ij}}$$

**Off-diagonal 3×3 super-element:**

$$\mathbf{H}_{ij} = -\gamma \, w_{ij} \, (\hat{e}_{ij} \otimes \hat{e}_{ij}) \qquad i \neq j$$

where $\otimes$ is the outer product and $\gamma = 1.0$ is the uniform spring constant.

**Diagonal super-element (force balance):**

$$\mathbf{H}_{ii} = -\sum_{j \neq i} \mathbf{H}_{ij}$$

**Output:** $\mathbf{H} \in \mathbb{R}^{3N \times 3N}$, real-symmetric positive semi-definite.

---

## 2. ANM Eigendecomposition (`anm.anm_modes`)

**Input:** $\mathbf{H} \in \mathbb{R}^{3N \times 3N}$

Solve the eigenvalue problem (via `torch.linalg.eigh` in float64):

$$\mathbf{H} \mathbf{u}_k = \lambda_k \mathbf{u}_k$$

yielding $3N$ eigenvalue/eigenvector pairs sorted ascending: $\lambda_1 \leq \lambda_2 \leq \cdots$

**Trivial mode removal:** First 6 modes (3 translation + 3 rotation) have $\lambda \approx 0$. Skip them.

**Output:**
- Eigenvalues: $\lambda_k$, $k = 1 \ldots K$ (default $K = 20$)
- Eigenvectors reshaped: $\mathbf{v}_k \in \mathbb{R}^{N \times 3}$ per-residue 3D displacement directions

---

## 3. B-factor Computation (`anm.anm_bfactors`)

$$B_i = \sum_{k=1}^{K} \frac{\|\mathbf{v}_{k,i}\|^2}{\lambda_k}$$

Physical meaning: mean-square fluctuation of residue $i$ summed over all modes, weighted by inverse eigenvalue (low-frequency modes contribute more).

---

## 4. Collectivity (`anm.collectivity`)

Measures how many residues participate in mode $k$.

**Normalized squared displacement:**

$$u^2_{k,i} = \frac{\|\mathbf{v}_{k,i}\|^2}{\sum_{j=1}^{N} \|\mathbf{v}_{k,j}\|^2}$$

**Shannon entropy:**

$$S_k = -\sum_{i=1}^{N} u^2_{k,i} \ln u^2_{k,i}$$

**Collectivity:**

$$\kappa_k = \frac{1}{N} \exp(S_k)$$

Range: $\kappa \in [1/N, 1]$. Localized mode $\to 1/N$, maximally collective $\to 1$.

Reference: Bruschweiler (1995) J Chem Phys 102:3396

---

## 5. Combined Collectivity for Mode Subsets (`anm.combo_collectivity`)

For a subset of modes $\mathcal{M} = \{m_1, m_2, \ldots, m_k\}$:

**Eigenvalue-weighted combination:**

$$a_{m} = \frac{1}{\sqrt{\lambda_m}} \qquad \tilde{a}_m = \frac{a_m}{\sum_{m' \in \mathcal{M}} a_{m'}}$$

**Combined displacement field:**

$$\mathbf{d}_i = \sum_{m \in \mathcal{M}} \tilde{a}_m \, \mathbf{v}_{m,i}$$

Then compute collectivity of the combined field $\mathbf{d}$ using the same $u^2, S, \kappa$ formulas from Section 4.

---

## 6. Batch Vectorized Collectivity (`anm.batch_combo_collectivity`)

For $C$ combos simultaneously:

**Mask matrix:** $\mathbf{M} \in \mathbb{R}^{C \times K}$

$$M_{c,m} = \begin{cases} \tilde{a}_m & \text{if mode } m \in \text{combo } c \\ 0 & \text{otherwise} \end{cases}$$

where $\tilde{a}_m$ are the normalized $1/\sqrt{\lambda_m}$ weights per combo.

**Batched combined displacement via einsum:**

$$\mathbf{D}_{c,i,d} = \sum_{m} M_{c,m} \, V_{i,m,d} \qquad \text{(einsum: } cm, imd \to cid\text{)}$$

yielding $\mathbf{D} \in \mathbb{R}^{C \times N \times 3}$.

**Batched collectivity:**

$$\text{sq}_{c,i} = \|\mathbf{D}_{c,i}\|^2, \qquad u^2_{c,i} = \frac{\text{sq}_{c,i}}{\sum_j \text{sq}_{c,j}}$$

$$S_c = -\sum_i u^2_{c,i} \ln u^2_{c,i}, \qquad \kappa_c = \frac{1}{N} \exp(S_c)$$

---

## 7. Mode Combination Strategies (`mode_combinator`)

### 7a. Collectivity Strategy (`collectivity_combinations`)

1. Enumerate all subsets of size $1 \ldots s_\text{max}$ from $K$ available modes: $\binom{K}{1} + \binom{K}{2} + \cdots + \binom{K}{s_\text{max}}$
2. Compute $\kappa_c$ for each subset via batch vectorized collectivity
3. Sort descending by $\kappa$, take top $N_\text{combos}$
4. Assign per-mode displacement factors:

$$\text{df}_m = \text{df}_\text{global} \cdot \tilde{a}_m$$

where $\tilde{a}_m = a_m / \sum a_{m'}$ and $a_m = 1/\sqrt{\lambda_m}$.

### 7b. Grid Strategy (`grid_combinations`)

Cartesian product: $\binom{K}{s} \times |\text{df\_values}|^s$

$$\text{df\_values} = \text{linspace}(\text{df\_min}, \text{df\_max}, n_\text{steps})$$

### 7c. Random Strategy (`random_combinations`)

**Mode sampling probability** (eigenvalue-weighted):

$$p(m) = \frac{1/\lambda_m}{\sum_{m'} 1/\lambda_{m'}}$$

Low-frequency modes sampled more frequently. Multinomial sampling without replacement.

**Per-mode df scale:**

$$s_m = \sqrt{\frac{1/\lambda_m}{\max_m(1/\lambda_m)}}$$

$$\text{df}_m \sim \mathcal{N}(0, \sigma \cdot s_m)$$

### 7d. Targeted Strategy (`targeted_combinations`)

**Displacement vector:** $\Delta = \mathbf{r}_\text{target} - \mathbf{r}_\text{current} \in \mathbb{R}^{3N}$

**Projection onto each mode:**

$$p_m = \mathbf{u}_m^\top \Delta$$

Select top $T$ modes by $|p_m|$ descending. Optimal combo: $\text{df}_m = p_m$.

Perturbed combos: $\text{df}_m = p_m \cdot (1 + \epsilon)$, $\epsilon \sim \mathcal{N}(0, \sigma_\text{pert})$

---

## 8. Eigenvalue-Weighted Displacement (`anm.displace`)

**Input:** current coords $\mathbf{r}_i$, selected modes $\mathbf{v}_{m,i}$, displacement factors $\text{df}_m$, eigenvalues $\lambda_m$

**Amplitude weighting:**

$$a_m = \frac{1}{\sqrt{\lambda_m}}, \qquad \tilde{a}_m = \frac{a_m}{\sum_{m'} a_{m'}}$$

**Effective weights:**

$$w_m = \text{df}_m \cdot \tilde{a}_m$$

**Displaced coordinates:**

$$\mathbf{r}'_i = \mathbf{r}_i + \sum_{m} w_m \, \mathbf{v}_{m,i}$$

Without eigenvalues: $w_m = \text{df}_m$ directly.

---

## 9. Coordinates to Soft Contact Map (`coords_to_contact`)

$$d_{ij} = \|\mathbf{r}'_i - \mathbf{r}'_j\|$$

$$C_{ij} = \sigma\!\left(-\frac{d_{ij} - r_\text{cut}}{\tau}\right), \qquad C_{ii} = 0$$

Default: $r_\text{cut} = 10\,\text{A}$, $\tau = 1.5$

Note: Different cutoff/tau from Hessian construction (Section 1).

---

## 10. ContactProjectionHead — Forward Path (`contact_head.forward`)

**Input:** $\mathbf{z} \in \mathbb{R}^{N \times N \times c_z}$ (pair representation, $c_z = 128$)

**Symmetrize:**

$$\mathbf{z}_\text{sym} = \frac{1}{2}(\mathbf{z} + \mathbf{z}^\top)$$

**Encode to bottleneck:**

$$\mathbf{h} = \mathbf{z}_\text{sym} \, \mathbf{W}_\text{enc} \qquad \in \mathbb{R}^{N \times N \times k}$$

$\mathbf{W}_\text{enc} \in \mathbb{R}^{c_z \times k}$, $k = 32$ (bottleneck dim), no bias.

**Dot product with contact vector:**

$$\ell_{ij} = \mathbf{h}_{ij} \cdot \mathbf{v} = \sum_{d=1}^{k} h_{ij,d} \, v_d$$

**Symmetrize logits:**

$$\ell_{ij} \leftarrow \frac{1}{2}(\ell_{ij} + \ell_{ji}), \qquad \ell_{ii} = 0$$

**Contact probability:**

$$\hat{C}_{ij} = \sigma(\ell_{ij})$$

---

## 11. ContactProjectionHead — Inverse Path (`contact_head.inverse`)

**Input:** $C \in \mathbb{R}^{N \times N}$ contact probabilities

**Sigmoid inverse (logit):**

$$\ell_{ij} = \ln \frac{C_{ij}}{1 - C_{ij}}$$

with clamping $C \in [10^{-6}, 1 - 10^{-6}]$.

**Broadcast through normalized contact vector:**

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\| + 10^{-8}}$$

$$\mathbf{h}_{ij} = \ell_{ij} \, \hat{\mathbf{v}} \qquad \in \mathbb{R}^k$$

**Decode to pair space:**

$$\tilde{\mathbf{z}}_{ij} = \mathbf{h}_{ij} \, \mathbf{W}_\text{dec} \qquad \in \mathbb{R}^{c_z}$$

$\mathbf{W}_\text{dec} \in \mathbb{R}^{k \times c_z}$, no bias.

---

## 12. z-Blending (`mode_drive._blend_z`)

**z-score normalization** (optional, default on):

$$\tilde{\mathbf{z}}_\text{pseudo} = \frac{\tilde{\mathbf{z}} - \mu(\tilde{\mathbf{z}})}{\sigma(\tilde{\mathbf{z}}) + 10^{-8}} \cdot \sigma(\mathbf{z}_\text{trunk}) + \mu(\mathbf{z}_\text{trunk})$$

**Linear blending:**

$$\mathbf{z}_\text{mod} = \alpha \, \tilde{\mathbf{z}}_\text{pseudo} + (1 - \alpha) \, \mathbf{z}_\text{trunk}$$

Default $\alpha = 0.3$.

---

## 13. Kabsch Superimposition (`mode_drive.kabsch_superimpose`)

Optimal rigid-body alignment of mobile $\mathbf{Q}$ onto reference $\mathbf{P}$.

**Center both:**

$$\bar{\mathbf{P}} = \mathbf{P} - \text{mean}(\mathbf{P}), \qquad \bar{\mathbf{Q}} = \mathbf{Q} - \text{mean}(\mathbf{Q})$$

**Cross-covariance matrix:**

$$\mathbf{H} = \bar{\mathbf{Q}}^\top \bar{\mathbf{P}} \qquad \in \mathbb{R}^{3 \times 3}$$

**SVD:**

$$\mathbf{H} = \mathbf{U} \, \mathbf{\Sigma} \, \mathbf{V}^\top$$

**Reflection correction:**

$$d = \det(\mathbf{V}^\top{}^\top \, \mathbf{U}^\top) = \det(\mathbf{V} \mathbf{U}^\top)$$

$$\mathbf{S} = \text{diag}(1, 1, \text{sign}(d))$$

**Optimal rotation:**

$$\mathbf{R} = \mathbf{V} \, \mathbf{S} \, \mathbf{U}^\top$$

**Aligned coordinates:**

$$\mathbf{Q}_\text{aligned} = \bar{\mathbf{Q}} \, \mathbf{R}^\top + \text{mean}(\mathbf{P})$$

---

## 14. RMSD (`mode_drive.compute_rmsd`)

After Kabsch superimposition:

$$\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{P}_i - \mathbf{Q}_{\text{aligned},i}\|^2}$$

---

## 15. TM-score (`mode_drive.tm_score`)

$$d_0 = 1.24 \cdot (N - 15)^{1/3} - 1.8, \qquad d_0 \geq 0.5$$

After Kabsch superimposition, per-residue distance:

$$d_i = \|\mathbf{P}_i - \mathbf{Q}_{\text{aligned},i}\|$$

$$\text{TM} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{1 + (d_i / d_0)^2}$$

Range: $\text{TM} \in [0, 1]$. Values $> 0.5$ indicate same fold.

Reference: Zhang & Skolnick (2004) Proteins 57:702

---

## 16. df Escalation Logic (`mode_drive.step`)

Collectivity strategy only. Goal: maximize RMSD from initial structure.

```
current_df ← df_min

while current_df ≤ df_max:
    combos ← generate_combos(current_df)        # sorted by κ descending
    for combo in combos:
        result ← evaluate(combo, current_df)
        if result.rmsd > prev_rmsd:
            return result                         # accept first improvement
    current_df ← current_df × df_escalation      # default ×1.5
```

Default: $\text{df\_min} = 0.3$, $\text{df\_max} = 3.0$, escalation $= 1.5\times$

---

## 17. GNM — Kirchhoff Matrix (`kirchhoff.soft_kirchhoff`)

From soft contact map $C$:

$$\Gamma_{ij} = -C_{ij} \quad (i \neq j), \qquad \Gamma_{ii} = \sum_{j \neq i} C_{ij}$$

With regularization: $\Gamma \leftarrow \Gamma + \epsilon \mathbf{I}$, $\epsilon = 10^{-6}$

---

## 18. GNM Eigendecomposition (`kirchhoff.gnm_decompose`)

$$\Gamma \mathbf{u}_k = \lambda_k \mathbf{u}_k$$

Skip trivial mode ($k = 0$, uniform eigenvector). Return modes $k = 1 \ldots K$.

**GNM B-factors:**

$$B_i = \sum_{k=1}^{K} \frac{u_{k,i}^2}{\lambda_k}$$

Note: GNM eigenvectors are scalar ($\mathbf{u}_k \in \mathbb{R}^N$), unlike ANM ($\mathbb{R}^{N \times 3}$).

---

## 19. Training Losses (`losses`)

### 19a. Focal Loss

$$\text{FL}(p_t) = -\alpha_t \, (1 - p_t)^\gamma \, \ln p_t$$

where $p_t = \hat{C}$ if $C_\text{gt} = 1$, else $1 - \hat{C}$.

Sequence separation filter: only pairs with $|i - j| \geq s_\text{min}$ contribute (default $s_\text{min} = 6$).

### 19b. Contact Loss (BCE)

$$\mathcal{L}_\text{contact} = -\frac{1}{|\mathcal{S}|} \sum_{(i,j) \in \mathcal{S}} \left[ C_\text{gt} \ln \hat{C} + (1 - C_\text{gt}) \ln(1 - \hat{C}) \right]$$

where $\mathcal{S} = \{(i,j) : |i-j| \geq s_\text{min}\}$.

### 19c. GNM Loss (Physics-Informed)

Both $\hat{C}$ and $C_\text{gt}$ pass through Kirchhoff $\to$ eigh:

**Eigenvalue loss** (normalized inverse eigenvalues):

$$\tilde{\lambda}_k^{-1} = \frac{1/\lambda_k}{\sum_m 1/\lambda_m}, \qquad \mathcal{L}_\text{eig} = \text{MSE}\!\left(\tilde{\lambda}^{-1}_\text{pred}, \tilde{\lambda}^{-1}_\text{gt}\right)$$

**B-factor loss** (normalized profiles):

$$\hat{B}_i = \frac{B_i}{\max_j B_j}, \qquad \mathcal{L}_\text{bf} = \text{MSE}(\hat{B}_\text{pred}, \hat{B}_\text{gt})$$

**Eigenvector loss** (phase-invariant cosine):

$$\mathcal{L}_\text{vec} = \frac{1}{K} \sum_k \left(1 - |\cos(\mathbf{u}_{k,\text{pred}}, \mathbf{u}_{k,\text{gt}})|\right)$$

**Combined:**

$$\mathcal{L}_\text{GNM} = w_\text{eig} \mathcal{L}_\text{eig} + w_\text{bf} \mathcal{L}_\text{bf} + w_\text{vec} \mathcal{L}_\text{vec}$$

**Surrogate gradient:** Since `eigh` is on CPU (numerical stability), gradient flows back via normalized surrogate:

$$\hat{g} = \frac{\nabla_{C_\text{pred}} \mathcal{L}_\text{GNM}}{\|\nabla_{C_\text{pred}} \mathcal{L}_\text{GNM}\| + 10^{-8}}$$

$$\mathcal{L}_\text{surrogate} = \sum_{i,j} \hat{C}_{ij} \cdot \hat{g}_{ij}$$

### 19d. Reconstruction Loss

$$\mathcal{L}_\text{recon} = \text{MSE}(\mathbf{z}_\text{sym}, \mathbf{z}_\text{recon})$$

where $\mathbf{z}_\text{recon} = \mathbf{W}_\text{dec}(\mathbf{W}_\text{enc}(\mathbf{z}_\text{sym}))$ (autoencoder roundtrip).

### 19e. Total Loss

$$\mathcal{L} = \alpha \, \mathcal{L}_\text{contact} + \beta \, \mathcal{L}_\text{GNM} + \gamma \, \mathcal{L}_\text{recon}$$

Default weights: $\alpha = 1.0$, $\beta = 0.5$, $\gamma = 0.1$

---

## Pipeline Execution Order (One Step)

| Step | Module | Operation | Shape Transform |
|------|--------|-----------|-----------------|
| 1 | `anm.build_hessian` | $\mathbf{r} \to \mathbf{H}$ | $[N,3] \to [3N,3N]$ |
| 2 | `anm.anm_modes` | $\mathbf{H} \to (\lambda, \mathbf{v})$ | $[3N,3N] \to [K], [N,K,3]$ |
| 3 | `anm.anm_bfactors` | $(\lambda, \mathbf{v}) \to B$ | $[K],[N,K,3] \to [N]$ |
| 4 | `mode_combinator` | Enumerate + rank by $\kappa$ | $[N,K,3] \to \text{list[ModeCombo]}$ |
| 5 | `anm.displace` | $\mathbf{r} + \sum w_m \mathbf{v}_m \to \mathbf{r}'$ | $[N,3] \to [N,3]$ |
| 6 | `coords_to_contact` | $\mathbf{r}' \to C$ | $[N,3] \to [N,N]$ |
| 7 | `contact_head.inverse` | $C \to \tilde{\mathbf{z}}$ | $[N,N] \to [N,N,128]$ |
| 8 | `mode_drive._blend_z` | $\alpha \tilde{\mathbf{z}} + (1-\alpha)\mathbf{z}_\text{trunk}$ | $[N,N,128] \to [N,N,128]$ |
| 9 | `diffusion_fn` (OF3) | $\mathbf{z}_\text{mod} \to \mathbf{r}_\text{new}$ | $[N,N,128] \to [N,3]$ |
| 10 | `kabsch + RMSD/TM` | Quality metrics | $[N,3] \to \text{scalar}$ |

Repeat for `n_steps` iterations. Each step uses the output coordinates and modified $\mathbf{z}$ from the previous step.
