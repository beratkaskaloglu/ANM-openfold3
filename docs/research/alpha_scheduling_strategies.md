# Alpha Scheduling Strategies for Mode-Drive Pipeline

*Research date: 2026-04-17*

## Context

Our Mode-Drive pipeline blends pseudo pair representations with trunk z_ij:
```
z_new = z_trunk + alpha * (z_pseudo - z_trunk)
```
Currently alpha is fixed (e.g., 0.3). This document summarizes literature findings on scheduling strategies.

---

## 1. How AF2/RF2 Handle Representation Mixing

**AlphaFold2 uses additive recycling, not interpolation.** The recycled pair representation goes through `LayerNorm -> LinearNoBias` and is **added** to fresh embeddings. No explicit alpha -- the learned projection implicitly controls mixing magnitude. AF2 uses 3 recycling iterations; AF3 uses `N_cycle = 4`.

**RoseTTAFold2** similarly uses additive projection through a three-track architecture.

**Key insight:** Our explicit `alpha` is a manual approximation of what these systems learn end-to-end.

## 2. Annealing Schedules

| Strategy | Description | When to use |
|----------|-------------|-------------|
| **Exponential decay** | `alpha_t = alpha_0 * decay^t` | Conservative, stable convergence |
| **Linear decay** | `alpha_t = max(0, alpha_0 - C*t)` | Simple, predictable |
| **Cosine decay** | `alpha_t = alpha_max * 0.5 * (1 + cos(pi*t/T))` | Smooth, avoids sharp transitions |
| **Cyclical** | Ramp alpha 0->max->0 multiple times | Multiple explore-exploit sweeps |
| **Warm-up then decay** | Low alpha initially, ramp up, then decay | When early z_pseudo is unreliable |

Cyclical annealing (Fu et al., 2019) showed that repeating ramp cycles outperformed monotonic schedules for VAE training -- each cycle explores then exploits.

## 3. Confidence-Guided Approaches

No established direct confidence-guided alpha in literature, but building blocks exist:

- **Per-residue alpha:** `alpha_ij = alpha_base * (1 - pLDDT_i/100) * (1 - pLDDT_j/100)`
  Low-confidence regions get more perturbation; high-confidence preserved.
- **Global gating:** `alpha_t = schedule(t) * (1 - min(pTM, 0.9))`
  As structure improves, perturbations naturally shrink.
- **Threshold-based:** Only apply perturbation to regions below pLDDT threshold.

**Warning (AFM-Refine-G, Oda et al. 2024):** Iterative re-refinement can produce misleadingly high confidence scores despite low actual accuracy. Don't blindly trust pTM/pLDDT in iterative settings.

## 4. Diffusion Model Analogies

The noise scale in diffusion models is functionally equivalent to our alpha:

| Method | Parameter | Finding |
|--------|-----------|---------|
| **RFDiffusion** | `noise_scale_ca` (default 1.0) | Reducing to 0.5 improves quality, reduces diversity |
| **Chroma** | Low-temperature sampling | Trades backbone quality for diversity |
| **FrameDiff** | `zeta in [0, 1]` | `zeta=0.5` dramatically improves designability |

**Consensus:** Reducing innovation rate from 1.0 to ~0.5 significantly improves quality while retaining adequate diversity. Our default alpha=0.3 is already conservative.

**Partial diffusion (RFDiffusion):** Controls how much of original structure is perturbed -- directly analogous to alpha. Small `partial_T` = conservative; large `partial_T` = aggressive.

## 5. Recommended Strategies (Ranked)

### 1. Cosine Decay (recommended starting point)
```python
alpha_t = alpha_max * 0.5 * (1 + cos(pi * t / T))
# alpha_max ~ 0.5, decays to ~0.05
```

### 2. Confidence-Gated Decay
```python
alpha_t = cosine_schedule(t) * (1 - min(pTM, 0.9))
```

### 3. Per-Residue Adaptive
```python
alpha_ij = alpha_base * (1 - pLDDT_i/100) * (1 - pLDDT_j/100)
```

### 4. Cyclical with Decay
2-3 cycles of alpha 0.1->0.5->0.1, peak decreasing each cycle.

### 5. Partial-Displacement Analog
Fix alpha, vary ANM/MD displacement magnitude per step (large early, small late).

## Key References

- Jumper et al. (2021) - AlphaFold2 recycling mechanism
- AF3 Supplementary (2024) - N_cycle=4 pair recycling
- Watson et al. (2023) - RFDiffusion noise_scale
- Ingraham et al. (2023) - Chroma low-temperature sampling
- Yim et al. (2023) - FrameDiff zeta parameter
- Fu et al. (2019) - Cyclical annealing for VAEs (arXiv:1903.10145)
- Uehara et al. (2025) - Reward-guided iterative refinement (arXiv:2502.14944)
- Oda et al. (2024) - AFM-Refine-G cautionary findings (bioRxiv:2022.12.27.521991)
