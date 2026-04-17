# Early-Stopping Criteria for IW-ENM Molecular Dynamics

*Research date: 2026-04-17*

## Context

Our IW-ENM MD pipeline uses two criteria to detect when to stop:
1. **Energy reversal:** total potential E(t) starts increasing after initial decrease
2. **Spring count reversal:** active springs N(t) starts increasing after decreasing

We pick the frame at `pk = tk - back_off` (or `pk = tk * fraction`) before the turning point.

---

## 1. Is Energy Reversal a Good Criterion?

**Yes, for our specific use case (IW-ENM with dynamic contact breaking).**

ENM potentials are harmonic: `V = (1/2) * sum(k_ij * (d_ij - d_ij_0)^2)`. Energy normally increases monotonically as structure deforms. Energy *reversal* happens when enough springs break (contacts lost as distance exceeds cutoff), reducing the remaining elastic energy. This signals **the network topology is fundamentally changing** -- the structure is leaving the basin of validity of the original ENM.

This is a **novel contribution** of our pipeline. The published literature doesn't define canonical stopping criteria because most ENM methods either:
- Use normal mode analysis without dynamics
- Have a target structure (TMD, eBDIMS)
- Use iterative deformation + energy minimization (ClustENM)

## 2. How Other Methods Decide When to Stop

### Target-based methods
| Method | Stopping criterion |
|--------|-------------------|
| **TMD** | RMSD to target reaches ~0 (holonomic constraint) |
| **SMD** | Pulling coordinate reaches target value |
| **eBDIMS** | RMSD to target falls within 1-3 A (thermal fluctuations) |

### Open-ended sampling
| Method | Stopping criterion |
|--------|-------------------|
| **ClustENM** | Fixed number of generations (3-5), step size `s` calibrated to avoid unphysical deformations |
| **All-atom MD** | RMSD plateau + energy stabilization + secondary structure stability |
| **Adaptive sampling** | Coverage metrics (state space coverage, MSM transition rate convergence) |

## 3. Setting the Back-Off Fraction

**No directly published method exists**, but several analogies provide guidance:

### From ENM validity regime
- ENMs reliably capture motions within **2-4 A RMSD** of reference
- Beyond this range, harmonic approximation and fixed-connectivity assumptions break down
- If turning point occurs at 6 A RMSD, backing off to 3-4 A (50-67%) keeps you in valid regime

### From ClustENM step size
- Step size `s` is calibrated so deformations are "broad but not unphysical"
- Structures that can't be relaxed by energy minimization are over-deformed
- **Optimal point is well before irrecoverability**

### Practical recommendations
- **back_off_fraction range: 0.1 to 0.3** (pick frame at 10-30% before turning point)
- This keeps: most springs intact, harmonic approximation valid, ensemble physically meaningful
- **Alternative metric:** Monitor spring survival fraction, pick where it's still >80-90%

### Multi-metric adaptive approach (synthesized)
Rather than single criterion, use:
1. **Primary:** Energy reversal (current approach)
2. **Secondary:** Spring survival fraction drops below threshold (70-80%)
3. **Tertiary:** RMSD acceleration -- if d(RMSD)/dt increases rapidly, leaving harmonic basin
4. **Back-off:** Choose frame where ALL metrics are in "healthy" regime

## 4. Risks of Too Long vs Too Short

### Too long (past turning point)
- **Harmonic breakdown:** Bond angles distorted, chain crossings, steric clashes
- **Network topology collapse:** Remaining network no longer represents physical protein
- **Backbone ruptures:** Overly weak/poorly connected springs cause unphysical ruptures (edENM paper)
- **Loss of collectivity:** Remaining motions become localized and non-physical
- **Energy landscape artifacts:** ENM potential no longer corresponds to true free energy

### Too short
- **Insufficient diversity:** Ensemble clusters too tightly around starting structure
- **Mode under-sampling:** Slowest (most biologically relevant) modes truncated
- **B-factor underestimation:** Fluctuation amplitudes too small
- **Missing intermediates:** Meaningful conformational intermediates at moderate RMSD (2-4 A) missed

## 5. Adaptive Step-Size / Adaptive Stopping

### Published approaches
- **eBDIMS:** Modulate unbiased BD steps `k` -- "increasing k provides slower but wider sampling"
- **ClustENM:** Iterative deformation + energy minimization + clustering as "safety net"
- **Iterative ENM updating:** Periodically refresh network connectivity during simulation (Togashi, 2018)
- **LJ-extended ENM:** Replace harmonic springs with Lennard-Jones for smoother energy at large extensions (Poma et al., 2018)

### Our recommended approach
Monitor multiple signals simultaneously:
```
healthy = (
    spring_survival > 0.80
    and rmsd_acceleration < threshold
    and energy_slope < 0  # still decreasing
)
pick = last frame where healthy == True
```

This naturally handles different proteins without fixed fraction tuning.

## Key References

- Orellana et al. (2016) Nature Comms - eBDIMS, RMSD-based convergence for ENM-driven BD
- Kurkcuoglu et al. (2016) JCTC - ClustENM step size optimization
- Bhatt & Bhatt (2011) PMC3009412 - Damped Network Model with contact fraction convergence
- Zheng (2010) PMC2884254 - Anharmonicity corrections to ENM
- edENM (2026) bioRxiv - MD-parametrized ENM, warns of backbone ruptures
- Poma et al. (2018) PCCP - LJ-extended ENM for beyond-harmonic dynamics
- Togashi (2018) PMC6320916 - Review of ENM dynamics, nonlinear effects
- Mardt et al. PLOS ONE - lmcENM, predicting which contacts break
- Shamsi et al. (2023) J. Phys. Chem. B - ML-driven adaptive sampling review
