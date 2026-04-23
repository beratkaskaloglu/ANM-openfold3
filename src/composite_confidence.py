"""Composite confidence scoring for mode-drive pipeline.

Instead of individual metric cutoffs (pTM > 0.6, pLDDT > 70, etc.),
combines all metrics into a single weighted score in [0, 1].

Usage:
    weights = CompositeWeights(w_ptm=0.25, w_plddt=0.20, ...)
    score = composite_score_from_step(step_result, weights)
    accept = score >= threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CompositeWeights:
    """Weights for composite confidence score. Must sum to ~1.0."""

    w_ptm: float = 0.25
    w_plddt: float = 0.20
    w_pae: float = 0.25
    w_rg: float = 0.15
    w_contact_recon: float = 0.15

    def as_dict(self) -> dict[str, float]:
        return {
            "w_ptm": self.w_ptm,
            "w_plddt": self.w_plddt,
            "w_pae": self.w_pae,
            "w_rg": self.w_rg,
            "w_contact_recon": self.w_contact_recon,
        }

    @property
    def total(self) -> float:
        return self.w_ptm + self.w_plddt + self.w_pae + self.w_rg + self.w_contact_recon


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def normalize_ptm(ptm: float | None) -> float:
    """pTM in [0, 1] -> normalized. 0.8 = perfect."""
    if ptm is None:
        return 0.0
    return _clamp(ptm / 0.8)


def normalize_plddt(plddt_mean: float | None) -> float:
    """Mean pLDDT in [0, 100] -> normalized. 50-90 mapped to 0-1."""
    if plddt_mean is None:
        return 0.0
    return _clamp((plddt_mean - 50.0) / 40.0)


def normalize_pae(mean_pae: float | None) -> float:
    """Mean PAE in Angstrom -> normalized. 0=perfect, 25=terrible."""
    if mean_pae is None:
        return 0.5  # neutral when unavailable
    return _clamp(1.0 - mean_pae / 25.0)


def normalize_rg(rg_ratio: float | None) -> float:
    """Rg ratio -> normalized. 1.0=ideal, quadratic penalty for deviation.

    Quadratic penalty makes Rg>1.5 drop very fast:
      Rg=1.0 -> 1.00, Rg=1.2 -> 0.96, Rg=1.5 -> 0.75
      Rg=1.8 -> 0.36, Rg=2.0 -> 0.00, Rg=2.5 -> 0.00
    """
    if rg_ratio is None:
        return 0.5
    dev = abs(rg_ratio - 1.0)
    # Quadratic: (dev/1.0)^2 — hits zero at deviation=1.0 (i.e. Rg=2.0)
    return _clamp(1.0 - (dev / 1.0) ** 2)


def normalize_contact_recon(cr: float | None) -> float:
    """Contact reconstruction Pearson r -> normalized. [-0.1, 0.7] -> [0, 1]."""
    if cr is None:
        return 0.5
    return _clamp((cr + 0.1) / 0.8)


def compute_composite(
    ptm: float | None = None,
    plddt_mean: float | None = None,
    mean_pae: float | None = None,
    rg_ratio: float | None = None,
    contact_recon: float | None = None,
    weights: CompositeWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute composite confidence score.

    Returns:
        (composite_score, component_dict) where component_dict has
        normalized values and weighted contributions for debugging.
    """
    if weights is None:
        weights = CompositeWeights()

    n_ptm = normalize_ptm(ptm)
    n_plddt = normalize_plddt(plddt_mean)
    n_pae = normalize_pae(mean_pae)
    n_rg = normalize_rg(rg_ratio)
    n_cr = normalize_contact_recon(contact_recon)

    score = (
        weights.w_ptm * n_ptm
        + weights.w_plddt * n_plddt
        + weights.w_pae * n_pae
        + weights.w_rg * n_rg
        + weights.w_contact_recon * n_cr
    )

    components = {
        "n_ptm": n_ptm,
        "n_plddt": n_plddt,
        "n_pae": n_pae,
        "n_rg": n_rg,
        "n_cr": n_cr,
        "c_ptm": weights.w_ptm * n_ptm,
        "c_plddt": weights.w_plddt * n_plddt,
        "c_pae": weights.w_pae * n_pae,
        "c_rg": weights.w_rg * n_rg,
        "c_cr": weights.w_contact_recon * n_cr,
    }

    return score, components


def composite_score_from_step(step_result, weights: CompositeWeights | None = None) -> tuple[float, dict]:
    """Extract metrics from a StepResult and compute composite score."""
    plddt_mean = None
    if step_result.plddt is not None:
        plddt_mean = float(step_result.plddt.mean().item())

    return compute_composite(
        ptm=step_result.ptm,
        plddt_mean=plddt_mean,
        mean_pae=step_result.mean_pae,
        rg_ratio=step_result.rg_ratio,
        contact_recon=step_result.contact_recon,
        weights=weights,
    )


# Preset weight sets for grid search
# Rg weight raised across all presets after V3.0 showed Rg=2.3 structures
# being accepted then cascading into full divergence.
WEIGHT_PRESETS: dict[str, CompositeWeights] = {
    "A_ptm_heavy": CompositeWeights(w_ptm=0.35, w_plddt=0.15, w_pae=0.20, w_rg=0.20, w_contact_recon=0.10),
    "B_pae_heavy": CompositeWeights(w_ptm=0.15, w_plddt=0.10, w_pae=0.35, w_rg=0.25, w_contact_recon=0.15),
    "C_balanced": CompositeWeights(w_ptm=0.20, w_plddt=0.15, w_pae=0.25, w_rg=0.25, w_contact_recon=0.15),
    "D_physical": CompositeWeights(w_ptm=0.10, w_plddt=0.10, w_pae=0.25, w_rg=0.35, w_contact_recon=0.20),
}

THRESHOLD_PRESETS: list[float] = [0.45, 0.50, 0.55]
