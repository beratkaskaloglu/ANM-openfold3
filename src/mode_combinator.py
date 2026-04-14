"""Combinatorial mode + displacement factor generation strategies.

Four strategies for exploring conformational space:
  - grid:          systematic cartesian product of modes x df values
  - random:        stochastic sampling of mode subsets and df amplitudes
  - targeted:      project displacement toward a target structure onto modes
  - collectivity:  rank mode combos by collectivity, apply global df
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import torch

from .anm import batch_combo_collectivity, combo_collectivity


@dataclass(frozen=True)
class ModeCombo:
    """A single combination of modes and displacement factors."""

    mode_indices: tuple[int, ...]
    dfs: tuple[float, ...]
    label: str = ""
    collectivity_score: float = 0.0

    def __post_init__(self) -> None:
        assert len(self.mode_indices) == len(self.dfs)


def collectivity_combinations(
    eigenvectors: torch.Tensor,
    n_modes_available: int,
    max_combo_size: int = 3,
    df: float = 0.6,
    max_combos: int = 50,
    eigenvalues: torch.Tensor | None = None,
) -> list[ModeCombo]:
    """Generate mode combos ranked by eigenvalue-weighted collectivity.

    For each subset size 1..max_combo_size, enumerates all mode subsets,
    computes combined collectivity (weighted by 1/sqrt(lambda)),
    ranks descending, and assigns eigenvalue-weighted displacement factors.

    Args:
        eigenvectors:      [N, n_modes, 3] per-residue mode vectors.
        n_modes_available: Total number of non-trivial modes.
        max_combo_size:    Maximum modes per combination (e.g. 3 or 5).
        df:                Global displacement factor (Angstrom scale).
        max_combos:        Maximum combinations to return.
        eigenvalues:       [n_modes] eigenvalues for amplitude weighting.

    Returns:
        List of ModeCombo sorted by collectivity (descending).
    """
    # Enumerate all mode subsets of size 1..max_combo_size
    all_subsets: list[tuple[int, ...]] = []
    for k in range(1, min(max_combo_size, n_modes_available) + 1):
        for mode_subset in combinations(range(n_modes_available), k):
            all_subsets.append(mode_subset)

    if not all_subsets:
        return []

    # Batch-vectorized collectivity computation (eigenvalue-weighted)
    scores = batch_combo_collectivity(eigenvectors, all_subsets, eigenvalues)

    # Sort by collectivity descending.
    # Each subset produces 2 combos (+df and -df), so take half.
    n_top = max(1, max_combos // 2)
    top_indices = scores.argsort(descending=True)[:n_top]

    # Precompute per-mode amplitude weights
    if eigenvalues is not None:
        amp = 1.0 / (eigenvalues.sqrt() + 1e-10)  # [n_modes]
    else:
        amp = None

    combos: list[ModeCombo] = []
    for rank, idx in enumerate(top_indices.tolist()):
        mode_subset = all_subsets[idx]
        score = scores[idx].item()
        k = len(mode_subset)

        if amp is not None:
            # Per-mode df weighted by 1/sqrt(lambda), normalized, scaled by df
            mode_amps = torch.stack([amp[m] for m in mode_subset])
            mode_amps = mode_amps / (mode_amps.sum() + 1e-10)  # normalize
            mode_dfs = tuple((df * mode_amps).tolist())
        else:
            weight = df / (k ** 0.5)
            mode_dfs = tuple([weight] * k)

        mode_label = f"m{'_'.join(map(str, mode_subset))}"

        # +df direction
        combos.append(ModeCombo(
            mode_indices=mode_subset,
            dfs=mode_dfs,
            label=f"coll_{rank:03d}_{mode_label}_pos",
            collectivity_score=score,
        ))

        # -df direction (negate all displacement factors)
        neg_dfs = tuple(-d for d in mode_dfs)
        combos.append(ModeCombo(
            mode_indices=mode_subset,
            dfs=neg_dfs,
            label=f"coll_{rank:03d}_{mode_label}_neg",
            collectivity_score=score,
        ))

    return combos


def grid_combinations(
    n_modes_available: int,
    select_modes: int = 3,
    df_range: tuple[float, float] = (-2.0, 2.0),
    df_steps: int = 5,
    max_combos: int = 100,
) -> list[ModeCombo]:
    """Generate mode+df combinations via cartesian product.

    Args:
        n_modes_available: Total modes from ANM.
        select_modes:      How many modes per combination.
        df_range:          (min_df, max_df) range.
        df_steps:          Number of df values to sample.
        max_combos:        Cap on total combinations.

    Returns:
        List of ModeCombo.
    """
    if n_modes_available < select_modes:
        select_modes = n_modes_available

    df_values = torch.linspace(df_range[0], df_range[1], df_steps).tolist()

    # All mode subsets
    mode_subsets = list(combinations(range(n_modes_available), select_modes))

    # All df tuples for k modes
    df_tuples = list(product(df_values, repeat=select_modes))

    combos: list[ModeCombo] = []
    for modes in mode_subsets:
        for dfs in df_tuples:
            combos.append(ModeCombo(
                mode_indices=modes,
                dfs=dfs,
                label=f"grid_m{'_'.join(map(str, modes))}_df{'_'.join(f'{d:.1f}' for d in dfs)}",
            ))
            if len(combos) >= max_combos:
                return combos

    return combos


def random_combinations(
    n_modes_available: int,
    n_combos: int = 50,
    select_modes_range: tuple[int, int] = (1, 5),
    df_scale: float = 2.0,
    seed: int | None = None,
    eigenvalues: torch.Tensor | None = None,
) -> list[ModeCombo]:
    """Generate random mode+df combinations weighted by eigenvalue.

    When eigenvalues are provided, modes with lower eigenvalues (larger
    collective motions) are sampled more frequently: p(k) ~ 1/lambda_k.
    The df amplitude for each mode is also scaled by 1/sqrt(lambda_k)
    so that low-frequency modes get larger displacements.

    Args:
        n_modes_available:  Total modes from ANM.
        n_combos:           Number of combinations to generate.
        select_modes_range: (min, max) modes per combination.
        df_scale:           Base standard deviation of Gaussian df sampling.
        seed:               Random seed for reproducibility.
        eigenvalues:        [n_modes] eigenvalues for importance weighting.

    Returns:
        List of ModeCombo.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    lo = max(1, select_modes_range[0])
    hi = min(select_modes_range[1], n_modes_available)

    # Mode selection weights: p(k) ~ 1/lambda_k (lower freq = more likely)
    if eigenvalues is not None:
        inv_lambda = 1.0 / (eigenvalues + 1e-10)
        mode_weights = inv_lambda / inv_lambda.sum()  # [n_modes] sums to 1
        # Per-mode df scale: low-freq modes get larger displacements
        mode_df_scale = (inv_lambda / inv_lambda.max()).sqrt()  # [n_modes]
    else:
        mode_weights = None
        mode_df_scale = None

    combos: list[ModeCombo] = []
    for i in range(n_combos):
        k = torch.randint(lo, hi + 1, (1,), generator=gen).item()

        if mode_weights is not None:
            # Weighted sampling without replacement
            indices = torch.multinomial(mode_weights, k, replacement=False)
            modes = tuple(sorted(indices.tolist()))
            # Scale df per mode
            scales = mode_df_scale[indices]
            dfs = tuple((torch.randn(k, generator=gen) * df_scale * scales).tolist())
        else:
            perm = torch.randperm(n_modes_available, generator=gen)
            modes = tuple(sorted(perm[:k].tolist()))
            dfs = tuple((torch.randn(k, generator=gen) * df_scale).tolist())

        combos.append(ModeCombo(
            mode_indices=modes,
            dfs=dfs,
            label=f"rand_{i:03d}",
        ))

    return combos


def targeted_combinations(
    current_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mode_vectors: torch.Tensor,
    n_combos: int = 20,
    top_modes: int = 5,
    perturbation_scale: float = 0.2,
    seed: int | None = None,
) -> list[ModeCombo]:
    """Generate combinations projected toward a target structure.

    Projects the displacement vector (target - current) onto mode basis
    and samples around the optimal df values.

    Args:
        current_coords: [N, 3] current CA positions.
        target_coords:  [N, 3] target CA positions.
        mode_vectors:   [N, n_modes, 3] eigenvectors from anm_modes.
        n_combos:       Number of combinations to generate.
        top_modes:      How many best-projecting modes to use.
        perturbation_scale: Gaussian noise scale around optimal dfs.
        seed:           Random seed.

    Returns:
        List of ModeCombo.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    N, n_modes, _ = mode_vectors.shape
    top_modes = min(top_modes, n_modes)

    # Displacement vector: [N, 3] -> [3N]
    delta = (target_coords - current_coords).reshape(-1)

    # Mode vectors: [N, n_modes, 3] -> [3N, n_modes]
    modes_flat = mode_vectors.reshape(N * 3, n_modes)

    # Project displacement onto each mode: [n_modes]
    projections = modes_flat.T @ delta

    # Select modes with largest absolute projection
    top_idx = projections.abs().argsort(descending=True)[:top_modes]
    top_idx_sorted = top_idx.sort().values

    combos: list[ModeCombo] = []

    # First combo: exact optimal projection with all top modes
    combos.append(ModeCombo(
        mode_indices=tuple(top_idx_sorted.tolist()),
        dfs=tuple(projections[top_idx_sorted].tolist()),
        label="targeted_optimal",
    ))

    # Remaining: subsets with perturbation
    for i in range(1, n_combos):
        k = torch.randint(1, top_modes + 1, (1,), generator=gen).item()
        perm = torch.randperm(top_modes, generator=gen)
        subset = top_idx_sorted[perm[:k]].sort().values
        optimal_dfs = projections[subset]
        noise = torch.randn(k, generator=gen) * perturbation_scale
        perturbed_dfs = optimal_dfs * (1.0 + noise)
        combos.append(ModeCombo(
            mode_indices=tuple(subset.tolist()),
            dfs=tuple(perturbed_dfs.tolist()),
            label=f"targeted_{i:03d}",
        ))

    return combos
