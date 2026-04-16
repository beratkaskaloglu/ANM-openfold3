"""ANM Mode-Drive Pipeline: iterative conformational exploration.

Combines ANM normal-mode displacements with OF3 diffusion to generate
physically meaningful protein conformational ensembles.

Pipeline:
    coords → ANM modes → collectivity rank → displace → contact → z_pseudo → blend z → diffusion → new coords → repeat

Strategy (goal: maximize displacement from initial structure):
    1. Compute ANM modes and rank combinations by collectivity
    2. Try most collective combo first with df_min
    3. If RMSD from initial doesn't increase → try next combo
    4. If all combos exhausted → escalate df toward df_max
    5. Run exactly n_steps iterations (no early stopping)

RMSD is measured from INITIAL structure — higher = more exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from .anm import anm_bfactors, anm_modes, build_hessian, collectivity, displace
from .converter import PairContactConverter
from .coords_to_contact import coords_to_contact
from .mode_combinator import (
    ModeCombo,
    collectivity_combinations,
    grid_combinations,
    random_combinations,
    targeted_combinations,
)
# Lazy import to avoid pulling OF3 dependencies when not needed
# DiffusionResult is checked via isinstance at runtime
try:
    from .of3_diffusion import DiffusionResult
except (ImportError, Exception):
    try:
        from src.of3_diffusion import DiffusionResult  # type: ignore[no-redef]
    except (ImportError, Exception):
        DiffusionResult = None  # type: ignore[assignment,misc]


@dataclass
class ModeDriveConfig:
    """Configuration for the ANM Mode-Drive pipeline."""

    # ANM parameters
    anm_cutoff: float = 15.0
    anm_gamma: float = 1.0
    anm_tau: float = 1.0
    n_anm_modes: int = 20

    # Contact map parameters
    contact_r_cut: float = 10.0
    contact_tau: float = 1.5

    # Pipeline parameters
    n_steps: int = 5                             # fixed number of steps (no early stop)
    combination_strategy: str = "collectivity"  # "collectivity", "grid", "random", "targeted", "manual"
    n_combinations: int = 20
    z_mixing_alpha: float = 0.3
    normalize_z: bool = True
    z_direction: str = "plus"                    # "plus" or "minus": add or subtract delta_z

    # Global displacement factor (collectivity strategy)
    df: float = 0.6                              # initial global df (Angstrom)
    df_min: float = 0.3                          # minimum df
    df_max: float = 3.0                          # maximum df
    df_escalation_factor: float = 1.5            # multiply df when combos exhausted
    max_combo_size: int = 3                      # max modes per combination (e.g. 3 or 5)

    # Manual mode selection
    manual_modes: tuple[int, ...] = ()           # e.g. (0, 1, 2) — mode indices to use

    # Random combinator defaults
    select_modes_range: tuple[int, int] = (1, 5)
    df_scale: float = 2.0

    # Grid combinator defaults
    grid_select_modes: int = 3
    grid_df_range: tuple[float, float] = (-2.0, 2.0)
    grid_df_steps: int = 5

    # Targeted combinator defaults
    targeted_top_modes: int = 5

    # Confidence & multi-sample
    num_diffusion_samples: int = 1               # K: samples per diffusion call
    confidence_ptm_cutoff: float = 0.5           # minimum pTM to accept step (0-1)
    confidence_plddt_cutoff: float = 70.0        # minimum mean pLDDT to accept (0-100 scale, OF3 native)
    confidence_ranking_cutoff: float = 0.5       # minimum ranking score to accept

    # Adaptive fallback
    enable_confidence_fallback: bool = False      # enable confidence-guided fallback
    fallback_combo_tries: int = 3                # Level 1: try next N combos by collectivity
    fallback_df_factor: float = 0.5              # Level 2: df *= this
    fallback_max_combo_size: int = 1             # Level 3: reduce to single mode
    fallback_alpha_factor: float = 0.5           # Level 4: alpha *= this
    max_fallback_retries: int = 4                # max retry levels per step


@dataclass
class StepResult:
    """Result from a single pipeline step."""

    combo: ModeCombo
    displaced_ca: torch.Tensor       # [N, 3]
    new_ca: torch.Tensor             # [N, 3]
    z_modified: torch.Tensor         # [N, N, 128]
    contact_map: torch.Tensor        # [N, N]
    rmsd: float
    eigenvalues: torch.Tensor        # [n_modes]
    eigenvectors: torch.Tensor       # [N, n_modes, 3]
    b_factors: torch.Tensor          # [N]
    df_used: float = 0.0             # actual df applied

    # Confidence metrics (populated when multi-sample diffusion is used)
    plddt: torch.Tensor | None = None         # [N] best sample per-residue pLDDT
    ptm: float | None = None                  # best sample pTM
    ranking_score: float | None = None        # best sample ranking
    all_ptm: torch.Tensor | None = None       # [K] all samples' pTM
    all_ranking: torch.Tensor | None = None   # [K] all samples' ranking
    fallback_level: int = 0                   # 0=normal, 1=df, 2=modes, 3=alpha
    num_samples: int = 1                      # K diffusion samples used


@dataclass
class ModeDriveResult:
    """Result from the full iterative pipeline."""

    trajectory: list[torch.Tensor] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    total_steps: int = 0


def kabsch_superimpose(
    ref: torch.Tensor,
    mobile: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kabsch superimposition: align mobile onto ref.

    Args:
        ref: [N, 3] reference coordinates
        mobile: [N, 3] mobile coordinates to align

    Returns:
        aligned: [N, 3] mobile after optimal rotation+translation
        rmsd: scalar RMSD after alignment
    """
    # 1. Center both
    ref_center = ref.mean(dim=0)
    mob_center = mobile.mean(dim=0)
    ref_centered = ref - ref_center
    mob_centered = mobile - mob_center

    # 2. Covariance matrix
    H = mob_centered.T @ ref_centered  # [3, 3]

    # 3. SVD
    U, S, Vt = torch.linalg.svd(H)

    # 4. Correct reflection
    d = torch.det(Vt.T @ U.T)
    sign_matrix = torch.diag(
        torch.tensor([1.0, 1.0, torch.sign(d)], device=ref.device, dtype=ref.dtype),
    )

    # 5. Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # 6. Apply
    aligned = (mob_centered @ R.T) + ref_center

    # 7. RMSD
    rmsd_val = ((ref - aligned) ** 2).sum(dim=-1).mean().sqrt()

    return aligned, rmsd_val


def compute_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    """RMSD between two coordinate sets after Kabsch superimposition [N, 3]."""
    _, rmsd_val = kabsch_superimpose(a, b)
    return rmsd_val.item()


def tm_score(
    coords_model: torch.Tensor,
    coords_ref: torch.Tensor,
) -> float:
    """Approximate TM-score between two CA coordinate sets.

    TM-score = (1/N) * sum_i 1 / (1 + (d_i / d0)^2)

    where d_i is per-residue distance after Kabsch superimposition
    and d0 = 1.24 * (N - 15)^(1/3) - 1.8

    Args:
        coords_model: [N, 3] model coordinates
        coords_ref:   [N, 3] reference coordinates

    Returns:
        TM-score in [0, 1]
    """
    aligned, _ = kabsch_superimpose(coords_ref, coords_model)
    N = coords_ref.shape[0]
    d0 = 1.24 * (max(N, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    di = ((coords_ref - aligned) ** 2).sum(dim=-1).sqrt()
    scores = 1.0 / (1.0 + (di / d0) ** 2)
    return scores.mean().item()


def _contact_to_distance(contact: torch.Tensor, r_cut: float, tau: float) -> torch.Tensor:
    """Invert sigmoid soft contact to approximate distances.

    d_ij = r_cut - tau * ln(C / (1 - C))

    Args:
        contact: [N, N] contact probabilities in (0, 1).
        r_cut:   Cutoff centre used in coords_to_contact.
        tau:     Sigmoid temperature.

    Returns:
        dist: [N, N] approximate pairwise distances.
    """
    c = contact.clamp(1e-6, 1.0 - 1e-6)
    logit = torch.log(c / (1.0 - c))  # sigmoid inverse
    dist = r_cut - tau * logit
    dist = dist.clamp(min=0.0)
    dist.fill_diagonal_(0.0)
    # Symmetrize
    dist = 0.5 * (dist + dist.T)
    return dist


def _classical_mds(dist_matrix: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """Classical multidimensional scaling: distance matrix → 3D coordinates.

    Args:
        dist_matrix: [N, N] symmetric distance matrix.
        dim: Embedding dimension (3 for 3D coords).

    Returns:
        coords: [N, dim] embedded coordinates.
    """
    N = dist_matrix.shape[0]
    D2 = dist_matrix ** 2

    # Centering matrix: H = I - (1/N) * 11^T
    H = torch.eye(N, device=dist_matrix.device, dtype=dist_matrix.dtype) - 1.0 / N

    # Double-centered matrix: B = -0.5 * H * D^2 * H
    B = -0.5 * H @ D2 @ H

    # Eigendecompose (float64 for stability)
    B64 = B.to(dtype=torch.float64, device="cpu")
    vals, vecs = torch.linalg.eigh(B64)
    vals = vals.to(dtype=dist_matrix.dtype, device=dist_matrix.device)
    vecs = vecs.to(dtype=dist_matrix.dtype, device=dist_matrix.device)

    # Take top `dim` eigenvalues (largest, at the end)
    top_vals = vals[-dim:].flip(0).clamp(min=0.0)
    top_vecs = vecs[:, -dim:].flip(1)

    # Coordinates: X = V * sqrt(Lambda)
    coords = top_vecs * top_vals.sqrt().unsqueeze(0)

    return coords


def make_pseudo_diffusion(
    converter: PairContactConverter,
    r_cut: float = 10.0,
    tau: float = 1.5,
    reference_coords: torch.Tensor | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a pseudo-diffusion function for testing without OF3.

    Converts blended z_mod back to 3D coordinates via:
        z_mod → contact (forward head) → distances (invert sigmoid) → MDS → coords

    If reference_coords is provided, the MDS output is Kabsch-aligned
    to the reference to maintain consistent orientation.

    Args:
        converter:        Trained PairContactConverter.
        r_cut:            Contact cutoff used in coords_to_contact.
        tau:              Sigmoid temperature.
        reference_coords: [N, 3] coords for alignment (typically initial structure).

    Returns:
        diffusion_fn: Callable([N, N, 128]) -> [N, 3]
    """
    def _pseudo_diffuse(z_mod: torch.Tensor) -> torch.Tensor:
        # z_mod [N, N, 128] → contact [N, N]
        contact = converter.z_to_contact(z_mod)

        # contact → approximate distance matrix
        dist = _contact_to_distance(contact, r_cut, tau)

        # distance → 3D coordinates via classical MDS
        coords = _classical_mds(dist, dim=3)

        # Align to reference if available
        if reference_coords is not None:
            coords, _ = kabsch_superimpose(reference_coords, coords)

        return coords

    return _pseudo_diffuse


class ModeDrivePipeline:
    """Iterative ANM mode-driven conformational exploration.

    Displaces structures along ANM normal modes ranked by collectivity,
    converts displaced coordinates to pseudo pair representations via
    the trained ContactProjectionHead inverse path, and feeds modified
    z_ij back to OF3's diffusion module.

    Collectivity strategy (exploration = maximize RMSD from initial):
        - Rank all mode combos by collectivity (most collective first)
        - Try best combo at current df
        - If RMSD doesn't increase (not exploring), try next combo
        - If all combos exhausted, escalate df and restart from best combo

    Args:
        converter: Trained PairContactConverter (z <-> contact).
        config:    Pipeline configuration.
        diffusion_fn: Callable(zij_trunk [N,N,128]) -> coords [N,3] CA.
                      Wraps OF3 diffusion + CA extraction.
    """

    def __init__(
        self,
        converter: PairContactConverter,
        config: ModeDriveConfig | None = None,
        diffusion_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.converter = converter
        self.config = config or ModeDriveConfig()
        self.diffusion_fn = diffusion_fn

    def _generate_combos(
        self,
        n_modes: int,
        coords_ca: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        df: float,
        target_coords: torch.Tensor | None = None,
    ) -> list[ModeCombo]:
        """Generate mode combinations based on configured strategy."""
        cfg = self.config

        if cfg.combination_strategy == "collectivity":
            return collectivity_combinations(
                eigenvectors=eigenvectors,
                n_modes_available=n_modes,
                max_combo_size=cfg.max_combo_size,
                df=df,
                max_combos=cfg.n_combinations,
                eigenvalues=eigenvalues,
            )
        elif cfg.combination_strategy == "grid":
            return grid_combinations(
                n_modes_available=n_modes,
                select_modes=cfg.grid_select_modes,
                df_range=cfg.grid_df_range,
                df_steps=cfg.grid_df_steps,
                max_combos=cfg.n_combinations,
            )
        elif cfg.combination_strategy == "manual":
            return self._manual_combo(eigenvalues, df)
        elif cfg.combination_strategy == "targeted":
            if target_coords is None:
                raise ValueError("targeted strategy requires target_coords")
            return targeted_combinations(
                current_coords=coords_ca,
                target_coords=target_coords,
                mode_vectors=eigenvectors,
                n_combos=cfg.n_combinations,
                top_modes=cfg.targeted_top_modes,
            )
        else:  # "random"
            return random_combinations(
                n_modes_available=n_modes,
                n_combos=cfg.n_combinations,
                select_modes_range=cfg.select_modes_range,
                df_scale=cfg.df_scale,
                eigenvalues=eigenvalues,
            )

    def _manual_combo(
        self,
        eigenvalues: torch.Tensor,
        df: float,
    ) -> list[ModeCombo]:
        """Build a single combo from user-specified modes.

        Eigenvalue-weighted normalize + single global df.
        Returns one combo (the specified modes with normalized df).
        """
        modes = self.config.manual_modes
        if not modes:
            raise ValueError("manual strategy requires manual_modes to be set")

        amp = 1.0 / (eigenvalues[list(modes)].sqrt() + 1e-10)
        amp = amp / (amp.sum() + 1e-10)
        mode_dfs = tuple((df * amp).tolist())

        label = f"manual_m{'_'.join(map(str, modes))}"
        return [ModeCombo(
            mode_indices=modes,
            dfs=mode_dfs,
            label=label,
        )]

    def _blend_z(
        self,
        z_pseudo: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delta_z to trunk z_ij in the configured direction.

        Computes delta_z = z_pseudo - zij_trunk (the displacement in z-space),
        then adds (+) or subtracts (-) it scaled by alpha.

        z_direction="plus":  zij_trunk + alpha * delta_z  (move toward displaced state)
        z_direction="minus": zij_trunk - alpha * delta_z  (move in opposite direction)
        """
        alpha = self.config.z_mixing_alpha

        if self.config.normalize_z:
            z_pseudo = (z_pseudo - z_pseudo.mean()) / (z_pseudo.std() + 1e-8)
            z_pseudo = z_pseudo * zij_trunk.std() + zij_trunk.mean()

        delta_z = z_pseudo - zij_trunk

        if self.config.z_direction == "minus":
            return zij_trunk - alpha * delta_z
        else:  # "plus"
            return zij_trunk + alpha * delta_z

    def _evaluate_combo(
        self,
        combo: ModeCombo,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        eigenvectors: torch.Tensor,
        zij_trunk: torch.Tensor,
        eigenvalues: torch.Tensor,
        b_factors: torch.Tensor,
        df_used: float,
    ) -> StepResult:
        """Evaluate a single mode combination.

        RMSD is computed from initial_coords_ca (not current iteration),
        because higher RMSD = more conformational exploration = better.
        """
        cfg = self.config
        device = coords_ca.device

        # Select modes
        indices = list(combo.mode_indices)
        modes_sel = eigenvectors[:, indices, :]  # [N, k, 3]
        dfs = torch.tensor(combo.dfs, device=device, dtype=coords_ca.dtype)
        eig_sel = eigenvalues[indices]  # [k]

        # Displace from current coords (eigenvalue-weighted)
        displaced = displace(coords_ca, modes_sel, dfs, eigenvalues=eig_sel)

        # Contact map from displaced coordinates
        contact = coords_to_contact(
            displaced, cfg.contact_r_cut, cfg.contact_tau,
        )

        # Contact -> pseudo z_ij
        z_pseudo = self.converter.contact_to_z(contact)

        # Blend with trunk z
        z_mod = self._blend_z(z_pseudo, zij_trunk)

        # Run diffusion if available, otherwise use displaced as proxy
        plddt_out = None
        ptm_out = None
        ranking_out = None
        all_ptm_out = None
        all_ranking_out = None
        num_samples_out = 1

        if self.diffusion_fn is not None:
            diff_result = self.diffusion_fn(z_mod)
            if hasattr(diff_result, "best_ca"):
                new_ca = diff_result.best_ca
                num_samples_out = diff_result.all_ca.shape[0]
                if diff_result.plddt is not None:
                    plddt_out = diff_result.plddt[diff_result.best_idx]
                if diff_result.ptm is not None:
                    ptm_out = diff_result.ptm[diff_result.best_idx].item()
                    all_ptm_out = diff_result.ptm
                if diff_result.ranking is not None:
                    ranking_out = diff_result.ranking[diff_result.best_idx].item()
                    all_ranking_out = diff_result.ranking
            else:
                new_ca = diff_result
        else:
            new_ca = displaced

        # Score: RMSD from INITIAL structure (higher = more exploration)
        rmsd = compute_rmsd(initial_coords_ca, new_ca)

        return StepResult(
            combo=combo,
            displaced_ca=displaced,
            new_ca=new_ca,
            z_modified=z_mod,
            contact_map=contact,
            rmsd=rmsd,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            b_factors=b_factors,
            df_used=df_used,
            plddt=plddt_out,
            ptm=ptm_out,
            ranking_score=ranking_out,
            all_ptm=all_ptm_out,
            all_ranking=all_ranking_out,
            num_samples=num_samples_out,
        )

    @torch.no_grad()
    def step(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        prev_rmsd: float = 0.0,
        target_coords: torch.Tensor | None = None,
    ) -> StepResult:
        """Run a single ANM mode-drive iteration with df escalation.

        Goal: MAXIMIZE RMSD from initial structure (conformational exploration).

        For collectivity strategy:
            1. Generate combos ranked by collectivity at df_min
            2. Try combos in order; accept first that increases RMSD
            3. If none increase RMSD, escalate df and retry
            4. Repeat until df_max reached

        Args:
            coords_ca:         [N, 3] current CA positions.
            initial_coords_ca: [N, 3] starting CA positions (RMSD reference).
            zij_trunk:         [N, N, 128] current pair representation.
            prev_rmsd:         RMSD from previous iteration (to beat).
            target_coords:     [N, 3] optional target for targeted strategy.

        Returns:
            StepResult with best combination's outputs.
        """
        cfg = self.config

        # Step 1: ANM Hessian + Eigendecomposition
        H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
        eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
        b_factors = anm_bfactors(eigenvalues, eigenvectors)
        n_modes = eigenvalues.shape[0]

        # Score function: if target given, minimize RMSD to target;
        # otherwise maximize RMSD from initial.
        has_target = target_coords is not None

        def _score(result: StepResult) -> float:
            if has_target:
                # Lower RMSD to target = better → negate for maximization
                return -compute_rmsd(result.new_ca, target_coords)
            return result.rmsd  # higher RMSD from initial = better

        # For non-collectivity strategies
        if cfg.combination_strategy != "collectivity":
            combos = self._generate_combos(
                n_modes, coords_ca, eigenvectors, eigenvalues,
                cfg.df, target_coords,
            )
            best_score = -float("inf")
            best: StepResult | None = None

            for combo in combos:
                result = self._evaluate_combo(
                    combo, coords_ca, initial_coords_ca, eigenvectors,
                    zij_trunk, eigenvalues, b_factors, cfg.df,
                )
                s = _score(result)
                if s > best_score:
                    best_score = s
                    best = result

            assert best is not None
            return best

        # Collectivity strategy with df escalation
        # +df and -df pairs are already generated by collectivity_combinations
        current_df = cfg.df_min
        best_overall: StepResult | None = None
        best_overall_score = -float("inf")

        while current_df <= cfg.df_max:
            # Generate combos at current df (includes +df and -df for each subset)
            combos = self._generate_combos(
                n_modes, coords_ca, eigenvectors, eigenvalues,
                current_df, target_coords,
            )

            improved = False
            for combo in combos:
                result = self._evaluate_combo(
                    combo, coords_ca, initial_coords_ca, eigenvectors,
                    zij_trunk, eigenvalues, b_factors, current_df,
                )

                s = _score(result)
                if s > best_overall_score:
                    best_overall_score = s
                    best_overall = result

                    # Check improvement vs previous step
                    if has_target:
                        # Improved if we got closer to target
                        improved = True
                        break
                    elif result.rmsd > prev_rmsd:
                        improved = True
                        break

            if improved:
                break

            # Escalate df for more displacement
            current_df *= cfg.df_escalation_factor

        # If nothing found (shouldn't happen), fall back to first combo
        if best_overall is None:
            combos = self._generate_combos(
                n_modes, coords_ca, eigenvectors, eigenvalues,
                cfg.df, target_coords,
            )
            best_overall = self._evaluate_combo(
                combos[0], coords_ca, initial_coords_ca, eigenvectors,
                zij_trunk, eigenvalues, b_factors, cfg.df,
            )

        return best_overall

    def _confidence_ok(self, result: StepResult) -> bool:
        """Check if step result passes ALL confidence cutoffs (AND logic).

        Gates: pTM, mean pLDDT (0-100 scale), ranking_score.
        If no confidence data is available, passes by default.
        """
        cfg = self.config

        # No confidence data at all → pass (K=1 without confidence, or disabled)
        if (
            result.ptm is None
            and result.plddt is None
            and result.ranking_score is None
        ):
            return True

        # pTM gate
        if result.ptm is not None and result.ptm < cfg.confidence_ptm_cutoff:
            return False
        # pLDDT gate (OF3 returns 0-100 scale; cutoff must match that scale)
        if result.plddt is not None:
            mean_plddt = result.plddt.mean().item()
            if mean_plddt < cfg.confidence_plddt_cutoff:
                return False
        # Ranking gate
        if (
            result.ranking_score is not None
            and result.ranking_score < cfg.confidence_ranking_cutoff
        ):
            return False
        return True

    @torch.no_grad()
    def step_with_fallback(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        prev_rmsd: float = 0.0,
        target_coords: torch.Tensor | None = None,
    ) -> StepResult:
        """Run step with confidence-guided adaptive fallback.

        Fallback levels:
            0: Normal step (best combo by RMSD/collectivity)
            1: Try next N combos with higher collectivity (combo fallback)
            2: Reduce df by fallback_df_factor
            3: Reduce max_combo_size to single mode
            4: Reduce z_mixing_alpha by fallback_alpha_factor

        If all levels fail, returns the attempt with highest ranking score.
        """
        cfg = self.config

        # Save originals
        orig_df = cfg.df
        orig_df_min = cfg.df_min
        orig_max_combo = cfg.max_combo_size
        orig_alpha = cfg.z_mixing_alpha

        best_result: StepResult | None = None
        best_ranking = -1.0

        def _track(result: StepResult) -> bool:
            """Track best result, return True if confidence OK."""
            nonlocal best_result, best_ranking
            r_score = result.ranking_score if result.ranking_score is not None else 0.0
            if r_score > best_ranking:
                best_ranking = r_score
                best_result = result
            return self._confidence_ok(result)

        try:
            # ── Level 0: Normal step (best combo) ──
            result = self.step(
                coords_ca, initial_coords_ca, zij_trunk,
                prev_rmsd, target_coords,
            )
            result.fallback_level = 0
            if _track(result):
                return result

            # ── Level 1: Try next combos by collectivity ──
            H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
            eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
            b_factors = anm_bfactors(eigenvalues, eigenvectors)
            n_modes = eigenvalues.shape[0]

            combos = self._generate_combos(
                n_modes, coords_ca, eigenvectors, eigenvalues,
                cfg.df, target_coords,
            )

            n_combo_tries = min(cfg.fallback_combo_tries, len(combos) - 1)
            for ci in range(1, 1 + n_combo_tries):
                combo_result = self._evaluate_combo(
                    combos[ci], coords_ca, initial_coords_ca, eigenvectors,
                    zij_trunk, eigenvalues, b_factors, cfg.df,
                )
                combo_result.fallback_level = 1
                if _track(combo_result):
                    return combo_result

            # ── Level 2: Reduce df ──
            cfg.df = orig_df * cfg.fallback_df_factor
            cfg.df_min = orig_df_min * cfg.fallback_df_factor
            result = self.step(
                coords_ca, initial_coords_ca, zij_trunk,
                prev_rmsd, target_coords,
            )
            result.fallback_level = 2
            if _track(result):
                return result

            # ── Level 3: Single mode (+ keep reduced df) ──
            cfg.max_combo_size = cfg.fallback_max_combo_size
            result = self.step(
                coords_ca, initial_coords_ca, zij_trunk,
                prev_rmsd, target_coords,
            )
            result.fallback_level = 3
            if _track(result):
                return result

            # ── Level 4: Reduce alpha (+ keep reduced df + single mode) ──
            cfg.z_mixing_alpha = orig_alpha * cfg.fallback_alpha_factor
            result = self.step(
                coords_ca, initial_coords_ca, zij_trunk,
                prev_rmsd, target_coords,
            )
            result.fallback_level = 4
            _track(result)

            # Forced accept: best attempt across all levels
            assert best_result is not None
            return best_result

        finally:
            # ALWAYS restore config — no matter which level returned
            cfg.df = orig_df
            cfg.df_min = orig_df_min
            cfg.max_combo_size = orig_max_combo
            cfg.z_mixing_alpha = orig_alpha

    @torch.no_grad()
    def run(
        self,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        target_coords: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> ModeDriveResult:
        """Run the full iterative ANM mode-drive pipeline.

        Runs exactly n_steps iterations (no early stopping).
        Each step tries to INCREASE RMSD from initial structure
        (maximize conformational exploration).

        Args:
            initial_coords_ca: [N, 3] starting CA positions.
            zij_trunk:         [N, N, 128] trunk pair representation.
            target_coords:     [N, 3] optional target structure.
            verbose:           Print live RMSD/TM-score per step.

        Returns:
            ModeDriveResult with trajectory and per-step results.
        """
        cfg = self.config
        result = ModeDriveResult()
        result.trajectory.append(initial_coords_ca.clone())

        coords_ca = initial_coords_ca
        z_current = zij_trunk
        prev_rmsd = 0.0  # initial RMSD from self = 0

        use_fallback = cfg.enable_confidence_fallback

        if verbose:
            N = initial_coords_ca.shape[0]
            has_target = target_coords is not None
            header = f"{'Step':>6} {'RMSD_init':>10} {'df':>6} {'Combo':>20}"
            if has_target:
                header += f" {'RMSD_tgt':>10} {'TM_tgt':>8}"
            header += f" {'pTM':>6} {'pLDDT':>6} {'FB':>3}"
            print(f"\n{'='*len(header)}")
            fb_str = "ON" if use_fallback else "OFF"
            print(
                f"Mode-Drive Pipeline | N={N} | strategy={cfg.combination_strategy} "
                f"| n_steps={cfg.n_steps} | z_dir={cfg.z_direction} | "
                f"K={cfg.num_diffusion_samples} | fallback={fb_str}"
            )
            print(f"{'='*len(header)}")
            print(header)
            print(f"{'-'*len(header)}")

            # Initial state
            init_line = f"{'Init':>6} {'0.000':>10} {'—':>6} {'—':>20}"
            if has_target:
                rmsd_tgt = compute_rmsd(initial_coords_ca, target_coords)
                tm_tgt = tm_score(initial_coords_ca, target_coords)
                init_line += f" {rmsd_tgt:>10.3f} {tm_tgt:>8.4f}"
            init_line += f" {'—':>6} {'—':>6} {'—':>3}"
            print(init_line)

        for step_idx in range(cfg.n_steps):
            if use_fallback:
                step_result = self.step_with_fallback(
                    coords_ca, initial_coords_ca, z_current,
                    prev_rmsd, target_coords,
                )
            else:
                step_result = self.step(
                    coords_ca, initial_coords_ca, z_current,
                    prev_rmsd, target_coords,
                )
            result.step_results.append(step_result)
            result.trajectory.append(step_result.new_ca.clone())
            result.total_steps = step_idx + 1

            if verbose:
                combo_label = step_result.combo.label[:20]
                ptm_str = f"{step_result.ptm:.3f}" if step_result.ptm is not None else "—"
                plddt_str = (
                    f"{step_result.plddt.mean().item():.3f}"
                    if step_result.plddt is not None else "—"
                )
                fb_str = str(step_result.fallback_level) if step_result.fallback_level > 0 else "—"

                line = (
                    f"{step_idx+1:>6} "
                    f"{step_result.rmsd:>10.3f} "
                    f"{step_result.df_used:>6.2f} "
                    f"{combo_label:>20}"
                )
                if has_target:
                    rmsd_t = compute_rmsd(step_result.new_ca, target_coords)
                    tm_t = tm_score(step_result.new_ca, target_coords)
                    line += f" {rmsd_t:>10.3f} {tm_t:>8.4f}"
                line += f" {ptm_str:>6} {plddt_str:>6} {fb_str:>3}"
                print(line)

            # Update for next step
            prev_rmsd = step_result.rmsd
            coords_ca = step_result.new_ca
            z_current = step_result.z_modified

        if verbose:
            print(f"{'-'*len(header)}")
            final_rmsd = result.step_results[-1].rmsd
            print(f"Final RMSD from initial: {final_rmsd:.3f} A")
            if has_target:
                final_tm = tm_score(result.step_results[-1].new_ca, target_coords)
                print(f"Final TM-score vs target: {final_tm:.4f}")

            # Confidence summary
            ptms = [r.ptm for r in result.step_results if r.ptm is not None]
            if ptms:
                print(f"pTM range: {min(ptms):.3f} — {max(ptms):.3f}")
            fallbacks = [r.fallback_level for r in result.step_results if r.fallback_level > 0]
            if fallbacks:
                print(f"Fallback triggered: {len(fallbacks)}/{len(result.step_results)} steps (levels: {fallbacks})")
            print()

        return result
