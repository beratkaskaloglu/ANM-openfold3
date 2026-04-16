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
    # Level 5: extended (combo × df × alpha) grid search
    fallback_extended_enabled: bool = True       # enable aggressive grid search when L0-L4 fail
    fallback_extended_combo_count: int = 10      # top N combos to iterate
    fallback_extended_df_scales: tuple[float, ...] = (0.5, 0.25)   # df multipliers
    fallback_extended_alpha_scales: tuple[float, ...] = (0.5, 0.25)  # alpha multipliers

    # ─────────────── Autostop strategy ───────────────
    # Replaces ANM mode-combo displacement with IW-ENM MD + early-stop picker.
    # Downstream (coords → contact → z_pseudo → blend → OF3 → QC) is unchanged.
    autostop_chain_id: str = "A"
    # Physics
    autostop_R_bb: float = 11.0
    autostop_R_sc: float = 2.0
    autostop_K_0: float = 0.8
    autostop_d_0: float = 3.8
    autostop_n_ref: float = 10.0
    # Integration
    autostop_dt: float = 0.01
    autostop_mass: float = 1.0
    autostop_damping: float = 0.0
    autostop_v_mode: str = "breathing"
    autostop_v_magnitude: float = 1.0
    # Run control
    autostop_n_steps: int = 5000
    autostop_save_every: int = 10
    autostop_back_off: int = 2
    autostop_crash_threshold_distance: float = 0.5
    # Monitor (early-stop)
    autostop_smooth_w: int = 11
    autostop_warmup_frac: float = 0.40
    autostop_patience: int = 3
    autostop_eps_E_rel: float = 0.0002
    autostop_eps_N_rel: float = 0.0005
    autostop_crash_window_saves: int = 20
    autostop_crash_threshold: int = 5
    autostop_min_saves_before_check: int = 15
    autostop_verbose: bool = True

    # Autostop fallback ladder (L0-L9).
    # See docs/plans/autostop_integration.md §2.4.
    # Per-level scale / delta sweeps; each entry is a NEW mutation relative to
    # the BASELINE config value (not cumulative across levels).
    autostop_fallback_v_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)
    autostop_fallback_back_off_adds: tuple[int, ...] = (0, 2, 4, 8)
    autostop_fallback_eps_E_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)
    autostop_fallback_eps_N_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)
    autostop_fallback_patience_deltas: tuple[int, ...] = (0, -1, 1, 2)
    autostop_fallback_smooth_w_deltas: tuple[int, ...] = (0, 4, -2, 8)
    autostop_fallback_warmup_frac_scales: tuple[float, ...] = (1.0, 1.5, 0.5)
    autostop_fallback_crash_window_scales: tuple[float, ...] = (1.0, 2.0, 0.5)
    autostop_fallback_crash_threshold_adds: tuple[int, ...] = (0, 2, -2)
    autostop_fallback_alpha_scales: tuple[float, ...] = (1.0, 0.5, 0.25)  # L4/L7 z_mixing_alpha multipliers
    # Which L0-L9 levels to enable (default: L0 + cheap replay-only levels
    # L1 back_off and L4 alpha; L9 is forced-accept safety net).
    # L2/L7 re-run MD (expensive), so they are disabled by default.
    # L3/L5/L6/L8 are replay-only but user's baseline strategy is
    # "go back in picked + change alpha" only, so they are off as well.
    autostop_fallback_levels: tuple[int, ...] = (0, 1, 4, 9)
    # Max cells per extended grid (L7, L8) — prevents combinatorial blow-up.
    autostop_fallback_grid_cap: int = 8


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
    alpha_used: float = 0.0          # actual z_mixing_alpha applied

    # Confidence metrics (populated when multi-sample diffusion is used)
    plddt: torch.Tensor | None = None         # [N] best sample per-residue pLDDT
    ptm: float | None = None                  # best sample pTM
    ranking_score: float | None = None        # best sample ranking
    all_ptm: torch.Tensor | None = None       # [K] all samples' pTM
    all_ranking: torch.Tensor | None = None   # [K] all samples' ranking
    fallback_level: int = 0                   # 0=normal, 1=df, 2=modes, 3=alpha
    rejected: bool = False                    # True if all fallback levels failed (forced-accept)
    num_samples: int = 1                      # K diffusion samples used

    # Autostop strategy diagnostics (None when strategy != "autostop")
    autostop_info: dict | None = None


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
        structure_ctx=None,  # autostop_adapter.StructureContext | None — lazy-typed to avoid importing iw_enm unless autostop is actually used
    ) -> None:
        self.converter = converter
        self.config = config or ModeDriveConfig()
        self.diffusion_fn = diffusion_fn
        self.structure_ctx = structure_ctx

        # Cache of most-recent autostop trace for cheap fallback replay.
        # Only populated when combination_strategy == "autostop".
        self._autostop_last_trace = None
        self._autostop_last_coords_key: int | None = None

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
        """Evaluate a single mode combination (ANM displace → downstream).

        RMSD is computed from initial_coords_ca (not current iteration),
        because higher RMSD = more conformational exploration = better.
        """
        device = coords_ca.device

        # Select modes
        indices = list(combo.mode_indices)
        modes_sel = eigenvectors[:, indices, :]  # [N, k, 3]
        dfs = torch.tensor(combo.dfs, device=device, dtype=coords_ca.dtype)
        eig_sel = eigenvalues[indices]  # [k]

        # Displace from current coords (eigenvalue-weighted)
        displaced = displace(coords_ca, modes_sel, dfs, eigenvalues=eig_sel)

        return self._downstream_from_displaced(
            combo=combo,
            displaced=displaced,
            initial_coords_ca=initial_coords_ca,
            zij_trunk=zij_trunk,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            b_factors=b_factors,
            df_used=df_used,
        )

    def _downstream_from_displaced(
        self,
        combo: ModeCombo,
        displaced: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        b_factors: torch.Tensor,
        df_used: float,
        autostop_info: dict | None = None,
    ) -> StepResult:
        """Common downstream path from a displaced-CA tensor.

        Shared by ANM-combo strategies and the autostop strategy:
        displaced → contact → z_pseudo → blend → diffusion → StepResult.
        """
        cfg = self.config

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
            alpha_used=cfg.z_mixing_alpha,
            plddt=plddt_out,
            ptm=ptm_out,
            ranking_score=ranking_out,
            all_ptm=all_ptm_out,
            all_ranking=all_ranking_out,
            num_samples=num_samples_out,
            autostop_info=autostop_info,
        )

    # ─────────────── Autostop strategy ───────────────

    def _autostop_params(self):
        """Build AutostopParams dataclass from current config values.

        Reads the *current* config fields — so fallback mutations (which
        patch config in place under try/finally) are picked up automatically.
        """
        from .autostop_adapter import AutostopParams
        cfg = self.config
        return AutostopParams(
            R_bb=cfg.autostop_R_bb,
            R_sc=cfg.autostop_R_sc,
            K_0=cfg.autostop_K_0,
            d_0=cfg.autostop_d_0,
            n_ref=cfg.autostop_n_ref,
            dt=cfg.autostop_dt,
            mass=cfg.autostop_mass,
            damping=cfg.autostop_damping,
            v_mode=cfg.autostop_v_mode,
            v_magnitude=cfg.autostop_v_magnitude,
            n_steps=cfg.autostop_n_steps,
            save_every=cfg.autostop_save_every,
            back_off=cfg.autostop_back_off,
            crash_threshold_distance=cfg.autostop_crash_threshold_distance,
            smooth_w=max(3, cfg.autostop_smooth_w),
            warmup_frac=max(0.0, min(0.5, cfg.autostop_warmup_frac)),
            patience=max(1, cfg.autostop_patience),
            eps_E_rel=max(1e-6, cfg.autostop_eps_E_rel),
            eps_N_rel=max(1e-6, cfg.autostop_eps_N_rel),
            crash_window_saves=max(1, cfg.autostop_crash_window_saves),
            crash_threshold=max(1, cfg.autostop_crash_threshold),
            min_saves_before_check=max(1, cfg.autostop_min_saves_before_check),
            verbose=cfg.autostop_verbose,
        )

    def _autostop_synthetic_combo(
        self,
        step_idx: int,
        picked_step: int,
        turn_k: int,
    ) -> ModeCombo:
        """Build a placeholder ModeCombo for autostop results.

        Contains no ANM modes — just a descriptive label so downstream
        code (notebook tables, StepResult serialization) keeps working.
        """
        return ModeCombo(
            mode_indices=(),
            dfs=(),
            label=f"autostop_s{step_idx}_pk{picked_step}_tk{turn_k}",
            collectivity_score=0.0,
        )

    def _autostop_step(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        step_idx: int = 0,
    ) -> StepResult:
        """Run one autostop MD + downstream evaluation.

        ANM modes are still computed (same cost as other strategies) so
        StepResult eigenvalues/eigenvectors/b_factors fields remain
        populated for notebook plots — but autostop does NOT use them
        for displacement.
        """
        from .autostop_adapter import run_autostop_from_tensor

        if self.structure_ctx is None:
            raise RuntimeError(
                "combination_strategy='autostop' requires a StructureContext. "
                "Pass structure_ctx=StructureContext.from_pdb(...) or "
                "StructureContext.from_ca_only(coords_ca, res_names=...) to "
                "ModeDrivePipeline(...)."
            )

        cfg = self.config

        # Compute ANM modes — kept for diagnostics (plots), not for displacement
        H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
        eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
        b_factors = anm_bfactors(eigenvalues, eigenvectors)

        # Run autostop — cache the trace for possible fallback replay
        params = self._autostop_params()
        pick, trace = run_autostop_from_tensor(coords_ca, self.structure_ctx, params)
        self._autostop_last_trace = trace
        self._autostop_last_coords_key = id(coords_ca)

        combo = self._autostop_synthetic_combo(
            step_idx=step_idx,
            picked_step=pick.picked_step_md,
            turn_k=pick.turn_k,
        )

        autostop_info = {
            "picked_save_index": pick.picked_save_index,
            "picked_step_md": pick.picked_step_md,
            "turn_k": pick.turn_k,
            "argmin_E_k": pick.argmin_E_k,
            "argmin_N_k": pick.argmin_N_k,
            "stop_step_md": pick.stop_step_md,
            "crashes_total": pick.crashes_total,
            "back_off_used": pick.back_off_used,
            "monitor_params": dict(pick.monitor_params),
            "stop_reason": pick.stop_reason,
            "n_saves": int(len(trace.steps)),
            "total_mdsteps_requested": trace.total_mdsteps_requested,
            "save_every": trace.save_every,
        }

        # df_used is 0.0 for autostop (no mode displacement factor).
        # alpha_used is read by downstream from current cfg (already set).
        return self._downstream_from_displaced(
            combo=combo,
            displaced=pick.picked_ca,
            initial_coords_ca=initial_coords_ca,
            zij_trunk=zij_trunk,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            b_factors=b_factors,
            df_used=0.0,
            autostop_info=autostop_info,
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

        # Autostop strategy — completely bypasses ANM mode-combo displacement.
        # Downstream (contact → z_pseudo → blend → diffusion → QC) unchanged.
        if cfg.combination_strategy == "autostop":
            return self._autostop_step(
                coords_ca=coords_ca,
                initial_coords_ca=initial_coords_ca,
                zij_trunk=zij_trunk,
                step_idx=0,
            )

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

        # Dispatch to autostop-specific ladder when strategy=autostop.
        if cfg.combination_strategy == "autostop":
            return self.step_with_autostop_fallback(
                coords_ca=coords_ca,
                initial_coords_ca=initial_coords_ca,
                zij_trunk=zij_trunk,
                step_idx=0,
            )

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
            if _track(result):
                return result

            # ── Level 5: Extended grid search (combo × df × alpha) ──
            # Aggressive exploration: iterate over top-N combos with multiple
            # df and alpha scales. First attempt that passes cutoffs wins.
            if cfg.fallback_extended_enabled:
                cfg.max_combo_size = cfg.fallback_max_combo_size  # single-mode combos
                n_ext = min(cfg.fallback_extended_combo_count, len(combos))
                for ci in range(n_ext):
                    for df_scale in cfg.fallback_extended_df_scales:
                        cfg.df = orig_df * df_scale
                        cfg.df_min = orig_df_min * df_scale
                        for alpha_scale in cfg.fallback_extended_alpha_scales:
                            cfg.z_mixing_alpha = orig_alpha * alpha_scale
                            ext_result = self._evaluate_combo(
                                combos[ci], coords_ca, initial_coords_ca,
                                eigenvectors, zij_trunk, eigenvalues,
                                b_factors, cfg.df,
                            )
                            ext_result.fallback_level = 5
                            if _track(ext_result):
                                return ext_result

            # Forced accept: best attempt across all levels.
            # Mark as rejected so the caller can keep previous base coords.
            assert best_result is not None
            best_result.rejected = True
            return best_result

        finally:
            # ALWAYS restore config — no matter which level returned
            cfg.df = orig_df
            cfg.df_min = orig_df_min
            cfg.max_combo_size = orig_max_combo
            cfg.z_mixing_alpha = orig_alpha

    # ─────────────── Autostop fallback ladder (L0-L9) ───────────────

    def _autostop_downstream_from_pick(
        self,
        pick,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        b_factors: torch.Tensor,
        step_idx: int,
        fallback_level: int,
        trace_stats: dict,
    ) -> StepResult:
        """Build a StepResult from an autostop pick — reused across fallback levels."""
        combo = self._autostop_synthetic_combo(
            step_idx=step_idx,
            picked_step=pick.picked_step_md,
            turn_k=pick.turn_k,
        )
        autostop_info = {
            "picked_save_index": pick.picked_save_index,
            "picked_step_md": pick.picked_step_md,
            "turn_k": pick.turn_k,
            "argmin_E_k": pick.argmin_E_k,
            "argmin_N_k": pick.argmin_N_k,
            "stop_step_md": pick.stop_step_md,
            "crashes_total": pick.crashes_total,
            "back_off_used": pick.back_off_used,
            "monitor_params": dict(pick.monitor_params),
            "stop_reason": pick.stop_reason,
            "fallback_level": fallback_level,
            **trace_stats,
        }
        result = self._downstream_from_displaced(
            combo=combo,
            displaced=pick.picked_ca,
            initial_coords_ca=initial_coords_ca,
            zij_trunk=zij_trunk,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            b_factors=b_factors,
            df_used=0.0,
            autostop_info=autostop_info,
        )
        result.fallback_level = fallback_level
        return result

    @torch.no_grad()
    def step_with_autostop_fallback(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        step_idx: int = 0,
    ) -> StepResult:
        """Run autostop strategy with L0-L9 confidence-guided fallback.

        Levels (see docs/plans/autostop_integration.md §2.4):
            L0: baseline autostop                                (MD + monitor + ds)
            L1: back_off adds  — replay_monitor                  (no MD)
            L2: v_magnitude scales — re-run MD                   (RERUN)
            L3: eps_E/eps_N scales — replay_monitor              (no MD)
            L4: z_alpha scales — re-run downstream only          (no MD, no monitor)
            L5: patience deltas — replay_monitor                 (no MD)
            L6: smooth_w deltas — replay_monitor                 (no MD)
            L7: grid (v × back_off × alpha) — re-run MD          (RERUN; capped)
            L8: grid (eps_E × eps_N × patience × smooth_w)       (no MD; capped)
            L9: skip — forced-accept best-so-far

        `cfg.autostop_fallback_levels` subsets which levels are attempted.
        """
        from .autostop_adapter import AutostopParams, run_autostop_from_tensor, replay_monitor

        if self.structure_ctx is None:
            raise RuntimeError(
                "combination_strategy='autostop' with fallback requires a StructureContext."
            )

        cfg = self.config

        # Save originals for try/finally restoration — includes autostop knobs
        # touched by mutations and z_mixing_alpha touched by L4/L7.
        orig_v = cfg.autostop_v_magnitude
        orig_back = cfg.autostop_back_off
        orig_eps_E = cfg.autostop_eps_E_rel
        orig_eps_N = cfg.autostop_eps_N_rel
        orig_patience = cfg.autostop_patience
        orig_smooth_w = cfg.autostop_smooth_w
        orig_warmup = cfg.autostop_warmup_frac
        orig_crash_win = cfg.autostop_crash_window_saves
        orig_crash_thr = cfg.autostop_crash_threshold
        orig_alpha = cfg.z_mixing_alpha

        best_result: StepResult | None = None
        best_ranking = -1.0

        def _track(
            result: StepResult,
            *,
            level: int = 0,
            desc: str = "",
        ) -> bool:
            nonlocal best_result, best_ranking
            r_score = result.ranking_score if result.ranking_score is not None else 0.0
            if r_score > best_ranking:
                best_ranking = r_score
                best_result = result
            ok = self._confidence_ok(result)
            if cfg.autostop_verbose:
                # pTM
                ptm_str = (
                    f"{result.ptm:.3f}" if result.ptm is not None else "  -  "
                )
                # mean pLDDT (0-100)
                if result.plddt is not None:
                    try:
                        plddt_mean = float(result.plddt.mean().item())
                    except Exception:
                        plddt_mean = float("nan")
                    plddt_str = f"{plddt_mean:5.1f}"
                else:
                    plddt_str = "  -  "
                rank_str = (
                    f"{result.ranking_score:.3f}"
                    if result.ranking_score is not None else "  -  "
                )
                ai = result.autostop_info or {}
                pk = ai.get("picked_step_md", "-")
                tk = ai.get("turn_k", "-")
                tag = "PASS" if ok else "FAIL"
                print(
                    f"      [FB L{level}] {tag}  {desc:<24s}  "
                    f"pk={pk!s:>5} tk={tk!s:>3}  "
                    f"pTM={ptm_str}  pLDDT={plddt_str}  rank={rank_str}  "
                    f"RMSD_init={result.rmsd:.2f}Å"
                )
            return ok

        # Compute ANM modes once for diagnostics — same cost as _autostop_step
        H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
        eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
        b_factors = anm_bfactors(eigenvalues, eigenvectors)

        device = coords_ca.device
        dtype = coords_ca.dtype
        enabled = set(cfg.autostop_fallback_levels)

        # Will be populated by L0 and reused by cheap levels.
        trace = None

        def _trace_stats(tr) -> dict:
            return {
                "n_saves": int(len(tr.steps)),
                "total_mdsteps_requested": tr.total_mdsteps_requested,
                "save_every": tr.save_every,
            }

        def _run_md(params: AutostopParams):
            nonlocal trace
            pick_loc, tr_loc = run_autostop_from_tensor(
                coords_ca, self.structure_ctx, params,
            )
            trace = tr_loc
            self._autostop_last_trace = tr_loc
            self._autostop_last_coords_key = id(coords_ca)
            return pick_loc, tr_loc

        def _replay(monitor_params: dict, back_off: int):
            assert trace is not None
            return replay_monitor(
                trace=trace,
                monitor_params=monitor_params,
                back_off=int(back_off),
                device=device,
                dtype=dtype,
            )

        try:
            # ── L0: baseline ─────────────────────────────────────────
            if 0 in enabled:
                params0 = self._autostop_params()
                pick0, tr0 = _run_md(params0)
                result0 = self._autostop_downstream_from_pick(
                    pick0, initial_coords_ca, zij_trunk,
                    eigenvalues, eigenvectors, b_factors,
                    step_idx, fallback_level=0, trace_stats=_trace_stats(tr0),
                )
                if _track(
                    result0, level=0,
                    desc=f"baseline v={orig_v:.2f} α={orig_alpha:.2f} back={orig_back}",
                ):
                    return result0

            # If L0 was skipped we still need a trace for replay-based levels.
            if trace is None:
                params_boot = self._autostop_params()
                pick_boot, tr_boot = _run_md(params_boot)
                _ = pick_boot

            # ── L1: back_off adds (replay only) ──────────────────────
            if 1 in enabled:
                base_mon = self._autostop_params().monitor_only()
                for add in cfg.autostop_fallback_back_off_adds:
                    if add == 0:
                        continue  # baseline = L0 already tried
                    new_back = max(0, orig_back + int(add))
                    pick = _replay(base_mon, new_back)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=1, trace_stats=_trace_stats(trace),
                    )
                    if _track(
                        res, level=1,
                        desc=f"back_off {orig_back}→{new_back} (add={add:+d})",
                    ):
                        return res

            # ── L2: v_magnitude scales (re-run MD) ───────────────────
            if 2 in enabled:
                for scale in cfg.autostop_fallback_v_scales:
                    if scale == 1.0:
                        continue
                    cfg.autostop_v_magnitude = orig_v * float(scale)
                    params = self._autostop_params()
                    pick, tr = _run_md(params)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=2, trace_stats=_trace_stats(tr),
                    )
                    if _track(
                        res, level=2,
                        desc=f"v_mag×{scale:.2f}→{cfg.autostop_v_magnitude:.2f} (RERUN MD)",
                    ):
                        return res
                cfg.autostop_v_magnitude = orig_v

            # ── L3: eps_E/eps_N scales (replay only) ─────────────────
            if 3 in enabled:
                for se in cfg.autostop_fallback_eps_E_scales:
                    for sn in cfg.autostop_fallback_eps_N_scales:
                        if se == 1.0 and sn == 1.0:
                            continue
                        mon = dict(self._autostop_params().monitor_only())
                        mon["eps_E_rel"] = max(1e-6, orig_eps_E * float(se))
                        mon["eps_N_rel"] = max(1e-6, orig_eps_N * float(sn))
                        pick = _replay(mon, orig_back)
                        res = self._autostop_downstream_from_pick(
                            pick, initial_coords_ca, zij_trunk,
                            eigenvalues, eigenvectors, b_factors,
                            step_idx, fallback_level=3, trace_stats=_trace_stats(trace),
                        )
                        if _track(
                            res, level=3,
                            desc=f"eps_E×{se:.1f} eps_N×{sn:.1f}",
                        ):
                            return res

            # ── L4: alpha scales (no MD, no monitor rerun — just re-blend) ──
            # Reuse the BASELINE monitor pick, just change alpha.
            if 4 in enabled:
                base_mon = self._autostop_params().monitor_only()
                for scale in cfg.autostop_fallback_alpha_scales:
                    if scale == 1.0:
                        continue
                    cfg.z_mixing_alpha = orig_alpha * float(scale)
                    pick = _replay(base_mon, orig_back)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=4, trace_stats=_trace_stats(trace),
                    )
                    if _track(
                        res, level=4,
                        desc=f"α×{scale:.2f}→{cfg.z_mixing_alpha:.2f}",
                    ):
                        return res
                cfg.z_mixing_alpha = orig_alpha

            # ── L5: patience deltas (replay only) ─────────────────────
            if 5 in enabled:
                for delta in cfg.autostop_fallback_patience_deltas:
                    if delta == 0:
                        continue
                    mon = dict(self._autostop_params().monitor_only())
                    new_pat = max(1, orig_patience + int(delta))
                    mon["patience"] = new_pat
                    pick = _replay(mon, orig_back)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=5, trace_stats=_trace_stats(trace),
                    )
                    if _track(
                        res, level=5,
                        desc=f"patience {orig_patience}→{new_pat} (Δ={delta:+d})",
                    ):
                        return res

            # ── L6: smooth_w deltas (replay only) ─────────────────────
            if 6 in enabled:
                for delta in cfg.autostop_fallback_smooth_w_deltas:
                    if delta == 0:
                        continue
                    mon = dict(self._autostop_params().monitor_only())
                    new_sw = max(3, orig_smooth_w + int(delta))
                    mon["smooth_w"] = new_sw
                    pick = _replay(mon, orig_back)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=6, trace_stats=_trace_stats(trace),
                    )
                    if _track(
                        res, level=6,
                        desc=f"smooth_w {orig_smooth_w}→{new_sw} (Δ={delta:+d})",
                    ):
                        return res

            # ── L7: grid (v × back_off × alpha) — RERUN MD per v ─────
            if 7 in enabled:
                cap = max(1, int(cfg.autostop_fallback_grid_cap))
                tried = 0
                for vs in cfg.autostop_fallback_v_scales:
                    if tried >= cap:
                        break
                    cfg.autostop_v_magnitude = orig_v * float(vs)
                    params = self._autostop_params()
                    pick_base, tr_v = _run_md(params)  # refresh trace under new v
                    for add in cfg.autostop_fallback_back_off_adds:
                        if tried >= cap:
                            break
                        for a_scale in cfg.autostop_fallback_alpha_scales:
                            if tried >= cap:
                                break
                            if vs == 1.0 and add == 0 and a_scale == 1.0:
                                continue  # baseline
                            cfg.z_mixing_alpha = orig_alpha * float(a_scale)
                            new_back = max(0, orig_back + int(add))
                            pick = _replay(params.monitor_only(), new_back)
                            res = self._autostop_downstream_from_pick(
                                pick, initial_coords_ca, zij_trunk,
                                eigenvalues, eigenvectors, b_factors,
                                step_idx, fallback_level=7, trace_stats=_trace_stats(tr_v),
                            )
                            tried += 1
                            if _track(
                                res, level=7,
                                desc=(
                                    f"grid v×{vs:.2f} back+{add} α×{a_scale:.2f} "
                                    f"(RERUN MD)"
                                ),
                            ):
                                return res
                cfg.autostop_v_magnitude = orig_v
                cfg.z_mixing_alpha = orig_alpha

            # ── L8: monitor-knob grid (replay only) ───────────────────
            if 8 in enabled:
                cap = max(1, int(cfg.autostop_fallback_grid_cap))
                tried = 0
                for se in cfg.autostop_fallback_eps_E_scales:
                    if tried >= cap:
                        break
                    for sn in cfg.autostop_fallback_eps_N_scales:
                        if tried >= cap:
                            break
                        for dp in cfg.autostop_fallback_patience_deltas:
                            if tried >= cap:
                                break
                            for ds in cfg.autostop_fallback_smooth_w_deltas:
                                if tried >= cap:
                                    break
                                if se == 1.0 and sn == 1.0 and dp == 0 and ds == 0:
                                    continue
                                mon = dict(self._autostop_params().monitor_only())
                                mon["eps_E_rel"] = max(1e-6, orig_eps_E * float(se))
                                mon["eps_N_rel"] = max(1e-6, orig_eps_N * float(sn))
                                mon["patience"] = max(1, orig_patience + int(dp))
                                mon["smooth_w"] = max(3, orig_smooth_w + int(ds))
                                pick = _replay(mon, orig_back)
                                res = self._autostop_downstream_from_pick(
                                    pick, initial_coords_ca, zij_trunk,
                                    eigenvalues, eigenvectors, b_factors,
                                    step_idx, fallback_level=8, trace_stats=_trace_stats(trace),
                                )
                                tried += 1
                                if _track(
                                    res, level=8,
                                    desc=(
                                        f"grid eps_E×{se:.1f} eps_N×{sn:.1f} "
                                        f"pat{dp:+d} sw{ds:+d}"
                                    ),
                                ):
                                    return res

            # ── L9: forced-accept best-so-far (or skip) ───────────────
            if best_result is not None:
                best_result.rejected = True
                best_result.fallback_level = 9
                if best_result.autostop_info is not None:
                    best_result.autostop_info["fallback_level"] = 9
                if cfg.autostop_verbose:
                    print(
                        f"      [FB L9] FORCE  all attempts failed confidence → "
                        f"accepting best-so-far (rank={best_ranking:.3f}, "
                        f"RMSD_init={best_result.rmsd:.2f}Å)"
                    )
                return best_result

            # Hard fallback: run one baseline pick and return rejected.
            params = self._autostop_params()
            pick, tr = _run_md(params)
            res = self._autostop_downstream_from_pick(
                pick, initial_coords_ca, zij_trunk,
                eigenvalues, eigenvectors, b_factors,
                step_idx, fallback_level=9, trace_stats=_trace_stats(tr),
            )
            res.rejected = True
            if cfg.autostop_verbose:
                print(
                    f"      [FB L9] FORCE  no prior attempts tracked → "
                    f"hard baseline accepted (rejected=True)"
                )
            return res

        finally:
            # ALWAYS restore — even if a level returned successfully.
            cfg.autostop_v_magnitude = orig_v
            cfg.autostop_back_off = orig_back
            cfg.autostop_eps_E_rel = orig_eps_E
            cfg.autostop_eps_N_rel = orig_eps_N
            cfg.autostop_patience = orig_patience
            cfg.autostop_smooth_w = orig_smooth_w
            cfg.autostop_warmup_frac = orig_warmup
            cfg.autostop_crash_window_saves = orig_crash_win
            cfg.autostop_crash_threshold = orig_crash_thr
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
            header = f"{'Step':>6} {'RMSD_init':>10} {'df':>6} {'α':>5} {'Combo':>20}"
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
            init_line = f"{'Init':>6} {'0.000':>10} {'—':>6} {'—':>5} {'—':>20}"
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
                if step_result.rejected:
                    fb_str = f"!{step_result.fallback_level}"
                elif step_result.fallback_level > 0:
                    fb_str = str(step_result.fallback_level)
                else:
                    fb_str = "—"

                line = (
                    f"{step_idx+1:>6} "
                    f"{step_result.rmsd:>10.3f} "
                    f"{step_result.df_used:>6.2f} "
                    f"{step_result.alpha_used:>5.2f} "
                    f"{combo_label:>20}"
                )
                if has_target:
                    rmsd_t = compute_rmsd(step_result.new_ca, target_coords)
                    tm_t = tm_score(step_result.new_ca, target_coords)
                    line += f" {rmsd_t:>10.3f} {tm_t:>8.4f}"
                line += f" {ptm_str:>6} {plddt_str:>6} {fb_str:>3}"
                print(line)

            # Update for next step only if not rejected.
            # Rejected steps are logged for visibility but pipeline keeps
            # previous base (coords_ca, z_current) to avoid cascading from
            # low-confidence structures.
            if not step_result.rejected:
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
            rejected = [i for i, r in enumerate(result.step_results) if r.rejected]
            if rejected:
                print(f"Rejected (base preserved): {len(rejected)}/{len(result.step_results)} steps (indices: {rejected})")
            print()

        return result
