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
    combination_strategy: str = "collectivity"  # "collectivity", "grid", "random", "targeted"
    n_combinations: int = 20
    z_mixing_alpha: float = 0.3
    normalize_z: bool = True

    # Global displacement factor (collectivity strategy)
    df: float = 0.6                              # initial global df (Angstrom)
    df_min: float = 0.3                          # minimum df
    df_max: float = 3.0                          # maximum df
    df_escalation_factor: float = 1.5            # multiply df when combos exhausted
    max_combo_size: int = 3                      # max modes per combination (e.g. 3 or 5)

    # Random combinator defaults
    select_modes_range: tuple[int, int] = (1, 5)
    df_scale: float = 2.0

    # Grid combinator defaults
    grid_select_modes: int = 3
    grid_df_range: tuple[float, float] = (-2.0, 2.0)
    grid_df_steps: int = 5

    # Targeted combinator defaults
    targeted_top_modes: int = 5


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
            )
        elif cfg.combination_strategy == "grid":
            return grid_combinations(
                n_modes_available=n_modes,
                select_modes=cfg.grid_select_modes,
                df_range=cfg.grid_df_range,
                df_steps=cfg.grid_df_steps,
                max_combos=cfg.n_combinations,
            )
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

    def _blend_z(
        self,
        z_pseudo: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> torch.Tensor:
        """Blend pseudo z_ij with original trunk z_ij.

        Optionally normalize z_pseudo to match trunk statistics.
        """
        alpha = self.config.z_mixing_alpha

        if self.config.normalize_z:
            z_pseudo = (z_pseudo - z_pseudo.mean()) / (z_pseudo.std() + 1e-8)
            z_pseudo = z_pseudo * zij_trunk.std() + zij_trunk.mean()

        return alpha * z_pseudo + (1.0 - alpha) * zij_trunk

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

        # Displace from current coords
        displaced = displace(coords_ca, modes_sel, dfs)

        # Contact map from displaced coordinates
        contact = coords_to_contact(
            displaced, cfg.contact_r_cut, cfg.contact_tau,
        )

        # Contact -> pseudo z_ij
        z_pseudo = self.converter.contact_to_z(contact)

        # Blend with trunk z
        z_mod = self._blend_z(z_pseudo, zij_trunk)

        # Run diffusion if available, otherwise use displaced as proxy
        if self.diffusion_fn is not None:
            new_ca = self.diffusion_fn(z_mod)
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

        # For non-collectivity strategies: maximize RMSD from initial
        if cfg.combination_strategy != "collectivity":
            combos = self._generate_combos(
                n_modes, coords_ca, eigenvectors, eigenvalues,
                cfg.df, target_coords,
            )
            best_score = -1.0
            best: StepResult | None = None

            for combo in combos:
                result = self._evaluate_combo(
                    combo, coords_ca, initial_coords_ca, eigenvectors,
                    zij_trunk, eigenvalues, b_factors, cfg.df,
                )
                if result.rmsd > best_score:
                    best_score = result.rmsd
                    best = result

            assert best is not None
            return best

        # Collectivity strategy with df escalation
        # Goal: find a combo that increases RMSD beyond prev_rmsd
        current_df = cfg.df_min
        best_overall: StepResult | None = None
        best_overall_score = -1.0

        while current_df <= cfg.df_max:
            # Generate combos at current df (already sorted by collectivity)
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

                # Accept if RMSD increased (more exploration)
                if result.rmsd > best_overall_score:
                    best_overall_score = result.rmsd
                    best_overall = result

                    # If we beat previous iteration's RMSD, accept
                    if result.rmsd > prev_rmsd:
                        improved = True
                        break  # Accept first improvement (most collective)

            if improved:
                break  # Found improvement, no need to escalate

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

        if verbose:
            N = initial_coords_ca.shape[0]
            has_target = target_coords is not None
            header = f"{'Step':>6} {'RMSD_init':>10} {'df':>6} {'Combo':>20}"
            if has_target:
                header += f" {'RMSD_tgt':>10} {'TM_tgt':>8}"
            print(f"\n{'='*len(header)}")
            print(f"Mode-Drive Pipeline | N={N} | strategy={cfg.combination_strategy} | n_steps={cfg.n_steps}")
            print(f"{'='*len(header)}")
            print(header)
            print(f"{'-'*len(header)}")

            # Initial state
            init_line = f"{'Init':>6} {'0.000':>10} {'—':>6} {'—':>20}"
            if has_target:
                rmsd_tgt = compute_rmsd(initial_coords_ca, target_coords)
                tm_tgt = tm_score(initial_coords_ca, target_coords)
                init_line += f" {rmsd_tgt:>10.3f} {tm_tgt:>8.4f}"
            print(init_line)

        for step_idx in range(cfg.n_steps):
            step_result = self.step(
                coords_ca, initial_coords_ca, z_current,
                prev_rmsd, target_coords,
            )
            result.step_results.append(step_result)
            result.trajectory.append(step_result.new_ca.clone())
            result.total_steps = step_idx + 1

            if verbose:
                combo_label = step_result.combo.label[:20]
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
            print()

        return result
