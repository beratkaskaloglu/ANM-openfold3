"""Configuration and result dataclasses for the ANM Mode-Drive pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .mode_combinator import ModeCombo

CombinationStrategy = Literal[
    "collectivity", "grid", "random", "targeted", "manual", "autostop",
]


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
    combination_strategy: CombinationStrategy = "collectivity"
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
    autostop_back_off_fraction: float | None = None  # if set, back_off = int(tk * fraction); overrides fixed back_off
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
    autostop_fallback_pick_fractions: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)  # L1: pk = tk * frac (1.0=turn point, 0.5=halfway, etc.)
    autostop_fallback_eps_E_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)
    autostop_fallback_eps_N_scales: tuple[float, ...] = (1.0, 2.0, 0.5, 0.25)
    autostop_fallback_patience_deltas: tuple[int, ...] = (0, -1, 1, 2)
    autostop_fallback_smooth_w_deltas: tuple[int, ...] = (0, 4, -2, 8)
    autostop_fallback_warmup_frac_scales: tuple[float, ...] = (1.0, 1.5, 0.5)
    autostop_fallback_crash_window_scales: tuple[float, ...] = (1.0, 2.0, 0.5)
    autostop_fallback_crash_threshold_adds: tuple[int, ...] = (0, 2, -2)
    autostop_fallback_alpha_scales: tuple[float, ...] = (1.0, 0.5, 0.35, 0.25, 0.15, 0.07)  # L4/L7 z_mixing_alpha multipliers
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
    fallback_level: int = 0                   # 0=normal, 1-8=fallback level, 9=forced-accept
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
