"""ANM Mode-Drive Pipeline: iterative conformational exploration.

Combines ANM normal-mode displacements with OF3 diffusion to generate
physically meaningful protein conformational ensembles.

Pipeline:
    coords -> ANM modes -> collectivity rank -> displace -> contact -> z_pseudo -> blend z -> diffusion -> new coords -> repeat

Strategy (goal: maximize displacement from initial structure):
    1. Compute ANM modes and rank combinations by collectivity
    2. Try most collective combo first with df_min
    3. If RMSD from initial doesn't increase -> try next combo
    4. If all combos exhausted -> escalate df toward df_max
    5. Run exactly n_steps iterations (no early stopping)

RMSD is measured from INITIAL structure -- higher = more exploration.
"""

from __future__ import annotations

from typing import Callable

import torch

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

# Re-export config/result dataclasses and utility functions so existing
# `from src.mode_drive import ModeDriveConfig, ...` keeps working.
from .mode_drive_config import CombinationStrategy, ModeDriveConfig, ModeDriveResult, StepResult  # noqa: F401
from .mode_drive_utils import (  # noqa: F401
    compute_rmsd,
    contact_to_distance,
    classical_mds,
    kabsch_superimpose,
    make_pseudo_diffusion,
    tm_score,
)

# Lazy import — DiffusionResult is only needed when OF3 is available.
try:
    from .of3_diffusion import DiffusionResult
except (ImportError, Exception):
    DiffusionResult = None  # type: ignore[assignment,misc]


_SENTINEL = object()  # default-from-config marker for _replay


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
        structure_ctx=None,  # autostop_adapter.StructureContext | None
    ) -> None:
        self.converter = converter
        self.config = config or ModeDriveConfig()
        self.diffusion_fn = diffusion_fn
        self.structure_ctx = structure_ctx

        # Cache of most-recent autostop trace for cheap fallback replay.
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
        """Build a single combo from user-specified modes."""
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
        """Apply delta_z to trunk z_ij in the configured direction."""
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
        """Evaluate a single mode combination (ANM displace -> downstream)."""
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
        displaced -> contact -> z_pseudo -> blend -> diffusion -> StepResult.
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

        # ── Confidence V2 metrics ──
        mean_pae_out = None
        has_clash_out = None
        consensus_out = None
        contact_recon_out = None
        contact_of3_out = None
        rg_ratio_out = None

        if self.diffusion_fn is not None and hasattr(diff_result, "mean_pae"):
            mean_pae_out = diff_result.mean_pae
            has_clash_out = diff_result.has_clash
            consensus_out = diff_result.consensus_score

            # Contact reconstruction: displaced→contact vs new_ca→contact
            output_contact = coords_to_contact(new_ca, cfg.contact_r_cut, cfg.contact_tau)
            flat_in = contact.flatten().float()
            flat_out = output_contact.flatten().float()
            if flat_in.std() > 1e-8 and flat_out.std() > 1e-8:
                contact_recon_out = float(
                    torch.corrcoef(torch.stack([flat_in, flat_out]))[0, 1].item()
                )

            # OF3 distogram contact vs input contact
            if diff_result.contact_probs is not None:
                flat_of3 = diff_result.contact_probs.flatten().float()
                if flat_of3.std() > 1e-8 and flat_in.std() > 1e-8:
                    contact_of3_out = float(
                        torch.corrcoef(torch.stack([flat_in, flat_of3]))[0, 1].item()
                    )

        # Rg ratio (always computable)
        N = new_ca.shape[0]
        centered = new_ca.float() - new_ca.float().mean(0)
        rg_obs = float(centered.pow(2).sum(1).mean().sqrt().item())
        rg_exp = 2.2 * (N ** 0.38)
        rg_ratio_out = rg_obs / max(rg_exp, 1e-6)

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
            mean_pae=mean_pae_out,
            has_clash=has_clash_out,
            consensus_score=consensus_out,
            contact_recon=contact_recon_out,
            contact_of3=contact_of3_out,
            rg_ratio=rg_ratio_out,
        )

    # ─────────────── Autostop strategy ───────────────

    def _autostop_params(self):
        """Build AutostopParams dataclass from current config values."""
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
            back_off_fraction=cfg.autostop_back_off_fraction,
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
        """Build a placeholder ModeCombo for autostop results."""
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
        """Run one autostop MD + downstream evaluation."""
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

        trace_stats = {
            "n_saves": int(len(trace.steps)),
            "total_mdsteps_requested": trace.total_mdsteps_requested,
            "save_every": trace.save_every,
        }

        return self._autostop_downstream_from_pick(
            pick, initial_coords_ca, zij_trunk,
            eigenvalues, eigenvectors, b_factors,
            step_idx, fallback_level=0, trace_stats=trace_stats,
        )

    @torch.no_grad()
    def step(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        prev_rmsd: float = 0.0,
        target_coords: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> StepResult:
        """Run a single ANM mode-drive iteration with df escalation."""
        cfg = self.config

        # Autostop strategy — completely bypasses ANM mode-combo displacement.
        if cfg.combination_strategy == "autostop":
            return self._autostop_step(
                coords_ca=coords_ca,
                initial_coords_ca=initial_coords_ca,
                zij_trunk=zij_trunk,
                step_idx=step_idx,
            )

        # Step 1: ANM Hessian + Eigendecomposition
        H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
        eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
        b_factors = anm_bfactors(eigenvalues, eigenvectors)
        n_modes = eigenvalues.shape[0]

        # Score function
        has_target = target_coords is not None

        def _score(result: StepResult) -> float:
            if has_target:
                return -compute_rmsd(result.new_ca, target_coords)
            return result.rmsd

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
        current_df = cfg.df_min
        best_overall: StepResult | None = None
        best_overall_score = -float("inf")

        while current_df <= cfg.df_max:
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

                    if has_target:
                        improved = True
                        break
                    elif result.rmsd > prev_rmsd:
                        improved = True
                        break

            if improved:
                break

            current_df *= cfg.df_escalation_factor

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

    def _confidence_check(self, result: StepResult, step_idx: int = 0) -> tuple[bool, str]:
        """Check if step result passes ALL confidence cutoffs (AND logic).

        Returns (ok, reason). reason is "" if ok, otherwise the first failing check.
        Supports warmup period with relaxed cutoffs and V2 physical filters.
        """
        cfg = self.config

        if (
            result.ptm is None
            and result.plddt is None
            and result.ranking_score is None
        ):
            return True, ""

        # Step-adaptive cutoffs (warmup)
        in_warmup = cfg.confidence_warmup_steps > 0 and step_idx < cfg.confidence_warmup_steps
        ptm_cut = cfg.confidence_warmup_ptm_cutoff if in_warmup else cfg.confidence_ptm_cutoff
        rank_cut = cfg.confidence_warmup_ranking_cutoff if in_warmup else cfg.confidence_ranking_cutoff

        # Physical filters (always active)
        if result.rg_ratio is not None:
            if result.rg_ratio > cfg.confidence_rg_max:
                return False, f"Rg={result.rg_ratio:.2f}>{cfg.confidence_rg_max}"
            if result.rg_ratio < cfg.confidence_rg_min:
                return False, f"Rg={result.rg_ratio:.2f}<{cfg.confidence_rg_min}"

        if cfg.confidence_clash_reject and result.has_clash:
            return False, "clash"

        # Core metrics (warmup-adjusted)
        if result.ptm is not None and result.ptm < ptm_cut:
            return False, f"pTM={result.ptm:.3f}<{ptm_cut}"
        if result.plddt is not None:
            mean_plddt = result.plddt.mean().item()
            if mean_plddt < cfg.confidence_plddt_cutoff:
                return False, f"pLDDT={mean_plddt:.1f}<{cfg.confidence_plddt_cutoff}"
        if (
            result.ranking_score is not None
            and result.ranking_score < rank_cut
        ):
            return False, f"rank={result.ranking_score:.3f}<{rank_cut}"

        # V2 metrics (None cutoff = disabled)
        if cfg.confidence_mean_pae_cutoff is not None and result.mean_pae is not None:
            if result.mean_pae > cfg.confidence_mean_pae_cutoff:
                return False, f"mPAE={result.mean_pae:.1f}>{cfg.confidence_mean_pae_cutoff}"
        if cfg.confidence_consensus_cutoff is not None and result.consensus_score is not None:
            if result.consensus_score < cfg.confidence_consensus_cutoff:
                return False, f"cons={result.consensus_score:.3f}<{cfg.confidence_consensus_cutoff}"
        if cfg.confidence_contact_recon_cutoff is not None and result.contact_recon is not None:
            if result.contact_recon < cfg.confidence_contact_recon_cutoff:
                return False, f"cR={result.contact_recon:.3f}<{cfg.confidence_contact_recon_cutoff}"
        if cfg.confidence_contact_of3_cutoff is not None and result.contact_of3 is not None:
            if result.contact_of3 < cfg.confidence_contact_of3_cutoff:
                return False, f"cOF3={result.contact_of3:.3f}<{cfg.confidence_contact_of3_cutoff}"

        return True, ""

    def _confidence_ok(self, result: StepResult, step_idx: int = 0) -> bool:
        """Check if step result passes ALL confidence cutoffs (AND logic)."""
        ok, _ = self._confidence_check(result, step_idx)
        return ok

    @torch.no_grad()
    def step_with_fallback(
        self,
        coords_ca: torch.Tensor,
        initial_coords_ca: torch.Tensor,
        zij_trunk: torch.Tensor,
        prev_rmsd: float = 0.0,
        target_coords: torch.Tensor | None = None,
        step_idx: int = 0,
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
                step_idx=step_idx,
                target_coords=target_coords,
            )

        # Save originals
        orig_df = cfg.df
        orig_df_min = cfg.df_min
        orig_max_combo = cfg.max_combo_size
        orig_alpha = cfg.z_mixing_alpha

        best_result: StepResult | None = None
        best_ranking = -1.0

        def _track(result: StepResult) -> bool:
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
            # Reuse ANM modes from L0's step result (avoids redundant O(N^3) eigendecomposition)
            eigenvalues = result.eigenvalues
            eigenvectors = result.eigenvectors
            b_factors = result.b_factors
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

            # ── Level 5: Extended grid search (combo x df x alpha) ──
            if cfg.fallback_extended_enabled:
                cfg.max_combo_size = cfg.fallback_max_combo_size
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
            assert best_result is not None
            best_result.rejected = True
            return best_result

        finally:
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
        """Build a StepResult from an autostop pick -- reused across fallback levels."""
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
        target_coords: torch.Tensor | None = None,
    ) -> StepResult:
        """Run autostop strategy with L0-L9 confidence-guided fallback.

        Levels (see docs/plans/autostop_integration.md S2.4):
            L0: baseline autostop                                (MD + monitor + ds)
            L1: back_off adds  -- replay_monitor                  (no MD)
            L2: v_magnitude scales -- re-run MD                   (RERUN)
            L3: eps_E/eps_N scales -- replay_monitor              (no MD)
            L4: z_alpha scales -- re-run downstream only          (no MD, no monitor)
            L5: patience deltas -- replay_monitor                 (no MD)
            L6: smooth_w deltas -- replay_monitor                 (no MD)
            L7: grid (v x back_off x alpha) -- re-run MD          (RERUN; capped)
            L8: grid (eps_E x eps_N x patience x smooth_w)       (no MD; capped)
            L9: skip -- forced-accept best-so-far

        `cfg.autostop_fallback_levels` subsets which levels are attempted.
        """
        from .autostop_adapter import AutostopParams, run_autostop_from_tensor, replay_monitor

        if self.structure_ctx is None:
            raise RuntimeError(
                "combination_strategy='autostop' with fallback requires a StructureContext."
            )

        cfg = self.config

        # Save originals for try/finally restoration
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

            # L9 Rg guard: reject physically nonsensical structures from candidate pool
            if result.rg_ratio is not None and result.rg_ratio > cfg.confidence_rg_max:
                if cfg.autostop_verbose:
                    skip_extra = f"  RMSD_init={result.rmsd:.2f}A"
                    if target_coords is not None and result.new_ca is not None:
                        skip_extra += (
                            f"  RMSD_tgt={compute_rmsd(result.new_ca, target_coords):.2f}A"
                            f"  TM_tgt={tm_score(result.new_ca, target_coords):.3f}"
                        )
                    print(
                        f"      [FB L{level}] SKIP  Rg={result.rg_ratio:.1f} "
                        f"> {cfg.confidence_rg_max} — {desc}{skip_extra}"
                    )
                return False

            r_score = result.ranking_score if result.ranking_score is not None else 0.0
            if r_score > best_ranking:
                best_ranking = r_score
                best_result = result
            ok = self._confidence_ok(result, step_idx=step_idx)
            if cfg.autostop_verbose:
                ptm_str = (
                    f"{result.ptm:.3f}" if result.ptm is not None else "  -  "
                )
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
                if not ok:
                    _, rej_reason = self._confidence_check(result, step_idx=step_idx)
                    tag = f"FAIL[{rej_reason}]" if rej_reason else "FAIL"
                else:
                    tag = "PASS"
                tgt_str = ""
                if target_coords is not None and result.new_ca is not None:
                    rmsd_t = compute_rmsd(result.new_ca, target_coords)
                    tm_t = tm_score(result.new_ca, target_coords)
                    tgt_str = f"  RMSD_tgt={rmsd_t:.2f}A  TM_tgt={tm_t:.3f}"
                # V2 metrics
                v2_str = ""
                if result.rg_ratio is not None:
                    v2_str += f"  Rg={result.rg_ratio:.2f}"
                if result.contact_recon is not None:
                    v2_str += f"  cR={result.contact_recon:.2f}"
                if result.contact_of3 is not None:
                    v2_str += f"  cOF3={result.contact_of3:.2f}"
                if result.mean_pae is not None:
                    v2_str += f"  mPAE={result.mean_pae:.1f}"
                if result.consensus_score is not None:
                    v2_str += f"  cons={result.consensus_score:.2f}"
                print(
                    f"      [FB L{level}] {tag}  {desc:<24s}  "
                    f"pk={pk!s:>5} tk={tk!s:>3}  "
                    f"pTM={ptm_str}  pLDDT={plddt_str}  rank={rank_str}  "
                    f"RMSD_init={result.rmsd:.2f}A{v2_str}{tgt_str}"
                )
            return ok

        # Compute ANM modes once for diagnostics
        H = build_hessian(coords_ca, cfg.anm_cutoff, cfg.anm_gamma, cfg.anm_tau)
        eigenvalues, eigenvectors = anm_modes(H, cfg.n_anm_modes)
        b_factors = anm_bfactors(eigenvalues, eigenvectors)

        device = coords_ca.device
        dtype = coords_ca.dtype
        enabled = set(cfg.autostop_fallback_levels)

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

        def _replay(
            monitor_params: dict,
            back_off: int,
            back_off_fraction: float | None = _SENTINEL,
        ):
            assert trace is not None
            frac = cfg.autostop_back_off_fraction if back_off_fraction is _SENTINEL else back_off_fraction
            return replay_monitor(
                trace=trace,
                monitor_params=monitor_params,
                back_off=int(back_off),
                device=device,
                dtype=dtype,
                back_off_fraction=frac,
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
                    desc=f"baseline v={orig_v:.2f} a={orig_alpha:.2f} back={orig_back}",
                ):
                    return result0

            # If L0 was skipped we still need a trace for replay-based levels.
            if trace is None:
                params_boot = self._autostop_params()
                pick_boot, tr_boot = _run_md(params_boot)
                _ = pick_boot

            # ── L1: pick_fractions — progressively earlier frames ─────
            # pk = tk * frac: 1.0 = turn point, 0.5 = halfway back, etc.
            # This replaces fixed back_off adds with proportional stepping.
            if 1 in enabled:
                base_mon = self._autostop_params().monitor_only()
                for frac in cfg.autostop_fallback_pick_fractions:
                    if frac >= 1.0:
                        continue  # baseline (L0) already tried
                    # back_off_fraction = 1 - frac  →  pk = tk * frac
                    bo_frac = 1.0 - float(frac)
                    pick = _replay(base_mon, orig_back, back_off_fraction=bo_frac)
                    res = self._autostop_downstream_from_pick(
                        pick, initial_coords_ca, zij_trunk,
                        eigenvalues, eigenvectors, b_factors,
                        step_idx, fallback_level=1, trace_stats=_trace_stats(trace),
                    )
                    if _track(
                        res, level=1,
                        desc=f"pk=tk*{frac:.2f} (bo_frac={bo_frac:.2f})",
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
                        desc=f"v_mag x{scale:.2f}->{cfg.autostop_v_magnitude:.2f} (RERUN MD)",
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
                            desc=f"eps_E x{se:.1f} eps_N x{sn:.1f}",
                        ):
                            return res

            # ── L4: alpha scales (no MD, no monitor rerun) ──
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
                        desc=f"a x{scale:.2f}->{cfg.z_mixing_alpha:.2f}",
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
                        desc=f"patience {orig_patience}->{new_pat} (d={delta:+d})",
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
                        desc=f"smooth_w {orig_smooth_w}->{new_sw} (d={delta:+d})",
                    ):
                        return res

            # ── L7: grid (v x back_off x alpha) -- RERUN MD per v ─────
            if 7 in enabled:
                cap = max(1, int(cfg.autostop_fallback_grid_cap))
                tried = 0
                for vs in cfg.autostop_fallback_v_scales:
                    if tried >= cap:
                        break
                    cfg.autostop_v_magnitude = orig_v * float(vs)
                    params = self._autostop_params()
                    pick_base, tr_v = _run_md(params)
                    for add in cfg.autostop_fallback_back_off_adds:
                        if tried >= cap:
                            break
                        for a_scale in cfg.autostop_fallback_alpha_scales:
                            if tried >= cap:
                                break
                            if vs == 1.0 and add == 0 and a_scale == 1.0:
                                continue
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
                                    f"grid v x{vs:.2f} back+{add} a x{a_scale:.2f} "
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
                                        f"grid eps_E x{se:.1f} eps_N x{sn:.1f} "
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
                        f"      [FB L9] FORCE  all attempts failed confidence -> "
                        f"accepting best-so-far (rank={best_ranking:.3f}, "
                        f"RMSD_init={best_result.rmsd:.2f}A)"
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
                    f"      [FB L9] FORCE  no prior attempts tracked -> "
                    f"hard baseline accepted (rejected=True)"
                )
            return res

        finally:
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
        """Run the full iterative ANM mode-drive pipeline."""
        cfg = self.config
        result = ModeDriveResult()
        result.trajectory.append(initial_coords_ca.clone())

        coords_ca = initial_coords_ca
        z_current = zij_trunk
        prev_rmsd = 0.0
        consecutive_rejected = 0
        orig_alpha = cfg.z_mixing_alpha

        use_fallback = cfg.enable_confidence_fallback

        if verbose:
            N = initial_coords_ca.shape[0]
            has_target = target_coords is not None
            header = f"{'Step':>6} {'RMSD_init':>10} {'df':>6} {'a':>5} {'Combo':>20}"
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
            init_line = f"{'Init':>6} {'0.000':>10} {'-':>6} {'-':>5} {'-':>20}"
            if has_target:
                rmsd_tgt = compute_rmsd(initial_coords_ca, target_coords)
                tm_tgt = tm_score(initial_coords_ca, target_coords)
                init_line += f" {rmsd_tgt:>10.3f} {tm_tgt:>8.4f}"
            init_line += f" {'-':>6} {'-':>6} {'-':>3}"
            print(init_line)

        for step_idx in range(cfg.n_steps):
            if use_fallback:
                step_result = self.step_with_fallback(
                    coords_ca, initial_coords_ca, z_current,
                    prev_rmsd, target_coords, step_idx=step_idx,
                )
            else:
                step_result = self.step(
                    coords_ca, initial_coords_ca, z_current,
                    prev_rmsd, target_coords, step_idx=step_idx,
                )
            result.step_results.append(step_result)
            result.trajectory.append(step_result.new_ca.clone())
            result.total_steps = step_idx + 1

            if verbose:
                combo_label = step_result.combo.label[:20]
                ptm_str = f"{step_result.ptm:.3f}" if step_result.ptm is not None else "-"
                plddt_str = (
                    f"{step_result.plddt.mean().item():.3f}"
                    if step_result.plddt is not None else "-"
                )
                if step_result.rejected:
                    fb_str = f"!{step_result.fallback_level}"
                elif step_result.fallback_level > 0:
                    fb_str = str(step_result.fallback_level)
                else:
                    fb_str = "-"

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
                # V2 compact summary
                v2_parts = []
                if step_result.rg_ratio is not None:
                    v2_parts.append(f"Rg={step_result.rg_ratio:.1f}")
                if step_result.contact_recon is not None:
                    v2_parts.append(f"cR={step_result.contact_recon:.2f}")
                if step_result.mean_pae is not None:
                    v2_parts.append(f"mPAE={step_result.mean_pae:.0f}")
                if v2_parts:
                    line += "  " + " ".join(v2_parts)
                # Show reject reason
                if step_result.rejected:
                    _, reason = self._confidence_check(step_result, step_idx=step_idx)
                    if reason:
                        line += f"  [{reason}]"
                print(line)

            # Update for next step only if not rejected.
            if not step_result.rejected:
                prev_rmsd = step_result.rmsd
                coords_ca = step_result.new_ca
                z_current = step_result.z_modified
                consecutive_rejected = 0
                cfg.z_mixing_alpha = orig_alpha  # restore on success
            else:
                consecutive_rejected += 1
                # Alpha decay on rejection (stall prevention)
                if cfg.rejected_alpha_decay < 1.0:
                    cfg.z_mixing_alpha = max(0.02, cfg.z_mixing_alpha * cfg.rejected_alpha_decay)
                # Max consecutive rejected check
                if cfg.max_consecutive_rejected > 0 and consecutive_rejected >= cfg.max_consecutive_rejected:
                    if verbose:
                        print(
                            f"  STOP: {consecutive_rejected} consecutive rejected steps "
                            f"— pipeline stalled"
                        )
                    break

        # Restore alpha to original after loop (in case alpha_decay was applied)
        cfg.z_mixing_alpha = orig_alpha

        if verbose and result.step_results:
            print(f"{'-'*len(header)}")
            final_rmsd = result.step_results[-1].rmsd
            print(f"Final RMSD from initial: {final_rmsd:.3f} A")
            if has_target:
                final_tm = tm_score(result.step_results[-1].new_ca, target_coords)
                print(f"Final TM-score vs target: {final_tm:.4f}")

            # Confidence summary
            ptms = [r.ptm for r in result.step_results if r.ptm is not None]
            if ptms:
                print(f"pTM range: {min(ptms):.3f} - {max(ptms):.3f}")
            fallbacks = [r.fallback_level for r in result.step_results if r.fallback_level > 0]
            if fallbacks:
                print(f"Fallback triggered: {len(fallbacks)}/{len(result.step_results)} steps (levels: {fallbacks})")
            rejected = [i for i, r in enumerate(result.step_results) if r.rejected]
            if rejected:
                print(f"Rejected (base preserved): {len(rejected)}/{len(result.step_results)} steps (indices: {rejected})")
            print()

        return result
