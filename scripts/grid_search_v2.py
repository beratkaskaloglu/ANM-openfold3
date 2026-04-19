#!/usr/bin/env python3
"""Confidence V2 grid search experiments.

Runs the mode-drive pipeline with different V2 config combinations
and collects per-step metrics for analysis.

Usage (Colab with OF3):
    python scripts/grid_search_v2.py --query path/to/query.json --n-steps 15

Usage (local mock, for smoke-testing):
    python scripts/grid_search_v2.py --mock --n-steps 5

Results are saved to results/grid_search_v2/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from src.mode_drive import ModeDrivePipeline
from src.mode_drive_config import ModeDriveConfig
from src.converter import PairContactConverter


# ── Experiment definitions ───────────────────────────────────────

EXPERIMENTS: dict[str, list[dict]] = {
    "exp1_warmup": [
        {"confidence_warmup_steps": 0},
        {"confidence_warmup_steps": 3, "confidence_warmup_ptm_cutoff": 0.35},
        {"confidence_warmup_steps": 5, "confidence_warmup_ptm_cutoff": 0.30},
        {"confidence_warmup_steps": 10, "confidence_warmup_ptm_cutoff": 0.25},
    ],
    "exp2_rg_filter": [
        {"confidence_rg_max": 999.0},
        {"confidence_rg_max": 3.0},
        {"confidence_rg_max": 2.5},
        {"confidence_rg_max": 2.0},
    ],
    "exp3_max_rejected": [
        {"max_consecutive_rejected": 0},
        {"max_consecutive_rejected": 3},
        {"max_consecutive_rejected": 5},
    ],
    "exp4_alpha_decay": [
        {"rejected_alpha_decay": 1.0},
        {"rejected_alpha_decay": 0.85},
        {"rejected_alpha_decay": 0.7},
        {"rejected_alpha_decay": 0.5},
    ],
    "exp5_pae_cutoff": [
        {"confidence_mean_pae_cutoff": None},
        {"confidence_mean_pae_cutoff": 15.0},
        {"confidence_mean_pae_cutoff": 10.0},
        {"confidence_mean_pae_cutoff": 8.0},
    ],
    "exp6_consensus": [
        {"num_diffusion_samples": 1},
        {"num_diffusion_samples": 3, "confidence_consensus_cutoff": 0.3},
        {"num_diffusion_samples": 3, "confidence_consensus_cutoff": 0.5},
    ],
    "exp7_contact_recon": [
        {"confidence_contact_recon_cutoff": None},
        {"confidence_contact_recon_cutoff": 0.3},
        {"confidence_contact_recon_cutoff": 0.5},
        {"confidence_contact_recon_cutoff": 0.7},
    ],
    "exp8_contact_of3": [
        {"confidence_contact_of3_cutoff": None},
        {"confidence_contact_of3_cutoff": 0.3},
        {"confidence_contact_of3_cutoff": 0.5},
    ],
}


@dataclass
class StepMetrics:
    """Serializable per-step metrics."""
    step: int
    rmsd: float
    ptm: float | None
    plddt_mean: float | None
    ranking: float | None
    rg_ratio: float | None
    contact_recon: float | None
    contact_of3: float | None
    mean_pae: float | None
    has_clash: bool | None
    consensus_score: float | None
    fallback_level: int
    rejected: bool
    df_used: float
    alpha_used: float


@dataclass
class RunResult:
    """Serializable run result."""
    experiment: str
    config_idx: int
    config_overrides: dict
    total_steps: int
    final_rmsd: float
    accepted_steps: int
    rejected_steps: int
    max_fallback_level: int
    step_metrics: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0


def _baseline_config(n_steps: int, strategy: str) -> ModeDriveConfig:
    """Create baseline config for grid search."""
    return ModeDriveConfig(
        n_steps=n_steps,
        combination_strategy=strategy,
        z_mixing_alpha=0.3,
        n_anm_modes=20,
        n_combinations=20,
        max_combo_size=3,
        num_diffusion_samples=1,
        confidence_ptm_cutoff=0.5,
        confidence_plddt_cutoff=70.0,
        confidence_ranking_cutoff=0.5,
        confidence_rg_max=2.5,
        confidence_rg_min=0.3,
        confidence_clash_reject=True,
        enable_confidence_fallback=True,
        fallback_extended_enabled=True,
        # Autostop defaults
        autostop_fallback_levels=(0, 1, 4, 9),
    )


def _apply_overrides(cfg: ModeDriveConfig, overrides: dict) -> ModeDriveConfig:
    """Apply config overrides (returns new config)."""
    import copy
    new_cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if not hasattr(new_cfg, k):
            raise ValueError(f"Unknown config field: {k}")
        setattr(new_cfg, k, v)
    return new_cfg


def _extract_step_metrics(result) -> list[dict]:
    """Extract per-step metrics from ModeDriveResult."""
    metrics = []
    for i, sr in enumerate(result.step_results):
        m = StepMetrics(
            step=i + 1,
            rmsd=sr.rmsd,
            ptm=sr.ptm,
            plddt_mean=float(sr.plddt.mean().item()) if sr.plddt is not None else None,
            ranking=sr.ranking_score,
            rg_ratio=sr.rg_ratio,
            contact_recon=sr.contact_recon,
            contact_of3=sr.contact_of3,
            mean_pae=sr.mean_pae,
            has_clash=sr.has_clash,
            consensus_score=sr.consensus_score,
            fallback_level=sr.fallback_level,
            rejected=sr.rejected,
            df_used=sr.df_used,
            alpha_used=sr.alpha_used,
        )
        metrics.append(asdict(m))
    return metrics


def run_experiment(
    exp_name: str,
    configs: list[dict],
    coords_ca: torch.Tensor,
    zij_trunk: torch.Tensor,
    converter: PairContactConverter,
    diffusion_fn,
    n_steps: int,
    strategy: str,
    target_coords: torch.Tensor | None = None,
) -> list[RunResult]:
    """Run one experiment with multiple config variants."""
    results = []
    for idx, overrides in enumerate(configs):
        cfg = _baseline_config(n_steps, strategy)
        cfg = _apply_overrides(cfg, overrides)

        pipe = ModeDrivePipeline(converter=converter, config=cfg, diffusion_fn=diffusion_fn)

        print(f"\n{'='*70}")
        print(f"[{exp_name}] Config {idx}: {overrides}")
        print(f"{'='*70}")

        t0 = time.time()
        result = pipe.run(
            coords_ca, zij_trunk,
            verbose=True,
            target_coords=target_coords,
        )
        wall_time = time.time() - t0

        step_metrics = _extract_step_metrics(result)
        accepted = sum(1 for sr in result.step_results if not sr.rejected)
        rejected = sum(1 for sr in result.step_results if sr.rejected)
        max_fb = max((sr.fallback_level for sr in result.step_results), default=0)

        rr = RunResult(
            experiment=exp_name,
            config_idx=idx,
            config_overrides=overrides,
            total_steps=result.total_steps,
            final_rmsd=result.step_results[-1].rmsd if result.step_results else 0.0,
            accepted_steps=accepted,
            rejected_steps=rejected,
            max_fallback_level=max_fb,
            step_metrics=step_metrics,
            wall_time_s=wall_time,
        )
        results.append(rr)

        print(f"\n  => {accepted}/{result.total_steps} accepted, "
              f"max_fb={max_fb}, RMSD={rr.final_rmsd:.2f}, "
              f"wall={wall_time:.1f}s")

    return results


def run_combined_experiment(
    best_overrides: dict,
    coords_ca: torch.Tensor,
    zij_trunk: torch.Tensor,
    converter: PairContactConverter,
    diffusion_fn,
    n_steps: int,
    strategy: str,
    target_coords: torch.Tensor | None = None,
) -> RunResult:
    """Run Experiment 9: combined best configs."""
    cfg = _baseline_config(n_steps, strategy)
    cfg = _apply_overrides(cfg, best_overrides)

    pipe = ModeDrivePipeline(converter=converter, config=cfg, diffusion_fn=diffusion_fn)

    print(f"\n{'='*70}")
    print(f"[exp9_combined] Overrides: {best_overrides}")
    print(f"{'='*70}")

    t0 = time.time()
    result = pipe.run(coords_ca, zij_trunk, verbose=True, target_coords=target_coords)
    wall_time = time.time() - t0

    step_metrics = _extract_step_metrics(result)
    accepted = sum(1 for sr in result.step_results if not sr.rejected)
    rejected = sum(1 for sr in result.step_results if sr.rejected)
    max_fb = max((sr.fallback_level for sr in result.step_results), default=0)

    return RunResult(
        experiment="exp9_combined",
        config_idx=0,
        config_overrides=best_overrides,
        total_steps=result.total_steps,
        final_rmsd=result.step_results[-1].rmsd if result.step_results else 0.0,
        accepted_steps=accepted,
        rejected_steps=rejected,
        max_fallback_level=max_fb,
        step_metrics=step_metrics,
        wall_time_s=wall_time,
    )


def _save_results(results: list[RunResult], out_dir: Path):
    """Save results as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


def _print_summary(all_results: list[RunResult]):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print(f"{'Experiment':<20} {'Config':<5} {'Steps':>5} {'Accept':>6} "
          f"{'Reject':>6} {'MaxFB':>5} {'RMSD':>8} {'Time':>6}")
    print(f"{'-'*90}")
    for r in all_results:
        print(f"{r.experiment:<20} {r.config_idx:<5} {r.total_steps:>5} "
              f"{r.accepted_steps:>6} {r.rejected_steps:>6} "
              f"{r.max_fallback_level:>5} {r.final_rmsd:>8.2f} "
              f"{r.wall_time_s:>5.1f}s")
    print(f"{'='*90}")


# ── Mock mode ────────────────────────────────────────────────────

class _MockConverter:
    """Lightweight converter for local testing."""
    def __init__(self):
        self.head = None

    def contact_to_z(self, C):
        N = C.shape[0]
        return torch.randn(N, N, 128)

    def z_to_contact(self, z):
        N = z.shape[-2]
        c = torch.rand(N, N)
        c = 0.5 * (c + c.T)
        c.fill_diagonal_(0.0)
        return c


class _MockDiffResult:
    """Mock diffusion result with V2 fields."""
    def __init__(self, z_mod):
        n = z_mod.shape[0]
        self.best_ca = torch.randn(n, 3) * 10.0
        self.all_ca = self.best_ca.unsqueeze(0)
        self.best_idx = 0
        self.plddt = torch.full((1, n), 85.0 + torch.rand(1).item() * 10)
        ptm_val = 0.3 + torch.rand(1).item() * 0.5
        self.ptm = torch.tensor([ptm_val])
        self.ranking = torch.tensor([0.8 * ptm_val + 0.2 * 0.85])
        self.mean_pae = 5.0 + torch.rand(1).item() * 15.0
        self.has_clash = torch.rand(1).item() > 0.8
        self.consensus_score = None
        self.contact_probs = torch.rand(n, n) * 0.3
        self.pae = None
        self.sample_rmsd = None
        self.sample_rmsf = None


def main():
    parser = argparse.ArgumentParser(description="Confidence V2 grid search")
    parser.add_argument("--query", type=str, help="OF3 query JSON path")
    parser.add_argument("--mock", action="store_true", help="Use mock diffusion (local testing)")
    parser.add_argument("--n-steps", type=int, default=15, help="Pipeline steps per run")
    parser.add_argument("--strategy", type=str, default="autostop", help="Combination strategy")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Which experiments to run (e.g. exp1_warmup exp2_rg_filter). Default: all.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results/grid_search_v2") / timestamp

    # Setup diffusion
    if args.mock:
        print("[MOCK MODE] Using mock diffusion — results are for testing only")
        converter = _MockConverter()
        diffusion_fn = lambda z: _MockDiffResult(z)
        n = 30
        torch.manual_seed(42)
        coords_ca = torch.randn(n, 3) * 10.0
        zij_trunk = torch.randn(n, n, 128)
        target_coords = None
    else:
        if not args.query:
            parser.error("--query is required when not using --mock")
        from src.of3_diffusion import load_of3_diffusion
        diffusion_fn, zij_trunk = load_of3_diffusion(
            query_json=args.query,
            device="cuda",
            num_samples=1,
        )
        # Extract initial CA from OF3 trunk output
        with torch.no_grad():
            init_result = diffusion_fn(zij_trunk)
        if hasattr(init_result, "best_ca"):
            coords_ca = init_result.best_ca
        else:
            coords_ca = init_result
        converter = PairContactConverter.from_pretrained()
        target_coords = None

    # Select experiments
    exp_names = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    # Run
    all_results = []
    for exp_name in exp_names:
        if exp_name not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment '{exp_name}', skipping")
            continue
        configs = EXPERIMENTS[exp_name]
        results = run_experiment(
            exp_name, configs,
            coords_ca, zij_trunk, converter, diffusion_fn,
            args.n_steps, args.strategy, target_coords,
        )
        all_results.extend(results)

    # Summary
    _print_summary(all_results)
    _save_results(all_results, out_dir)

    # Print best per experiment
    print("\n── Best per experiment (by accepted_steps, then RMSD) ──")
    from itertools import groupby
    for exp_name, group in groupby(all_results, key=lambda r: r.experiment):
        group_list = list(group)
        best = max(group_list, key=lambda r: (r.accepted_steps, r.final_rmsd))
        print(f"  {exp_name}: config {best.config_idx} "
              f"({best.accepted_steps} accepted, RMSD={best.final_rmsd:.2f})")


if __name__ == "__main__":
    main()
