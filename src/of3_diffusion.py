"""OF3 diffusion wrapper for mode-drive pipeline.

Wraps OpenFold3's SampleDiffusion module as a callable that takes
a modified pair representation z_mod and returns CA coordinates
with optional confidence scoring (pLDDT, pTM).

Usage (Colab with GPU):
    from src.of3_diffusion import load_of3_diffusion

    diffusion_fn, zij_trunk = load_of3_diffusion(
        query_json="path/to/query.json",   # OF3 inference query
        device="cuda",
        num_samples=5,                      # multi-sample diffusion
    )

    pipeline = ModeDrivePipeline(converter, config, diffusion_fn=diffusion_fn)

The wrapper:
    1. Runs OF3 trunk ONCE to get si_input, si_trunk, zij_trunk, batch
    2. Caches everything except zij_trunk
    3. On each call: replaces zij_trunk with z_mod → runs SampleDiffusion → extracts CA
    4. Optionally runs aux_heads for confidence scores (pLDDT, pTM)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class DiffusionResult:
    """Result from multi-sample diffusion with confidence scoring."""

    all_ca: torch.Tensor          # [K, N, 3] all sample CA coordinates
    best_ca: torch.Tensor         # [N, 3] best sample by ranking score
    best_idx: int                 # index of best sample
    plddt: torch.Tensor | None    # [K, N] per-sample per-residue pLDDT (0-1)
    ptm: torch.Tensor | None      # [K] per-sample pTM (0-1)
    ranking: torch.Tensor | None  # [K] per-sample ranking score


def _ensure_of3_importable() -> None:
    """Add openfold3-repo to sys.path if needed."""
    project_root = Path(__file__).resolve().parent.parent
    of3_dir = project_root / "openfold3-repo"
    for p in [str(project_root), str(of3_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_of3_diffusion(
    query_json: str | Path,
    device: str = "cuda",
    num_rollout_steps: int | None = None,
    num_samples: int = 1,
    use_msa_server: bool = False,
    use_templates: bool = False,
) -> tuple[Callable[[torch.Tensor], DiffusionResult | torch.Tensor], torch.Tensor]:
    """Load OF3 model and return a diffusion_fn for ModeDrivePipeline.

    Runs trunk inference ONCE, then caches si_input, si_trunk, batch.
    Each call to the returned function replaces zij_trunk with z_mod
    and runs only the diffusion sampling (no re-running trunk).

    Args:
        query_json:        Path to OF3 inference query JSON.
        device:            'cuda' or 'cpu'.
        num_rollout_steps: Override diffusion rollout steps (None = use default).
        num_samples:       Number of diffusion samples per call (K). Default 1.
        use_msa_server:    Whether to use MSA server.
        use_templates:     Whether to use templates.

    Returns:
        Tuple of:
            diffusion_fn: Callable([N, N, 128]) -> DiffusionResult (if K>1) or [N, 3].
            zij_trunk:     [N_token, N_token, C_z] original trunk pair representation.
    """
    _ensure_of3_importable()

    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.core.metrics.aggregate_confidence_ranking import (
        get_confidence_scores,
    )

    # ── Load model ──────────────────────────────────────────────
    print("[OF3] Loading model...")

    config = InferenceExperimentConfig(
        output_writer_settings={
            "structure_format": "cif",
            "write_latent_outputs": True,
            "write_features": True,
            "write_full_confidence_scores": False,
        },
        model_update={
            "custom": {
                "settings": {
                    "memory": {
                        "eval": {
                            "use_deepspeed_evo_attention": False,
                            "use_cueq_triangle_kernels": False,
                            "use_triton_triangle_kernels": False,
                        }
                    }
                }
            }
        },
    )

    runner = InferenceExperimentRunner(
        config,
        num_diffusion_samples=num_samples,
        num_model_seeds=1,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    runner.setup()

    lightning_module = runner.lightning_module
    model = lightning_module.model
    model = model.to(device)
    model.eval()

    # Cache the runner config for confidence computation
    runner_config = lightning_module.config

    print("[OF3] Model loaded.")

    # ── Build batch from query ─────────────────────────────────
    print("[OF3] Preparing input batch...")

    query_set = InferenceQuerySet.from_json(str(query_json))

    runner.inference_query_set = query_set
    data_module = runner.lightning_data_module

    data_module.prepare_data()
    data_module.setup(stage="predict")

    predict_dl = data_module.predict_dataloader()
    cached_batch = next(iter(predict_dl))

    cached_batch = tensor_tree_map(
        lambda t: t.to(device) if isinstance(t, torch.Tensor) else t,
        cached_batch,
    )

    print("[OF3] Input batch ready.")

    # ── Run trunk once to get cached representations ────────────
    print("[OF3] Running trunk inference (one-time)...")

    with torch.no_grad():
        num_cycles = model.shared.num_recycles + 1
        si_input_cached, si_trunk_cached, zij_trunk_cached = model.run_trunk(
            batch=cached_batch, num_cycles=num_cycles,
            inplace_safe=True,
        )
        # Add sample dimension: [*, N_token, C] -> [*, 1, N_token, C]
        si_input_cached = si_input_cached.unsqueeze(1)
        si_trunk_cached = si_trunk_cached.unsqueeze(1)
        zij_trunk_cached = zij_trunk_cached.unsqueeze(1)

    # Expand batch for sampling dimension (same as model.forward does)
    ref_space_uid_to_perm = cached_batch.pop("ref_space_uid_to_perm", None)
    cached_batch = tensor_tree_map(lambda t: t.unsqueeze(1), cached_batch)
    if ref_space_uid_to_perm is not None:
        cached_batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm

    # Get token→atom mapping for CA extraction
    start_atom_index = cached_batch.get("start_atom_index")

    N_token = si_trunk_cached.shape[-2]
    C_z = zij_trunk_cached.shape[-1]

    # Noise schedule
    no_rollout_steps = (
        num_rollout_steps
        if num_rollout_steps is not None
        else model.shared.diffusion.no_full_rollout_steps
    )
    noise_schedule = create_noise_schedule(
        no_rollout_steps=no_rollout_steps,
        **model.config.architecture.noise_schedule,
        dtype=si_trunk_cached.dtype,
        device=torch.device(device),
    )

    K = num_samples

    print(f"[OF3] Trunk cached: N_token={N_token}, C_z={C_z}")
    print(f"[OF3] Diffusion: {no_rollout_steps} steps, {K} samples")

    # ── Build closure ───────────────────────────────────────────
    def _extract_ca_multi(atom_positions: torch.Tensor) -> torch.Tensor:
        """Extract CA coordinates from all-atom positions for K samples.

        Args:
            atom_positions: [1, K, N_atom, 3]

        Returns:
            ca_coords: [K, N_token, 3]
        """
        # Remove batch dim: [K, N_atom, 3]
        pos = atom_positions.squeeze(0)

        if pos.dim() == 2:
            # Single sample: [N_atom, 3] → [1, N_atom, 3]
            pos = pos.unsqueeze(0)

        k_actual = pos.shape[0]
        cas = []
        for ki in range(k_actual):
            p = pos[ki]  # [N_atom, 3]
            if start_atom_index is not None:
                idx = start_atom_index.squeeze().long()  # [N_token]
                cas.append(p[idx])
            else:
                cas.append(p[:N_token])

        return torch.stack(cas, dim=0)  # [K, N_token, 3]

    def _compute_confidence(
        atom_positions: torch.Tensor,
        zij_modified: torch.Tensor,
    ) -> dict | None:
        """Run aux_heads and compute confidence scores.

        Returns dict with pLDDT [K, N] and pTM [K], or None on failure.
        """
        try:
            output = {
                "si_trunk": si_trunk_cached,
                "zij_trunk": zij_modified,
                "atom_positions_predicted": atom_positions,
            }

            cast_dtype = si_trunk_cached.dtype
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=cast_dtype):
                aux_output = model.aux_heads(
                    batch=cached_batch,
                    si_input=si_input_cached,
                    output=output,
                    use_zij_trunk_embedding=True,
                )
                output.update(aux_output)

            confidence = get_confidence_scores(
                batch=cached_batch,
                outputs=output,
                config=runner_config,
            )
            return confidence
        except Exception as e:
            print(f"[OF3] Warning: confidence computation failed: {e}")
            return None

    @torch.no_grad()
    def diffusion_fn(z_mod: torch.Tensor) -> DiffusionResult | torch.Tensor:
        """Run OF3 diffusion with modified pair representation.

        Args:
            z_mod: [N, N, 128] modified pair representation.

        Returns:
            DiffusionResult if num_samples > 1 or confidence available,
            otherwise [N, 3] CA coordinates (backward compat).
        """
        # Reshape z_mod to match OF3 format: [1, 1, N_token, N_token, C_z]
        zij_modified = z_mod.unsqueeze(0).unsqueeze(0).to(device)

        # Run SampleDiffusion with modified z_ij
        atom_positions = model.sample_diffusion(
            batch=cached_batch,
            si_input=si_input_cached,
            si_trunk=si_trunk_cached,
            zij_trunk=zij_modified,
            noise_schedule=noise_schedule,
            no_rollout_samples=K,
            use_conditioning=True,
        )
        # atom_positions: [1, K, N_atom, 3]

        # Extract CA for all samples: [K, N_token, 3]
        all_ca = _extract_ca_multi(atom_positions)

        if K == 1:
            # Single sample, no confidence → backward compatible
            confidence = _compute_confidence(atom_positions, zij_modified)
            ca = all_ca[0].to(z_mod.device)

            if confidence is not None:
                plddt_raw = confidence.get("pLDDT")
                ptm_raw = confidence.get("pTM")
                plddt = plddt_raw.squeeze() if plddt_raw is not None else None
                ptm = ptm_raw.squeeze() if ptm_raw is not None else None

                ranking_val = 0.0
                if ptm is not None:
                    ptm_f = ptm.item() if ptm.dim() == 0 else ptm.mean().item()
                    plddt_f = plddt.mean().item() if plddt is not None else 0.0
                    ranking_val = 0.8 * ptm_f + 0.2 * plddt_f

                return DiffusionResult(
                    all_ca=all_ca.to(z_mod.device),
                    best_ca=ca,
                    best_idx=0,
                    plddt=plddt.unsqueeze(0) if plddt is not None else None,
                    ptm=torch.tensor([ptm_f]) if ptm is not None else None,
                    ranking=torch.tensor([ranking_val]),
                )
            else:
                # No confidence available → return raw coords
                return ca

        # Multi-sample: compute confidence for ranking
        confidence = _compute_confidence(atom_positions, zij_modified)

        if confidence is not None:
            plddt_raw = confidence.get("pLDDT")  # [K, N] or similar
            ptm_raw = confidence.get("pTM")       # [K] or similar

            # Normalize shapes
            if plddt_raw is not None:
                plddt = plddt_raw.squeeze(0) if plddt_raw.dim() > 2 else plddt_raw
                if plddt.dim() == 1:
                    plddt = plddt.unsqueeze(0).expand(K, -1)
            else:
                plddt = None

            if ptm_raw is not None:
                ptm = ptm_raw.squeeze()
                if ptm.dim() == 0:
                    ptm = ptm.unsqueeze(0).expand(K)
            else:
                ptm = None

            # Ranking: 0.8 * pTM + 0.2 * mean(pLDDT)
            if ptm is not None and plddt is not None:
                ranking = 0.8 * ptm + 0.2 * plddt.mean(dim=-1)
            elif ptm is not None:
                ranking = ptm
            else:
                ranking = plddt.mean(dim=-1) if plddt is not None else torch.zeros(K)

            best_idx = ranking.argmax().item()
        else:
            # No confidence → pick first sample
            plddt = None
            ptm = None
            ranking = None
            best_idx = 0

        best_ca = all_ca[best_idx].to(z_mod.device)

        return DiffusionResult(
            all_ca=all_ca.to(z_mod.device),
            best_ca=best_ca,
            best_idx=best_idx,
            plddt=plddt,
            ptm=ptm,
            ranking=ranking,
        )

    # Return zij_trunk without batch/sample dims: [N_token, N_token, C_z]
    zij_trunk_out = zij_trunk_cached.squeeze(0).squeeze(0).detach().cpu()

    print("[OF3] Diffusion function ready.")
    return diffusion_fn, zij_trunk_out
