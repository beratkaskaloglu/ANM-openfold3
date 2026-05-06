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

    # Confidence V2
    pae: torch.Tensor | None = None           # [N, N] best sample PAE matrix
    contact_probs: torch.Tensor | None = None  # [N, N] OF3 distogram contact probs
    has_clash: bool | None = None              # OF3 clash detection
    mean_pae: float | None = None              # mean PAE of best sample
    sample_rmsd: torch.Tensor | None = None    # [K*(K-1)/2] pairwise inter-sample RMSD
    sample_rmsf: torch.Tensor | None = None    # [N] per-residue RMSF across K samples
    consensus_score: float | None = None       # 1/(1 + mean_inter_sample_rmsd)


# ── V2 confidence helpers (module-level for testability) ──────────


def _extract_pae(
    confidence: dict | None, best_idx: int = 0,
) -> tuple[torch.Tensor | None, float | None]:
    """Extract PAE matrix for best sample. Returns (pae, mean_pae)."""
    if confidence is None:
        return None, None
    pae_raw = confidence.get("pae")
    if pae_raw is None:
        return None, None
    # pae_raw: [1, K, N, N] or [1, N, N]
    pae = pae_raw.squeeze(0)  # [K, N, N] or [N, N]
    if pae.dim() == 3:
        pae_best = pae[best_idx]  # [N, N]
    else:
        pae_best = pae  # [N, N]
    mean_pae = float(pae_best.mean().item())
    return pae_best.detach().cpu(), mean_pae


def _extract_contact_probs(confidence: dict | None) -> torch.Tensor | None:
    """Extract OF3 distogram-derived contact probabilities."""
    if confidence is None:
        return None
    cp_raw = confidence.get("contact_probs")
    if cp_raw is None:
        return None
    # [1, N, N] → [N, N]
    return cp_raw.squeeze(0).detach().cpu()


def _extract_has_clash(confidence: dict | None) -> bool | None:
    """Extract OF3 clash detection flag."""
    if confidence is None:
        return None
    hc = confidence.get("has_clash")
    if hc is None:
        return None
    if isinstance(hc, torch.Tensor):
        return bool(hc.item())
    return bool(hc)


def _compute_sample_consistency(
    all_ca: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float | None]:
    """Compute inter-sample RMSD and per-residue RMSF across K samples.

    Returns (sample_rmsd, sample_rmsf, consensus_score).
    All None if K == 1.
    """
    k_actual = all_ca.shape[0]
    if k_actual < 2:
        return None, None, None

    coords = all_ca.float()  # [K, N, 3]

    # Pairwise RMSD
    pairwise = []
    for i in range(k_actual):
        for j in range(i + 1, k_actual):
            diff = coords[i] - coords[j]
            rmsd_ij = float(diff.pow(2).sum(-1).mean().sqrt().item())
            pairwise.append(rmsd_ij)
    sample_rmsd = torch.tensor(pairwise)
    mean_inter = sample_rmsd.mean().item()
    consensus = 1.0 / (1.0 + mean_inter)

    # Per-residue RMSF
    mean_pos = coords.mean(dim=0)  # [N, 3]
    deviations = coords - mean_pos.unsqueeze(0)  # [K, N, 3]
    msf = deviations.pow(2).sum(dim=-1).mean(dim=0)  # [N]
    sample_rmsf = msf.sqrt()

    return sample_rmsd, sample_rmsf, consensus


def _ensure_of3_importable() -> None:
    """Add openfold3-repo to sys.path if needed."""
    project_root = Path(__file__).resolve().parent.parent
    of3_dir = project_root / "openfold3-repo"
    for p in [str(project_root), str(of3_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_diffusion_closure(
    model, cached_batch, si_input_cached, si_trunk_cached,
    noise_schedule, start_atom_index, N_token, K, device,
    runner_config, get_confidence_scores,
):
    """Build the diffusion_fn closure used by both load_of3_diffusion and OF3ModelCache."""

    def _extract_ca_multi(atom_positions: torch.Tensor) -> torch.Tensor:
        pos = atom_positions.squeeze(0)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        k_actual = pos.shape[0]
        cas = []
        for ki in range(k_actual):
            p = pos[ki]
            if start_atom_index is not None:
                idx = start_atom_index.squeeze().long()
                cas.append(p[idx])
            else:
                cas.append(p[:N_token])
        return torch.stack(cas, dim=0)

    def _compute_confidence(atom_positions, zij_modified):
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
            import traceback
            print(f"[OF3] Warning: confidence computation failed: {e}")
            traceback.print_exc()
            return None

    @torch.no_grad()
    def diffusion_fn(z_mod: torch.Tensor) -> DiffusionResult | torch.Tensor:
        zij_modified = z_mod.unsqueeze(0).unsqueeze(0).to(device)

        atom_positions = model.sample_diffusion(
            batch=cached_batch,
            si_input=si_input_cached,
            si_trunk=si_trunk_cached,
            zij_trunk=zij_modified,
            noise_schedule=noise_schedule,
            no_rollout_samples=K,
            use_conditioning=True,
        )

        all_ca = _extract_ca_multi(atom_positions)

        if K == 1:
            confidence = _compute_confidence(atom_positions, zij_modified)
            ca = all_ca[0].to(z_mod.device)

            if confidence is not None:
                plddt_raw = confidence.get("plddt")
                ptm_raw = confidence.get("ptm")
                plddt = plddt_raw.squeeze(0).squeeze(0) if plddt_raw is not None else None
                ptm = ptm_raw.squeeze() if ptm_raw is not None else None

                ptm_f = 0.0
                ranking_val = 0.0
                if ptm is not None:
                    ptm_f = ptm.item() if ptm.dim() == 0 else ptm.mean().item()
                    plddt_f = plddt.mean().item() / 100.0 if plddt is not None else 0.0
                    ranking_val = 0.8 * ptm_f + 0.2 * plddt_f

                pae_out, mean_pae_out = _extract_pae(confidence, best_idx=0)
                cp_out = _extract_contact_probs(confidence)
                clash_out = _extract_has_clash(confidence)

                return DiffusionResult(
                    all_ca=all_ca.to(z_mod.device),
                    best_ca=ca,
                    best_idx=0,
                    plddt=plddt.unsqueeze(0) if plddt is not None else None,
                    ptm=torch.tensor([ptm_f]) if ptm is not None else None,
                    ranking=torch.tensor([ranking_val]),
                    pae=pae_out,
                    contact_probs=cp_out,
                    has_clash=clash_out,
                    mean_pae=mean_pae_out,
                )
            else:
                return DiffusionResult(
                    all_ca=all_ca.to(z_mod.device),
                    best_ca=ca,
                    best_idx=0,
                    plddt=None,
                    ptm=None,
                    ranking=None,
                )

        # Multi-sample
        confidence = _compute_confidence(atom_positions, zij_modified)

        if confidence is not None:
            plddt_raw = confidence.get("plddt")
            ptm_raw = confidence.get("ptm")

            if plddt_raw is not None:
                plddt = plddt_raw.squeeze(0) if plddt_raw.dim() > 2 else plddt_raw
                if plddt.dim() == 1:
                    plddt = plddt.unsqueeze(0).expand(K, -1)
            else:
                plddt = None

            if ptm_raw is not None:
                ptm = ptm_raw.squeeze(0) if ptm_raw.dim() > 1 else ptm_raw
                if ptm.dim() == 0:
                    ptm = ptm.unsqueeze(0).expand(K)
            else:
                ptm = None

            if ptm is not None and plddt is not None:
                ranking = 0.8 * ptm + 0.2 * plddt.mean(dim=-1) / 100.0
            elif ptm is not None:
                ranking = ptm
            else:
                ranking = plddt.mean(dim=-1) / 100.0 if plddt is not None else torch.zeros(K)

            best_idx = ranking.argmax().item()
        else:
            plddt = None
            ptm = None
            ranking = None
            best_idx = 0

        best_ca = all_ca[best_idx].to(z_mod.device)

        pae_out, mean_pae_out = _extract_pae(confidence, best_idx=best_idx)
        cp_out = _extract_contact_probs(confidence)
        clash_out = _extract_has_clash(confidence)
        s_rmsd, s_rmsf, consensus = _compute_sample_consistency(all_ca)

        return DiffusionResult(
            all_ca=all_ca.to(z_mod.device),
            best_ca=best_ca,
            best_idx=best_idx,
            plddt=plddt,
            ptm=ptm,
            ranking=ranking,
            pae=pae_out,
            contact_probs=cp_out,
            has_clash=clash_out,
            mean_pae=mean_pae_out,
            sample_rmsd=s_rmsd,
            sample_rmsf=s_rmsf,
            consensus_score=consensus,
        )

    return diffusion_fn


def load_of3_model(
    device: str = "cuda",
    num_samples: int = 1,
    use_msa_server: bool = False,
    use_templates: bool = False,
) -> "OF3ModelCache":
    """Load OF3 model ONCE and return a reusable cache object.

    Use this to avoid reloading the model for each protein in a benchmark.
    Then call cache.prepare_sequence(sequence) for each new protein.

    Args:
        device:         'cuda' or 'cpu'.
        num_samples:    Number of diffusion samples per call (K).
        use_msa_server: Whether to use MSA server.
        use_templates:  Whether to use templates.

    Returns:
        OF3ModelCache object with .prepare_sequence() method.
    """
    _ensure_of3_importable()

    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner

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

    runner_config = lightning_module.config
    print("[OF3] Model loaded.")

    return OF3ModelCache(
        runner=runner,
        model=model,
        runner_config=runner_config,
        device=device,
        num_samples=num_samples,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )


class OF3ModelCache:
    """Cached OF3 model that can prepare different sequences without reload."""

    def __init__(self, runner, model, runner_config, device, num_samples,
                 use_msa_server, use_templates):
        self.runner = runner
        self.model = model
        self.runner_config = runner_config
        self.device = device
        self.num_samples = num_samples
        self.use_msa_server = use_msa_server
        self.use_templates = use_templates

    def prepare_sequence(
        self,
        sequence: str,
        query_name: str = "protein",
        chain_id: str = "A",
        num_rollout_steps: int | None = None,
    ) -> tuple[Callable[[torch.Tensor], "DiffusionResult | torch.Tensor"], torch.Tensor]:
        """Prepare diffusion_fn and zij_trunk for a new sequence.

        This re-runs the data pipeline and trunk for the new sequence
        but reuses the already-loaded model weights.

        Returns:
            (diffusion_fn, zij_trunk) same as load_of3_diffusion.
        """
        return _prepare_sequence_impl(
            model=self.model,
            runner=self.runner,
            runner_config=self.runner_config,
            device=self.device,
            num_samples=self.num_samples,
            sequence=sequence,
            query_name=query_name,
            chain_id=chain_id,
            num_rollout_steps=num_rollout_steps,
            use_msa_server=self.use_msa_server,
            use_templates=self.use_templates,
        )


def _prepare_sequence_impl(
    model, runner, runner_config, device, num_samples,
    sequence, query_name, chain_id,
    num_rollout_steps, use_msa_server, use_templates,
):
    """Internal: prepare batch + run trunk for a given sequence."""
    import json as _json
    import tempfile

    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.core.metrics.aggregate_confidence_ranking import (
        get_confidence_scores,
    )

    # Build query JSON
    _query = {
        "queries": {
            query_name: {
                "chains": [{"molecule_type": "protein", "chain_ids": [chain_id], "sequence": sequence}]
            }
        }
    }
    _query_dir = tempfile.mkdtemp(prefix="of3_query_")
    _query_path = f"{_query_dir}/{query_name}.json"
    with open(_query_path, "w") as f:
        _json.dump(_query, f)

    # Prepare batch via data pipeline — create FRESH runner for data only
    print(f"[OF3] Preparing batch for {query_name} (N={len(sequence)})...")
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner

    _data_config = InferenceExperimentConfig(
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
    _data_runner = InferenceExperimentRunner(
        _data_config,
        num_diffusion_samples=num_samples,
        num_model_seeds=1,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    _data_runner.setup()

    query_set = InferenceQuerySet.from_json(_query_path)
    _data_runner.inference_query_set = query_set
    data_module = _data_runner.lightning_data_module
    data_module.prepare_data()
    data_module.setup(stage="predict")

    predict_dl = data_module.predict_dataloader()
    cached_batch = next(iter(predict_dl))

    # Debug: check batch validity
    _batch_keys = [k for k in cached_batch.keys() if isinstance(cached_batch.get(k), torch.Tensor)]
    print(f"[OF3] Batch tensor keys ({len(_batch_keys)}): {sorted(_batch_keys)[:10]}...")
    if cached_batch.get("valid_sample") is not None:
        _valid = cached_batch["valid_sample"]
        if isinstance(_valid, torch.Tensor) and not _valid.all():
            raise RuntimeError(
                "[OF3] Data pipeline failed — batch marked as invalid. "
                "Check MSA server connectivity and query JSON format."
            )

    cached_batch = tensor_tree_map(
        lambda t: t.to(device) if isinstance(t, torch.Tensor) else t,
        cached_batch,
    )

    # Safety: ensure token_mask exists
    if "token_mask" not in cached_batch:
        _n_tok = len(sequence)
        cached_batch["token_mask"] = torch.ones(
            (1, _n_tok), dtype=torch.float32, device=torch.device(device)
        )
        print(f"[OF3] WARNING: token_mask missing, created all-ones (N={_n_tok})")

    print("[OF3] Running trunk...")
    with torch.no_grad():
        num_cycles = model.shared.num_recycles + 1
        si_input_cached, si_trunk_cached, zij_trunk_cached = model.run_trunk(
            batch=cached_batch, num_cycles=num_cycles,
            inplace_safe=True,
        )
        si_input_cached = si_input_cached.unsqueeze(1)
        si_trunk_cached = si_trunk_cached.unsqueeze(1)
        zij_trunk_cached = zij_trunk_cached.unsqueeze(1)

    # Expand batch for sampling dimension
    ref_space_uid_to_perm = cached_batch.pop("ref_space_uid_to_perm", None)
    cached_batch = tensor_tree_map(lambda t: t.unsqueeze(1), cached_batch)
    if ref_space_uid_to_perm is not None:
        cached_batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm

    start_atom_index = cached_batch.get("start_atom_index")
    N_token = si_trunk_cached.shape[-2]
    C_z = zij_trunk_cached.shape[-1]

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

    print(f"[OF3] Trunk cached: N_token={N_token}, C_z={C_z}, diffusion={no_rollout_steps} steps")

    # Build diffusion closure (same logic as before)
    diffusion_fn = _build_diffusion_closure(
        model=model,
        cached_batch=cached_batch,
        si_input_cached=si_input_cached,
        si_trunk_cached=si_trunk_cached,
        noise_schedule=noise_schedule,
        start_atom_index=start_atom_index,
        N_token=N_token,
        K=K,
        device=device,
        runner_config=runner_config,
        get_confidence_scores=get_confidence_scores,
    )

    # Return zij_trunk without batch/sample dims: [N_token, N_token, C_z]
    zij_out = zij_trunk_cached.squeeze(0).squeeze(0)
    return diffusion_fn, zij_out


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

    # Debug: check if batch is valid (failed samples have minimal keys)
    _batch_keys = [k for k in cached_batch.keys() if isinstance(cached_batch.get(k), torch.Tensor)]
    print(f"[OF3] Batch tensor keys ({len(_batch_keys)}): {sorted(_batch_keys)[:15]}...")
    if cached_batch.get("valid_sample") is not None:
        _valid = cached_batch["valid_sample"]
        if isinstance(_valid, torch.Tensor) and not _valid.all():
            raise RuntimeError(
                "[OF3] Data pipeline failed — batch marked as invalid. "
                "Check MSA server connectivity and query JSON format."
            )

    cached_batch = tensor_tree_map(
        lambda t: t.to(device) if isinstance(t, torch.Tensor) else t,
        cached_batch,
    )

    # Safety: ensure token_mask exists (required by model.run_trunk).
    if "token_mask" not in cached_batch:
        for _key in ("restype", "residue_index", "ref_pos"):
            if _key in cached_batch and isinstance(cached_batch[_key], torch.Tensor):
                _n_tok = cached_batch[_key].shape[-1] if cached_batch[_key].dim() <= 2 else cached_batch[_key].shape[-2]
                break
        else:
            _n_tok = None
            # Try to get from query JSON
            import json as _json
            with open(str(query_json)) as _f:
                _q = _json.load(_f)
            for _qname, _qval in _q.get("queries", {}).items():
                for _chain in _qval.get("chains", []):
                    if _chain.get("sequence"):
                        _n_tok = len(_chain["sequence"])
                        break
                if _n_tok:
                    break
            if _n_tok is None:
                raise RuntimeError(
                    "[OF3] Cannot determine N_token — batch keys: "
                    f"{[k for k in cached_batch if isinstance(cached_batch.get(k), torch.Tensor)]}"
                )
        cached_batch["token_mask"] = torch.ones(
            (1, _n_tok), dtype=torch.float32, device=torch.device(device)
        )
        print(f"[OF3] WARNING: token_mask missing from batch, created all-ones mask (N_token={_n_tok})")

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
    diffusion_fn = _build_diffusion_closure(
        model=model,
        cached_batch=cached_batch,
        si_input_cached=si_input_cached,
        si_trunk_cached=si_trunk_cached,
        noise_schedule=noise_schedule,
        start_atom_index=start_atom_index,
        N_token=N_token,
        K=K,
        device=device,
        runner_config=runner_config,
        get_confidence_scores=get_confidence_scores,
    )

    # Return zij_trunk without batch/sample dims: [N_token, N_token, C_z]
    zij_trunk_out = zij_trunk_cached.squeeze(0).squeeze(0).detach().cpu()

    print("[OF3] Diffusion function ready.")
    return diffusion_fn, zij_trunk_out
