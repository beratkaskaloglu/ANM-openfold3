"""OF3 diffusion wrapper for mode-drive pipeline.

Wraps OpenFold3's SampleDiffusion module as a callable that takes
a modified pair representation z_mod and returns CA coordinates.

Usage (Colab with GPU):
    from src.of3_diffusion import load_of3_diffusion

    diffusion_fn, zij_trunk = load_of3_diffusion(
        query_json="path/to/query.json",   # OF3 inference query
        device="cuda",
    )

    pipeline = ModeDrivePipeline(converter, config, diffusion_fn=diffusion_fn)

The wrapper:
    1. Runs OF3 trunk ONCE to get si_input, si_trunk, zij_trunk, batch
    2. Caches everything except zij_trunk
    3. On each call: replaces zij_trunk with z_mod → runs SampleDiffusion → extracts CA
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


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
    use_msa_server: bool = False,
    use_templates: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Load OF3 model and return a diffusion_fn for ModeDrivePipeline.

    Runs trunk inference ONCE, then caches si_input, si_trunk, batch.
    Each call to the returned function replaces zij_trunk with z_mod
    and runs only the diffusion sampling (no re-running trunk).

    Args:
        query_json:        Path to OF3 inference query JSON.
        device:            'cuda' or 'cpu'.
        num_rollout_steps: Override diffusion rollout steps (None = use default).
        use_msa_server:    Whether to use MSA server.
        use_templates:     Whether to use templates.

    Returns:
        Tuple of:
            diffusion_fn: Callable([N, N, 128]) -> [N, 3] CA coordinates.
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
        num_diffusion_samples=1,
        num_model_seeds=1,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    runner.setup()

    # Get the actual PyTorch model via lightning_module
    # runner.lightning_module = OpenFold3AllAtom (LightningModule)
    # runner.lightning_module.model = OpenFold3 (nn.Module)
    lightning_module = runner.lightning_module
    model = lightning_module.model
    model = model.to(device)
    model.eval()

    print("[OF3] Model loaded.")

    # ── Build batch from query ─────────────────────────────────
    print("[OF3] Preparing input batch...")

    query_set = InferenceQuerySet.from_json(str(query_json))

    # Set query set on runner and build data module
    runner.inference_query_set = query_set
    data_module = runner.lightning_data_module
    data_module.setup(stage="predict")

    # Get the first batch from the predict dataloader
    predict_dl = data_module.predict_dataloader()
    cached_batch = next(iter(predict_dl))

    # Move batch tensors to device
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
    num_atoms_per_token = cached_batch.get("num_atoms_per_token")
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
    no_rollout_samples = model.shared.diffusion.no_full_rollout_samples

    print(f"[OF3] Trunk cached: N_token={N_token}, C_z={C_z}")
    print(f"[OF3] Diffusion: {no_rollout_steps} steps, {no_rollout_samples} samples")

    # ── Build closure ───────────────────────────────────────────
    def _extract_ca(atom_positions: torch.Tensor) -> torch.Tensor:
        """Extract CA coordinates from all-atom positions.

        Args:
            atom_positions: [1, 1, N_atom, 3] or [N_atom, 3]

        Returns:
            ca_coords: [N_token, 3]
        """
        # Remove batch/sample dims
        pos = atom_positions.squeeze()  # [N_atom, 3]

        if start_atom_index is not None:
            # Use token→atom mapping: CA is the first atom of each token for proteins
            idx = start_atom_index.squeeze().long()  # [N_token]
            return pos[idx]
        else:
            # Fallback: assume 1 atom per token (CA-only)
            return pos[:N_token]

    @torch.no_grad()
    def diffusion_fn(z_mod: torch.Tensor) -> torch.Tensor:
        """Run OF3 diffusion with modified pair representation.

        Args:
            z_mod: [N, N, 128] modified pair representation.

        Returns:
            ca_coords: [N, 3] CA coordinates.
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
            no_rollout_samples=no_rollout_samples,
            use_conditioning=True,
        )

        # Extract CA
        ca = _extract_ca(atom_positions)
        return ca.to(z_mod.device)

    # Return zij_trunk without batch/sample dims: [N_token, N_token, C_z]
    zij_trunk_out = zij_trunk_cached.squeeze(0).squeeze(0).detach().cpu()

    print("[OF3] Diffusion function ready.")
    return diffusion_fn, zij_trunk_out
