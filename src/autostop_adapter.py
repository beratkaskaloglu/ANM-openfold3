"""Torch ↔ NumPy bridge for run_autostop → mode-drive pipeline integration.

This adapter wraps the pure-NumPy `run_autostop` / `EarlyStopMonitor` logic in a
torch-tensor-friendly API suitable for embedding into
`src.mode_drive.ModeDrivePipeline`.

Public API
----------

- `AutostopParams`      — dataclass grouping ALL autostop knobs (physics +
  monitor). Mirrors the kwargs of `run_autostop.run_with_autostop`.
- `StructureContext`    — residue/atom metadata cache. Built ONCE at pipeline
  start from a PDB (preferred) or from a CA-only tensor (idealized fallback).
  Supports `update_ca(coords_ca)` to rigid-translate atoms with new CA.
- `AutostopTrace`       — raw signals from one autostop MD run. Sufficient to
  replay the monitor with different monitor params — no re-integration.
- `AutostopPick`        — the picked frame + diagnostics (what mode-drive uses).
- `run_autostop_from_tensor(coords_ca, ctx, params)` → (AutostopPick, AutostopTrace)
- `replay_monitor(trace, monitor_params, back_off)` → AutostopPick
  *Cheap fallback path — re-scans the cached trajectory under new monitor knobs.*

Device / dtype contract
-----------------------
- Input tensors may be on any device/dtype. The adapter always pulls them to
  CPU float64 for NumPy physics, then returns picked CA as a torch tensor on
  the ORIGINAL input device/dtype.
- Trajectory frames in `AutostopTrace.trajectory` are kept as NumPy arrays
  (float64) to avoid wasteful device round-trips during replay_monitor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

# Vendored iw_enm lives at src/iw_enm/*.
# Two import paths are supported so this works whether the caller puts
# `src/` on PYTHONPATH (tests) or the project root on PYTHONPATH (notebooks).
try:
    from iw_enm.analysis import compute_kinetic_energy
    from iw_enm.config import SimulationConfig
    from iw_enm.integrator import VelocityVerletIntegrator
    from iw_enm.network import InteractionWeightedENM
    from iw_enm.structure import ProteinStructure
except ImportError:  # pragma: no cover — notebook-style import
    from src.iw_enm.analysis import compute_kinetic_energy
    from src.iw_enm.config import SimulationConfig
    from src.iw_enm.integrator import VelocityVerletIntegrator
    from src.iw_enm.network import InteractionWeightedENM
    from src.iw_enm.structure import ProteinStructure


# ======================================================================
# Parameter dataclass
# ======================================================================

@dataclass
class AutostopParams:
    """All knobs for one autostop MD + monitor run.

    Grouped so the mode-drive pipeline can mutate them per-fallback-level
    without spreading kwargs everywhere.
    """

    # --- ENM physics ---
    R_bb: float = 11.0
    R_sc: float = 2.0
    K_0: float = 0.8
    d_0: float = 3.8
    n_ref: float = 10.0

    # --- Integration ---
    dt: float = 0.01
    mass: float = 1.0
    damping: float = 0.0
    v_mode: str = "breathing"
    v_magnitude: float = 1.0

    # --- Run control ---
    n_steps: int = 5000
    save_every: int = 10
    back_off: int = 2
    crash_threshold_distance: float = 0.5

    # --- Monitor (early-stop) ---
    smooth_w: int = 11
    warmup_frac: float = 0.20
    patience: int = 3
    eps_E_rel: float = 0.002
    eps_N_rel: float = 0.005
    crash_window_saves: int = 20
    crash_threshold: int = 5   # crashes required in window to trigger onset
    min_saves_before_check: int = 15

    verbose: bool = False

    def monitor_only(self) -> dict:
        """Extract just the monitor knobs (for replay_monitor)."""
        return dict(
            smooth_w=self.smooth_w,
            warmup_frac=self.warmup_frac,
            patience=self.patience,
            eps_E_rel=self.eps_E_rel,
            eps_N_rel=self.eps_N_rel,
            crash_window_saves=self.crash_window_saves,
            crash_threshold=self.crash_threshold,
            min_saves_before_check=self.min_saves_before_check,
        )


# ======================================================================
# Structure context
# ======================================================================

# Idealized CB offset in the local N-CA-C frame (Å).
# Placeholder for true sidechain geometry — GOOD ENOUGH for packing count and
# crash detection at CA-level granularity.
_IDEAL_CB_OFFSET = np.array([1.0, 0.5, 0.5], dtype=np.float64)


@dataclass
class StructureContext:
    """Residue/atom metadata needed by iw_enm, cached across pipeline steps.

    The mode-drive pipeline owns ONE StructureContext for the whole run.
    Every autostop step reuses the same residue names and atom layout; only
    the CA (and rigid-translated atoms) change frame-to-frame.
    """

    struct: ProteinStructure
    # Baseline atom offsets relative to the corresponding residue's CA,
    # cached at context-build time. On update_ca(coords_ca_new), atom
    # coordinates are regenerated as atom_offset + coords_ca_new[atom_res_idx].
    atom_offsets_from_ca: np.ndarray = field(repr=False)
    cb_offsets_from_ca: np.ndarray = field(repr=False)

    @classmethod
    def from_pdb(cls, path: str, chain_id: str = "A") -> "StructureContext":
        struct = ProteinStructure.from_pdb(path, chain_id=chain_id)
        return cls._from_structure(struct)

    @classmethod
    def from_cif(cls, path: str, chain_id: str = "A") -> "StructureContext":
        struct = ProteinStructure.from_cif(path, chain_id=chain_id)
        return cls._from_structure(struct)

    @classmethod
    def from_ca_only(
        cls,
        coords_ca: torch.Tensor | np.ndarray,
        res_names: Sequence[str] | None = None,
        res_ids: Sequence[int] | None = None,
        chain_id: str = "A",
    ) -> "StructureContext":
        """Fallback builder — PDB not available mid-pipeline.

        Generates CB with a canonical CA-frame offset (no true sidechain).
        `res_names` may be given (all-GLY if omitted — conservative for
        volume weighting since GLY has V_ref lowest).
        """
        if isinstance(coords_ca, torch.Tensor):
            ca_np = coords_ca.detach().cpu().numpy().astype(np.float64, copy=False)
        else:
            ca_np = np.asarray(coords_ca, dtype=np.float64)

        N = ca_np.shape[0]
        if res_names is None:
            res_names = ["GLY"] * N
        if res_ids is None:
            res_ids = list(range(1, N + 1))

        # Idealized CB = CA + fixed offset (chiral shape that iw_enm's
        # packing count can work with). All-same-offset is degenerate for
        # contact counting at step 0, but after the first MD step coords
        # diverge so the packing signal recovers.
        cb_np = ca_np + _IDEAL_CB_OFFSET

        atom_coords = cb_np.copy()
        atom_res_idx = np.arange(N, dtype=np.int32)
        atom_names = ["CB"] * N

        struct = ProteinStructure(
            coords_ca=ca_np,
            coords_cb=cb_np,
            res_names=res_names,
            res_ids=res_ids,
            chain_ids=[chain_id] * N,
            atom_coords=atom_coords,
            atom_res_idx=atom_res_idx,
            atom_names=atom_names,
        )
        return cls._from_structure(struct)

    @classmethod
    def _from_structure(cls, struct: ProteinStructure) -> "StructureContext":
        # Offsets = current atom position − corresponding residue's CA.
        # Stored ONCE; for every new CA tensor we regenerate atom_coords
        # as offset + new_ca[res_idx]. Same rigid-translate convention iw_enm
        # uses internally (integrator.step line 45-46).
        ca_per_atom = struct.coords_ca[struct.atom_res_idx]
        atom_offsets = struct.atom_coords - ca_per_atom
        cb_offsets = struct.coords_cb - struct.coords_ca
        return cls(
            struct=struct,
            atom_offsets_from_ca=atom_offsets,
            cb_offsets_from_ca=cb_offsets,
        )

    @property
    def N(self) -> int:
        return self.struct.N

    def rebuild_from_ca(self, coords_ca_np: np.ndarray) -> ProteinStructure:
        """Produce a fresh ProteinStructure with the given CA and rigid atoms.

        Does NOT mutate `self.struct` — safe to call per pipeline step.
        """
        if coords_ca_np.shape != self.struct.coords_ca.shape:
            raise ValueError(
                f"coords_ca shape mismatch: got {coords_ca_np.shape}, "
                f"expected {self.struct.coords_ca.shape}"
            )
        coords_ca = np.asarray(coords_ca_np, dtype=np.float64)
        coords_cb = coords_ca + self.cb_offsets_from_ca
        atom_coords = coords_ca[self.struct.atom_res_idx] + self.atom_offsets_from_ca
        return ProteinStructure(
            coords_ca=coords_ca,
            coords_cb=coords_cb,
            res_names=self.struct.res_names,
            res_ids=self.struct.res_ids,
            chain_ids=self.struct.chain_ids,
            atom_coords=atom_coords,
            atom_res_idx=self.struct.atom_res_idx.copy(),
            atom_names=list(self.struct.atom_names),
        )


# ======================================================================
# Monitor (same logic as run_autostop.EarlyStopMonitor, but state-exposable)
# ======================================================================

def _smooth_window(buf, w: int) -> float:
    """Causal box filter. Replicate-pads if short. Matches run_autostop."""
    if len(buf) == 0:
        return 0.0
    take = list(buf)[-w:]
    if len(take) < w:
        take = [take[0]] * (w - len(take)) + take
    return float(np.mean(take))


class _EarlyStopMonitor:
    """Internal re-implementation — identical numerics to
    run_autostop.EarlyStopMonitor, but written so both `update()` (live MD)
    and `replay()` (from cached arrays) are supported.
    """

    def __init__(
        self,
        total_steps: int,
        save_every: int,
        smooth_w: int = 11,
        warmup_frac: float = 0.20,
        patience: int = 3,
        eps_E_rel: float = 0.002,
        eps_N_rel: float = 0.005,
        crash_window_saves: int = 20,
        crash_threshold: int = 5,
        min_saves_before_check: int = 15,
    ) -> None:
        self.save_every = save_every
        self.smooth_w = max(3, smooth_w)
        self.patience = max(1, patience)
        self.eps_E_rel = eps_E_rel
        self.eps_N_rel = eps_N_rel
        self.crash_window_saves = max(1, crash_window_saves)
        self.crash_threshold = max(1, crash_threshold)
        self.min_saves_before_check = min_saves_before_check
        self.warmup_saves = max(
            min_saves_before_check,
            int((total_steps / save_every) * warmup_frac),
        )
        self.E_raw = deque(maxlen=4 * self.smooth_w)
        self.N_raw = deque(maxlen=4 * self.smooth_w)
        self.steps: list[int] = []
        self.E_smooth: list[float] = []
        self.N_smooth: list[float] = []
        self.crashes_cum: list[int] = []
        self.E_min = float("inf")
        self.N_min = float("inf")
        self.argmin_E = 0
        self.argmin_N = 0
        self.hold_E = 0
        self.hold_N = 0
        self.hold_C = 0
        self.stopped = False
        self.stop_reason: str | None = None

    def update(
        self, step: int, E_tot: float, n_spr: int, crashes_cum: int
    ) -> bool:
        self.E_raw.append(E_tot)
        self.N_raw.append(n_spr)
        Es = _smooth_window(self.E_raw, self.smooth_w)
        Ns = _smooth_window(self.N_raw, self.smooth_w)

        k = len(self.steps)
        self.steps.append(step)
        self.E_smooth.append(Es)
        self.N_smooth.append(Ns)
        self.crashes_cum.append(crashes_cum)

        if k >= self.warmup_saves:
            if Es < self.E_min:
                self.E_min = Es
                self.argmin_E = k
            if Ns < self.N_min:
                self.N_min = Ns
                self.argmin_N = k

        if k < self.warmup_saves + self.min_saves_before_check:
            return False

        eps_E_abs = self.eps_E_rel * max(abs(self.E_min), 1.0)
        reversed_E = Es >= self.E_min + eps_E_abs
        reversed_N = Ns >= self.N_min * (1.0 + self.eps_N_rel)
        self.hold_E = self.hold_E + 1 if reversed_E else 0
        self.hold_N = self.hold_N + 1 if reversed_N else 0

        w = self.crash_window_saves
        lo = max(0, k - w)
        recent_crashes = crashes_cum - self.crashes_cum[lo]
        crash_onset = recent_crashes >= self.crash_threshold
        self.hold_C = self.hold_C + 1 if crash_onset else 0

        if (
            self.hold_E >= self.patience
            and self.hold_N >= self.patience
            and self.hold_C >= self.patience
        ):
            self.stopped = True
            self.stop_reason = (
                f"reversal(E:{self.hold_E},N:{self.hold_N}) + "
                f"crash_onset({recent_crashes} in last {w*self.save_every} steps)"
            )
            return True
        return False

    def turnpoint_index(self) -> int:
        return min(self.argmin_E, self.argmin_N)


# ======================================================================
# Trace / Pick dataclasses
# ======================================================================

@dataclass
class AutostopTrace:
    """Raw signals from ONE autostop MD run — cache for cheap fallback replay."""

    # One entry per save (len == number of saves)
    steps: np.ndarray               # int raw-step at each save, shape (S,)
    E_tot: np.ndarray               # float, shape (S,)
    n_springs: np.ndarray           # int, shape (S,)
    crashes_cum_at_save: np.ndarray # int cumulative crash count at each save, shape (S,)

    # trajectory[0] = initial, trajectory[i+1] is save i. Length S+1.
    trajectory: list[np.ndarray]    # each (N, 3) float64

    # Book-keeping
    total_mdsteps_requested: int    # n_steps originally requested
    save_every: int
    stop_step_md: int | None        # raw MD step where monitor stopped (None if not triggered)


@dataclass
class AutostopPick:
    """Picked frame + diagnostics (what mode-drive consumes)."""

    picked_ca: torch.Tensor         # (N, 3) on original input device/dtype
    picked_save_index: int          # index into trace (0-based over saves)
    picked_step_md: int             # raw MD step of the picked frame
    turn_k: int                     # save-index of turnpoint (before back_off)
    argmin_E_k: int
    argmin_N_k: int
    stop_step_md: int | None        # raw MD step where monitor triggered (None if exhausted)
    crashes_total: int
    back_off_used: int
    monitor_params: dict            # monitor-only params applied for this pick
    stop_reason: str | None


# ======================================================================
# Main entry points
# ======================================================================

def run_autostop_from_tensor(
    coords_ca: torch.Tensor,
    ctx: StructureContext,
    params: AutostopParams,
) -> tuple[AutostopPick, AutostopTrace]:
    """Run one autostop MD and pick a frame.

    Args:
        coords_ca: (N, 3) current CA — on any device/dtype.
        ctx:       StructureContext built once at pipeline init.
        params:    AutostopParams bundle.

    Returns:
        (AutostopPick, AutostopTrace).
        `picked_ca` is on the same device/dtype as input `coords_ca`.
        The trace is reusable for cheap monitor-only replay.
    """
    if coords_ca.dim() != 2 or coords_ca.shape[1] != 3:
        raise ValueError(f"coords_ca must be [N, 3], got {tuple(coords_ca.shape)}")

    device = coords_ca.device
    dtype = coords_ca.dtype
    ca_np = coords_ca.detach().cpu().numpy().astype(np.float64, copy=False)

    # Rebuild ProteinStructure with fresh CA (atoms rigid-translated).
    struct = ctx.rebuild_from_ca(ca_np)

    cfg = SimulationConfig(
        R_bb=params.R_bb,
        R_sc=params.R_sc,
        K_0=params.K_0,
        d_0=params.d_0,
        n_ref=params.n_ref,
        dt=params.dt,
        mass=params.mass,
        n_steps=params.n_steps,
        save_every=params.save_every,
        damping=params.damping,
        v_mode=params.v_mode,
        v_magnitude=params.v_magnitude,
        crash_threshold=params.crash_threshold_distance,
        output_prefix="autostop_adapter",
        chain_id=(struct.chain_ids[0] if struct.chain_ids else "A"),
    )

    enm = InteractionWeightedENM(
        R_bb=cfg.R_bb, R_sc=cfg.R_sc, K_0=cfg.K_0,
        d_0=cfg.d_0, n_ref=cfg.n_ref,
    )
    integrator = VelocityVerletIntegrator(
        mass=cfg.mass, dt=cfg.dt, damping=cfg.damping,
    )

    coords = struct.coords_ca.copy()
    coords_sc = struct.coords_cb.copy()
    atom_coords = struct.atom_coords.copy()
    atom_res_idx = struct.atom_res_idx
    enm.set_equilibrium_distances(coords)
    velocities = integrator.initialize_velocities(
        struct.N, mode=cfg.v_mode, magnitude=cfg.v_magnitude, coords=coords,
    )

    trajectory: list[np.ndarray] = [coords.copy()]
    steps_saved: list[int] = []
    E_tot_saved: list[float] = []
    n_spr_saved: list[int] = []
    crashes_cum_saved: list[int] = []
    crash_events = 0

    monitor = _EarlyStopMonitor(
        total_steps=cfg.n_steps,
        save_every=cfg.save_every,
        **params.monitor_only(),
    )
    thr = cfg.crash_threshold
    stop_step = None

    for step in range(1, cfg.n_steps + 1):
        coords, velocities, coords_sc, atom_coords, _, network_info = (
            integrator.step(
                coords, velocities, coords_sc, enm,
                struct.res_names, atom_coords, atom_res_idx,
            )
        )
        neighbors, K_matrix, n_springs, _ = network_info

        # Crash tracking (atomic, non-adjacent residues)
        tree = cKDTree(atom_coords)
        pairs_near = tree.query_pairs(r=max(thr * 2.0, 1.0), output_type="ndarray")
        if len(pairs_near) > 0:
            ri = atom_res_idx[pairs_near[:, 0]]
            rj = atom_res_idx[pairs_near[:, 1]]
            nonadj = np.abs(ri - rj) > 1
            if nonadj.any():
                pp = pairs_near[nonadj]
                diffs = atom_coords[pp[:, 0]] - atom_coords[pp[:, 1]]
                d_inter = np.linalg.norm(diffs, axis=1)
                n_below = int((d_inter < thr).sum())
                if n_below > 0:
                    crash_events += n_below

        if step % cfg.save_every == 0 or step == 1:
            E_pot = enm.compute_energy(coords, neighbors, K_matrix)
            E_kin = compute_kinetic_energy(velocities, cfg.mass)
            E_tot = float(E_pot + E_kin)

            trajectory.append(coords.copy())
            steps_saved.append(step)
            E_tot_saved.append(E_tot)
            n_spr_saved.append(int(n_springs))
            crashes_cum_saved.append(crash_events)

            if monitor.update(step, E_tot, n_springs, crash_events):
                stop_step = step
                break

    if len(steps_saved) == 0:
        raise RuntimeError(
            "No saves produced — is save_every > n_steps?"
        )

    trace = AutostopTrace(
        steps=np.asarray(steps_saved, dtype=np.int64),
        E_tot=np.asarray(E_tot_saved, dtype=np.float64),
        n_springs=np.asarray(n_spr_saved, dtype=np.int64),
        crashes_cum_at_save=np.asarray(crashes_cum_saved, dtype=np.int64),
        trajectory=trajectory,
        total_mdsteps_requested=cfg.n_steps,
        save_every=cfg.save_every,
        stop_step_md=stop_step,
    )

    pick = _pick_from_monitor(monitor, trace, back_off=params.back_off,
                              monitor_params=params.monitor_only(),
                              stop_step_md=stop_step,
                              device=device, dtype=dtype)
    return pick, trace


def replay_monitor(
    trace: AutostopTrace,
    monitor_params: dict,
    back_off: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AutostopPick:
    """Cheap fallback path — re-run ONLY the monitor over the cached trace.

    Does NOT re-integrate MD. Produces a new picked frame from the existing
    trajectory under new monitor knobs.

    Args:
        trace:           Cached trace from a prior `run_autostop_from_tensor`.
        monitor_params:  Dict with keys from AutostopParams.monitor_only().
        back_off:        Per-pick, may differ from original run.
        device/dtype:    For the returned `picked_ca` tensor.
    """
    monitor = _EarlyStopMonitor(
        total_steps=trace.total_mdsteps_requested,
        save_every=trace.save_every,
        **monitor_params,
    )
    stop_step = None
    for i in range(len(trace.steps)):
        triggered = monitor.update(
            int(trace.steps[i]),
            float(trace.E_tot[i]),
            int(trace.n_springs[i]),
            int(trace.crashes_cum_at_save[i]),
        )
        if triggered:
            stop_step = int(trace.steps[i])
            break
    return _pick_from_monitor(
        monitor, trace, back_off=back_off,
        monitor_params=dict(monitor_params),
        stop_step_md=stop_step,
        device=device, dtype=dtype,
    )


# ======================================================================
# Internal helper
# ======================================================================

def _pick_from_monitor(
    monitor: _EarlyStopMonitor,
    trace: AutostopTrace,
    back_off: int,
    monitor_params: dict,
    stop_step_md: int | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> AutostopPick:
    """Turn a monitor's argmin state into a concrete frame pick."""
    # Same arithmetic as run_autostop.run_with_autostop lines 310-320.
    k_turn = monitor.turnpoint_index() if monitor.argmin_E != 0 else 0
    k_best = max(0, k_turn - max(0, int(back_off)))
    # trajectory[0] = initial, trajectory[i+1] == save i
    traj_idx = min(k_best + 1, len(trace.trajectory) - 1)
    picked_np = trace.trajectory[traj_idx]
    picked_step_md = int(trace.steps[k_best])

    picked_ca = torch.from_numpy(
        np.asarray(picked_np, dtype=np.float64)
    ).to(device=device, dtype=dtype)

    crashes_total = int(trace.crashes_cum_at_save[-1]) if len(trace.crashes_cum_at_save) else 0

    return AutostopPick(
        picked_ca=picked_ca,
        picked_save_index=k_best,
        picked_step_md=picked_step_md,
        turn_k=int(k_turn),
        argmin_E_k=int(monitor.argmin_E),
        argmin_N_k=int(monitor.argmin_N),
        stop_step_md=stop_step_md,
        crashes_total=crashes_total,
        back_off_used=int(back_off),
        monitor_params=dict(monitor_params),
        stop_reason=monitor.stop_reason,
    )


__all__ = [
    "AutostopParams",
    "StructureContext",
    "AutostopTrace",
    "AutostopPick",
    "run_autostop_from_tensor",
    "replay_monitor",
]
