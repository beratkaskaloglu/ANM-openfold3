"""Simulation parameters."""

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    R_bb: float = 10.0        # backbone cutoff (Å)
    R_sc: float = 6.0         # sidechain cutoff (Å)
    K_0: float = 1.0          # base spring constant
    d_0: float = 3.8          # equilibrium distance (Å)
    n_ref: float = 7.0        # reference interaction count
    dt: float = 0.01          # time step
    mass: float = 1.0         # atom mass (uniform)
    n_steps: int = 10000      # total steps
    save_every: int = 100     # save interval
    damping: float = 0.1      # damping coefficient
    v_mode: str = "random"    # initial velocity mode
    v_magnitude: float = 1.0  # velocity magnitude
    output_prefix: str = "iwenm"
    chain_id: str = "A"       # which chain to use
    crash_threshold: float = 0.5  # sidechain-sidechain distance below this = crash (Å)
