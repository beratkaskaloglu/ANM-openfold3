"""IW-ENM fine-tuning + analysis subpackage.

Bundles:
  - grid search over ENM hyperparameters (parallel)
  - MLX surrogate model fit on the grid results + param-space optimization
  - turning-point based best-frame selector
  - IO helpers (CSV)

Typical pipeline:

    from iw_enm.finetune import (
        run_grid_search, train_surrogate, optimize_params,
        save_results_csv, save_suggestions_csv,
    )
    from iw_enm.finetune.turnpoint import select_best_frame

    results = run_grid_search("1AKE.pdb", "4ake.cif")
    save_results_csv("fine_tune_results.csv", results)

    model, stats = train_surrogate(results)
    suggestions = optimize_params(model, stats)
    save_suggestions_csv("surrogate_suggestions.csv", suggestions)
"""

from .grid import (
    DEFAULT_GRID,
    DEFAULT_FIXED,
    run_grid_search,
)
from .io import (
    load_results_csv,
    save_results_csv,
    save_suggestions_csv,
)
from . import turnpoint  # re-export iw_enm.turnpoint at iw_enm.finetune.turnpoint

# Lazy-import surrogate (requires mlx) to avoid hard dependency
def _lazy_surrogate():
    from . import surrogate
    return surrogate


def train_surrogate(*args, **kwargs):
    """Train MLX surrogate MLP on grid results. Requires `mlx`."""
    return _lazy_surrogate().train_surrogate(*args, **kwargs)


def optimize_params(*args, **kwargs):
    """Gradient-descent optimize params through trained surrogate. Requires `mlx`."""
    return _lazy_surrogate().optimize_params(*args, **kwargs)


def composite_loss(*args, **kwargs):
    """Return the composite scalar loss used by the surrogate."""
    from .loss import composite_loss as _cl
    return _cl(*args, **kwargs)


__all__ = [
    "DEFAULT_GRID",
    "DEFAULT_FIXED",
    "run_grid_search",
    "train_surrogate",
    "optimize_params",
    "composite_loss",
    "load_results_csv",
    "save_results_csv",
    "save_suggestions_csv",
    "turnpoint",
]
