"""CSV IO helpers for finetune pipeline (pandas-free)."""

import csv


# Columns exported for grid results (in this order)
DEFAULT_PARAM_KEYS = ["R_bb", "K_0", "n_ref", "v_magnitude"]
DEFAULT_METRIC_KEYS = [
    "delta_rmsd", "min_rmsd", "best_tm",
    "crash_events", "e_drift", "e_plateau_std",
]


def save_results_csv(path, results, *,
                     param_keys=None, metric_keys=None,
                     sort_by="delta_rmsd", descending=True):
    """Save raw `run_grid_search` results as a flat CSV.

    Each row = parameter combo + selected metrics, sorted by `sort_by`.
    """
    param_keys = list(param_keys or DEFAULT_PARAM_KEYS)
    metric_keys = list(metric_keys or DEFAULT_METRIC_KEYS)

    rows = []
    for r in results:
        row = dict(r.get("params", {}))
        for mk in metric_keys:
            row[mk] = r.get(mk, "")
        rows.append(row)

    if sort_by:
        rows.sort(
            key=lambda r: r.get(sort_by, 0) if isinstance(r.get(sort_by), (int, float)) else 0,
            reverse=descending,
        )

    cols = param_keys + metric_keys
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    return rows


def load_results_csv(path, *,
                     param_keys=None, metric_keys=None):
    """Load grid results CSV written by `save_results_csv`. Returns list[dict]."""
    param_keys = list(param_keys or DEFAULT_PARAM_KEYS)
    metric_keys = list(metric_keys or DEFAULT_METRIC_KEYS)

    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            out.append(parsed)
    return out


def save_suggestions_csv(path, params_array, pred_losses, *,
                         param_keys=None, top_k=None):
    """Save surrogate-optimized parameter suggestions, sorted by predicted loss.

    Args:
        path: output CSV
        params_array: shape (N, D) array/list
        pred_losses: length-N array/list of predicted composite losses
        param_keys: column names matching D (defaults to DEFAULT_PARAM_KEYS)
        top_k: limit output to best-K rows (by lowest pred_loss)
    """
    import numpy as np
    param_keys = list(param_keys or DEFAULT_PARAM_KEYS)
    params_array = np.asarray(params_array)
    pred_losses = np.asarray(pred_losses)

    order = np.argsort(pred_losses)
    if top_k is not None:
        order = order[:top_k]

    cols = list(param_keys) + ["pred_loss"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for idx in order:
            row = list(params_array[idx]) + [float(pred_losses[idx])]
            w.writerow([f"{x:.4f}" if isinstance(x, float) else str(x)
                        for x in row])
    return order
