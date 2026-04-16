"""MLX surrogate MLP — trains on grid results, optimizes params in normalized space."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .loss import composite_loss, DEFAULT_WEIGHTS
from .io import DEFAULT_PARAM_KEYS


@dataclass
class NormStats:
    """Normalization stats captured at training time."""
    x_min: np.ndarray  # (D,) float32
    x_max: np.ndarray  # (D,) float32
    x_span: np.ndarray  # (D,) float32
    y_mean: float
    y_std: float
    param_keys: List[str] = field(default_factory=list)

    def encode_x(self, x):
        return (np.asarray(x, dtype=np.float32) - self.x_min) / self.x_span

    def decode_x(self, xn):
        return np.asarray(xn, dtype=np.float32) * self.x_span + self.x_min

    def decode_y(self, yn):
        return np.asarray(yn, dtype=np.float32) * self.y_std + self.y_mean


class SurrogateMLP(nn.Module):
    """Small MLP: D → hidden → hidden → 1, GELU activations."""

    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def __call__(self, x):
        x = nn.gelu(self.l1(x))
        x = nn.gelu(self.l2(x))
        return self.l3(x).squeeze(-1)


def _extract_arrays(results, param_keys):
    """Pull X, delta, crash, edrift, pstd arrays out of a results list-of-dicts."""
    rows = []
    for r in results:
        params = r.get("params", r)
        rows.append([float(params[k]) for k in param_keys])
    X = np.asarray(rows, dtype=np.float32)

    def grab(key):
        return np.array([float(r.get(key, 0.0)) for r in results], dtype=np.float32)

    delta = grab("delta_rmsd")
    crash = np.maximum(grab("crash_events"), 0.0)
    edrift = grab("e_drift")
    pstd = grab("e_plateau_std")
    return X, delta, crash, edrift, pstd


def train_surrogate(results, *,
                    param_keys=None,
                    epochs=600, lr=3e-3, hidden=64,
                    weights=None, verbose=True):
    """Fit a SurrogateMLP to the composite loss over grid results.

    Args:
        results: list of dicts from `run_grid_search` OR from `load_results_csv`
                 (both support dict-lookup of param_keys + metric keys).
        param_keys: parameter columns (order determines model input dim)
        epochs, lr, hidden: training hyperparams
        weights: dict overriding composite-loss weights (alpha/beta/gamma)
        verbose: print progress

    Returns:
        (model, NormStats) — both are needed to re-use / optimize later.
    """
    param_keys = list(param_keys or DEFAULT_PARAM_KEYS)
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    X, delta, crash, edrift, pstd = _extract_arrays(results, param_keys)
    y = composite_loss(delta, crash, edrift, pstd,
                       alpha=w["alpha"], beta=w["beta"], gamma=w["gamma"]
                       ).astype(np.float32)

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_span = np.maximum(x_max - x_min, 1e-9)
    Xn = (X - x_min) / x_span

    y_mean = float(y.mean())
    y_std = float(max(y.std(), 1e-6))
    yn = ((y - y_mean) / y_std).astype(np.float32)

    model = SurrogateMLP(in_dim=X.shape[1], hidden=hidden)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=lr)

    def mse_loss(m, xb, yb):
        return ((m(xb) - yb) ** 2).mean()

    loss_and_grad_fn = nn.value_and_grad(model, mse_loss)
    xb_all = mx.array(Xn)
    yb_all = mx.array(yn)

    for epoch in range(epochs):
        loss_val, grads = loss_and_grad_fn(model, xb_all, yb_all)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if verbose and (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"  epoch {epoch+1:4d}  mse={loss_val.item():.4f}", flush=True)

    stats = NormStats(x_min=x_min, x_max=x_max, x_span=x_span,
                      y_mean=y_mean, y_std=y_std,
                      param_keys=param_keys)
    return model, stats


def optimize_params(model, stats, *,
                    n_starts=128, steps=500, lr=0.02,
                    seed=0):
    """Run gradient descent through the surrogate to find low-loss params.

    Optimization is performed in normalized [0, 1]^D space and then decoded
    back. Params are clipped to the training-set box.

    Returns:
        (params_real, pred_loss_real)
            params_real: (n_starts, D) array in original units
            pred_loss_real: (n_starts,) predicted composite loss (original scale)
    """
    D = len(stats.x_min)
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0.0, 1.0, size=(n_starts, D)).astype(np.float32)
    p = mx.array(starts)

    def _pred_mean(x):
        return model(x).mean()

    grad_fn = mx.grad(_pred_mean)
    for _ in range(steps):
        g = grad_fn(p)
        p = p - lr * g
        p = mx.clip(p, 0.0, 1.0)
        mx.eval(p)

    p_np = np.array(p)
    preds = model(mx.array(p_np))
    mx.eval(preds)
    preds_np = np.array(preds) * stats.y_std + stats.y_mean
    params_real = p_np * stats.x_span + stats.x_min
    return params_real, preds_np


def predict(model, stats, params_dict_or_array):
    """Predict composite loss for a single params dict or array."""
    if isinstance(params_dict_or_array, dict):
        x = np.array([[params_dict_or_array[k] for k in stats.param_keys]],
                     dtype=np.float32)
    else:
        x = np.asarray(params_dict_or_array, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
    xn = (x - stats.x_min) / stats.x_span
    pred = model(mx.array(xn))
    mx.eval(pred)
    return np.asarray(pred) * stats.y_std + stats.y_mean
