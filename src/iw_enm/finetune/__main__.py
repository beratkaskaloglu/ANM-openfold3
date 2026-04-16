"""Command-line interface for the finetune pipeline.

Usage:
    python -m iw_enm.finetune grid --pdb 1AKE.pdb --target 4ake.cif [...]
    python -m iw_enm.finetune surrogate --csv fine_tune_results.csv [...]
    python -m iw_enm.finetune turnpoint --pdb 1AKE.pdb --target 4ake.cif \\
                                        --params "R_bb=11,K_0=0.8,n_ref=10,v_magnitude=1.0"
"""

import argparse
import sys

from . import (
    run_grid_search, save_results_csv,
    load_results_csv, save_suggestions_csv,
)


def _parse_params(s):
    """Parse 'k=v,k=v' into a dict of floats."""
    out = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def cmd_grid(args):
    results = run_grid_search(
        args.pdb, args.target,
        chain_id=args.chain,
        n_workers=args.workers,
    )
    save_results_csv(args.out, results)
    print(f"Saved: {args.out}")


def cmd_surrogate(args):
    from . import train_surrogate, optimize_params
    results = load_results_csv(args.csv)
    print(f"Loaded {len(results)} grid points")
    model, stats = train_surrogate(results, epochs=args.epochs, lr=args.lr)
    params, preds = optimize_params(
        model, stats, n_starts=args.n_starts, steps=args.steps,
    )
    save_suggestions_csv(args.out, params, preds, top_k=args.top_k)
    print(f"\nSaved: {args.out}")


def cmd_turnpoint(args):
    from .runner import run_with_params
    from .turnpoint import select_best_frame
    params = _parse_params(args.params)
    sim, baseline = run_with_params(
        args.pdb, args.target, params, chain_id=args.chain,
        output_prefix="turnpoint", verbose=False,
    )
    picked = select_best_frame(
        sim, back_off=args.back_off, weight_crash=args.crash_weight,
        smooth_w=args.smooth, warmup_skip=args.warmup,
    )
    print(f"Baseline: {baseline:.3f} Å")
    print(f"Turn step (E_tot): {picked['step_turn_e']}")
    print(f"Turn step (springs): {picked['step_turn_s']}")
    print(f"Picked step: {picked['step']}")
    print(f"RMSD_target: {picked['rmsd_to_target']:.3f} Å")
    print(f"TM_target: {picked['tm_to_target']:.3f}")
    print(f"Crashes until pick: {picked['crashes_until_best']}")
    print(f"Composite score: {picked['score']:.3f}")
    if args.out:
        sim.structure.to_pdb(args.out, coords=picked["coords"])
        print(f"Saved model PDB: {args.out}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m iw_enm.finetune")
    sp = p.add_subparsers(dest="cmd", required=True)

    g = sp.add_parser("grid", help="run parallel grid search")
    g.add_argument("--pdb", required=True)
    g.add_argument("--target", required=True)
    g.add_argument("--chain", default="A")
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--out", default="fine_tune_results.csv")
    g.set_defaults(func=cmd_grid)

    s = sp.add_parser("surrogate", help="train MLX surrogate, output top params")
    s.add_argument("--csv", default="fine_tune_results.csv")
    s.add_argument("--out", default="surrogate_suggestions.csv")
    s.add_argument("--epochs", type=int, default=600)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--n-starts", type=int, default=128)
    s.add_argument("--steps", type=int, default=500)
    s.add_argument("--top-k", type=int, default=20)
    s.set_defaults(func=cmd_surrogate)

    t = sp.add_parser("turnpoint", help="run sim + select best frame via turnpoint")
    t.add_argument("--pdb", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--chain", default="A")
    t.add_argument("--params", required=True,
                   help="e.g. 'R_bb=11,K_0=0.8,n_ref=10,v_magnitude=1.0'")
    t.add_argument("--back-off", type=int, default=1)
    t.add_argument("--crash-weight", type=float, default=0.30)
    t.add_argument("--smooth", type=int, default=11)
    t.add_argument("--warmup", type=float, default=0.30)
    t.add_argument("--out", default="model_turnpoint.pdb")
    t.set_defaults(func=cmd_turnpoint)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
