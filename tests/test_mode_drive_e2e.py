"""End-to-end test for ModeDrivePipeline using pseudo-diffusion (no OF3)."""

import sys
import math
import traceback

import torch

sys.path.insert(0, ".")

from src.contact_head import ContactProjectionHead
from src.converter import PairContactConverter
from src.mode_drive import (
    ModeDriveConfig,
    ModeDrivePipeline,
    ModeDriveResult,
    StepResult,
    compute_rmsd,
    kabsch_superimpose,
    make_pseudo_diffusion,
    tm_score,
)


def make_helix_coords(n_residues: int = 50) -> torch.Tensor:
    """Generate ideal alpha-helix CA coordinates."""
    # Alpha helix: rise=1.5A, radius=2.3A, 100deg per residue
    coords = []
    for i in range(n_residues):
        angle = math.radians(100.0 * i)
        x = 2.3 * math.cos(angle)
        y = 2.3 * math.sin(angle)
        z = 1.5 * i
        coords.append([x, y, z])
    return torch.tensor(coords, dtype=torch.float32)


def make_converter_and_diffusion(
    n_residues: int,
) -> tuple[PairContactConverter, callable]:
    """Build a PairContactConverter with random weights and pseudo-diffusion."""
    # Build converter without checkpoint (random weights)
    converter = PairContactConverter(checkpoint=None, device="cpu")

    # Create pseudo-diffusion function
    # We need reference coords for alignment
    ref_coords = make_helix_coords(n_residues)
    diffusion_fn = make_pseudo_diffusion(
        converter, r_cut=10.0, tau=1.5, reference_coords=ref_coords,
    )
    return converter, diffusion_fn


def check_step_result(sr: StepResult, n: int, n_modes: int, label: str) -> list[str]:
    """Validate tensor shapes in a StepResult. Returns list of error messages."""
    errors = []

    def _check(name: str, tensor: torch.Tensor, expected_shape: tuple):
        if tensor.shape != torch.Size(expected_shape):
            errors.append(
                f"  [{label}] {name}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )

    _check("displaced_ca", sr.displaced_ca, (n, 3))
    _check("new_ca", sr.new_ca, (n, 3))
    _check("z_modified", sr.z_modified, (n, n, 128))
    _check("contact_map", sr.contact_map, (n, n))
    _check("eigenvalues", sr.eigenvalues, (n_modes,))
    _check("eigenvectors", sr.eigenvectors, (n, n_modes, 3))
    _check("b_factors", sr.b_factors, (n,))

    if not math.isfinite(sr.rmsd):
        errors.append(f"  [{label}] RMSD is not finite: {sr.rmsd}")
    if sr.rmsd < 0:
        errors.append(f"  [{label}] RMSD is negative: {sr.rmsd}")

    return errors


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────

def test_compute_rmsd():
    a = make_helix_coords(30)
    rmsd_self = compute_rmsd(a, a)
    assert abs(rmsd_self) < 1e-5, f"RMSD(a,a) should be ~0, got {rmsd_self}"
    print("  PASS: compute_rmsd(a, a) == 0")


def test_kabsch_superimpose():
    a = make_helix_coords(30)
    # Rotate + translate
    b = a @ torch.tensor([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]]) + torch.tensor([5.0, -3.0, 2.0])
    aligned, rmsd_val = kabsch_superimpose(a, b)
    assert aligned.shape == (30, 3), f"aligned shape {aligned.shape}"
    assert rmsd_val.shape == (), f"rmsd shape {rmsd_val.shape}"
    assert rmsd_val.item() < 1e-4, f"RMSD after Kabsch should be ~0, got {rmsd_val.item()}"
    print("  PASS: kabsch_superimpose returns correct shapes and near-zero RMSD")


def test_tm_score():
    a = make_helix_coords(30)
    ts = tm_score(a, a)
    assert 0.0 <= ts <= 1.0, f"TM-score out of range: {ts}"
    assert ts > 0.99, f"TM-score(a,a) should be ~1.0, got {ts}"
    print(f"  PASS: tm_score(a,a) = {ts:.4f} in [0, 1]")


def test_pseudo_diffusion_callable():
    n = 40
    converter, diffusion_fn = make_converter_and_diffusion(n)
    z_input = torch.randn(n, n, 128)
    coords_out = diffusion_fn(z_input)
    assert coords_out.shape == (n, 3), f"Expected ({n}, 3), got {coords_out.shape}"
    assert torch.isfinite(coords_out).all(), "pseudo_diffusion output has non-finite values"
    print(f"  PASS: make_pseudo_diffusion returns callable producing [{n}, 3]")


# ──────────────────────────────────────────────
# Pipeline strategy tests
# ──────────────────────────────────────────────

def test_strategy(
    strategy: str,
    n_residues: int = 50,
    n_steps: int = 3,
    n_modes: int = 20,
    extra_cfg: dict | None = None,
):
    """Run ModeDrivePipeline with a given strategy and validate outputs."""
    coords = make_helix_coords(n_residues)
    converter, diffusion_fn = make_converter_and_diffusion(n_residues)

    # Synthetic trunk z
    zij_trunk = torch.randn(n_residues, n_residues, 128)

    cfg_kwargs = dict(
        n_steps=n_steps,
        combination_strategy=strategy,
        n_anm_modes=n_modes,
        n_combinations=10,
    )
    if extra_cfg:
        cfg_kwargs.update(extra_cfg)

    config = ModeDriveConfig(**cfg_kwargs)
    pipeline = ModeDrivePipeline(converter, config, diffusion_fn)
    result = pipeline.run(coords, zij_trunk, verbose=True)

    # Validate trajectory length
    errors = []
    expected_traj = n_steps + 1
    if len(result.trajectory) != expected_traj:
        errors.append(
            f"trajectory length: expected {expected_traj}, got {len(result.trajectory)}"
        )

    if len(result.step_results) != n_steps:
        errors.append(
            f"step_results length: expected {n_steps}, got {len(result.step_results)}"
        )

    # Validate each step result
    for i, sr in enumerate(result.step_results):
        step_errors = check_step_result(sr, n_residues, n_modes, f"step_{i}")
        errors.extend(step_errors)

    return errors, result


def run_all_tests():
    passed = 0
    failed = 0
    total = 0

    # ── Helper tests ──
    print("\n=== Helper function tests ===")
    helper_tests = [
        ("compute_rmsd", test_compute_rmsd),
        ("kabsch_superimpose", test_kabsch_superimpose),
        ("tm_score", test_tm_score),
        ("pseudo_diffusion_callable", test_pseudo_diffusion_callable),
    ]
    for name, fn in helper_tests:
        total += 1
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} — {e}")
            traceback.print_exc()

    # ── Strategy tests ──
    strategies = [
        ("collectivity", {}),
        ("manual", {"manual_modes": (0, 1, 2)}),
        ("grid", {}),
        ("random", {}),
    ]

    for strategy, extra_cfg in strategies:
        total += 1
        label = f"strategy={strategy}"
        print(f"\n=== Pipeline test: {label} ===")
        try:
            errors, result = test_strategy(strategy, extra_cfg=extra_cfg)
            if errors:
                failed += 1
                print(f"  FAIL: {label}")
                for e in errors:
                    print(f"    {e}")
            else:
                passed += 1
                final_rmsd = result.step_results[-1].rmsd
                print(f"  PASS: {label} | final RMSD={final_rmsd:.3f}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {label} — {e}")
            traceback.print_exc()

    # ── Collectivity-specific: trajectory grows, RMSD increases ──
    total += 1
    print("\n=== Collectivity: trajectory growth & RMSD increase ===")
    try:
        n_steps = 4
        coords = make_helix_coords(50)
        converter, diffusion_fn = make_converter_and_diffusion(50)
        zij = torch.randn(50, 50, 128)
        config = ModeDriveConfig(
            n_steps=n_steps,
            combination_strategy="collectivity",
            n_anm_modes=20,
            n_combinations=10,
        )
        pipeline = ModeDrivePipeline(converter, config, diffusion_fn)
        result = pipeline.run(coords, zij, verbose=False)

        # Trajectory should grow each step
        assert len(result.trajectory) == n_steps + 1
        # Check at least one RMSD > 0 (exploration happened)
        rmsds = [sr.rmsd for sr in result.step_results]
        assert all(r >= 0 for r in rmsds), f"Negative RMSD found: {rmsds}"
        # With collectivity strategy, RMSD should generally be > 0 after step 1
        assert rmsds[0] > 0, f"First step RMSD should be > 0, got {rmsds[0]}"
        print(f"  PASS: trajectory has {len(result.trajectory)} entries, RMSDs={[f'{r:.3f}' for r in rmsds]}")
        passed += 1
    except Exception as e:
        failed += 1
        print(f"  FAIL: {e}")
        traceback.print_exc()

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
