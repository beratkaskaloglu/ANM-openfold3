"""Test suite for ContactProjectionHead, PairContactConverter, Kirchhoff, and GNM preservation."""

import sys
import math
import torch
import numpy as np

sys.path.insert(0, ".")

from src.contact_head import ContactProjectionHead
from src.converter import PairContactConverter
from src.kirchhoff import soft_kirchhoff, gnm_decompose
from src.ground_truth import compute_gt_probability_matrix

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def make_helix_coords(n: int = 50) -> torch.Tensor:
    """Ideal alpha-helix Calpha coordinates."""
    coords = []
    for i in range(n):
        # 3.6 residues per turn, 1.5A rise per residue, 2.3A radius
        theta = i * 2 * math.pi / 3.6
        x = 2.3 * math.cos(theta)
        y = 2.3 * math.sin(theta)
        z = 1.5 * i
        coords.append([x, y, z])
    return torch.tensor(coords, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# 1. ContactProjectionHead tests
# ──────────────────────────────────────────────────────────────
print("\n=== 1. ContactProjectionHead ===")

head = ContactProjectionHead(c_z=128, bottleneck_dim=32)
N = 20

# 1a. Forward
print("\n-- 1a. Forward --")
z = torch.randn(1, N, N, 128)
c_pred = head(z)
check("shape [1, N, N]", c_pred.shape == (1, N, N), str(c_pred.shape))
check("values in [0,1]", (c_pred >= 0).all() and (c_pred <= 1).all(),
      f"min={c_pred.min():.4f}, max={c_pred.max():.4f}")
check("symmetric", torch.allclose(c_pred, c_pred.transpose(-1, -2), atol=1e-6))
diag = torch.diagonal(c_pred, dim1=-2, dim2=-1)
check("diagonal=0", (diag.abs() < 1e-6).all(), f"max diag={diag.abs().max():.6f}")

# 1b. Inverse
print("\n-- 1b. Inverse --")
c_input = torch.rand(N, N)
c_input = 0.5 * (c_input + c_input.T)
c_input.fill_diagonal_(0.0)
pseudo_z = head.inverse(c_input)
check("shape [N, N, 128]", pseudo_z.shape == (N, N, 128), str(pseudo_z.shape))

# 1c. Roundtrip z -> forward -> c -> inverse -> z_recon
print("\n-- 1c. Roundtrip z->c->z --")
z_in = torch.randn(1, N, N, 128)
z_sym = 0.5 * (z_in + z_in.transpose(1, 2))  # symmetrise for fair comparison
c_mid = head(z_sym)
z_recon = head.inverse(c_mid.squeeze(0))
mse = ((z_sym.squeeze(0) - z_recon) ** 2).mean().item()
check("roundtrip MSE is finite", math.isfinite(mse), f"MSE={mse:.4f}")
print(f"    (z->c->z roundtrip MSE = {mse:.4f})")

# 1d. Roundtrip c_gt -> inverse -> z -> forward -> c_recon
print("\n-- 1d. Roundtrip c->z->c --")
coords = make_helix_coords(N)
c_gt = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)
z_pseudo = head.inverse(c_gt)
c_recon = head(z_pseudo.unsqueeze(0)).squeeze(0)
mse_c = ((c_gt - c_recon) ** 2).mean().item()
check("contact roundtrip MSE is finite", math.isfinite(mse_c), f"MSE={mse_c:.4f}")
print(f"    (c->z->c roundtrip MSE = {mse_c:.4f})")

# 1e. encode_bottleneck
print("\n-- 1e. encode_bottleneck --")
h = head.encode_bottleneck(z)
check("shape [B, N, N, bottleneck_dim]", h.shape == (1, N, N, 32), str(h.shape))


# ──────────────────────────────────────────────────────────────
# 2. PairContactConverter tests
# ──────────────────────────────────────────────────────────────
print("\n=== 2. PairContactConverter ===")

# 2a. Load with no checkpoint
print("\n-- 2a. No-checkpoint load --")
conv = PairContactConverter(checkpoint=None, device="cpu")
check("loads without checkpoint", conv.head is not None)

# 2b. z_to_contact
print("\n-- 2b. z_to_contact --")
z_in = torch.randn(N, N, 128)
c_out = conv.z_to_contact(z_in)
check("shape [N, N]", c_out.shape == (N, N), str(c_out.shape))
check("values in [0,1]", (c_out >= 0).all() and (c_out <= 1).all(),
      f"min={c_out.min():.4f}, max={c_out.max():.4f}")

# 2c. contact_to_z
print("\n-- 2c. contact_to_z --")
c_in = torch.rand(N, N)
c_in = 0.5 * (c_in + c_in.T)
c_in.fill_diagonal_(0.0)
z_out = conv.contact_to_z(c_in)
check("shape [N, N, 128]", z_out.shape == (N, N, 128), str(z_out.shape))

# 2d. roundtrip
print("\n-- 2d. roundtrip --")
z_rt = torch.randn(N, N, 128)
contact, z_recon, mse = conv.roundtrip(z_rt)
check("returns contact shape [N, N]", contact.shape == (N, N), str(contact.shape))
check("returns z_recon shape [N, N, 128]", z_recon.shape == (N, N, 128), str(z_recon.shape))
check("returns finite mse", math.isfinite(mse), f"MSE={mse:.4f}")

# 2e. analyze with contact input
print("\n-- 2e. analyze (contact input) --")
c_analyze = torch.rand(N, N)
c_analyze = 0.5 * (c_analyze + c_analyze.T)
c_analyze.fill_diagonal_(0.0)
result = conv.analyze(c_analyze, n_modes=10)
expected_keys = {"contact", "kirchhoff", "eigenvalues", "eigenvectors", "b_factors"}
check("all keys present", set(result.keys()) == expected_keys,
      f"got {set(result.keys())}")
check("eigenvalues shape [10]", result["eigenvalues"].shape == (10,),
      str(result["eigenvalues"].shape))
check("eigenvectors shape [N, 10]", result["eigenvectors"].shape == (N, 10),
      str(result["eigenvectors"].shape))
check("b_factors shape [N]", result["b_factors"].shape == (N,),
      str(result["b_factors"].shape))

# 2f. analyze with pair repr input (is_contact=False auto-detection)
print("\n-- 2f. analyze (pair repr, auto-detect) --")
z_analyze = torch.randn(N, N, 128)
result2 = conv.analyze(z_analyze, n_modes=5)
check("auto-detects pair repr (3D)", result2["contact"].shape == (N, N),
      str(result2["contact"].shape))
check("eigenvalues shape [5]", result2["eigenvalues"].shape == (5,),
      str(result2["eigenvalues"].shape))


# ──────────────────────────────────────────────────────────────
# 3. GNM preservation test
# ──────────────────────────────────────────────────────────────
print("\n=== 3. GNM Preservation (helix, random weights) ===")

N_helix = 50
coords = make_helix_coords(N_helix)
c_gt = compute_gt_probability_matrix(coords, r_cut=10.0, tau=1.5)

# GT GNM
gamma_gt = soft_kirchhoff(c_gt)
evals_gt, evecs_gt, bfac_gt = gnm_decompose(gamma_gt, n_modes=20)

# Roundtrip through random-weight head
head_rt = ContactProjectionHead(c_z=128, bottleneck_dim=32)
head_rt.eval()
with torch.no_grad():
    z_pseudo = head_rt.inverse(c_gt)
    c_recon = head_rt(z_pseudo.unsqueeze(0)).squeeze(0)

gamma_recon = soft_kirchhoff(c_recon)
evals_recon, evecs_recon, bfac_recon = gnm_decompose(gamma_recon, n_modes=20)

# B-factor Pearson correlation
bfac_gt_np = bfac_gt.numpy()
bfac_recon_np = bfac_recon.numpy()
pearson_r = np.corrcoef(bfac_gt_np, bfac_recon_np)[0, 1]

check("B-factor Pearson r > 0", pearson_r > 0, f"r={pearson_r:.4f}")
print(f"    B-factor Pearson r = {pearson_r:.4f}")

mse_contact = ((c_gt - c_recon) ** 2).mean().item()
print(f"    Contact map MSE = {mse_contact:.4f}")

evals_corr = np.corrcoef(evals_gt.numpy(), evals_recon.numpy())[0, 1]
check("eigenvalue correlation > 0", evals_corr > 0, f"r={evals_corr:.4f}")
print(f"    Eigenvalue correlation = {evals_corr:.4f}")


# ──────────────────────────────────────────────────────────────
# 4. Kirchhoff tests
# ──────────────────────────────────────────────────────────────
print("\n=== 4. Kirchhoff & GNM decompose ===")

N_k = 30
c_k = torch.rand(N_k, N_k)
c_k = 0.5 * (c_k + c_k.T)
c_k.fill_diagonal_(0.0)

# 4a. soft_kirchhoff
print("\n-- 4a. soft_kirchhoff --")
gamma = soft_kirchhoff(c_k, eps=1e-6)
check("symmetric", torch.allclose(gamma, gamma.T, atol=1e-6))

# Check off-diagonal = -C_ij
mask = ~torch.eye(N_k, dtype=torch.bool)
offdiag_diff = (gamma[mask] + c_k[mask]).abs().max().item()
check("off-diagonal = -C_ij", offdiag_diff < 1e-5, f"max diff={offdiag_diff:.6f}")

# Check diagonal = sum of row contacts + eps
diag_expected = c_k.sum(dim=-1) + 1e-6
diag_actual = torch.diagonal(gamma)
diag_diff = (diag_actual - diag_expected).abs().max().item()
check("diagonal = coordination + eps", diag_diff < 1e-5, f"max diff={diag_diff:.6f}")

# PSD check: all eigenvalues >= 0
eigvals = torch.linalg.eigvalsh(gamma)
check("PSD (all eigenvalues >= -1e-5)", (eigvals >= -1e-5).all(),
      f"min eigenvalue={eigvals.min():.6f}")

# 4b. gnm_decompose
print("\n-- 4b. gnm_decompose --")
evals, evecs, bfacs = gnm_decompose(gamma, n_modes=10)
check("eigenvalues ascending", (evals[1:] >= evals[:-1] - 1e-6).all(),
      str(evals.tolist()))
check("eigenvalues shape [10]", evals.shape == (10,), str(evals.shape))
check("eigenvectors shape [N, 10]", evecs.shape == (N_k, 10), str(evecs.shape))

# Orthogonality
gram = evecs.T @ evecs
identity = torch.eye(10)
ortho_err = (gram - identity).abs().max().item()
check("eigenvectors orthogonal", ortho_err < 1e-4, f"max deviation={ortho_err:.6f}")

check("B-factors all positive", (bfacs > 0).all(), f"min={bfacs.min():.6f}")
check("B-factors shape [N]", bfacs.shape == (N_k,), str(bfacs.shape))


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL} total")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
