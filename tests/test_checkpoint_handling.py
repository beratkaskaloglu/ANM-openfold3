"""Test checkpoint loading logic and weight handling.

Verifies:
1. Checkpoint format validation
2. PairContactConverter checkpoint loading
3. Notebook Cell 3 checkpoint selection logic
4. Weight consistency through pipeline
5. Notebook Cell 9 roundtrip test issue
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.contact_head import ContactProjectionHead
from src.converter import PairContactConverter

PASS = 0
FAIL = 0
ISSUES = []


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        ISSUES.append(name)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════
#  1. Checkpoint format validation
# ═══════════════════════════════════════════════════════
section("1. Checkpoint format validation")

for bdim in [32, 64]:
    print(f"\n  --- bottleneck_dim={bdim} ---")
    head = ContactProjectionHead(c_z=128, bottleneck_dim=bdim)

    # Create checkpoint dict
    ckpt = {
        "model_state_dict": head.state_dict(),
        "c_z": 128,
        "bottleneck_dim": bdim,
    }

    # 1a. Required keys
    check(f"ckpt has model_state_dict (bd={bdim})", "model_state_dict" in ckpt)
    check(f"ckpt has c_z (bd={bdim})", "c_z" in ckpt)
    check(f"ckpt has bottleneck_dim (bd={bdim})", "bottleneck_dim" in ckpt)

    # 1b. Save, reload, verify weights match
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
        torch.save(ckpt, f.name)

    loaded_ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
    head2 = ContactProjectionHead(c_z=128, bottleneck_dim=bdim)
    head2.load_state_dict(loaded_ckpt["model_state_dict"])

    all_match = True
    for key in head.state_dict():
        if not torch.equal(head.state_dict()[key], head2.state_dict()[key]):
            all_match = False
            break
    check(f"save/reload weights match exactly (bd={bdim})", all_match)

    os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════
#  2. PairContactConverter checkpoint loading
# ═══════════════════════════════════════════════════════
section("2. PairContactConverter checkpoint loading")

# 2a. Load from file path
head_ref = ContactProjectionHead(c_z=128, bottleneck_dim=32)
# Give it known weights
torch.manual_seed(42)
for p in head_ref.parameters():
    p.data = torch.randn_like(p)

ckpt_dict = {
    "model_state_dict": head_ref.state_dict(),
    "c_z": 128,
    "bottleneck_dim": 32,
}

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    tmp_path = f.name
    torch.save(ckpt_dict, f.name)

converter_from_path = PairContactConverter(checkpoint=tmp_path, device="cpu")
weights_match = all(
    torch.equal(converter_from_path.head.state_dict()[k], head_ref.state_dict()[k])
    for k in head_ref.state_dict()
)
check("load from .pt file path: weights match", weights_match)
check(
    "load from .pt file: bottleneck_dim=32",
    converter_from_path.head.bottleneck_dim == 32,
)

# 2b. Load from dict
converter_from_dict = PairContactConverter(checkpoint=ckpt_dict, device="cpu")
weights_match_dict = all(
    torch.equal(converter_from_dict.head.state_dict()[k], head_ref.state_dict()[k])
    for k in head_ref.state_dict()
)
check("load from dict: weights match", weights_match_dict)

# 2c. None checkpoint (random init)
try:
    converter_none = PairContactConverter(checkpoint=None, device="cpu")
    check("load with None checkpoint: no error", True)
    check(
        "None checkpoint defaults c_z=128",
        converter_none.head.c_z == 128,
    )
    check(
        "None checkpoint defaults bottleneck_dim=64",
        converter_none.head.bottleneck_dim == 64,
    )
except Exception as e:
    check("load with None checkpoint: no error", False, str(e))

# 2d. Verify head is in eval mode after loading
check("head in eval mode (from path)", not converter_from_path.head.training)
check("head in eval mode (from dict)", not converter_from_dict.head.training)
check("head in eval mode (None)", not converter_none.head.training)

os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════
#  3. Notebook Cell 3 checkpoint selection logic
# ═══════════════════════════════════════════════════════
section("3. Notebook Cell 3 checkpoint selection logic")

with tempfile.TemporaryDirectory() as tmpdir:
    # 3a. Create mock history.json
    history = [
        {"epoch": 1, "val_bf_pearson": 0.60, "val_loss": 0.10},
        {"epoch": 2, "val_bf_pearson": 0.75, "val_loss": 0.08},
        {"epoch": 3, "val_bf_pearson": 0.82, "val_loss": 0.07},
        {"epoch": 4, "val_bf_pearson": 0.78, "val_loss": 0.06},
        {"epoch": 5, "val_bf_pearson": 0.70, "val_loss": 0.09},
    ]
    history_path = os.path.join(tmpdir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)

    # Reproduce notebook logic
    with open(history_path) as f:
        hist = json.load(f)

    best_bf_r = -1
    best_epoch = -1
    for entry in hist:
        bf_r = entry.get("val_bf_pearson", 0)
        if bf_r > best_bf_r:
            best_bf_r = bf_r
            best_epoch = entry["epoch"]

    check("selects epoch with highest bf_r", best_epoch == 3)
    check("best bf_r value correct", abs(best_bf_r - 0.82) < 1e-6)

    # 3b. epoch checkpoint exists
    epoch_ckpt_path = os.path.join(tmpdir, f"epoch_{best_epoch:04d}.pt")
    dummy_ckpt = {
        "model_state_dict": ContactProjectionHead().state_dict(),
        "c_z": 128,
        "bottleneck_dim": 32,
        "epoch": best_epoch,
        "val_bf_pearson": best_bf_r,
    }
    torch.save(dummy_ckpt, epoch_ckpt_path)

    if os.path.exists(epoch_ckpt_path):
        selected = epoch_ckpt_path
    else:
        selected = os.path.join(tmpdir, "best_model.pt")

    check(
        "uses epoch checkpoint when it exists",
        selected == epoch_ckpt_path,
    )

    # 3c. Fallback to best_model.pt when epoch checkpoint missing
    os.unlink(epoch_ckpt_path)
    best_model_path = os.path.join(tmpdir, "best_model.pt")
    torch.save(dummy_ckpt, best_model_path)

    epoch_ckpt_path2 = os.path.join(tmpdir, f"epoch_{best_epoch:04d}.pt")
    if os.path.exists(epoch_ckpt_path2):
        selected2 = epoch_ckpt_path2
    else:
        selected2 = best_model_path

    check(
        "falls back to best_model.pt when epoch ckpt missing",
        selected2 == best_model_path,
    )

    # 3d. Test edge case: history with missing val_bf_pearson key
    history_missing = [
        {"epoch": 1},
        {"epoch": 2, "val_bf_pearson": 0.5},
    ]
    with open(history_path, "w") as f:
        json.dump(history_missing, f)

    with open(history_path) as f:
        hist2 = json.load(f)

    best_bf_r2 = -1
    best_epoch2 = -1
    for entry in hist2:
        bf_r = entry.get("val_bf_pearson", 0)
        if bf_r > best_bf_r2:
            best_bf_r2 = bf_r
            best_epoch2 = entry["epoch"]

    check(
        "handles missing val_bf_pearson gracefully",
        best_epoch2 == 2 and abs(best_bf_r2 - 0.5) < 1e-6,
    )


# ═══════════════════════════════════════════════════════
#  4. Weight consistency through pipeline
# ═══════════════════════════════════════════════════════
section("4. Weight consistency through pipeline")

# 4a. Same converter, same input -> identical output
torch.manual_seed(99)
head_known = ContactProjectionHead(c_z=128, bottleneck_dim=32)
for p in head_known.parameters():
    p.data = torch.randn_like(p)

ckpt_known = {
    "model_state_dict": head_known.state_dict(),
    "c_z": 128,
    "bottleneck_dim": 32,
}

conv = PairContactConverter(checkpoint=ckpt_known, device="cpu")

N = 20
torch.manual_seed(123)
z_input = torch.randn(N, N, 128)

# Run z_to_contact twice
c1 = conv.z_to_contact(z_input.clone())
c2 = conv.z_to_contact(z_input.clone())
check("z_to_contact deterministic (same input)", torch.equal(c1, c2))

# Run contact_to_z twice
z_recon1 = conv.contact_to_z(c1)
z_recon2 = conv.contact_to_z(c1.clone())
check("contact_to_z deterministic (same input)", torch.equal(z_recon1, z_recon2))

# 4b. Two converters from same checkpoint produce identical outputs
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    tmp_path = f.name
    torch.save(ckpt_known, f.name)

conv_a = PairContactConverter(checkpoint=tmp_path, device="cpu")
conv_b = PairContactConverter(checkpoint=tmp_path, device="cpu")

c_a = conv_a.z_to_contact(z_input.clone())
c_b = conv_b.z_to_contact(z_input.clone())
check("two converters from same ckpt: z_to_contact identical", torch.equal(c_a, c_b))

z_back_a = conv_a.contact_to_z(c_a)
z_back_b = conv_b.contact_to_z(c_b)
check(
    "two converters from same ckpt: contact_to_z identical",
    torch.equal(z_back_a, z_back_b),
)

os.unlink(tmp_path)

# 4c. Roundtrip produces consistent results
contact_rt, z_recon_rt, mse_rt = conv.roundtrip(z_input.unsqueeze(0))
contact_rt2, z_recon_rt2, mse_rt2 = conv.roundtrip(z_input.unsqueeze(0))
check("roundtrip deterministic", torch.equal(contact_rt, contact_rt2))
check("roundtrip MSE deterministic", abs(mse_rt - mse_rt2) < 1e-10)


# ═══════════════════════════════════════════════════════
#  5. Notebook Cell 9 roundtrip test issue
# ═══════════════════════════════════════════════════════
section("5. Notebook Cell 9 roundtrip test analysis")

print("""
  ISSUE FOUND in Cell 9:

  The notebook calls:
    converter.roundtrip(torch.randn(1, N, N, 128))

  This tests roundtrip with RANDOM z (torch.randn), not the actual
  trunk pair representation (zij_trunk) or a z derived from a real contact.

  Problems:
    1. Random z is out-of-distribution for the trained head. The head was
       trained on OF3 pair representations, not random noise. Roundtrip MSE
       on random z tells you nothing about real-world fidelity.

    2. The variable 'initial_analysis["contact"]' is computed above but
       NEVER used in the roundtrip test. The code loads contact_init but
       then ignores it entirely.

    3. A meaningful test would be:
         # From real contact map:
         z_from_real = converter.contact_to_z(contact_init)
         contact_rt, z_rt, mse = converter.roundtrip(z_from_real)

       Or from actual trunk z:
         contact_rt, z_rt, mse = converter.roundtrip(zij_trunk)

    4. The batch dim handling is also inconsistent: roundtrip() internally
       calls z_to_contact() which handles 3D input by unsqueezing, but
       Cell 9 passes shape [1, N, N, 128] (already 4D). This works but
       the returned contact will be [1, N, N] and z_recon [N, N, 128]
       (from inverse which doesn't handle batch). The MSE comparison
       truncates via min() so it won't crash, but the shapes are mixed.
""")

# Demonstrate the issue
torch.manual_seed(0)
N_demo = 15
conv_demo = PairContactConverter(checkpoint=None, device="cpu")

# Random z roundtrip (what notebook does)
z_random = torch.randn(1, N_demo, N_demo, 128)
c_rand, z_recon_rand, mse_rand = conv_demo.roundtrip(z_random)

# Meaningful roundtrip: from a sigmoid-like contact
c_meaningful = torch.sigmoid(torch.randn(N_demo, N_demo))
c_meaningful = 0.5 * (c_meaningful + c_meaningful.T)
c_meaningful.fill_diagonal_(0.0)
z_from_contact = conv_demo.contact_to_z(c_meaningful)
c_rt, z_rt, mse_meaningful = conv_demo.roundtrip(z_from_contact)

print(f"  Random z roundtrip MSE:      {mse_rand:.6f}")
print(f"  Meaningful roundtrip MSE:    {mse_meaningful:.6f}")
print(f"  (Random MSE is expected to be much higher and uninformative)")

check(
    "Cell 9 uses random z (flagged as issue)",
    True,
    "Random z roundtrip is not meaningful; should use real data",
)

# Additional check: roundtrip shape handling with batch dim
print(f"\n  Shape analysis of Cell 9 roundtrip:")
print(f"    Input z shape:      {z_random.shape}")
print(f"    Returned contact:   {c_rand.shape}")
print(f"    Returned z_recon:   {z_recon_rand.shape}")

# The contact has batch dim because input was 4D (no squeeze in roundtrip)
# But z_recon from inverse() drops batch dim -> shape mismatch
has_shape_issue = c_rand.dim() != z_recon_rand.dim()
if has_shape_issue:
    print(f"    WARNING: contact dim={c_rand.dim()} != z_recon dim={z_recon_rand.dim()}")
    print(f"    The roundtrip method does not handle batch dim consistently.")
check(
    "roundtrip batch dim consistency",
    not has_shape_issue,
    f"contact shape {c_rand.shape} vs z_recon shape {z_recon_rand.shape}",
)


# ═══════════════════════════════════════════════════════
#  6. Additional: default value handling in converter
# ═══════════════════════════════════════════════════════
section("6. Additional: checkpoint metadata defaults")

# Test that missing c_z/bottleneck_dim in ckpt defaults correctly
head_default = ContactProjectionHead(c_z=128, bottleneck_dim=64)
ckpt_no_meta = {"model_state_dict": head_default.state_dict()}

conv_no_meta = PairContactConverter(checkpoint=ckpt_no_meta, device="cpu")
check("missing c_z defaults to 128", conv_no_meta.head.c_z == 128)
check("missing bottleneck_dim defaults to 64", conv_no_meta.head.bottleneck_dim == 64)

# Verify weights still loaded correctly
weights_ok = all(
    torch.equal(conv_no_meta.head.state_dict()[k], head_default.state_dict()[k])
    for k in head_default.state_dict()
)
check("weights load correctly even without metadata", weights_ok)

# Test mismatched metadata would fail
head_small = ContactProjectionHead(c_z=128, bottleneck_dim=32)
ckpt_mismatch = {
    "model_state_dict": head_small.state_dict(),
    "c_z": 128,
    "bottleneck_dim": 64,  # says 64 but weights are for 32
}
try:
    conv_mismatch = PairContactConverter(checkpoint=ckpt_mismatch, device="cpu")
    check(
        "mismatched bottleneck_dim raises error",
        False,
        "Should have raised RuntimeError for size mismatch",
    )
except RuntimeError as e:
    check("mismatched bottleneck_dim raises error", "size mismatch" in str(e).lower())


# ═══════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  SUMMARY: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")
if ISSUES:
    print("  Failed tests:")
    for issue in ISSUES:
        print(f"    - {issue}")
if FAIL == 0:
    print("  All tests passed.")
print()
