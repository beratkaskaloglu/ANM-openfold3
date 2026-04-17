"""Tests for GNMContactLearner model."""

import torch
import torch.nn as nn
import pytest

from src.model import GNMContactLearner
from src.contact_head import ContactProjectionHead


class MockTrunk(nn.Module):
    """Mock OpenFold3 trunk that returns random pair representations."""

    def __init__(self, c_z: int = 128, n_res: int = 10) -> None:
        super().__init__()
        self.c_z = c_z
        self.n_res = n_res
        # Add a parameter so freeze logic has something to iterate
        self.dummy = nn.Parameter(torch.randn(c_z))

    def run_trunk(self, batch: dict) -> tuple:
        n = batch.get("n_res", self.n_res)
        b = batch.get("batch_size", 1)
        s_input = torch.randn(b, n, self.c_z)
        s = torch.randn(b, n, self.c_z)
        z = torch.randn(b, n, n, self.c_z)
        return s_input, s, z


def _make_model(c_z: int = 128, bottleneck_dim: int = 32, n_res: int = 10):
    """Create a GNMContactLearner with a mock trunk."""
    trunk = MockTrunk(c_z=c_z, n_res=n_res)
    model = GNMContactLearner(trunk, c_z=c_z, bottleneck_dim=bottleneck_dim)
    return model


def _make_batch(n_res: int = 10):
    """Create a minimal batch dict for mock trunk."""
    return {"n_res": n_res, "batch_size": 1}


class TestForwardOutputKeys:
    def test_output_contains_required_keys(self):
        model = _make_model()
        out = model(_make_batch())
        assert "C_pred" in out
        assert "pair_repr" in out
        assert "pair_repr_recon" in out

    def test_output_values_are_tensors(self):
        model = _make_model()
        out = model(_make_batch())
        for key in ("C_pred", "pair_repr", "pair_repr_recon"):
            assert isinstance(out[key], torch.Tensor)


class TestCPredProperties:
    def test_c_pred_shape(self):
        n = 12
        model = _make_model(n_res=n)
        out = model(_make_batch(n_res=n))
        assert out["C_pred"].shape == (1, n, n)

    def test_c_pred_range_zero_one(self):
        model = _make_model()
        out = model(_make_batch())
        c = out["C_pred"]
        assert (c >= 0.0).all()
        assert (c <= 1.0).all()

    def test_c_pred_symmetry(self):
        model = _make_model()
        out = model(_make_batch())
        c = out["C_pred"]
        assert torch.allclose(c, c.transpose(-1, -2), atol=1e-6)

    def test_c_pred_diagonal_zero(self):
        n = 8
        model = _make_model(n_res=n)
        out = model(_make_batch(n_res=n))
        diag = out["C_pred"][0].diag()
        assert torch.allclose(diag, torch.zeros(n), atol=1e-7)


class TestPairReprOutput:
    def test_pair_repr_shape(self):
        n, c_z = 10, 128
        model = _make_model(c_z=c_z, n_res=n)
        out = model(_make_batch(n_res=n))
        assert out["pair_repr"].shape == (1, n, n, c_z)

    def test_pair_repr_detached(self):
        model = _make_model()
        out = model(_make_batch())
        assert not out["pair_repr"].requires_grad

    def test_pair_repr_recon_shape(self):
        n, c_z = 10, 128
        model = _make_model(c_z=c_z, n_res=n)
        out = model(_make_batch(n_res=n))
        assert out["pair_repr_recon"].shape == (1, n, n, c_z)


class TestFrozenBackbone:
    def test_trunk_params_frozen(self):
        model = _make_model()
        for param in model.openfold.parameters():
            assert not param.requires_grad

    def test_trunk_stays_eval_after_train(self):
        model = _make_model()
        model.train()
        assert not model.openfold.training


class TestTrainableHead:
    def test_head_params_trainable(self):
        model = _make_model()
        for param in model.contact_head.parameters():
            assert param.requires_grad

    def test_gradient_flows_to_head(self):
        model = _make_model(n_res=6)
        out = model(_make_batch(n_res=6))
        # Use both C_pred and pair_repr_recon so gradients reach forward-path params
        loss = out["C_pred"].sum() + out["pair_repr_recon"].sum()
        loss.backward()
        # Forward path uses w_enc, v, w_dec; w_inv is inverse-only
        forward_params = {"w_enc", "v", "w_dec"}
        for name, param in model.contact_head.named_parameters():
            top_name = name.split(".")[0]
            if top_name in forward_params:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_gradient_to_trunk(self):
        model = _make_model(n_res=6)
        out = model(_make_batch(n_res=6))
        loss = out["C_pred"].sum()
        loss.backward()
        for param in model.openfold.parameters():
            assert param.grad is None


class TestParamCount:
    def test_trainable_param_count_default(self):
        """Default bottleneck_dim=32: W_enc(4096) + v(32) + W_dec(4096) + w_inv(4288) = 12512."""
        model = _make_model(c_z=128, bottleneck_dim=32)
        trainable = sum(
            p.numel() for p in model.contact_head.parameters() if p.requires_grad
        )
        assert trainable == 12512

    def test_trainable_param_count_small(self):
        """bottleneck_dim=16: W_enc(1024) + v(16) + W_dec(1024) + w_inv(1120) = 3184."""
        model = _make_model(c_z=64, bottleneck_dim=16)
        trainable = sum(
            p.numel() for p in model.contact_head.parameters() if p.requires_grad
        )
        assert trainable == 3184
