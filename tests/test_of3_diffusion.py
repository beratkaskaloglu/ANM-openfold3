"""Tests for DiffusionResult dataclass from of3_diffusion."""

import pytest
import torch

from src.of3_diffusion import DiffusionResult


class TestDiffusionResult:
    def test_basic_construction(self):
        K, N = 5, 20
        result = DiffusionResult(
            all_ca=torch.randn(K, N, 3),
            best_ca=torch.randn(N, 3),
            best_idx=2,
            plddt=torch.rand(K, N),
            ptm=torch.rand(K),
            ranking=torch.rand(K),
        )
        assert result.all_ca.shape == (K, N, 3)
        assert result.best_ca.shape == (N, 3)
        assert result.best_idx == 2

    def test_none_confidence(self):
        K, N = 3, 10
        result = DiffusionResult(
            all_ca=torch.randn(K, N, 3),
            best_ca=torch.randn(N, 3),
            best_idx=0,
            plddt=None,
            ptm=None,
            ranking=None,
        )
        assert result.plddt is None
        assert result.ptm is None
        assert result.ranking is None

    def test_best_idx_in_range(self):
        K, N = 5, 20
        all_ca = torch.randn(K, N, 3)
        best_idx = 3
        result = DiffusionResult(
            all_ca=all_ca,
            best_ca=all_ca[best_idx],
            best_idx=best_idx,
            plddt=None,
            ptm=None,
            ranking=None,
        )
        assert torch.allclose(result.best_ca, result.all_ca[result.best_idx])

    def test_single_sample(self):
        N = 15
        result = DiffusionResult(
            all_ca=torch.randn(1, N, 3),
            best_ca=torch.randn(N, 3),
            best_idx=0,
            plddt=torch.rand(1, N),
            ptm=torch.rand(1),
            ranking=torch.rand(1),
        )
        assert result.all_ca.shape[0] == 1
