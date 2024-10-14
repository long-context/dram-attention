import pytest
import torch

from .max_ip_triton_kernel import max_inner_product, max_ip_ref


def test_max_inner_product_shape():
    N, H, D = 2, 4, 64
    L = 16
    q = torch.randn(N, H, D, device="cuda")
    min_max = torch.randn(H, L, 2, D, device="cuda")

    output = max_inner_product(q, min_max)

    assert output.shape == (N, H, L)


def test_max_inner_product_dtype():
    N, H, D = 2, 4, 64
    L = 16
    q = torch.randn(N, H, D, dtype=torch.float16, device="cuda")
    min_max = torch.randn(H, L, 2, D, dtype=torch.float16, device="cuda")

    output = max_inner_product(q, min_max)

    assert output.dtype == torch.float16


def test_max_inner_product_values():
    N, H, D = 2, 4, 64
    L = 16
    q = torch.randn(N, H, D, device="cuda")
    min_max = torch.randn(H, L, 2, D, device="cuda")

    triton_output = max_inner_product(q, min_max)
    ref_output = max_ip_ref(q, min_max)

    assert torch.allclose(triton_output, ref_output, rtol=1e-3, atol=1e-3)


def test_max_inner_product_edge_case():
    N, H, D = 1, 1, 1
    L = 1
    q = torch.randn(N, H, D, device="cuda")
    min_max = torch.randn(H, L, 2, D, device="cuda")

    output = max_inner_product(q, min_max)

    assert output.shape == (N, H, L)


def test_max_inner_product_large_input():
    N, H, D = 32, 32, 128
    L = 1024
    q = torch.randn(N, H, D, device="cuda")
    min_max = torch.randn(H, L, 2, D, device="cuda")

    output = max_inner_product(q, min_max)

    assert output.shape == (N, H, L)
