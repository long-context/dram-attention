import pytest
import torch

from dram_attention.dram_attention import DRAMAttention


@pytest.fixture
def dram_attention():
    return DRAMAttention(
        max_output_length=64,
        lru_hbm_cache_size=32 * 1024,
        local_cache_size=4096 + 512,
        page_size=16,
        top_k=4096,
        n_kv_heads=8,
        head_dim=128,
        device="cuda",
    )


def test_init(dram_attention):
    assert dram_attention.max_output_length == 64
    assert dram_attention.lru_hbm_cache_size == 32 * 1024
    assert dram_attention.local_cache_size == 4096 + 512
    assert dram_attention.page_size == 16
    assert dram_attention.top_k == 4096
    assert dram_attention.n_kv_heads == 8
    assert dram_attention.head_dim == 128


def test_forward_prefill(dram_attention):
    bsz, seqlen, n_heads, head_dim = 1, 66 * 1024, 8, 128
    xq = torch.randn(
        bsz,
        seqlen,
        4 * n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xv = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )

    output = dram_attention(xq, xk, xv, start_pos=0, stage="prefill")

    assert output.shape == (bsz, seqlen, 4 * n_heads, head_dim)


def test_forward_generate(dram_attention):
    bsz, seqlen, n_heads, head_dim = 1, 66 * 1024, 8, 128
    xq = torch.randn(
        bsz,
        seqlen,
        4 * n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xv = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )

    # First, run a prefill to initialize the cache
    dram_attention(xq, xk, xv, start_pos=0, stage="prefill")

    # Now test the generate stage
    bsz, seqlen, n_heads, head_dim = 1, 1, 8, 128
    xq = torch.randn(
        bsz,
        seqlen,
        4 * n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xv = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    output = dram_attention(xq, xk, xv, start_pos=64 * 1024 + 4096, stage="generate")

    assert output.shape == (bsz, 1, 4 * n_heads, head_dim)


def test_invalid_stage(dram_attention):
    bsz, seqlen, n_heads, head_dim = 1, 1, 8, 128
    xq = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xv = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )

    with pytest.raises(RuntimeError, match="Unsupported stage"):
        dram_attention(xq, xk, xv, start_pos=0, stage="invalid")


def test_batch_size_assertion(dram_attention):
    bsz, seqlen, n_heads, head_dim = 2, 1, 8, 128
    xq = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )
    xv = torch.randn(
        bsz,
        seqlen,
        n_heads,
        head_dim,
        device=dram_attention.cache_k.device,
        dtype=torch.bfloat16,
    )

    with pytest.raises(AssertionError, match="Only support batch size 1"):
        dram_attention(xq, xk, xv, start_pos=0, stage="prefill")
