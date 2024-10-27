# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from dram_attention import DRAMAttention, LRUCache
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from torch import nn


def ColumnParallelLinear(in_dim, out_dim, bias: bool, *args, **kvargs):
    del args, kvargs
    return torch.nn.Linear(in_dim, out_dim, bias=bias)


RowParallelLinear = ColumnParallelLinear


def VocabParallelEmbedding(vocab_size, dim, init_method=lambda x: x):
    del init_method
    return torch.nn.Embedding(vocab_size, dim)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    lru_hbm_cache_size: int = 32 * 1024
    local_cache_size: int = 4 * 1024 + 1024
    cache_page_size: int = 16
    cache_top_k: int = 4 * 1024
    max_output_length: int = 128

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # vision model params
    vision_chunk_size: int = -1  # image resolution for image models
    vision_max_num_chunks: int = 4
    vision_num_cross_attention_layers: int = -1

    # cache
    cache_dir: str = "cache"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1  # fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.cache_dir = args.cache_dir

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        # self.dram_attention = DRAMAttention(
        #     max_output_length=args.max_output_length,
        #     lru_hbm_cache_size=args.lru_hbm_cache_size,
        #     local_cache_size=args.local_cache_size,
        #     page_size=args.cache_page_size,
        #     top_k=args.cache_top_k,
        #     n_kv_heads=args.n_kv_heads,
        #     head_dim=self.head_dim,
        #     device="cuda",
        #     dtype=torch.bfloat16,
        # )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        index: int,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        stage = "prefill" if seqlen > 1 and start_pos == 0 else "generate"
        if stage == "prefill":
            output = flash_attn_func(xq, xk, xv, causal=True)
            import os

            cache_path = os.path.join(self.cache_dir, f"{index:02d}_cache.pth")
            torch.save(
                {
                    "cache_k": xk.to(device="cpu", non_blocking=False),
                    "cache_v": xv.to(device="cpu", non_blocking=False),
                },
                cache_path,
            )
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Saved KV cache for layer {index:02d} to {cache_path}")
        else:
            output = self.dram_attention(xq, xk, xv, start_pos=start_pos, stage=stage)
        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward_(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward(self, x):
        chunk_size = 1024 * 8
        chunks = x.split(chunk_size, dim=1)
        results = []

        for chunk in chunks:
            result = self.forward_(chunk)
            results.append(result)

        return torch.cat(results, dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        index: int,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, index)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        with torch.device("cuda"):
            self.freqs_cis = precompute_freqs_cis(
                params.dim // params.n_heads,
                params.max_seq_len * 2,
                params.rope_theta,
                params.use_scaled_rope,
            )

    def forward(self, tokens: torch.Tensor, *, start_pos: int, stage: str):
        _bsz, seqlen = tokens.shape

        if stage == "prefill":
            self.tok_embeddings.cuda()

        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if stage == "prefill":
            self.tok_embeddings.cpu()
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = self.freqs_cis.index_select(0, start_pos)

        for index, layer in enumerate(self.layers):
            if stage == "prefill":
                layer.cuda()

            h = layer(h, start_pos, freqs_cis, index)

            if stage == "prefill":
                layer.cpu()

        if stage == "prefill":
            self.norm.cuda()
            self.output.cuda()

        h = self.norm(h[:, -1:, :])
        output = self.output(h).float()

        if stage == "prefill":
            self.norm.cpu()
            self.output.cpu()

        return output
