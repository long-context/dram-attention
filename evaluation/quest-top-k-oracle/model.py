# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
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

    max_batch_size: int = 32
    max_seq_len: int = 2048

    # vision model params
    vision_chunk_size: int = -1  # image resolution for image models
    vision_max_num_chunks: int = 4
    vision_num_cross_attention_layers: int = -1

    # quest args
    quest_top_k: int = 4096
    quest_page_size: int = 16
    quest_local_attn_window: int = 4096

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
        self.local_attn_window = args.quest_local_attn_window
        self.top_k = args.quest_top_k
        self.page_size = args.quest_page_size

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
        self.cache_leftpad = None
        with torch.device("cuda"):
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        index: int,
    ):
        del index
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        if seqlen > 1:
            output = flash_attn_with_kvcache(
                q=xq,
                k_cache=self.cache_k,
                v_cache=self.cache_v,
                k=xk,
                v=xv,
                cache_leftpad=None,
                cache_seqlens=start_pos,
                causal=True,
                rotary_interleaved=True,
                return_softmax_lse=False,
            )

            prefix_cache_size = seqlen - self.local_attn_window
            assert prefix_cache_size % self.page_size == 0
            cache_k = self.cache_k[:, :prefix_cache_size]
            cache_v = self.cache_v[:, :prefix_cache_size]
            N, L, H, D = cache_v.shape
            self.cache_leftpad = torch.tensor(
                [prefix_cache_size], dtype=torch.int32, device=xq.device
            )
            self.cache_k_paged = cache_k.view(
                N, L // self.page_size, self.page_size, H, D
            )
            self.cache_k_page_min, self.cache_k_page_max = torch.aminmax(
                self.cache_k_paged, dim=2, keepdim=False
            )
            self.cache_v_paged = cache_v.view(
                N, L // self.page_size, self.page_size, H, D
            )
            self.num_top_pages = self.top_k // self.page_size
        else:
            suffix_output, suffix_lse = flash_attn_with_kvcache(
                q=xq,
                k_cache=self.cache_k,
                v_cache=self.cache_v,
                k=xk,
                v=xv,
                cache_leftpad=self.cache_leftpad,
                cache_seqlens=start_pos,
                causal=True,
                rotary_interleaved=True,
                return_softmax_lse=True,
            )

            xq_ = xq.view(bsz, seqlen, self.n_local_kv_heads, self.n_rep, self.head_dim)
            # amin = xq_ * self.cache_k_page_min[:, :, :, None, :]
            # amax = xq_ * self.cache_k_page_max[:, :, :, None, :]
            # scores = torch.sum(torch.maximum(amin, amax), dim=-1)
            scores = torch.einsum(
                "NSPHD,NLHRD->NSPHR",
                self.cache_k_paged,
                xq_
            ).amax(dim=2) # NPHR
            top_indices = torch.argsort(scores, dim=1, descending=True)
            num_top_pages = min(top_indices.shape[1], self.num_top_pages)
            top_indices = top_indices[:, :num_top_pages]
            top_indices = top_indices.swapaxes(2, 3)  # N K rep H
            top_indices = top_indices.reshape(
                bsz, num_top_pages * self.n_rep, self.n_local_kv_heads
            )
            top_indices = top_indices[:, :, None, :, None].expand(
                -1, -1, self.page_size, -1, self.head_dim
            )

            prefix_cache_k = torch.gather(self.cache_k_paged, 1, top_indices)
            prefix_cache_k = prefix_cache_k.view(
                bsz,
                num_top_pages,
                self.n_rep,
                self.page_size,
                self.n_local_kv_heads,
                self.head_dim,
            )
            prefix_cache_k = prefix_cache_k.permute(0, 1, 3, 4, 2, 5)
            prefix_cache_k = prefix_cache_k.reshape(
                bsz, num_top_pages * self.page_size, self.n_local_heads, self.head_dim
            )

            prefix_cache_v = torch.gather(self.cache_v_paged, 1, top_indices)
            prefix_cache_v = prefix_cache_v.view(
                bsz,
                num_top_pages,
                self.n_rep,
                self.page_size,
                self.n_local_kv_heads,
                self.head_dim,
            )
            prefix_cache_v = prefix_cache_v.permute(0, 1, 3, 4, 2, 5)
            prefix_cache_v = prefix_cache_v.reshape(
                bsz, num_top_pages * self.page_size, self.n_local_heads, self.head_dim
            )

            prefix_output, prefix_lse = flash_attn_with_kvcache(
                q=xq,
                k_cache=prefix_cache_k,
                v_cache=prefix_cache_v,
                k=None,
                v=None,
                cache_seqlens=prefix_cache_k.shape[1],
                causal=True,
                rotary_interleaved=True,
                return_softmax_lse=True,
            )
            lse = torch.logaddexp(prefix_lse, suffix_lse)
            prefix_weight = prefix_lse.sub_(lse).exp_()
            suffix_weight = suffix_lse.sub_(lse).exp_()
            # Swap axes 1 and 2 for prefix_weight and suffix_weight
            prefix_weight = prefix_weight.transpose(1, 2)[..., None]
            suffix_weight = suffix_weight.transpose(1, 2)[..., None]
            output = prefix_output * prefix_weight + suffix_output * suffix_weight
            output = output.to(dtype=xq.dtype)
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

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for index, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, index)

        h = self.norm(h[:, -1:, :])
        output = self.output(h).float()
        return output
