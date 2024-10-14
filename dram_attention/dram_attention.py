import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from .lru_cache import DummyCache, LRUCache


class DRAMAttention(torch.nn.Module):
    """
    Implements DRAM Attention.

    This module manages long-term key-value caches, utilizing both DRAM and HBM for storage.

    There are two caches:
    1. Local cache: Stored on HBM for most recent tokens (suffix tokens).
    2. Prefix LRU cache: Managed by a LRUCache and resides on DRAM.

    At every decoding step, a small number of "important" pages will be transferred
    from DRAM to HBM for attention computation.

    Args:
        dram_kv_cache_size (int): Size of the DRAM key-value cache.
        lru_hbm_cache_size (int): Size of the HBM LRU cache.
        local_cache_size (int): Size of (on GPU) local cache.
        page_size (int): Size of each page in the LRU cache.
        top_k (int): Number of top elements to retrieve from the LRU cache.
        n_kv_heads (int): Number of key-value heads.
        head_dim (int): Dimension of each head.
        device (str, optional): Device to store the cache tensors. Defaults to "cuda".
    """

    def __init__(
        self,
        max_output_length: int,
        lru_hbm_cache_size: int,
        local_cache_size: int,
        page_size: int,
        top_k: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.max_output_length = max_output_length
        self.page_size = page_size
        self.dram_kv_cache_size = None  # will be initialized later
        self.lru_hbm_cache_size = lru_hbm_cache_size
        self.local_cache_size = local_cache_size
        self.top_k = top_k
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_page = self.lru_hbm_cache_size // self.page_size
        self.device = device

        # Initialize cache tensors
        cache_shape = (1, local_cache_size + max_output_length, n_kv_heads, head_dim)
        self.register_buffer(
            "cache_k", torch.zeros(cache_shape, device=device, dtype=dtype)
        )
        self.register_buffer(
            "cache_v", torch.zeros(cache_shape, device=device, dtype=dtype)
        )

    def init_cache(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """
        Initialize the key-value caches.

        Args:
            k_cache (torch.Tensor): Initial key cache.
            v_cache (torch.Tensor): Initial value cache.
        """
        bsz, seqlen, H, D = k_cache.shape
        assert bsz == 1, "Only support batch size 1"
        # Copy suffix tokens to permanent HBM cache (local window)
        dram_kv_cache_size = seqlen - self.local_cache_size + self.page_size
        self.dram_kv_cache_size = dram_kv_cache_size // self.page_size * self.page_size
        L = seqlen - self.dram_kv_cache_size
        self.cache_k[:, :L].copy_(
            k_cache[:, self.dram_kv_cache_size :], non_blocking=True
        )
        self.cache_v[:, :L].copy_(
            v_cache[:, self.dram_kv_cache_size :], non_blocking=True
        )

        # DRAM caches
        dram_k_cache = k_cache[:, : self.dram_kv_cache_size]
        dram_v_cache = v_cache[:, : self.dram_kv_cache_size]

        # Initialize LRU cache
        if self.dram_kv_cache_size > self.lru_hbm_cache_size:
            self.lru_cache = LRUCache(
                k_cache=dram_k_cache[0].transpose(0, 1),
                v_cache=dram_v_cache[0].transpose(0, 1),
                page_size=self.page_size,
                top_k=self.top_k // self.page_size,
                lru_hbm_cache_size=self.lru_hbm_cache_size,
                device=self.device,
            )
        else:
            self.lru_cache = DummyCache(
                k_cache=dram_k_cache[0].transpose(0, 1),
                v_cache=dram_v_cache[0].transpose(0, 1),
                device=self.device,
            )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        start_pos: int,
        stage: str,
    ):
        """
        Forward pass of the DRAM Attention module.

        Args:
            xq (torch.Tensor): Query tensor.
            xk (torch.Tensor): Key tensor.
            xv (torch.Tensor): Value tensor.
            start_pos (int): Starting position in the sequence.
            stage (str): Current stage of processing ("prefill" or "generate").

        Returns:
            torch.Tensor: Output of the attention mechanism.

        Raises:
            RuntimeError: If an unsupported stage is provided.
        """
        bsz, seqlen, H, D = xq.shape
        assert bsz == 1, "Only support batch size 1"

        if stage == "prefill":
            output = flash_attn_func(xq, xk, xv, causal=True)
            self.init_cache(xk, xv)
            return output
        elif stage == "generate":
            return self._generate(xq, xk, xv, start_pos)
        else:
            raise RuntimeError(f"Unsupported stage: {stage}")

    def _generate(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, start_pos: int
    ):
        """
        Compute attention during the generate stage.

        Args:
            xq (torch.Tensor): Query tensor.
            xk (torch.Tensor): Key tensor.
            xv (torch.Tensor): Value tensor.
            start_pos (int): Starting position in the sequence.

        Returns:
            torch.Tensor: Output of the attention mechanism.
        """
        bsz, seqlen, _, _ = xq.shape

        # Retrieve HBM cache
        hbm_cache_k, hbm_cache_v = self._get_hbm_cache(xq)

        # Compute prefix attention
        prefix_output, prefix_lse = self._compute_prefix_attention(
            xq, hbm_cache_k, hbm_cache_v
        )

        # Compute suffix attention
        suffix_output, suffix_lse = self._compute_suffix_attention(
            xq, xk, xv, start_pos
        )

        # Combine prefix and suffix attention
        return self._combine_attention(
            prefix_output, suffix_output, prefix_lse, suffix_lse, xq.dtype
        )

    def _get_hbm_cache(self, xq: torch.Tensor):
        """Retrieve and reshape HBM cache."""
        hbm_cache_k, hbm_cache_v = self.lru_cache.get_kv_cache_(xq)
        bsz = xq.shape[0]
        hbm_cache_k = hbm_cache_k.transpose(0, 1).view(
            bsz, -1, self.n_kv_heads, self.head_dim
        )
        hbm_cache_v = hbm_cache_v.transpose(0, 1).view(
            bsz, -1, self.n_kv_heads, self.head_dim
        )
        return hbm_cache_k, hbm_cache_v

    def _compute_prefix_attention(
        self, xq: torch.Tensor, hbm_cache_k: torch.Tensor, hbm_cache_v: torch.Tensor
    ):
        """Compute attention for the prefix)."""
        return flash_attn_with_kvcache(
            xq,
            hbm_cache_k,
            hbm_cache_v,
            None,
            None,
            causal=True,
            cache_seqlens=hbm_cache_k.shape[1],
            rotary_interleaved=True,
            return_softmax_lse=True,
        )

    def _compute_suffix_attention(
        self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, start_pos: int
    ):
        """Compute attention for the suffix."""
        return flash_attn_with_kvcache(
            xq,
            self.cache_k,
            self.cache_v,
            xk,
            xv,
            rotary_cos=None,
            rotary_sin=None,
            causal=True,
            cache_seqlens=start_pos - self.dram_kv_cache_size,
            rotary_interleaved=True,
            return_softmax_lse=True,
        )

    def _combine_attention(
        self,
        prefix_output: torch.Tensor,
        suffix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_lse: torch.Tensor,
        dtype: torch.dtype,
    ):
        """Combine prefix and suffix attention outputs."""
        lse = torch.logaddexp(prefix_lse, suffix_lse)
        prefix_weight = prefix_lse.sub_(lse).exp_().transpose(1, 2)[..., None]
        suffix_weight = suffix_lse.sub_(lse).exp_().transpose(1, 2)[..., None]
        output = prefix_output * prefix_weight + suffix_output * suffix_weight
        return output.to(dtype=dtype)
