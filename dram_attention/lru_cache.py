"""
GPU HBM as cache with host DRAM as main storage for KV Cache management.

This module implements an LRUCache object that manages DRAM KV Caches
using a Least Recently Used (LRU) caching mechanism. KV caches are loaded
from DRAM to HBM (High Bandwidth Memory) when needed.

For efficiency, KV caches are stored in pages and are loaded and evicted at the page level.
"""

from typing import NamedTuple

import torch

from .lru_triton_kernel import load_lru_cache_
from .topk_selector import IPUpperboundTopKSelector


class CacheOutput(NamedTuple):
    k_cache: torch.Tensor
    v_cache: torch.Tensor


class DummyCache(torch.nn.Module):
    """
    A static KV Cache on HBM
    """

    def __init__(
        self, k_cache: torch.Tensor, v_cache: torch.Tensor, device: str, *args, **kvargs
    ):
        super().__init__()
        self.k_cache = k_cache.to(device=device, non_blocking=True)
        self.v_cache = v_cache.to(device=device, non_blocking=True)

    def get_kv_cache_(self, q: torch.Tensor) -> CacheOutput:
        return CacheOutput(k_cache=self.k_cache, v_cache=self.v_cache)


class LRUCache(torch.nn.Module):
    """
    LRUCache manages KV (Key-Value) caches in DRAM and HBM using an LRU caching mechanism.

    This object handles KV cache at the layer level. Each self-attention layer has its own LRUCache object.

    Attributes:
        current_step (torch.Tensor): Current step counter for LRU tracking. Shape: (1,)
        dram_kv_cache (torch.Tensor): KV cache stored in DRAM. Shape: (num_kv_heads, num_dram_pages, page_size * 2 * head_dim)
        hbm_kv_cache (torch.Tensor): KV cache stored in HBM (GPU memory). Shape: (num_kv_heads, num_hbm_pages, page_size * 2 * head_dim)
        page_size (int): Size of each page in the cache.
        num_kv_heads (int): Number of attention heads.
        num_dram_pages (int): Number of pages in DRAM.
        num_hbm_pages (int): Number of pages in HBM.
        head_dim (int): Dimension of each attention head.
        page_access_time (torch.Tensor): Tracks the last access time of each page in HBM. Shape: (num_kv_heads, num_hbm_pages)
        dram_page_to_hbm_page_mapping (torch.Tensor): Maps DRAM pages to HBM pages. Shape: (num_kv_heads, num_dram_pages)
        hbm_page_to_dram_page_mapping (torch.Tensor): Maps HBM pages to DRAM pages. Shape: (num_kv_heads, num_hbm_pages)
        top_k (int): Number of pages to fetch from DRAM to HBM.
        top_k_selector (IPUpperboundTopKSelector): Selector for finding most relevant DRAM pages.
    """

    def __init__(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_size: int,
        top_k: int,
        lru_hbm_cache_size: int | None = None,
        hbm_kv_cache_fraction: float = 0.2,
        device: str = "cuda",
    ):
        """
        Initialize the LRUCache.

        Args:
            k_cache (torch.Tensor): Key vectors. Shape: (num_kv_heads, length, head_dim)
            v_cache (torch.Tensor): Value vectors. Shape: (num_kv_heads, length, head_dim)
            page_size (int): Size of each page in the cache.
            top_k (int): Maximum number of pages to fetch from DRAM to HBM for each query.
            lru_hbm_cache_size (int, optional): Size of the HBM KV cache. If None, it's calculated based on hbm_kv_cache_fraction.
            hbm_kv_cache_fraction (float, optional): Fraction of DRAM cache size to use for HBM cache. Defaults to 0.2.
            device (str, optional): Device to use for computations. Defaults to "cuda".

        Raises:
            AssertionError: If k_cache and v_cache shapes don't match or if KV cache size is not divisible by page size.
        """
        super().__init__()
        assert k_cache.shape == v_cache.shape, "Key and value cache shapes must match."
        num_kv_heads, dram_kv_cache_size, head_dim = k_cache.shape
        dtype = k_cache.dtype

        assert (
            dram_kv_cache_size % page_size == 0
        ), "KV cache size must be divisible by page size."
        num_dram_pages = dram_kv_cache_size // page_size
        num_hbm_pages = (
            int(num_dram_pages * hbm_kv_cache_fraction)
            if lru_hbm_cache_size is None
            else lru_hbm_cache_size // page_size
        )

        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_dram_pages = num_dram_pages
        self.num_hbm_pages = num_hbm_pages
        self.top_k = top_k

        self.top_k_selector = IPUpperboundTopKSelector(
            k_cache, num_dram_pages, device=device
        )
        self.register_buffer(
            "current_step",
            torch.full(
                size=(1,),
                fill_value=0,
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "dram_kv_cache",
            torch.empty(
                size=(num_kv_heads, num_dram_pages, page_size * 2 * head_dim),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            ),
        )
        # Copy to DRAM cache pages
        _cache = self.dram_kv_cache.view(
            num_kv_heads, num_dram_pages, page_size, 2, head_dim
        )
        _cache[..., 0, :].copy_(
            k_cache.view(num_kv_heads, num_dram_pages, page_size, head_dim),
            non_blocking=True,
        )
        _cache[..., 1, :].copy_(
            v_cache.view(num_kv_heads, num_dram_pages, page_size, head_dim),
            non_blocking=True,
        )
        del _cache

        self.register_buffer(
            "hbm_kv_cache",
            torch.empty(
                num_kv_heads,
                num_hbm_pages,
                page_size * 2 * head_dim,
                dtype=dtype,
                device=device,
            ),
        )

        self.register_buffer(
            "page_access_time",
            torch.zeros(
                num_kv_heads,
                num_hbm_pages,
                dtype=torch.float32,
                device=device,
            ),
        )

        # dram_page_to_hbm_page_mapping: Tracks the location of DRAM pages in HBM
        # Values:
        #   -1: The DRAM page is not currently loaded in HBM
        #   0 to (num_hbm_pages - 1): The index of the HBM page where the DRAM page is loaded
        self.register_buffer(
            "dram_page_to_hbm_page_mapping",
            torch.full(
                (num_kv_heads, num_dram_pages),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            ),
        )
        # hbm_page_to_dram_page_mapping: Maps HBM pages to DRAM pages
        self.register_buffer(
            "hbm_page_to_dram_page_mapping",
            torch.full(
                (num_kv_heads, num_hbm_pages),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            ),
        )
        assert (
            num_hbm_pages <= num_dram_pages
        ), "HBM cache size must be smaller than DRAM cache size."
        # Load the rightmost pages of DRAM KV cache to HBM, assuming recent pages are more important
        self.hbm_kv_cache.copy_(
            self.dram_kv_cache[:, -num_hbm_pages:],
            non_blocking=True,
        )
        self.dram_page_to_hbm_page_mapping[:, -num_hbm_pages:].copy_(
            torch.arange(num_hbm_pages, dtype=torch.int64, device=device)[None],
            non_blocking=True,
        )
        # Initialize hbm_page_to_dram_page_mapping
        self.hbm_page_to_dram_page_mapping.copy_(
            torch.arange(
                num_dram_pages - num_hbm_pages,
                num_dram_pages,
                dtype=torch.int64,
                device=device,
            )[None],
            non_blocking=True,
        )

    def load_(self, top_k_dram_page_index: torch.Tensor) -> CacheOutput:
        """
        Transfer the specified pages from DRAM to GPU (HBM) based on the given top-k DRAM page indices.
        This is an inline loading method, where loaded KV vectors will override the current vectors in cache.

        Args:
            top_k_dram_page_index (torch.Tensor): A tensor containing the indices of the top-k DRAM pages to transfer. Shape: (num_kv_heads, k)

        Returns:
            CacheOutput: A named tuple containing the loaded key and value caches.
                k_cache: Shape: (num_kv_heads, num_hbm_pages * page_size, head_dim)
                v_cache: Shape: (num_kv_heads, num_hbm_pages * page_size, head_dim)
        """
        self.current_step.add_(1)

        load_lru_cache_(
            dram_kv_cache=self.dram_kv_cache,
            page_access_time=self.page_access_time,
            dram_page_to_hbm_page_mapping=self.dram_page_to_hbm_page_mapping,
            hbm_page_to_dram_page_mapping=self.hbm_page_to_dram_page_mapping,
            topk_dram_page_index=top_k_dram_page_index,
            current_step=self.current_step,
            hbm_kv_cache=self.hbm_kv_cache,
        )

        _cache = self.hbm_kv_cache.view(
            self.num_kv_heads, self.num_hbm_pages * self.page_size, 2, self.head_dim
        )
        k_cache, v_cache = _cache.unbind(dim=2)
        return CacheOutput(k_cache=k_cache, v_cache=v_cache)

    def update_page_access_time_(self, scores: torch.Tensor):
        """
        Update the access time of pages based on scores.

        Args:
            scores (torch.Tensor): Scores to add to page access times.
        """
        self.page_access_time.add_(scores)

    def get_kv_cache_(self, q: torch.Tensor) -> CacheOutput:
        """
        Query the KVCache using query vectors and return the corresponding key-value cache.
        This method performs inline loading, where loaded KV vectors will override the current vectors in cache.

        This method performs the following steps:
        1. Uses the top-k selector to find the most relevant DRAM page indices.
        2. Loads the selected pages from DRAM to HBM (if not already present), overriding existing cache entries.
        3. Returns the cached HBM key-value cache.

        Args:
            q (torch.Tensor): Query vectors. Shape: (seq_len, num_q_heads, head_dim)

        Returns:
            CacheOutput: A named tuple containing the loaded key and value caches.
                k_cache: Shape: (num_kv_heads, num_hbm_pages * page_size, head_dim)
                v_cache: Shape: (num_kv_heads, num_hbm_pages * page_size, head_dim)
        """
        top_k_indices = self.top_k_selector.top_k(q=q, k=self.top_k)
        N, H, R, K = top_k_indices.shape
        assert H % self.num_kv_heads == 0
        top_k_indices = top_k_indices.swapaxes(0, 1).contiguous()
        top_k_indices = top_k_indices.view(self.num_kv_heads, -1)
        return self.load_(top_k_indices)
