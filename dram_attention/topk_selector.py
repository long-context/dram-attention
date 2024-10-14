"""
Implement QUEST top-k selection heuristic for efficient attention computation.
"""

import torch

from .max_ip_triton_kernel import max_inner_product


class IPUpperboundTopKSelector:
    """
    Inner Product Upperbound TopK selector.
    Selects top k attention pages (clusters) based on the maximum inner product.
    """

    def __init__(self, k_cache: torch.Tensor, num_pages: int, device="cuda"):
        """
        Initialize the IPUpperboundTopKSelector.

        Args:
            k_cache: Key vectors cache. Shape: (num_heads, length, head_dim)
            num_pages: Number of pages to divide the cache into.
        """
        super().__init__()
        num_heads, dram_kv_cache_size, head_dim = k_cache.shape
        self.num_kv_heads = num_heads
        assert (
            dram_kv_cache_size % num_pages == 0
        ), "Cache size must be divisible by num_pages."
        page_size = dram_kv_cache_size // num_pages
        chunked_k_cache = k_cache.view(num_heads, num_pages, page_size, head_dim)
        self.chunk_min_max = torch.stack(
            torch.aminmax(chunked_k_cache, dim=2, keepdim=False), dim=-2
        ).to(device=device)
        # (num_heads, num_pages, 2, head_dim)

    def top_k(self, q: torch.Tensor, k: int) -> torch.Tensor:
        """
        Filter and return the top k page indices for each query and head.

        Args:
            q: Query vectors. Shape: (bsz, seq_len, num_q_heads, head_dim)
            k: Number of top pages to select.

        Returns:
            Indices of top k pages for each query and head. Shape: (bsz, seq_len, num_kv_heads, rep, k)
        """
        bsz, seq_len, num_q_heads, head_dim = q.shape
        assert bsz == 1, f"Only batch size 1 is supported, but got batch size {bsz}"
        assert num_q_heads % self.num_kv_heads == 0
        rep = num_q_heads // self.num_kv_heads
        q = q.view(seq_len, self.num_kv_heads, rep, head_dim)
        # seq_len, rep, self.num_kv_heads, head_dim
        q = q.swapaxes(1, 2)
        q = q.view(seq_len * rep, self.num_kv_heads, head_dim)
        ip = max_inner_product(q=q, min_max=self.chunk_min_max)
        ip = ip.view(seq_len, rep, self.num_kv_heads, -1)
        # seq_len, self.num_kv_heads, rep, -1)
        ip = ip.swapaxes(1, 2)
        # Get the indices of the top k values for each query and head
        return torch.argsort(ip, dim=-1, descending=True)[..., :k]
