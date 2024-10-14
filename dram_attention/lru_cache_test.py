import pytest
import torch

from dram_attention.lru_cache import LRUCache


@pytest.fixture
def sample_cache():
    k_cache = torch.randn(
        4, 1024 * 8, 64, device="cuda"
    )  # (num_kv_heads, length, head_dim)
    v_cache = torch.randn(4, 1024 * 8, 64, device="cuda")
    return LRUCache(k_cache, v_cache, page_size=16, top_k=10)


def test_lru_cache_initialization(sample_cache):
    assert sample_cache.num_kv_heads == 4
    assert sample_cache.num_dram_pages == 512  # 1024*8 / 16
    assert sample_cache.num_hbm_pages == 102  # Default 20% of DRAM pages
    assert sample_cache.head_dim == 64
    assert sample_cache.page_size == 16
    assert sample_cache.top_k == 10


def test_lru_cache_dram_to_hbm_initial_load(sample_cache):
    assert torch.all(
        sample_cache.dram_page_to_hbm_page_mapping[:, -102:] == torch.arange(102).cuda()
    )
    assert torch.all(sample_cache.dram_page_to_hbm_page_mapping[:, :-102] == -1)


def test_lru_cache_load_inline_method(sample_cache):
    top_k_dram_page_index = torch.randint(0, 512, (4, 10), device="cuda")
    k_cache, v_cache = sample_cache.load_(top_k_dram_page_index)

    assert k_cache.shape == (4, 102 * 16, 64)
    assert v_cache.shape == (4, 102 * 16, 64)
    assert sample_cache.current_step == 1


@pytest.mark.parametrize("page_size", [8, 16, 32])
def test_lru_cache_different_page_sizes(page_size):
    k_cache = torch.randn(4, 1024 * 8, 64)
    v_cache = torch.randn(4, 1024 * 8, 64)
    cache = LRUCache(k_cache, v_cache, page_size=page_size, top_k=10)

    assert cache.page_size == page_size
    assert cache.num_dram_pages == 1024 * 8 // page_size
    assert cache.top_k == 10


def test_lru_cache_invalid_initialization():
    k_cache = torch.randn(4, 1007 * 8, 64)  # Not divisible by page size 64
    v_cache = torch.randn(4, 1007 * 8, 64)

    with pytest.raises(AssertionError):
        LRUCache(k_cache, v_cache, page_size=16, top_k=10)


def test_lru_cache_hbm_size():
    k_cache = torch.randn(4, 1024 * 8, 64)
    v_cache = torch.randn(4, 1024 * 8, 64)
    cache = LRUCache(k_cache, v_cache, page_size=16, lru_hbm_cache_size=1024, top_k=10)

    assert cache.num_hbm_pages == 64  # 1024 / 16
    assert cache.top_k == 10


def test_lru_cache_get_kv_cache_inline(sample_cache):
    # Create a sample query tensor
    # (bsz, seq_len, num_kv_heads, head_dim)
    query = torch.randn(1, 10, 4, 64, device="cuda")

    # Call the get_kv_cache method
    cache_output = sample_cache.get_kv_cache_(query)

    # Check the shapes of k_cache and v_cache
    assert cache_output.k_cache.shape == (
        4,
        102 * 16,
        64,
    )  # (num_kv_heads, num_hbm_pages * page_size, head_dim)
    assert cache_output.v_cache.shape == (4, 102 * 16, 64)

    # Check if the current_step has been incremented
    assert sample_cache.current_step == 1

    # Call get_kv_cache again to check if current_step increments
    sample_cache.get_kv_cache_(query)
    assert sample_cache.current_step == 2

    # Check if the returned caches are on the correct device
    assert cache_output.k_cache.device.type == "cuda"
    assert cache_output.v_cache.device.type == "cuda"

    # Verify that the returned caches are not empty
    assert torch.any(cache_output.k_cache != 0)
    assert torch.any(cache_output.v_cache != 0)
