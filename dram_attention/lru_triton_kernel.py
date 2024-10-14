import torch
import triton
import triton.language as tl


@triton.jit
def _load_lru_cache_kernel(
    dram_kv_cache_ptr,
    dram_kv_cache_head_stride,
    dram_kv_cache_page_stride,
    hbm_kv_cache_ptr,
    hbm_kv_cache_head_stride,
    hbm_kv_cache_page_stride,
    dram_page_to_hbm_page_mapping_ptr,
    dram_page_to_hbm_page_mapping_head_stride,
    topk_dram_page_index_ptr,
    topk_dram_page_index_head_stride,
    hbm_cache_length_ptr,
    PAGE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
):
    head_idx = tl.program_id(0).to(tl.int64)
    eviction_pointer = 0

    for k in tl.range(K):
        dram_page = tl.load(
            topk_dram_page_index_ptr + topk_dram_page_index_head_stride * head_idx + k
        )
        hbm_page = tl.load(
            dram_page_to_hbm_page_mapping_ptr
            + dram_page_to_hbm_page_mapping_head_stride * head_idx
            + dram_page
        )
        if hbm_page == -1:
            # Page not in HBM, get the next page to evict
            lru_hbm_page = eviction_pointer
            tl.store(
                dram_page_to_hbm_page_mapping_ptr
                + dram_page_to_hbm_page_mapping_head_stride * head_idx
                + dram_page,
                -2 - lru_hbm_page,
            )
            # Load page from DRAM to HBM
            for i in tl.range(PAGE_DIM, step=BLOCK_SIZE, num_stages=4):
                block = tl.load(
                    dram_kv_cache_ptr
                    + dram_kv_cache_head_stride * head_idx
                    + dram_kv_cache_page_stride * dram_page
                    + i
                    + tl.arange(0, BLOCK_SIZE),
                    mask=((i + tl.arange(0, BLOCK_SIZE)) < PAGE_DIM),
                )
                tl.store(
                    hbm_kv_cache_ptr
                    + hbm_kv_cache_head_stride * head_idx
                    + hbm_kv_cache_page_stride * lru_hbm_page
                    + i
                    + tl.arange(0, BLOCK_SIZE),
                    block,
                    mask=((i + tl.arange(0, BLOCK_SIZE)) < PAGE_DIM),
                )
            eviction_pointer = eviction_pointer + 1
    tl.store(hbm_cache_length_ptr + head_idx, eviction_pointer)


@triton.jit
def _load_lru_cache_inline_kernel(
    dram_kv_cache_ptr,
    dram_kv_cache_head_stride,
    dram_kv_cache_page_stride,
    hbm_kv_cache_ptr,
    hbm_kv_cache_head_stride,
    hbm_kv_cache_page_stride,
    page_access_time_ptr,
    page_access_time_head_stride,
    dram_page_to_hbm_page_mapping_ptr,
    dram_page_to_hbm_page_mapping_head_stride,
    hbm_page_to_dram_page_mapping_ptr,
    hbm_page_to_dram_page_mapping_head_stride,
    topk_dram_page_index_ptr,
    topk_dram_page_index_head_stride,
    sorted_page_indices_ptr,
    sorted_page_indices_head_stride,
    current_step_ptr,
    PAGE_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    NUM_HBM_PAGES: tl.constexpr,
):
    head_idx = tl.program_id(0).to(tl.int64)
    eviction_pointer = 0
    current_step = tl.load(current_step_ptr)

    for k in tl.range(K):
        dram_page = tl.load(
            topk_dram_page_index_ptr + topk_dram_page_index_head_stride * head_idx + k
        )
        hbm_page = tl.load(
            dram_page_to_hbm_page_mapping_ptr
            + dram_page_to_hbm_page_mapping_head_stride * head_idx
            + dram_page
        )
        if hbm_page < 0:
            # Page not in HBM, get the next page to evict
            lru_hbm_page = tl.load(
                sorted_page_indices_ptr
                + sorted_page_indices_head_stride * head_idx
                + eviction_pointer
            )
            tl.store(
                page_access_time_ptr
                + page_access_time_head_stride * head_idx
                + lru_hbm_page,
                current_step,
            )
            lru_dram_page = tl.load(
                hbm_page_to_dram_page_mapping_ptr
                + hbm_page_to_dram_page_mapping_head_stride * head_idx
                + lru_hbm_page
            )
            tl.store(
                dram_page_to_hbm_page_mapping_ptr
                + dram_page_to_hbm_page_mapping_head_stride * head_idx
                + lru_dram_page,
                -1,
            )
            tl.store(
                hbm_page_to_dram_page_mapping_ptr
                + hbm_page_to_dram_page_mapping_head_stride * head_idx
                + lru_hbm_page,
                dram_page,
            )
            tl.store(
                dram_page_to_hbm_page_mapping_ptr
                + dram_page_to_hbm_page_mapping_head_stride * head_idx
                + dram_page,
                lru_hbm_page,
            )

            # Load page from DRAM to HBM
            for i in tl.range(PAGE_DIM, step=BLOCK_SIZE, num_stages=4):
                block = tl.load(
                    dram_kv_cache_ptr
                    + dram_kv_cache_head_stride * head_idx
                    + dram_kv_cache_page_stride * dram_page
                    + i
                    + tl.arange(0, BLOCK_SIZE),
                    mask=((i + tl.arange(0, BLOCK_SIZE)) < PAGE_DIM),
                )
                tl.store(
                    hbm_kv_cache_ptr
                    + hbm_kv_cache_head_stride * head_idx
                    + hbm_kv_cache_page_stride * lru_hbm_page
                    + i
                    + tl.arange(0, BLOCK_SIZE),
                    block,
                    mask=((i + tl.arange(0, BLOCK_SIZE)) < PAGE_DIM),
                )
            # Update eviction pointer
            eviction_pointer = (eviction_pointer + 1) % NUM_HBM_PAGES
        else:
            tl.store(
                page_access_time_ptr
                + page_access_time_head_stride * head_idx
                + hbm_page,
                current_step,
            )


@triton.jit
def _update_page_access_time_(
    page_access_time_ptr,
    page_access_time_head_stride,
    dram_page_to_hbm_page_mapping_ptr,
    dram_page_to_hbm_page_mapping_head_stride,
    topk_dram_page_index_ptr,
    topk_dram_page_index_head_stride,
    current_step_ptr,
    K: tl.constexpr,
):
    head_idx = tl.program_id(0).to(tl.int64)
    current_step = tl.load(current_step_ptr)
    for k in tl.range(K):
        # Load the DRAM page index
        dram_page = tl.load(
            topk_dram_page_index_ptr + topk_dram_page_index_head_stride * head_idx + k
        )
        hbm_page = tl.load(
            dram_page_to_hbm_page_mapping_ptr
            + dram_page_to_hbm_page_mapping_head_stride * head_idx
            + dram_page
        )
        if hbm_page >= 0:
            # Update the access time for this page to the current step
            tl.store(
                page_access_time_ptr
                + page_access_time_head_stride * head_idx
                + hbm_page,
                current_step,
            )


def load_lru_cache_(
    dram_kv_cache: torch.Tensor,
    page_access_time: torch.Tensor,
    dram_page_to_hbm_page_mapping: torch.Tensor,
    hbm_page_to_dram_page_mapping: torch.Tensor,
    topk_dram_page_index: torch.Tensor,
    *,
    current_step: torch.Tensor,
    hbm_kv_cache: torch.Tensor,
):
    num_heads, num_dram_pages, page_dim = dram_kv_cache.shape
    del num_dram_pages
    _, k = topk_dram_page_index.shape
    assert (
        dram_kv_cache.shape[0] == hbm_kv_cache.shape[0]
        and dram_kv_cache.shape[2] == hbm_kv_cache.shape[2]
    )

    # Update page access times
    _update_page_access_time_[(num_heads,)](
        page_access_time,
        page_access_time.stride(0),
        dram_page_to_hbm_page_mapping,
        dram_page_to_hbm_page_mapping.stride(0),
        topk_dram_page_index,
        topk_dram_page_index.stride(0),
        current_step_ptr=current_step,
        K=k,
    )

    # Sort page access times and get indices
    sorted_page_indices = torch.argsort(page_access_time, dim=-1, stable=True)

    _load_lru_cache_inline_kernel[(num_heads,)](
        dram_kv_cache,
        dram_kv_cache.stride(0),
        dram_kv_cache.stride(1),
        hbm_kv_cache,
        hbm_kv_cache.stride(0),
        hbm_kv_cache.stride(1),
        page_access_time,
        page_access_time.stride(0),
        dram_page_to_hbm_page_mapping,
        dram_page_to_hbm_page_mapping.stride(0),
        hbm_page_to_dram_page_mapping,
        hbm_page_to_dram_page_mapping.stride(0),
        topk_dram_page_index,
        topk_dram_page_index.stride(0),
        sorted_page_indices,
        sorted_page_indices.stride(0),
        current_step_ptr=current_step,
        PAGE_DIM=page_dim,
        BLOCK_SIZE=min(1024, page_dim),
        K=k,
        NUM_HBM_PAGES=hbm_kv_cache.shape[1],
    )
