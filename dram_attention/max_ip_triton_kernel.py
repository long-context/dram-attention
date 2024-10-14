import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_L": 2, "BLOCK_SIZE_N": 2, "BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_L": 2, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 1}),
        triton.Config({"BLOCK_SIZE_L": 2, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_L": 4, "BLOCK_SIZE_N": 1, "BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_L": 8, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_L": 16, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_L": 32, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_L": 64, "BLOCK_SIZE_N": 4, "BLOCK_SIZE_H": 1}),
    ],
    key=["L", "N", "H"],
)
@triton.jit
def _max_inner_product_kernel(
    query_ptr,
    query_n_stride,
    query_head_stride,
    min_max_ptr,
    min_max_head_stride,
    min_max_L_stride,
    min_max_R_stride,
    output_ptr,
    output_n_stride,
    output_head_stride,
    L,
    N,
    H,
    D: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_l, pid_n, pid_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    l_start = pid_l * BLOCK_SIZE_L
    n_start = pid_n * BLOCK_SIZE_N
    h_start = pid_h * BLOCK_SIZE_H

    l_offs = l_start + tl.arange(0, BLOCK_SIZE_L)
    n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
    h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)

    # Corrected indexing to ensure proper access to query elements
    query = tl.load(
        query_ptr
        + n_offs[:, None, None] * query_n_stride
        + h_offs[None, :, None] * query_head_stride
        + tl.arange(0, D)[None, None, :],
        mask=(n_offs[:, None, None] < N) & (h_offs[None, :, None] < H),
        other=0.0,
    )

    # Precompute min and max data pointers to minimize redundant index calculations
    min_data_max_data_ptr = (
        min_max_ptr
        + h_offs[:, None, None] * min_max_head_stride
        + l_offs[None, :, None] * min_max_L_stride
    )

    minmax_data = tl.load(
        min_data_max_data_ptr + 0 * min_max_R_stride + tl.arange(0, D)[None, None, :],
        mask=(h_offs[:, None, None] < H) & (l_offs[None, :, None] < L),
        other=0.0,
    )
    # Perform the maximum inner product calculation
    m1 = query[:, :, None, :] * minmax_data[None]

    minmax_data = tl.load(
        min_data_max_data_ptr + 1 * min_max_R_stride + tl.arange(0, D)[None, None, :],
        mask=(h_offs[:, None, None] < H) & (l_offs[None, :, None] < L),
        other=0.0,
    )
    m2 = query[:, :, None, :] * minmax_data[None]
    value = tl.sum(tl.maximum(m1, m2), axis=-1)

    # Store the result in output, ensuring minimal masking and coalesced access
    tl.store(
        output_ptr
        + n_offs[:, None, None] * output_n_stride
        + h_offs[None, :, None] * output_head_stride
        + l_offs[None, None, :],
        value,
        mask=(n_offs[:, None, None] < N)
        & (h_offs[None, :, None] < H)
        & (l_offs[None, None, :] < L),
    )


def max_inner_product(q, min_max):
    N, H, D = q.shape
    H1, L, R, D1 = min_max.shape
    assert H1 == H and D1 == D and R == 2
    output = torch.empty(N, H, L, dtype=q.dtype, device=q.device)

    def grid(META):
        return (
            triton.cdiv(L, META["BLOCK_SIZE_L"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
            triton.cdiv(H, META["BLOCK_SIZE_H"]),
        )

    _max_inner_product_kernel[grid](
        q,
        q.stride(0),
        q.stride(1),
        min_max,
        min_max.stride(0),
        min_max.stride(1),
        min_max.stride(2),
        output,
        output.stride(0),
        output.stride(1),
        L=L,
        N=N,
        H=H,
        D=D,
    )
    return output


def max_ip_ref(q, min_max):
    import math

    N, H, D = q.shape
    H1, L, R, D1 = min_max.shape
    assert H1 == H and D1 == D
    q = q[:, :, None, None, :]
    min_max = min_max[None, :, :, :, :]
    return torch.amax(q * min_max, dim=-2).sum(-1)
