import time

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


# =====================================================================
#  Implementation for smaller TopK (≤2048) without online softmax
# =====================================================================
def get_basse_configs():
    return [
        triton.Config({'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 64}, num_warps=8),
    ]


def get_autotune_configs():
    configs = []
    for block_d in [32, 64, 128]:
        for warps in [2, 4, 8]:
            for stages in [2, 3, 4, 5]:
                # filter
                if block_d == 32 and warps == 8:
                    continue
                configs.append(
                    triton.Config({"BLOCK_D": block_d}, num_warps=warps, num_stages=stages)
                )
    return configs


@triton.autotune(
    # configs=get_autotune_configs(),
    configs=get_basse_configs(),
    key=["D", "TopK"],
)
@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Idx_ptr, Out_ptr,
    sm_scale, N, TopK, D,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_in, stride_ik,
    stride_on, stride_od,
    BLOCK_K: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    # -----------------------------------------------------------
    # Step 1: Load Top-K indices
    # -----------------------------------------------------------
    idx_ptrs = Idx_ptr + pid * stride_in + tl.arange(0, BLOCK_K) * stride_ik
    mask_k = tl.arange(0, BLOCK_K) < TopK
    idx = tl.load(idx_ptrs, mask=mask_k, other=0)
    # NOTE: always use float32 for internal accumulation to prevent bf16/fp16 overflow
    scores = tl.zeros([BLOCK_K], dtype=tl.float32)

    # -----------------------------------------------------------
    # Step 2: Compute Q @ K^T 
    # -----------------------------------------------------------
    for d_offset in range(0, D, BLOCK_D):
        d_idx = d_offset + tl.arange(0, BLOCK_D)
        mask_d = d_idx < D
        
        q_ptrs = Q_ptr + pid * stride_qn + d_idx * stride_qd
        # NOTE: safely upcast input (fp16/bf16/fp32) to fp32 on the fly
        q_chunk = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        
        k_ptrs = K_ptr + idx[:, None] * stride_kn + d_idx[None, :] * stride_kd
        mask_kd = mask_k[:, None] & mask_d[None, :]
        k_chunk = tl.load(k_ptrs, mask=mask_kd, other=0.0).to(tl.float32)

        scores += tl.sum(q_chunk[None, :] * k_chunk, axis=1)

    # -----------------------------------------------------------
    # Step 3: Softmax (Computed in fp32)
    # -----------------------------------------------------------
    scores = scores * sm_scale
    scores = tl.where(mask_k, scores, float("-inf"))
    
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    p = tl.where(mask_k, p, 0.0)
    denom = tl.sum(p, axis=0)
    
    attn_weights = p / denom

    # -----------------------------------------------------------
    # Step 4: Compute Attention @ V 
    # -----------------------------------------------------------
    for d_offset in range(0, D, BLOCK_D):
        d_idx = d_offset + tl.arange(0, BLOCK_D)
        mask_d = d_idx < D

        v_ptrs = V_ptr + idx[:, None] * stride_vn + d_idx[None, :] * stride_vd
        mask_vd = mask_k[:, None] & mask_d[None, :]
        # NOTE: upcast V to fp32 before multiplication
        v_chunk = tl.load(v_ptrs, mask=mask_vd, other=0.0).to(tl.float32)

        out_chunk = tl.sum(attn_weights[:, None] * v_chunk, axis=0)

        # Cast the fp32 result directly back to the original pointer's dtype (e.g., bf16)
        o_ptrs = Out_ptr + pid * stride_on + d_idx * stride_od
        tl.store(o_ptrs, out_chunk.to(Out_ptr.dtype.element_ty), mask=mask_d)


def fused_sparse_topk_attention(q, k, v, kv_indices, scale=None):
    """
    Compute sparse attention where each query attends to a sparse subset of keys/values
    defined by pre-computed top-k indices.

    Args:
        q: [N, D]
        k: [N, D]
        v: [N, D]
        kv_indices: [N, TopK]
        scale: optional scaling factor for attention scores (default: 1/sqrt(D))
    Returns:
        out: [N, Dv]
    """
    N, D = q.shape
    TopK = kv_indices.shape[1]

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(v)

    # find the largest BLOCK_K that is a power of 2 and <= TopK for better performance
    BLOCK_K = triton.next_power_of_2(TopK)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    kv_indices = kv_indices.contiguous()

    grid = (N, )
    _sparse_attn_fwd_kernel[grid](
        q, k, v, kv_indices, out,
        scale, N, TopK, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        kv_indices.stride(0), kv_indices.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_K=BLOCK_K
        # autotune BLOCK_D
    )

    return out



# =====================================================================
#  Implementation for larger TopK (>2048) with online softmax
# =====================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 256}, num_warps=4),
    ],
    key=["D", "TopK"],
)
@triton.jit
def _sparse_flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Idx_ptr, Out_ptr,
    sm_scale, N, TopK, D,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_in, stride_ik,
    stride_on, stride_od,
    BLOCK_K: tl.constexpr, 
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    # 1. Load Q and reshape to a 2D matrix [1, BLOCK_D]
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q_ptrs = Q_ptr + pid * stride_qn + offs_d[None, :] * stride_qd
    
    # Cast to float16 to feed the Tensor Cores
    q = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float16)

    # 2. Initialize the online-softmax state variables (must be float32 to prevent overflow)
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([1, BLOCK_D], dtype=tl.float32)

    # 3. Loop over TopK
    for k_offset in range(0, TopK, BLOCK_K):
        offs_k = k_offset + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TopK

        # Load indices
        idx_ptrs = Idx_ptr + pid * stride_in + offs_k * stride_ik
        idx = tl.load(idx_ptrs, mask=mask_k, other=0).to(tl.int64)
        
        # If using padding index -1, uncomment the line below
        # mask_k = mask_k & (idx >= 0)

        # Load K block: [BLOCK_K, BLOCK_D] -> cast to float16
        k_ptrs = K_ptr + idx[:, None] * stride_kn + offs_d[None, :] * stride_kd
        mask_kd = mask_k[:, None] & mask_d[None, :]
        k_chunk = tl.load(k_ptrs, mask=mask_kd, other=0.0).to(tl.float16)

        # q [1, BLOCK_D] @ k_chunk.T [BLOCK_D, BLOCK_K] -> qk [1, BLOCK_K]
        qk = tl.dot(q, tl.trans(k_chunk)) * sm_scale
        qk = tl.where(mask_k[None, :], qk, float("-inf"))

        # --- Online softmax computation ---
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Load V block: [BLOCK_K, BLOCK_D] -> cast to float16
        v_ptrs = V_ptr + idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=mask_kd, other=0.0).to(tl.float16)

        # Trigger the Tensor Cores again: Attention @ V
        # p_f16 [1, BLOCK_K] @ v_chunk [BLOCK_K, BLOCK_D] -> v_out [1, BLOCK_D]
        p_f16 = p.to(tl.float16) 
        v_out = tl.dot(p_f16, v_chunk)

        # Update state
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + v_out
        m_i = m_i_new

    # 4. Final normalization and write-back
    acc = acc / l_i[:, None]
    
    o_ptrs = Out_ptr + pid * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_d[None, :])


def fused_sparse_topk_attention_2(q, k, v, kv_indices, scale=None):
    N, D = q.shape
    TopK = kv_indices.shape[1]

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(v)
    
    BLOCK_D = triton.next_power_of_2(D)

    grid = (N, )
    _sparse_flash_attn_fwd_kernel[grid](
        q, k, v, kv_indices, out,
        scale, N, TopK, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        kv_indices.stride(0), kv_indices.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D
    )

    return out
