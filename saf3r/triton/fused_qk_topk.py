"""
Fused per-frame QK top-k: for each query, segment the keys by frame and independently
select topk indices per frame.

Compared to the naive PyTorch implementation, this implementation assumes:
    - `per_frame=True` only; the global branch has been removed.
    - No capping/truncation; assumes `max_topk >= topk * num_frames`.
    - No first-frame retention.

NOTE: This fused implementation has slower GEMM than the naive PyTorch version
(especially for large hidden dimension D and long sequence), but significantly
faster TopK selection via bitonic sorting when K is small.
It is well suited for per-frame TopK on per-head projected QK (small D and small K).
"""

import time

import torch
import triton
import triton.language as tl

from .topk_utils import bitonic_merge, fpval_to_key, topk


@triton.jit
def _fused_perframe_qk_topk_kernel(
    Q, K, Out,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_ob, stride_om, stride_ok,
    Nq, Nk_padded, D,
    segment_len_real,                 # runtime: real token count per frame N (used to mask intra-frame padding)
    segment_len: tl.constexpr,        # padded frame length (a multiple of BLOCK_N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,       # = next_pow2(D), loaded as a full-D tile in one shot
    K_VAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_b = tl.program_id(1).to(tl.int64)

    # Promote output/sequence pointer arithmetic to int64 to avoid 32-bit overflow at the
    # Nq * (num_imgs*K_VAL) scale
    offs_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    m_mask = offs_m < Nq
    offs_dm = tl.arange(0, BLOCK_DMODEL)

    # Load q once and keep it resident ([BLOCK_M, BLOCK_DMODEL], reused across the N loop)
    q = tl.load(
        Q + off_b * stride_qb + (offs_m[:, None] * stride_qm + offs_dm[None, :] * stride_qd),
        mask=m_mask[:, None] & (offs_dm[None, :] < D),
        other=0.0,
    )

    acc = tl.zeros([BLOCK_M, K_VAL], dtype=tl.uint64)

    for start_n in tl.range(0, Nk_padded, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Compute QK on-chip: single full-D dot (k tile is small, smem is ample for D<=128)
        k = tl.load(
            K + off_b * stride_kb + (offs_n[:, None] * stride_kn + offs_dm[None, :] * stride_kd),
            mask=(offs_dm[None, :] < D),  # offs_n is always < Nk_padded (aligned to BLOCK_N)
            other=0.0,
        )  # # [BLOCK_N, BLOCK_DMODEL] 
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)  # [BLOCK_M, BLOCK_N] fp32

        # Intra-frame padding mask: mask out the fake tokens in each frame's [N, padded_N) from F.pad
        offs_n_in_seg = (start_n % segment_len) + tl.arange(0, BLOCK_N)
        valid_n = offs_n_in_seg < segment_len_real       # [BLOCK_N]
        qk = tl.where(valid_n[None, :], qk, float("-inf"))

        # Composite key: high 32 bits = sortable float key, low 32 bits = reversed index
        # (deterministic tie-break, favoring the smaller index)
        val_bits = qk.to(tl.uint32, bitcast=True)
        val_keys = fpval_to_key(val_bits).to(tl.uint64)
        idx_mat = tl.broadcast_to(offs_n[None, :], qk.shape).to(tl.uint32)
        stable_idx = ((Nk_padded - 1) - idx_mat).to(tl.uint64)
        composite = (val_keys << 32) | stable_idx
        composite = tl.where(valid_n[None, :], composite, 0)  # invalid positions sink to the bottom

        # Running per-frame top-K_VAL reduction: ascending acc + descending block -> max
        # (standard bitonic merge, drops no candidates)
        acc = bitonic_merge(acc, descending=False)
        blk = topk(composite, k=K_VAL, descending=True)
        acc = tl.maximum(acc, blk)

        # Frame boundary: decode and write back, then clear acc for the next frame
        next_start_n = start_n + BLOCK_N
        do_flush = (next_start_n % segment_len == 0) or (next_start_n >= Nk_padded)
        if do_flush:
            acc_final = bitonic_merge(acc, descending=True)
            idx_out = ((Nk_padded - 1) - (acc_final & 0xFFFFFFFF)).to(tl.int32)
            idx_out = tl.where(acc_final > 0, idx_out, -1)

            seg_idx = (start_n // segment_len).to(tl.int64)
            offs_k = tl.arange(0, K_VAL).to(tl.int64)
            o_ptrs = Out + off_b * stride_ob + \
                     (offs_m[:, None] * stride_om + (seg_idx * K_VAL + offs_k[None, :]) * stride_ok)
            tl.store(o_ptrs, idx_out, mask=m_mask[:, None])

        acc = tl.where(do_flush, tl.zeros([BLOCK_M, K_VAL], dtype=tl.uint64), acc)


def _reference_perframe(q, k, topk, num_toks_per_img):
    """
    Pure-PyTorch per-frame top-k (CPU fallback / reference), returns [B, Nq, num_imgs*topk_eff] int32 global indices.
    """
    B, Nq, D = q.shape
    Nk = k.shape[1]
    ni = Nk // num_toks_per_img
    te = min(topk, num_toks_per_img)
    scores = torch.matmul(q.float(), k.float().transpose(1, 2))
    sf = scores.view(B, Nq, ni, num_toks_per_img)
    _, idx = sf.topk(te, dim=-1, largest=True, sorted=False)
    base = (torch.arange(ni, device=q.device, dtype=torch.long) * num_toks_per_img).view(1, 1, ni, 1)
    return (idx + base).reshape(B, Nq, ni * te).to(torch.int32)


def compute_qk_topk_indices_fused(
    q: torch.Tensor,                    # (B, Nq, D)
    k: torch.Tensor,                    # (B, Nk, D)
    topk: int,
    num_toks_per_img: int = None,
) -> torch.Tensor:
    """
    Fused per-frame QK top-k. Returns (B, Nq, num_imgs * topk) global key indices (int32),
    frame-major layout [frame0's topk, frame1's topk, ...]; empty slots are -1.

    NOTE: only supports D <= 128.
    NOTE: this function implements per-frame topk.
    TODO: autotune BLOCK_M/BLOCK_N/num_warps, fixed D=64, etc.

    Args:
        q: (B, Nq, D) query tensor
        k: (B, Nk, D) key tensor
        topk: number of top-k indices to select per frame
        num_toks_per_img: number of tokens per image/frame (N); must be > 0
    Returns:
        idx: (B, Nq, num_imgs * topk) int32 tensor of global key indices
    """
    assert num_toks_per_img is not None and num_toks_per_img > 0
    assert q.dim() == 3 and k.dim() == 3 and q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2]
    assert k.shape[1] % num_toks_per_img == 0, "Nk must be divisible by num_toks_per_img"

    B, Nq, D = q.shape
    Nk = k.shape[1]
    device = q.device
    if device.type != "cuda":
        return _reference_perframe(q, k, topk, num_toks_per_img)

    assert D <= 128, "this fused impl targets D <= 128 (per-head)"

    N = num_toks_per_img
    num_imgs = Nk // N
    topk_eff = min(topk, N)

    K_VAL = triton.next_power_of_2(topk_eff)
    BLOCK_N = max(64, K_VAL)  # per-block candidate pool; must be >= K_VAL
    assert K_VAL <= BLOCK_N, f"K_VAL({K_VAL}) must be <= BLOCK_N({BLOCK_N})"
    assert topk_eff <= 128, "this impl supports per-frame topk up to 128"

    BLOCK_M = 64
    BLOCK_DMODEL = triton.next_power_of_2(D)

    # Independently pad each frame to a multiple of BLOCK_N
    pad_len = (BLOCK_N - (N % BLOCK_N)) % BLOCK_N
    padded_N = N + pad_len

    ### Timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    q_bf = q.to(torch.bfloat16).contiguous()
    k_bf = k.to(torch.bfloat16).contiguous()
    if pad_len > 0:
        k_split = k_bf.view(B, num_imgs, N, D)
        k_split = torch.nn.functional.pad(k_split, (0, 0, 0, pad_len))
        k_pad = k_split.reshape(B, num_imgs * padded_N, D)
    else:
        k_pad = k_bf
    Nk_padded = num_imgs * padded_N

    out = torch.full((B, Nq, num_imgs * K_VAL), -1, device=device, dtype=torch.int32)

    grid = (triton.cdiv(Nq, BLOCK_M), B)
    _fused_perframe_qk_topk_kernel[grid](
        q_bf, k_pad, out,
        q_bf.stride(0), q_bf.stride(1), q_bf.stride(2),
        k_pad.stride(0), k_pad.stride(1), k_pad.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        Nq, Nk_padded, D,
        N,  # segment_len_real
        segment_len=padded_N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        K_VAL=K_VAL,
        num_warps=8,
        num_stages=2,
    )

    # Slice back to the real topk_eff and restore the index offset introduced by per-frame padding
    idx = out.view(B, Nq, num_imgs, K_VAL)[..., :topk_eff]
    if pad_len > 0:
        offsets = (torch.arange(num_imgs, device=device, dtype=torch.int32) * pad_len).view(1, 1, num_imgs, 1)
        idx = idx - offsets  # restore valid indices; -1 becomes smaller
        idx = idx.clamp_(min=-1)  # clamp negative values to -1

    ### Timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Topk={num_imgs * topk_eff}] Time of computing top-k indices: {elapsed:.3f} seconds")
    
    return idx.reshape(B, Nq, num_imgs * topk_eff)
