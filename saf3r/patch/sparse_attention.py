import math

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

from ..triton.sparse_topk_attention import (
    fused_sparse_topk_attention, fused_sparse_topk_attention_2
)


def sparse_attention(
    q, k, v, est_topk_idx,
    cfg, p_h, p_w, n_spc, head_idx
):
    """
    Sparse attention interface that selects and applies one of the supported
    sparse-attention patterns to a given head.

    Args:
        q: (1, 1, N, D)
        k: (1, 1, N, D)
        v: (1, 1, N, Dv)
        est_topk_idx: (num_img_tokens, topk) estimated top-k indices for each image token
        cfg: dict containing the sparse attention configuration
        p_h: number of tokens along height
        p_w: number of tokens along width
        n_spc: number of special tokens
        head_idx: index of the current head
    Returns:
        out: (1, 1, N, Dv)
    """
    num_toks = q.shape[2]
    num_toks_per_img = p_h * p_w + n_spc
    num_imgs = num_toks // num_toks_per_img
    spec_idx, img_idx = get_separate_indices(num_imgs, num_toks_per_img, n_spc)

    # select form one of the following modes
    mode = cfg["mode"]
    if mode == "skip":
        return torch.zeros_like(v)
    
    elif mode == "broadcast_first":
        out = broadcast_anchorframe_attention(q, k, v, num_toks_per_img)
    
    elif mode == "all_to_first":
        out = all_to_first(q, k, v, num_toks_per_img)
    
    elif mode == "global_to_frame":
        out = global_to_frame_attention(q, k, v, num_toks_per_img)

    elif mode == "q_probe_topk":
        # prioritize stride (i.e., topk = N / stride)
        if "stride" in cfg:
            topk = max(1, num_toks // cfg["stride"])
        else:
            topk = min(cfg.get("topk", 1024), num_toks)
        q_sample_ratio = cfg.get("q_sample_ratio", 1)

        include_as = cfg.get("include_as", False)
        if include_as:
            anchor_idx = torch.arange(num_toks_per_img)
            indices = torch.cat([spec_idx, anchor_idx])
        else:
            indices = None

        out = q_probe_topk_attention(q, k, v, topk, q_sample_ratio, indices=indices)
    
    elif mode == "dino_topk":
        dino_sp2sp_stride = cfg.get("dino_sp2sp_stride", 1)
        out = dino_topk_attention(q, k, v, spec_idx, img_idx, est_topk_idx, dino_sp2sp_stride)

    elif mode == "stride":
        stride = cfg.get("stride", 4)
        shift = cfg.get("shift", False)
        start_pos = head_idx % stride if shift else 0

        include_as = cfg.get("include_as", True)
        if include_as:
            anchor_idx = torch.arange(num_toks_per_img)
            indices = torch.cat([spec_idx, anchor_idx])
        else:
            indices = None

        out = stride_attention(q, k, v, stride, start_pos, indices=indices)

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return out


def get_separate_indices(num_imgs, num_toks_per_img, num_special_tokens):
    """
    Returns two 1D tensors of indices for special tokens and image tokens, respectively.
    """
    base = torch.arange(num_imgs) * num_toks_per_img                # [num_imgs]

    # special: base + [0..num_special_tokens-1]
    spec_local = torch.arange(num_special_tokens)                   # [S]
    spec_idx = (base[:, None] + spec_local[None, :]).reshape(-1)    # [num_imgs*S]

    # image: base + [S..num_toks_per_img-1]
    img_local = torch.arange(num_special_tokens, num_toks_per_img)  # [T-S]
    img_idx = (base[:, None] + img_local[None, :]).reshape(-1)      # [num_imgs*(T-S)]

    return spec_idx, img_idx


def broadcast_anchorframe_attention(q, k, v, num_toks_per_img):
    """
    Copy the attention map of the first frame to all frames.
    """
    # q0, k0: (B, H, N, D)
    q0 = q[:, :, :num_toks_per_img, :]
    k0 = k[:, :, :num_toks_per_img, :]
    v = v.reshape(v.shape[0], v.shape[1], -1, num_toks_per_img, v.shape[-1])  # (B, H, S, N, D)

    # attn: (B, H, N, N)
    attn = torch.matmul(q0, k0.transpose(-1, -2)) / (q0.size(-1) ** 0.5)
    attn = torch.softmax(attn, dim=-1)

    # attn: (B, H, 1, N, N)
    # v:    (B, H, S, N, D)
    attn = attn.unsqueeze(2) 

    # output: (B, H, S, N, D)
    output = torch.matmul(attn, v)

    # (B, H, SN, D)
    return output.flatten(2, 3)


def all_to_first(q, k, v, num_toks_per_img):
    """
    Each query attends to all tokens in the first image.
    """
    k = k[:, :, :num_toks_per_img, :]
    v = v[:, :, :num_toks_per_img, :]
    return F.scaled_dot_product_attention(q, k, v)


def q_probe_topk_attention(q, k, v, topk, q_sample_ratio=1, indices=None):
    """
    Using a small subset of queries to probe the keys and select top-k keys for all queries.
    """
    assert q.shape[0] == 1 and q.shape[1] == 1
    assert k.shape[0] == 1 and k.shape[1] == 1
    assert v.shape[0] == 1 and v.shape[1] == 1
    _, _, N, D = q.shape

    # 0) Sample queries
    q_sel = q
    if q_sample_ratio < 1:
        num_sampled_q = max(1, int(q.size(2) * q_sample_ratio))
        q_idx = torch.linspace(0, q.size(2)-1, steps=num_sampled_q).long()
        q_sel = q[:, :, q_idx, :]             # (1, 1, num_sampled_q, D)

    # 1) Compute q_bar using ALL queries
    q_bar = q_sel.mean(dim=2, keepdim=True)   # (1, 1, 1, D)

    # 2) Compute scores for all keys
    # use float32 for stability
    scores = torch.matmul(
        q_bar.to(torch.float32), k.transpose(-1, -2).to(torch.float32)
    ) / math.sqrt(D)                          # (1, 1, 1, N)
    scores = scores.squeeze()                 # (N,)

    # 3) Select topk keys
    topk_idx = torch.topk(scores, k=topk, dim=-1, sorted=False).indices  # (topk,)

    # 3.5) If there are additional indices to attend to, combine them with the topk_idx
    if indices is not None:
        indices = indices.to(device=topk_idx.device, dtype=torch.long)
        combined_idx = torch.cat([topk_idx, indices])
        topk_idx = torch.unique(combined_idx)
    else:
        topk_idx = torch.sort(topk_idx).values

    # 4) Direct advanced indexing (NO gather)
    k_topk = k[:, :, topk_idx, :]             # (1, 1, topk, D)
    v_topk = v[:, :, topk_idx, :]             # (1, 1, topk, Dv)

    # 5) Final attention over reduced KV
    out = F.scaled_dot_product_attention(q, k_topk, v_topk)
    return out


def dino_topk_attention(q, k, v, spec_tok_idx, img_tok_idx, est_topk_idx, dino_sp2sp_stride=1):
    """
    Separate special and image tokens: special tokens only attend to each other / all tokens,
    while image tokens attend to a subset of keys/values.
    Selected keys/values of each query are given by est_topk_idx.
    """
    assert q.shape[0] == 1 and q.shape[1] == 1

    # 1) Process special tokens (attend to all tokens)
    q_spec = q[:, :, spec_tok_idx, :]  # [1, 1, num_special_tokens*num_imgs, D]
    if dino_sp2sp_stride == 1:
        k_spec = k  # attend to all keys
        v_spec = v  # attend to all values
    else:
        k_spec = k[:, :, ::dino_sp2sp_stride, :]
        v_spec = v[:, :, ::dino_sp2sp_stride, :]

    # NOTE: force to use flash or mem-eff SDPA
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out_spec = F.scaled_dot_product_attention(q_spec, k_spec, v_spec)  # [1, 1, num_special_tokens*num_imgs, Dv]

    # 2) Process image tokens (attend only to estimated top-k tokens)
    D = q.shape[-1]
    Dv = v.shape[-1]
    num_img_tokens = img_tok_idx.shape[0]
    q_img = q[:, :, img_tok_idx, :].reshape(-1, D)   # [num_img_tokens, D]
    k_img = k[:, :, img_tok_idx, :].reshape(-1, D)   # [num_img_tokens, D]
    v_img = v[:, :, img_tok_idx, :].reshape(-1, Dv)  # [num_img_tokens, Dv]

    # NOTE: `fused_sparse_topk_attention` fails when topk is large (e.g., 2048)
    topk = est_topk_idx.shape[-1]
    if topk <= 2048:
        out_img_flat = fused_sparse_topk_attention(q_img, k_img, v_img, est_topk_idx)
    else:
        out_img_flat = fused_sparse_topk_attention_2(q_img, k_img, v_img, est_topk_idx)
    out_img = out_img_flat.reshape(1, 1, num_img_tokens, Dv)

    # 3) Combine special and image token outputs
    out = torch.empty_like(v)
    out[:, :, spec_tok_idx, :] = out_spec
    out[:, :, img_tok_idx, :] = out_img

    return out


def global_to_frame_attention(q, k, v, num_toks_per_img):
    """
    Attend only to tokens within the same image.
    """
    B, H, num_toks, D = q.shape
    _, _, _, Dv = v.shape
    num_imgs = num_toks // num_toks_per_img

    q = q.view(B, H, num_imgs, num_toks_per_img, D)  # [B, 1, num_imgs, toks_per_img, head_dim]
    k = k.view(B, H, num_imgs, num_toks_per_img, D)  # [B, 1, num_imgs, toks_per_img, head_dim]
    v = v.view(B, H, num_imgs, num_toks_per_img, Dv) # [B, 1, num_imgs, toks_per_img, head_dim]
    
    q = q.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, D)  # [B*num_imgs, 1, toks_per_img, head_dim]
    k = k.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, D)  # [B*num_imgs, 1, toks_per_img, head_dim]
    v = v.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, Dv) # [B*num_imgs, 1, toks_per_img, head_dim]

    out = F.scaled_dot_product_attention(q, k, v)
    out = out.view(B, num_imgs, H, num_toks_per_img, Dv).permute(0, 2, 1, 3, 4).view(B, H, num_toks, Dv)  # [B, 1, N, head_dim]

    return out


def stride_attention(q, k, v, stride, start_pos=0, indices=None):
    """
    Each query attends to strided keys/values, optionally combined with additional indices. 
    """
    if indices is None:
        k_strided = k[:, :, start_pos::stride, :]
        v_strided = v[:, :, start_pos::stride, :]
        return F.scaled_dot_product_attention(q, k_strided, v_strided)
    
    _, _, N, _ = k.shape
    strided_idx = torch.arange(start_pos, N, step=stride, device=k.device, dtype=torch.long)
    indices = indices.to(device=k.device, dtype=torch.long)
    combined_idx = torch.cat([strided_idx, indices])
    final_idx = torch.unique(combined_idx)

    k_strided = k[:, :, final_idx, :]
    v_strided = v[:, :, final_idx, :]
    return F.scaled_dot_product_attention(q, k_strided, v_strided)
