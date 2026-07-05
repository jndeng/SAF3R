import time

import torch
import torch.nn.functional as F


def getattr_nested(obj, attr_path):
    """
    Get a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    return obj


def setattr_nested(obj, attr_path, value):
    """
    Set a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def replace_modules(model, replace_dict):
    """
    Replace modules in a model in-place based on a replacement dictionary.

    Args:
        model (nn.Module): The original model.
        replace_dict (Dict): A dictionary mapping module names to replacement block classes.
    """
    for module_name, replace_args in replace_dict.items():
        try:
            orig_module = getattr_nested(model, module_name)
        except:
            raise ValueError(f"Failed to get module {module_name} from the model.")
        new_module = replace_args.module(orig_module, **replace_args)
        setattr_nested(model, module_name, new_module)


def compute_qk_topk_indices_batched(
    q: torch.Tensor,                    # (B, Nq, D)
    k: torch.Tensor,                    # (B, Nk, D)
    topk: int,
    per_frame: bool = False,
    num_toks_per_img: int = None,
    q_blocksize: int = 4096,
    max_topk: int = 2048,
    include_first_frame: bool = False,  # keep a (subsampled) set of first-frame tokens
    first_frame_stride: int = 4,        # NEW: stride for uniform sampling of first-frame tokens
) -> torch.Tensor:
    """
    Batched blockwise streaming compute scores = q @ k^T, return top-k key indices per query.

    Args:
        q, k, topk, per_frame, num_toks_per_img, q_blocksize, max_topk: as before.
        include_first_frame: if True, a uniformly subsampled set of the first frame's tokens
                             (indices arange(0, num_toks_per_img, first_frame_stride)) is
                             guaranteed in every query's index set and is NOT subject to the
                             max_topk cap. The whole first frame is excluded from the top-k
                             over the remaining frames (no duplicates). The remaining budget
                             (max_topk - F_sel) is filled by top-k from frames >= 1.
        first_frame_stride: stride for sampling first-frame tokens (1 = all tokens).

    Returns:
        topk_idx: (B, Nq, out_width) int32 indices into key dimension [0, Nk).
                  If include_first_frame, the first F_sel columns are the sampled first-frame
                  indices, the rest are top-k from frames >= 1. (order not guaranteed otherwise)
    """
    assert q.dim() == 3 and k.dim() == 3, "q and k must be 3D: (B, N, D)"
    assert q.shape[0] == k.shape[0], "q and k must have same batch size B"
    assert q.shape[2] == k.shape[2], "q and k must have same D"
    assert topk > 0, "topk must be > 0"
    assert first_frame_stride >= 1, "first_frame_stride must be >= 1"

    B, Nq, D = q.shape
    _, Nk, _ = k.shape
    device = q.device
    NEG = torch.finfo(torch.bfloat16).min

    # ---- first-frame bookkeeping ----
    if include_first_frame:
        assert num_toks_per_img is not None and num_toks_per_img > 0, \
            "num_toks_per_img is required when include_first_frame=True"
        assert Nk % num_toks_per_img == 0, "Nk must be divisible by num_toks_per_img"
        F = num_toks_per_img                                   # full first-frame size (for masking)
        first_idx = torch.arange(0, F, first_frame_stride,
                                 device=device, dtype=torch.int32)  # uniform samples
        F_sel = first_idx.numel()                              # guaranteed token count
        assert F_sel <= max_topk, f"sampled first-frame tokens ({F_sel}) exceed max_topk ({max_topk})"
    else:
        F = 0
        F_sel = 0

    # ---- how many tokens to draw from the *rest* (non-first-frame) ----
    if per_frame:
        assert num_toks_per_img is not None and num_toks_per_img > 0
        assert Nk % num_toks_per_img == 0, "Nk must be divisible by num_toks_per_img when per_frame=True"
        num_imgs = Nk // num_toks_per_img
        topk = min(topk, num_toks_per_img)
        num_topk_imgs = num_imgs - 1 if include_first_frame else num_imgs  # exclude frame 0 if guaranteed
        rest_eff = topk * num_topk_imgs
    else:
        rest_eff = topk
    rest_eff = min(rest_eff, Nk - F)

    rest_cap = max_topk - F_sel                                # first frame survives within max_topk
    rest_final = min(rest_eff, rest_cap)
    out_width = F_sel + rest_final

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    k_t = k.transpose(1, 2).contiguous()  # (B, D, Nk)

    if per_frame:
        if include_first_frame:
            base = (torch.arange(1, num_imgs, device=device, dtype=torch.int32)
                    .view(1, 1, -1, 1) * num_toks_per_img)     # (1,1,num_imgs-1,1)
        else:
            base = (torch.arange(num_imgs, device=device, dtype=torch.int32)
                    .view(1, 1, -1, 1) * num_toks_per_img)     # (1,1,num_imgs,1)

    topk_idx_out = torch.empty((B, Nq, out_width), device=device, dtype=torch.int32)
    warned = False
    for qs in range(0, Nq, q_blocksize):
        qe = min(qs + q_blocksize, Nq)
        Bq = qe - qs
        q_blk = q[:, qs:qe, :]                 # (B, Bq, D)
        scores = torch.bmm(q_blk, k_t)         # (B, Bq, Nk)

        # ---------- select from the "rest" ----------
        if per_frame:
            scores_f = scores.view(B, Bq, num_imgs, num_toks_per_img)
            if include_first_frame:
                scores_f = scores_f[:, :, 1:, :]               # drop frame 0 entirely
            val, idx = torch.topk(scores_f, k=topk, dim=-1, largest=True, sorted=False)
            idx = idx + base
            n_rest = topk * (num_imgs - 1 if include_first_frame else num_imgs)
            idx = idx.reshape(B, Bq, n_rest)
            val = val.reshape(B, Bq, n_rest)
        else:
            if include_first_frame:
                scores[..., :F] = NEG                          # exclude whole first frame from rest
            val, idx = torch.topk(scores, k=rest_eff, dim=-1, largest=True, sorted=False)

        # ---------- cap the rest (sampled first frame is never capped) ----------
        if val.shape[-1] > rest_final:
            if not warned:
                print(f"[Warning] rest tokens={val.shape[-1]} capped at {rest_final} "
                      f"(first_frame_sampled={F_sel}, stride={first_frame_stride}, total={out_width}).")
                warned = True
            _, pos = torch.topk(val, rest_final, dim=-1, largest=True, sorted=False)
            idx = torch.gather(idx, dim=-1, index=pos)

        # ---------- prepend guaranteed (sampled) first-frame indices ----------
        if include_first_frame:
            first_blk = first_idx.view(1, 1, F_sel).expand(B, Bq, F_sel)
            idx = torch.cat([first_blk, idx], dim=-1)          # (B, Bq, out_width)

        topk_idx_out[:, qs:qe, :] = idx

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"[Topk={out_width}] Time of computing top-k indices: {elapsed:.3f} seconds")

    return topk_idx_out
