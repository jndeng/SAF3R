import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_attention import sparse_attention, get_separate_indices
from ..triton.fused_qk_topk import compute_qk_topk_indices_fused


class BaseHeadwiseSparseAttentionWrapper(nn.Module):
    """
    A base sparse-attention wrapper for monkey-patching arbitrary attention modules.
    """
    def __init__(self, orig_module, num_special_tokens, num_heads, **kwargs):
        super().__init__()
        self.orig_module = orig_module
        self.num_special_tokens = num_special_tokens
        self.num_heads = num_heads

        # lazy init, to be updated before each forward pass
        self.patch_width = kwargs.get("patch_width", None)
        self.patch_height = kwargs.get("patch_height", None)
        self.est_topk_idx = torch.empty(0) # init as empty, to be updated as [SP, topk]

        # for lazy topk computation
        self.lazy_dino_topk = kwargs.get("lazy_dino_topk", False)
        self.dino_topk = kwargs.get("dino_topk", 4)
        self.first_dino_flag = False  # init as False
        
        # parse config
        cfg = kwargs.get("config", [])
        self.head_cfg = [{"mode": "full"} for _ in range(self.num_heads)]  # default to "full"

        # parse group configs and update head configs
        for group_cfg in cfg:
            group_head_ids = group_cfg["head_ids"]
            group_cfg.pop("head_ids")
            for head_id in group_head_ids:
                # sanity check: each head can only be assigned to one pattern
                if self.head_cfg[head_id]["mode"] != "full":
                    raise ValueError(f"Head {head_id} assigned to multiple patterns!")
                self.head_cfg[head_id].update(group_cfg)

    def sparse_attention(self, q, k, v, cfg, head_idx=0):
        # lazy topk: compute and store QK top-k indices from the first DINO TopK head
        if (
            cfg["mode"] == "dino_topk" and
            self.lazy_dino_topk and cfg.get("is_first_dino", False) and not self.first_dino_flag
        ):
            print(f"Computing QK TopK indices (K={self.dino_topk} each frame) for subsequent DINO TopK heads ...")
            self.first_dino_flag = True  # avoid recomputing for subsequent DINO TopK heads at the same block

            # remove special tokens and extract image tokens
            num_toks = q.shape[2]
            num_toks_per_img = self.patch_width * self.patch_height + self.num_special_tokens
            num_imgs = num_toks // num_toks_per_img
            _, img_idx = get_separate_indices(num_imgs, num_toks_per_img, self.num_special_tokens)
            q_img = q.squeeze(1)[:, img_idx, :]  # (B, N_img, D)
            k_img = k.squeeze(1)[:, img_idx, :]  # (B, N_img, D)

            # compute and update indices
            self.est_topk_idx = compute_qk_topk_indices_fused(
                q_img, k_img, self.dino_topk,
                num_toks_per_img=(self.patch_width * self.patch_height),
            ).squeeze()  # [SP, topk]

        # compute sparse attention
        return sparse_attention(
            q, k, v, self.est_topk_idx, cfg, self.patch_height, self.patch_width,
            self.num_special_tokens, head_idx,
        )

    def compute_multihead_attention(self, q, k, v):
        attn_outs = []
        for h in range(self.num_heads):
            q_h, k_h, v_h = q[:, h:h+1], k[:, h:h+1], v[:, h:h+1]  # [B, 1, N, D]

            if self.head_cfg[h]["mode"] == "full":
                attn_out = F.scaled_dot_product_attention(q_h, k_h, v_h)  # [B, 1, N, D]
            else:
                attn_out = self.sparse_attention(q_h, k_h, v_h, self.head_cfg[h], head_idx=h)  # [B, 1, N, D]

            attn_outs.append(attn_out)

        return torch.cat(attn_outs, dim=1)  # [B, H, N, D]
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a base wrapper and should not be used directly.")


class BaseSparseAttentionProfiler(nn.Module):
    """
    A profiler that selects the best sparse-attention pattern for each head by exhaustively
    searching a predefined pattern set and comparing each output against full attention.

    NOTE: This profiler is for analysis only and is not optimized for speed.
    It can be much slower than the original attention layer because it performs
    multiple attention computations per head.
    """
    def __init__(self, orig_module, num_special_tokens, num_heads, **kwargs):
        super().__init__()
        self.orig_module = orig_module
        self.num_special_tokens = num_special_tokens
        self.num_heads = num_heads

        # lazy init, to be updated before each forward pass
        self.patch_width = kwargs.get("patch_width", None)
        self.patch_height = kwargs.get("patch_height", None)
        self.est_topk_idx = torch.empty(0) # [SP, topk]

        self.verbose = kwargs.get("verbose", True)
        self.layer_idx = kwargs.get("layer_idx", None)
        self.profile_metric = kwargs.get("profile_metric", "cmp_mse")

        # candidate sparse patterns under given budget
        self.upper_stride = kwargs.get("upper_stride", 128)
        self.lower_stride = kwargs.get("lower_stride", 4)
        self.switch_layer = kwargs.get("switch_layer", 9)

        # NOTE: by default, we will use all available sparse attention patterns for profiling
        self.sparse_patterns = kwargs.get(
            "sparse_patterns",
            ["broadcast_first", "all_to_first", "q_probe_topk", "dino_topk", "global_to_frame", "stride"]
        )

        # accumulate per-head errors across multiple scenes in the calibration dataset
        heads = kwargs.get("heads", list(range(self.num_heads)))
        self.head_errors = {
            h: {} for h in range(self.num_heads) if h in heads
        }

    def sparse_attention(self, q, k, v, cfg, head_idx=0):
        return sparse_attention(
            q, k, v, self.est_topk_idx, cfg, self.patch_height, self.patch_width,
            self.num_special_tokens, head_idx,
        )

    def profile_attention_cmp_mse(self, q, k, v, head_idx):
        # compute full attention
        fa_out = F.scaled_dot_product_attention(q, k, v)  # [B, 1, N, D]

        # define cfgs
        if int(self.layer_idx) < self.switch_layer:
            base_stride = self.upper_stride
        else:
            base_stride = self.lower_stride

        cfgs = [
            # base cfg
            {"mode": "stride", "stride": base_stride, "include_as": True, "is_base": True},
            {"mode": "broadcast_first", "is_base": False},
            {"mode": "all_to_first", "is_base": False},
            {"mode": "global_to_frame", "is_base": False}
        ]

        # stride
        for mul in range(1, 6):
            stride = base_stride * (2 ** mul)
            cfgs.append({"mode": "stride", "stride": stride, "is_base": False})

        # q probe topk with different K
        for mul in range(6):
            stride = base_stride * (2 ** mul)
            cfgs.append({"mode": "q_probe_topk", "stride": stride, "is_base": False})

        # dino topk
        # TODO: this `topk` is dummy and will never be used, to be removed
        cfgs.append({"mode": "dino_topk", "topk": 32768, "dino_sp2sp_stride": 1, "is_base": False})

        # loop over different sparse attention configs and find the best one
        for i, cfg in enumerate(cfgs):
            # compute sparse attention output
            sa_out = self.sparse_attention(q, k, v, cfg, head_idx)  # [B, 1, N, D]

            # compute NMSE with full attention output
            mse = F.mse_loss(sa_out, fa_out)
            energy = torch.mean(fa_out ** 2) + 1e-10
            mse = (mse / energy).item()

            # record
            if i not in self.head_errors[head_idx]:
                self.head_errors[head_idx][i] = {"cfg": cfg, "mse": [mse]}
            else:
                self.head_errors[head_idx][i]["mse"].append(mse)

            if self.verbose:
                print(f"[L{self.layer_idx:>2} | H{head_idx:>2}] config {cfg}, MSE={mse:.6f}")

        # output
        return fa_out

    def profile_multihead_attention(self, q, k, v):
        attn_outs = []
        for h in range(self.num_heads):
            q_h, k_h, v_h = q[:, h:h+1], k[:, h:h+1], v[:, h:h+1]  # [B, 1, N, D]
            attn_out = self.profile_attention_cmp_mse(q_h, k_h, v_h, head_idx=h)
            attn_outs.append(attn_out)
        return torch.cat(attn_outs, dim=1)  # [B, H, N, D]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a base profiler and should not be used directly.")
