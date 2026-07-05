from types import MethodType
import json
from addict import Dict
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..models.official_da3.model.dinov2.vision_transformer import (
    THRESH_FOR_REF_SELECTION, select_reference_view, reorder_by_reference, restore_original_order
)
from ..models.official_da3.model.dinov2.layers.block import Block
from ..models.official_da3.model.dinov2.layers.attention import Attention as DA3Attention
from .wrapper import BaseHeadwiseSparseAttentionWrapper, BaseSparseAttentionProfiler
from .utils import replace_modules, compute_qk_topk_indices_batched


NUM_SPECIAL_TOKENS = 1
NUM_LOCAL_BLOCKS = 13
NUM_GLOBAL_BLOCKS = 14
NUM_ATTENTION_HEADS = 24
PATCH_SIZE = 14

 
def rel2abs(x):
    """
    Convert the relative (global) block index to the actual block index in DA3 model.
    """
    if isinstance(x, (int, str)):
        return int(x) * 2 + 13
    return [int(idx) * 2 + 13 for idx in x]


def patch_model(model, model_cfg):
    # Headwise dynamic sparse attention
    if model_cfg.patch_module.type == "headsparse":
        # load the detailed sparse config of each layer
        with open(model_cfg.sparse_config_path, "r") as f:
            config = json.load(f)

        # replace forward function of aggregator to precompute TopK DINO features
        if (
            any("dino_topk" == g["mode"] for layer_config in config.values() for g in layer_config)
        ):
            assert model_cfg.patch_module.topk_mode == "token"
            if model_cfg.patch_module.lazy_dino_topk:
                # compute top-k indices from queries and keys from the first correspondence head
                print("=> Using lazy TopK computation (from the first correspondence head)")
                model.model.backbone.pretrained._get_intermediate_layers_not_chunked = MethodType(
                    customized_forward_lazy_topk(),
                    model.model.backbone.pretrained
                )
            else:
                # pre-computed topk indices from DINO features
                model.model.backbone.pretrained._get_intermediate_layers_not_chunked = MethodType(
                    customized_forward(
                        topk_mode=model_cfg.patch_module.topk_mode,
                        topk=model_cfg.patch_module.topk,
                        per_frame=model_cfg.patch_module.per_frame_topk,
                        max_topk=model_cfg.patch_module.max_topk,
                        include_first_frame=model_cfg.patch_module.include_first_frame,
                        first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
                    ),
                    model.model.backbone.pretrained
                )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"model.backbone.pretrained.blocks.{rel2abs(layer_idx)}.attn": Dict(
                    module=HeadwiseSparseAttentionWrapper,
                    layer_idx=layer_idx,
                    dino_topk=model_cfg.patch_module.topk,
                    lazy_dino_topk=model_cfg.patch_module.lazy_dino_topk,
                    config=layer_config,
                )
                for layer_idx, layer_config in config.items()  # NOTE: config provides "relative layer indices"
            }
        )

    # Offline profile
    elif model_cfg.patch_module.type == "headprofile":
        # load a file specifying which heads to profile
        if model_cfg.get("profile_config_path", None):
            with open(model_cfg.profile_config_path, "r") as f:
                config = json.load(f)
            print(f"=> Using profile config:{config}\n")
        else:
            # if no profile config file is provided, default to profiling all heads in all layers
            config = {str(i): list(range(NUM_ATTENTION_HEADS)) for i in range(NUM_GLOBAL_BLOCKS)}
            print(f"=> No profile config file provided, defaulting to profiling all heads in all layers")

        # pre-computed topk indices from DINO features
        model.model.backbone.pretrained._get_intermediate_layers_not_chunked = MethodType(
            customized_forward(
                topk_mode="token",
                topk=model_cfg.patch_module.topk,
                per_frame=model_cfg.patch_module.per_frame_topk,
                max_topk=model_cfg.patch_module.max_topk,
                include_first_frame=model_cfg.patch_module.include_first_frame,
                first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
            ),
            model.model.backbone.pretrained
        )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"model.backbone.pretrained.blocks.{rel2abs(layer_idx)}.attn": Dict(
                    module=SparseAttentionProfiler,
                    layer_idx=layer_idx,
                    heads=heads,  # a list of head indices to profile for the current layer
                    verbose=model_cfg.patch_module.verbose,
                    profile_metric=model_cfg.patch_module.profile_metric,
                    upper_stride=model_cfg.patch_module.upper_stride,
                    lower_stride=model_cfg.patch_module.lower_stride,
                    switch_layer=model_cfg.patch_module.switch_layer
                )
                for layer_idx, heads in config.items()  # NOTE: config provides "relative layer indices"
            }
        )

    return model


def update_args_per_forward(model, img_h, img_w):
    """
    Update wrapper args (e.g., patch size) before each inference.
    """
    # sweep all global blocks to update patch grid size
    for i in rel2abs(range(NUM_GLOBAL_BLOCKS)):
        block = model.model.backbone.pretrained.blocks[i]
        if isinstance(
            block.attn, (HeadwiseSparseAttentionWrapper, SparseAttentionProfiler)
        ):
            block.attn.patch_height = img_h // PATCH_SIZE
            block.attn.patch_width = img_w // PATCH_SIZE
            if hasattr(block.attn, "first_dino_flag"):
                block.attn.first_dino_flag = False  # reset the first_dino_flag for each forward pass


def customized_forward(
    topk_mode="token", per_frame=True, topk=4, max_topk=2048,
    include_first_frame=False, first_frame_stride=1
):
    def forward(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
                # Select reference view using configured strategy
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                b_idx = select_reference_view(x, strategy=strategy)
                # Reorder views to place reference view first
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token


            # ================================================================ #
            # NOTE: update for each global wrapper on-the-fly
            # ---------------------------------------------------------------- #
            if isinstance(
                blk.attn,
                (HeadwiseSparseAttentionWrapper, SparseAttentionProfiler)
            ):
                blk.attn.est_topk_idx = est_topk_idx
            # ================================================================ #


            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x
            

            # ================================================================ #
            # NOTE: add dino output features to patched attention blocks
            # ---------------------------------------------------------------- #
            # alt_start = 13
            # x: [B, S, N, D] where N=patch_h*patch_w + num_special_tokens; D = 1536
            # cam_token: [B, S, D]
            # NOTE: compute dino KV indices before the first global attention block
            if self.alt_start != -1 and i == self.alt_start - 1:
                patch_tokens = x[:, :, 1:]  # [1, S, P, C]
                _, _, P, C = patch_tokens.shape
                dino_feat = F.normalize(patch_tokens.reshape(1, S*P, C), p=2, dim=-1)  # [1, SP, C]
                est_topk_idx = compute_qk_topk_indices_batched(
                    dino_feat, dino_feat, topk=topk, q_blocksize=4096,
                    per_frame=per_frame, num_toks_per_img=P, max_topk=max_topk,
                    include_first_frame=include_first_frame, first_frame_stride=first_frame_stride
                ).reshape(S*P, -1)  # [SP, topk]
            # ================================================================ #


            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                # Restore original view order if reordering was applied
                if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and 'b_idx' in locals():
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        del est_topk_idx
        torch.cuda.empty_cache()
        
        return output, aux_output
    return forward


def customized_forward_lazy_topk():
    def forward(self, x, n=1, export_feat_layers=[], **kwargs):
        B, S, _, H, W = x.shape
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.blocks):
            if i < self.rope_start or self.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            if self.alt_start != -1 and (i == self.alt_start - 1) and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token", None) is None:
                # Select reference view using configured strategy
                strategy = kwargs.get("ref_view_strategy", "saddle_balanced")
                b_idx = select_reference_view(x, strategy=strategy)
                # Reorder views to place reference view first
                x = reorder_by_reference(x, b_idx)
                local_x = reorder_by_reference(local_x, b_idx)

            if self.alt_start != -1 and i == self.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim=1)
                x[:, :, 0] = cam_token

            if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                x = self.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.process_attention(x, blk, "local", pos=l_pos)
                local_x = x
            
            # ================================================================ #
            # NOTE: Update the indices for all DINO TopK heads after they are computed by the first DINO TopK head
            # ---------------------------------------------------------------- #
            if (
                isinstance(blk.attn, HeadwiseSparseAttentionWrapper) and
                blk.attn.first_dino_flag  # only do this for the block containing the first DINO TopK head
            ):
                est_topk_idx = blk.attn.est_topk_idx
                for j in range(i + 1, len(self.blocks)):
                    _blk = self.blocks[j]
                    if isinstance(_blk.attn, HeadwiseSparseAttentionWrapper):
                        _blk.attn.est_topk_idx = est_topk_idx
            # ================================================================ #

            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                # Restore original view order if reordering was applied
                if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and 'b_idx' in locals():
                    out_x = restore_original_order(out_x, b_idx)
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)

        del est_topk_idx
        torch.cuda.empty_cache()
        
        return output, aux_output
    return forward


def save_profile_results(model, save_path, metric):
    print("Profiling results:")
    
    # get and accumulate profile results
    err_metric = metric.split("_")[-1]
    results = {}  # {layer_idx: {head_idx: {pattern: [e1, e2, ...]}}}
    first_dino_flag = False
    for i in range(NUM_GLOBAL_BLOCKS):
        abs_idx = rel2abs(i)
        attn = model.model.backbone.pretrained.blocks[abs_idx].attn
        if isinstance(attn, SparseAttentionProfiler):
            layer_results = {}
            # for each head
            for head_idx, cfgs in attn.head_errors.items():
                # find the best pattern with minimum MSE for each head
                best_idx, best_cfg, best_metric = None, None, math.inf
                for cfg_idx, cfg in cfgs.items():
                    avg_metric = np.mean(cfg[err_metric]) # average over all samples 
                    if avg_metric < best_metric:
                        best_idx, best_cfg, best_metric = cfg_idx, cfg["cfg"], avg_metric
                    print(f"[L{i:>2}|H{head_idx:>2}] cfg: {cfg['cfg']}, Metric: {avg_metric:.6f}")

                # add results
                if best_idx not in layer_results:
                    # mark the first DINO TopK head in the layer
                    if best_cfg["mode"] == "dino_topk" and not first_dino_flag:
                        first_dino_flag = True
                        best_cfg["is_first_dino"] = True
                    layer_results[best_idx] = best_cfg
                    layer_results[best_idx]["head_ids"] = [head_idx]
                else:
                    layer_results[best_idx]["head_ids"].append(head_idx)

            # add layer results 
            results[i] = layer_results

    # convert format
    results_formatted = {}
    for layer_idx, pat_dict in results.items():
        results_formatted[layer_idx] = [cfg for _, cfg in pat_dict.items()]

    # save results to json
    with open(save_path, "w") as f:
        json.dump(results_formatted, f, indent=4)
    print("\n=> Saved profile results to:", save_path)



# Wrappers
class HeadwiseSparseAttentionWrapper(BaseHeadwiseSparseAttentionWrapper):
    def __init__(self, orig_module: DA3Attention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.orig_module.qkv(x)
            .reshape(B, N, 3, self.orig_module.num_heads, C // self.orig_module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None and pos is not None:
            q = self.orig_module.rope(q, pos)
            k = self.orig_module.rope(k, pos)

        x = self.compute_multihead_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.orig_module.proj(x)
        x = self.orig_module.proj_drop(x)

        return x


class SparseAttentionProfiler(BaseSparseAttentionProfiler):
    """
    A profiler to find the best sparse attention pattern for each head, by exhaustively
    searching over a predefined set of sparse attention patterns and comparing their
    outputs with the full attention output.

    NOTE: this profiler is designed for analysis and is NOT optimized for speed.
    It will run significantly slower than the original attention layer due to the exhaustive
    search and multiple attention computations per head.
    """
    def __init__(self, orig_module: DA3Attention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)
    
    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.orig_module.qkv(x)
            .reshape(B, N, 3, self.orig_module.num_heads, C // self.orig_module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None and pos is not None:
            q = self.orig_module.rope(q, pos)
            k = self.orig_module.rope(k, pos)

        x = self.profile_multihead_attention(q, k, v)  # [B, H, N, D]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.orig_module.proj(x)
        x = self.orig_module.proj_drop(x)

        return x
