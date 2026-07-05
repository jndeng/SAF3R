from types import MethodType
from addict import Dict
import json
import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from ..models.official_vggt.models.aggregator import slice_expand_and_flatten
from ..models.official_vggt.layers.attention import Attention as VGGTAttention
from .wrapper import BaseHeadwiseSparseAttentionWrapper, BaseSparseAttentionProfiler
from .utils import replace_modules, compute_qk_topk_indices_batched


NUM_SPECIAL_TOKENS = 5
NUM_LOCAL_BLOCKS = 24
NUM_GLOBAL_BLOCKS = 24
NUM_ATTENTION_HEADS = 16
PATCH_SIZE = 14


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
                model.aggregator.forward = MethodType(
                    customized_forward_lazy_topk(),
                    model.aggregator
                )
            else:
                # pre-computed topk indices from DINO features
                model.aggregator.forward = MethodType(
                    customized_forward(
                        topk_mode=model_cfg.patch_module.topk_mode,
                        per_frame=model_cfg.patch_module.per_frame_topk,
                        topk=model_cfg.patch_module.topk,
                        max_topk=model_cfg.patch_module.max_topk,
                        include_first_frame=model_cfg.patch_module.include_first_frame,
                        first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
                    ),
                    model.aggregator
                )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"aggregator.global_blocks.{layer_idx}.attn": Dict(
                    module=HeadwiseSparseAttentionWrapper,
                    layer_idx=layer_idx,
                    dino_topk=model_cfg.patch_module.topk,
                    lazy_dino_topk=model_cfg.patch_module.lazy_dino_topk,
                    config=layer_config,
                )
                for layer_idx, layer_config in config.items()
            }
        )


    # Offline profile
    elif model_cfg.patch_module.type == "headprofile":
        # load a file specifying which heads to profile
        # NOTE: this is different from the sparse config file used for headwise sparse attention,
        # which specifies the sparse pattern for each head. This file only specifies which heads to profile,
        # and the profiler will automatically search over all supported sparse patterns for those heads.
        if model_cfg.get("profile_config_path", None):
            with open(model_cfg.profile_config_path, "r") as f:
                config = json.load(f)
            print(f"=> Using profile config:{config}\n")
        else:
            # if no profile config file is provided, default to profiling all heads in all layers
            config = {str(i): list(range(NUM_ATTENTION_HEADS)) for i in range(NUM_GLOBAL_BLOCKS)}
            print(f"=> No profile config file provided, defaulting to profiling all heads in all layers")

        # pre-computed topk indices from DINO features
        model.aggregator.forward = MethodType(
            customized_forward(
                topk_mode=model_cfg.patch_module.topk_mode,
                per_frame=model_cfg.patch_module.per_frame_topk,
                topk=model_cfg.patch_module.topk,
                max_topk=model_cfg.patch_module.max_topk,
                include_first_frame=model_cfg.patch_module.include_first_frame,
                first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
            ),
            model.aggregator
        )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"aggregator.global_blocks.{layer_idx}.attn": Dict(
                    module=SparseAttentionProfiler,
                    layer_idx=layer_idx,
                    heads=heads,  # a list of head indices to profile for the current layer
                    verbose=model_cfg.patch_module.verbose,
                    profile_metric=model_cfg.patch_module.profile_metric,
                    upper_stride=model_cfg.patch_module.upper_stride,
                    lower_stride=model_cfg.patch_module.lower_stride,
                    switch_layer=model_cfg.patch_module.switch_layer
                )
                for layer_idx, heads in config.items()
            }
        )

    return model


def update_args_per_forward(model, img_h, img_w):
    """
    Update wrapper args (e.g., patch size) before each inference.
    """
    # compute patch size based on input image size
    patch_height = img_h // PATCH_SIZE
    patch_width = img_w // PATCH_SIZE

    # loop overall all patched modules
    for i in range(NUM_GLOBAL_BLOCKS):
        block = model.aggregator.global_blocks[i]
        if isinstance(
            block.attn, (HeadwiseSparseAttentionWrapper, SparseAttentionProfiler)
        ):
            block.attn.patch_height = patch_height
            block.attn.patch_width = patch_width
            if hasattr(block.attn, "first_dino_flag"):
                block.attn.first_dino_flag = False  # reset the first_dino_flag for each forward pass


def customized_forward(
    topk_mode="token", per_frame=True, topk=4, max_topk=2048,
    include_first_frame=False, first_frame_stride=1,
):
    def forward(self, images: torch.Tensor):
        B, S, C_in, H, W = images.shape
        assert B == 1, "Currently only support batch size of 1 for simplicity"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape


        # ===================================================================== #
        # NOTE: add dino output features to patched attention blocks
        # --------------------------------------------------------------------- #
        # NOTE: using the final output (post-norm) DINO features
        # NOTE: `patch_tokens` only contain patch tokens (without special tokens)
        dino_feat = F.normalize(patch_tokens.reshape(1, S*P, C), p=2, dim=-1)  # [1, SP, D]
        est_topk_idx = compute_qk_topk_indices_batched(
            dino_feat, dino_feat, topk=topk, q_blocksize=4096,
            per_frame=per_frame, num_toks_per_img=P, max_topk=max_topk,
            include_first_frame=include_first_frame, first_frame_stride=first_frame_stride,
        ).reshape(S*P, -1)  # [SP, topk]

        # register the precomputed top-k indices for each patched block
        for block in self.global_blocks:
            if isinstance(
                block.attn,
                (HeadwiseSparseAttentionWrapper, SparseAttentionProfiler)
            ):
                block.attn.est_topk_idx = est_topk_idx
        # ===================================================================== #


        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_num in range(self.aa_block_num):
            # whether to save output features of the current layer
            save_feats = True
            if not self.save_intermediates:
                save_feats = block_num in self.DPT_BLOCKS

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, save_feats=save_feats
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, save_feats=save_feats
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
            if save_feats:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                del concat_inter
            
            # clean up
            del frame_intermediates
            del global_intermediates
            torch.cuda.empty_cache()

        return output_list, self.patch_start_idx
    return forward


def customized_forward_lazy_topk():
    def forward(self, images: torch.Tensor):
        B, S, C_in, H, W = images.shape
        assert B == 1, "Currently only support batch size of 1 for simplicity"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_num in range(self.aa_block_num):
            # whether to save output features of the current layer
            save_feats = True
            if not self.save_intermediates:
                save_feats = block_num in self.DPT_BLOCKS

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, save_feats=save_feats
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, save_feats=save_feats
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
            if save_feats:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                del concat_inter

            # ================================================================ #
            # NOTE: Update the indices for all DINO TopK heads after they are computed by the first DINO TopK head
            # ---------------------------------------------------------------- #
            if attn_type == "global":
                # NOTE: `global_idx` already switched to the next layer after `_process_global_attention`
                cur_global_idx = global_idx - 1
                cur_block = self.global_blocks[cur_global_idx]
                if (
                    isinstance(cur_block.attn, HeadwiseSparseAttentionWrapper) and
                    cur_block.attn.first_dino_flag  # only do this for the block containing the first DINO TopK head
                ):
                    est_topk_idx = cur_block.attn.est_topk_idx
                    for bid in range(cur_global_idx + 1, self.aa_block_num):
                        block = self.global_blocks[bid]
                        if isinstance(block.attn, HeadwiseSparseAttentionWrapper):
                            block.attn.est_topk_idx = est_topk_idx
            # ================================================================ #
            
            # clean up
            del frame_intermediates
            del global_intermediates
            torch.cuda.empty_cache()

        return output_list, self.patch_start_idx
    return forward


def save_profile_results(model, save_path, metric):
    print("Profiling results:")

    # get and accumulate profile results
    err_metric = metric.split("_")[-1]
    results = {}  # {layer_idx: {head_idx: {pattern: [e1, e2, ...]}}}
    first_dino_flag = False
    for i in range(NUM_GLOBAL_BLOCKS):
        attn = model.aggregator.global_blocks[i].attn
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
                    print(f"[L{i:>2}|H{head_idx:>2}] cfg: {cfg['cfg']}, {err_metric}: {avg_metric:.6f}")

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

    # convert format and save results to json
    results_formatted = {}
    for layer_idx, pat_dict in results.items():
        results_formatted[layer_idx] = [cfg for _, cfg in pat_dict.items()]

    # save results to json
    with open(save_path, "w") as f:
        json.dump(results_formatted, f, indent=4)
    print("\n=> Saved profile results to:", save_path)



# Wrappers
class HeadwiseSparseAttentionWrapper(BaseHeadwiseSparseAttentionWrapper):
    def __init__(self, orig_module: VGGTAttention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.orig_module.qkv(x).reshape(B, N, 3, self.orig_module.num_heads, self.orig_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None:
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
    def __init__(self, orig_module: VGGTAttention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.orig_module.qkv(x).reshape(B, N, 3, self.orig_module.num_heads, self.orig_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.orig_module.q_norm(q), self.orig_module.k_norm(k)
        if self.orig_module.rope is not None:
            q = self.orig_module.rope(q, pos)
            k = self.orig_module.rope(k, pos)

        x = self.profile_multihead_attention(q, k, v)  # [B, H, N, D]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.orig_module.proj(x)
        x = self.orig_module.proj_drop(x)

        return x
