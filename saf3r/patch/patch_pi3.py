from types import MethodType
import json
from addict import Dict
import math

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from ..models.official_pi3.models.pi3 import homogenize_points
from ..models.official_pi3.models.layers.attention import FlashAttentionRope as Pi3Attention
from .wrapper import BaseHeadwiseSparseAttentionWrapper, BaseSparseAttentionProfiler
from .utils import replace_modules, compute_qk_topk_indices_batched


NUM_SPECIAL_TOKENS = 5
NUM_LOCAL_BLOCKS = 18
NUM_GLOBAL_BLOCKS = 18
NUM_ATTENTION_HEADS = 16
PATCH_SIZE = 14


def rel2abs(x):
    """
    Convert the relative (global) block index to the actual block index in Pi3 model.
    """
    if isinstance(x, (int, str)):
        return int(x) * 2 + 1
    return [int(idx) * 2 + 1 for idx in x]


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
                model.decode = MethodType(
                    customized_decode_lazy_topk(),
                    model
                )
            else:
                # pre-computed topk indices from DINO features
                model.forward = MethodType(
                    customized_forward(
                        topk_mode=model_cfg.patch_module.topk_mode,
                        topk=model_cfg.patch_module.topk,
                        per_frame=model_cfg.patch_module.per_frame_topk,
                        max_topk=model_cfg.patch_module.max_topk,
                        include_first_frame=model_cfg.patch_module.include_first_frame,
                        first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
                    ),
                    model
                )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"decoder.{rel2abs(layer_idx)}.attn": Dict(
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
        model.forward = MethodType(
            customized_forward(
                topk_mode="token",
                topk=model_cfg.patch_module.topk,
                per_frame=model_cfg.patch_module.per_frame_topk,
                max_topk=model_cfg.patch_module.max_topk,
                include_first_frame=model_cfg.patch_module.include_first_frame,
                first_frame_stride=model_cfg.patch_module.get("first_frame_stride", 4),
            ),
            model
        )

        # replace all global attention modules
        replace_modules(
            model,
            replace_dict={
                f"decoder.{rel2abs(layer_idx)}.attn": Dict(
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
        block = model.decoder[i]
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
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]


        # ===================================================================== #
        # NOTE: add dino output features to patched attention blocks
        # --------------------------------------------------------------------- #
        # NOTE: P=patch_h*patch_w w/o special tokens, D = 1024
        S, P, C = hidden.shape
        dino_feat = F.normalize(hidden.reshape(1, S*P, C), p=2, dim=-1)  # [1, SP, D]
        est_topk_idx = compute_qk_topk_indices_batched(
            dino_feat, dino_feat, topk=topk, q_blocksize=4096,
            per_frame=per_frame, num_toks_per_img=P, max_topk=max_topk,
            include_first_frame=include_first_frame, first_frame_stride=first_frame_stride
        ).reshape(S*P, -1)  # [SP, topk]

        # register the precomputed top-k indices for each patched block
        for block in self.decoder:
            if isinstance(
                block.attn,
                (HeadwiseSparseAttentionWrapper, SparseAttentionProfiler)
            ):
                if topk_mode == "token":
                    block.attn.est_topk_idx = est_topk_idx

        # ===================================================================== #
        

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
    return forward


def customized_decode_lazy_topk():
    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

            # ================================================================ #
            # NOTE: Update the indices for all DINO TopK heads after they are computed by the first DINO TopK head
            # ---------------------------------------------------------------- #
            if (
                isinstance(blk.attn, HeadwiseSparseAttentionWrapper) and
                blk.attn.first_dino_flag  # only do this for the block containing the first DINO TopK head
            ):
                est_topk_idx = blk.attn.est_topk_idx
                for j in range(i + 1, len(self.decoder)):
                    _blk = self.decoder[j]
                    if isinstance(_blk.attn, HeadwiseSparseAttentionWrapper):
                        _blk.attn.est_topk_idx = est_topk_idx
            # ================================================================ #

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    return decode


def save_profile_results(model, save_path, metric):
    print("Profiling results:")

    # get and accumulate profile results
    err_metric = metric.split("_")[-1]
    results = {}  # {layer_idx: {head_idx: {pattern: [e1, e2, ...]}}}
    first_dino_flag = False
    for i in range(NUM_GLOBAL_BLOCKS):
        abs_idx = rel2abs(i)
        attn = model.decoder[abs_idx].attn
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
    def __init__(self, orig_module: Pi3Attention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)

    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.orig_module.qkv(x).reshape(B, N, 3, self.orig_module.num_heads, C // self.orig_module.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.orig_module.q_norm(q).to(v.dtype), self.orig_module.k_norm(k).to(v.dtype)
        if self.orig_module.rope is not None:
            q = self.orig_module.rope(q, xpos)
            k = self.orig_module.rope(k, xpos)

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
    def __init__(self, orig_module: Pi3Attention, **kwargs):
        super().__init__(orig_module, NUM_SPECIAL_TOKENS, NUM_ATTENTION_HEADS, **kwargs)
    
    def forward(self, x: Tensor, attn_bias=None, xpos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.orig_module.qkv(x).reshape(B, N, 3, self.orig_module.num_heads, C // self.orig_module.num_heads).transpose(1, 3)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k = self.orig_module.q_norm(q).to(v.dtype), self.orig_module.k_norm(k).to(v.dtype)
        if self.orig_module.rope is not None:
            q = self.orig_module.rope(q, xpos)
            k = self.orig_module.rope(k, xpos)

        x = self.profile_multihead_attention(q, k, v)  # [B, H, N, D]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.orig_module.proj(x)
        x = self.orig_module.proj_drop(x)

        return x
