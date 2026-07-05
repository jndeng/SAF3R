"""
Utility functions for model probing and analysis.
"""
from functools import partial
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ Register functions ============================ #
def register_dino_output_hooks(model, model_name, dino_out_feats):
    """
    Register hooks to extract "DINO output features" from different models.
    NOTE: The output of final block is not before normalization.
    """
    dino_outfeat_hooks = []

    if model_name == "vggt":
        dino_blocks = model.aggregator.patch_embed.blocks
        for i in range(len(dino_blocks)):
            m = dino_blocks[i]
            hook_fn = partial(get_output_hook, results=dino_out_feats)
            dino_outfeat_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
    else:
        raise NotImplementedError(f"Model {model_name} not supported for DINO output feature hook registration.")
    
    return dino_outfeat_hooks


def register_dino_attn_hooks(model, model_name, dino_attn_maps):
    """
    Register hooks to extract "DINO Attention" maps from different models.
    """
    dino_attn_hooks = []

    if model_name == "vggt":
        dino_blocks = model.aggregator.patch_embed.blocks
        for i in range(len(dino_blocks)):
            m = dino_blocks[i].attn
            hook_fn = partial(get_vggt_attn_hook, results=dino_attn_maps)
            dino_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
    else:
        raise NotImplementedError(f"Model {model_name} not supported for DINO attention hook registration.")
    
    return dino_attn_hooks


def register_alt_attn_hooks(
    model, model_name, frame_attn_maps, global_attn_maps, g_q=None, g_k=None, g_v=None
):
    """
    Register hooks to extract "Alternating Attention (AA)" maps from different models.
    """
    frame_attn_hooks, global_attn_hooks = [], []

    if model_name == "vggt":
        frame_blocks = model.aggregator.frame_blocks
        global_blocks = model.aggregator.global_blocks
        for i in range(len(frame_blocks)):
            # frame attention
            m = frame_blocks[i].attn
            hook_fn = partial(get_vggt_attn_hook, results=frame_attn_maps)
            frame_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
        
            # global attention
            m = global_blocks[i].attn
            hook_fn = partial(get_vggt_attn_hook, results=global_attn_maps, q_list=g_q, k_list=g_k, v_list=g_v)
            global_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

    elif model_name == "pi3":
        blocks = model.decoder
        for i in range(len(blocks)):
            m = blocks[i].attn

            # frame attention
            if i % 2 == 0:
                hook_fn = partial(get_pi3_attn_hook, results=frame_attn_maps)
                frame_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

            # global attention
            else:
                hook_fn = partial(get_pi3_attn_hook, results=global_attn_maps)
                global_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
    
    elif model_name == "mapanything":
        global_blocks = model.info_sharing.self_attention_blocks[0::2]
        frame_blocks = model.info_sharing.self_attention_blocks[1::2]
        for i in range(len(frame_blocks)):
            # frame attention
            m = frame_blocks[i].attn
            hook_fn = partial(get_mapanything_attn_hook, results=frame_attn_maps)
            frame_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

            # global attention
            m = global_blocks[i].attn
            hook_fn = partial(get_mapanything_attn_hook, results=global_attn_maps)
            global_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

    elif model_name == "da3":
        blocks = model.model.backbone.pretrained.blocks[13:]
        for i in range(len(blocks)):
            m = blocks[i].attn

            # frame attention
            if i % 2 == 1:
                hook_fn = partial(get_da3_attn_hook, results=frame_attn_maps)
                frame_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

            # global attention (The 14-th layer is global attention)
            else:
                hook_fn = partial(get_da3_attn_hook, results=global_attn_maps, q_list=g_q, k_list=g_k, v_list=g_v)
                global_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
    
    elif model_name == "streamvggt":
        frame_blocks = model.aggregator.frame_blocks
        global_blocks = model.aggregator.global_blocks
        for i in range(len(frame_blocks)):
            # TODO: support and test frame attention
            # frame attention
            # m = frame_blocks[i].attn
            # hook_fn = partial(get_streamvggt_attn_hook, results=frame_attn_maps)
            # frame_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
        
            # global attention
            m = global_blocks[i].attn
            hook_fn = partial(get_streamvggt_attn_hook, results=global_attn_maps, q_list=g_q, k_list=g_k, v_list=g_v)
            global_attn_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

    else:
        raise NotImplementedError(f"Model {model_name} not supported for attention hook registration.")
    
    return frame_attn_hooks, global_attn_hooks


def register_block_internal_hooks(
    model, model_name, frame_block_feats: defaultdict, global_block_feats: defaultdict
):
    """
    Register hooks to extract internal features (e.g., features before/after attention or FFN) from each block.
    """
    frame_block_hooks, global_block_hooks = [], []

    if model_name == "vggt":
        # frame blocks
        if frame_block_feats is not None:
            frame_blocks = model.aggregator.frame_blocks
            for i in range(len(frame_blocks)):
                m = frame_blocks[i]
                hook_fn = partial(get_block_internal_hook, results=frame_block_feats)
                frame_block_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

        # global blocks
        if global_block_feats is not None:
            global_blocks = model.aggregator.global_blocks
            for i in range(len(global_blocks)):
                m = global_blocks[i]
                hook_fn = partial(get_block_internal_hook, results=global_block_feats)
                global_block_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))
    
    elif model_name == "da3":
        blocks = model.model.backbone.pretrained.blocks[13:]

        for i in range(len(blocks)):
            m = blocks[i]

            # frame block
            if i % 2 == 1:
                if frame_block_feats is not None:
                    hook_fn = partial(get_block_internal_hook, results=frame_block_feats)
                    frame_block_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

            # global block (The 14-th block is a global block)
            else:
                if global_block_feats is not None:
                    hook_fn = partial(get_block_internal_hook, results=global_block_feats)
                    global_block_hooks.append(m.register_forward_hook(hook_fn, with_kwargs=True))

    else:
        raise NotImplementedError(f"Model {model_name} not supported for block internal hook registration.")
    
    return frame_block_hooks, global_block_hooks


def remove_hooks(*hook_lists):
    """
    Remove registered hooks.
    """
    for hook_list in hook_lists:
        for hook in hook_list:
            hook.remove()


# ============================== Hook functions ============================== #
def get_vggt_attn_hook(
    m, input_args, input_kwargs, output,
    results: list, q_list: list = None, k_list: list = None, v_list: list = None 
):
    """
    Hook function to extract attention maps from an Attention module.
    This function re-runs the attention computation to get the attention weights.
    NOTE: Assume `with_kwargs=True` when registering the hook.
    NOTE: Compatible with global/local/dino attention in VGGT model.

    Args:
        m (nn.Module): The Attention module.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (list): A list to store the attention maps.
    """
    x = input_args[0] # [B, N, C] where N = img_toks + reg_toks (4) + cls/cam tok (1)
    pos = input_kwargs.get("pos", None) # [B, N, 2]

    B, N, C = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = m.q_norm(q), m.k_norm(k)

    if m.rope is not None:
        q = m.rope(q, pos)
        k = m.rope(k, pos)

    # NOTE: By default, fused_attn is set to True in VGGT model.
    # Yet here we force using naive attn instead fused attn to get attn maps
    q = q * m.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = m.proj(x)
    x = m.proj_drop(x)

    # sanity check
    if not torch.allclose(x.mean(), output.mean(), atol=1e-4):
    # if not torch.allclose(x, output, atol=1e-2):
        raise ValueError(
            f"Re-computed attention output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )
    
    # save attention maps and optionally q, k, v
    results.append(attn.detach().cpu())  # [B, num_heads, N, N]
    if q_list is not None:
        q_list.append(q.detach().cpu()) # NOTE: divided by sqrt(d_k) already
    if k_list is not None:
        k_list.append(k.detach().cpu())
    if v_list is not None:
        v_list.append(v.detach().cpu())


def get_mapanything_attn_hook(
    m, input_args, input_kwargs, output,
    results: list, q_list: list = None, k_list: list = None, v_list: list = None
):
    """
    Hook function to extract attention maps from an Attention module.
    This function re-runs the attention computation to get the attention weights.
    NOTE: Assume `with_kwargs=True` when registering the hook.
    NOTE: Assume using default settings in MapAnything attention module.

    Args:
        m (nn.Module): The Attention module.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (list): A list to store the attention maps.
    """
    x = input_args[0] # [B, N, C] where N = img_toks + reg_toks (4) + cls/cam tok (1)

    B, N, C = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = m.q_norm(q), m.k_norm(k)

    q = q * m.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)
    x = m.proj(x)
    x = m.proj_drop(x)
    
    # sanity check
    if not torch.allclose(x.mean(), output.mean(), atol=1e-2):
        raise ValueError(
            f"Re-computed attention output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )

    # save attention maps and optionally q, k, v
    results.append(attn.detach().cpu())  # [B, num_heads, N, N]
    if q_list is not None:
        q_list.append(q.detach().cpu()) # NOTE: divided by sqrt(d_k) already
    if k_list is not None:
        k_list.append(k.detach().cpu())
    if v_list is not None:
        v_list.append(v.detach().cpu())


def get_pi3_attn_hook(
    m, input_args, input_kwargs, output,
    results: list, q_list: list = None, k_list: list = None, v_list: list = None
):
    """
    Hook function to extract attention maps from an Attention module.
    This function re-runs the attention computation to get the attention weights.
    NOTE: Assume `with_kwargs=True` when registering the hook.

    Args:
        m (nn.Module): The Attention module.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (list): A list to store the attention maps.
    """
    x = input_args[0]
    xpos = input_kwargs.get("xpos", None) # [B, N, 2]

    B, N, C = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, C // m.num_heads).transpose(1, 3)

    # q, k, v = unbind(qkv, 2)
    q, k, v = [qkv[:,:,i] for i in range(3)]
    q, k = m.q_norm(q).to(v.dtype), m.k_norm(k).to(v.dtype)

    if m.rope is not None:
        q = m.rope(q, xpos)
        k = m.rope(k, xpos)

    q = q * m.scale
    attn = q @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = m.proj(x)
    x = m.proj_drop(x)

    # sanity check
    if not torch.allclose(x.mean(), output.mean(), atol=1e-4):
        raise ValueError(
            f"Re-computed attention output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )
    
    # save attention maps
    results.append(attn.detach().cpu())  # [B, num_heads, N, N]
    if q_list is not None:
        q_list.append(q.detach().cpu()) # NOTE: divided by sqrt(d_k) already
    if k_list is not None:
        k_list.append(k.detach().cpu())
    if v_list is not None:
        v_list.append(v.detach().cpu())
    

def get_da3_attn_hook(
    m, input_args, input_kwargs, output,
    results: list, q_list: list = None, k_list: list = None, v_list: list = None 
):
    """
    Hook function to extract attention maps from a DA3 Attention module.
    This function re-runs the attention computation to get the attention weights.
    NOTE: Assume `with_kwargs=True` when registering the hook.

    Args:
        m (nn.Module): The Attention module.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (list): A list to store the attention maps.
    """
    x = input_args[0] # [B, N, C] where N = img_toks + reg_toks (4) + cls/cam tok (1)
    pos = input_kwargs.get("pos", None) # [B, N, 2]

    B, N, C = x.shape
    qkv = (
        m.qkv(x)
        .reshape(B, N, 3, m.num_heads, C // m.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = m.q_norm(q), m.k_norm(k)
    if m.rope is not None and pos is not None:
        q = m.rope(q, pos)
        k = m.rope(k, pos)

    q = q * m.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = m.proj(x)
    x = m.proj_drop(x)

    # sanity check
    # NOTE: DA3 use SDPA which may introduce lager numerical differences
    if not torch.allclose(x.mean(), output.mean(), atol=1e-2):
        raise ValueError(
            f"Re-computed attention output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )

    # save attention maps
    results.append(attn.detach().cpu())  # [B, num_heads, N, N]
    if q_list is not None:
        q_list.append(q.detach().cpu()) # NOTE: divided by sqrt(d_k) already
    if k_list is not None:
        k_list.append(k.detach().cpu())
    if v_list is not None:
        v_list.append(v.detach().cpu())


def get_streamvggt_attn_hook(
    m, input_args, input_kwargs, output,
    results: defaultdict, q_list: defaultdict = None, k_list: defaultdict = None, v_list: defaultdict = None
):
    """
    Hook function to extract attention maps from an Attention module.
    This function re-runs the attention computation to get the attention weights.
    NOTE: Assume `with_kwargs=True` when registering the hook.
    NOTE: Compatible with global/local/dino attention in StreamVGGT model.

    Args:
        m (nn.Module): The Attention module.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (defaultdict): A dictionary of list to store the attention maps.
        NOTE: results[s] -> list of attention maps for the s-th frame (s starts from 0)
    """
    x = input_args[0] # [B, N, C] where N = img_toks + reg_toks (4) + cls/cam tok (1)
    pos = input_kwargs.get("pos", None) # [B, N, 2]
    attn_mask = input_kwargs.get("attn_mask", None)
    past_key_values = input_kwargs.get("past_key_values", None)
    use_cache = input_kwargs.get("use_cache", False)

    assert use_cache == True, "Currently only support use_cache=True setting in the hook function."
    assert attn_mask is None, "Attention mask is not supported in the hook function yet."


    B, N, C = x.shape
    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # [B, H, N, D_h]

    pos_k = pos
    if use_cache:
        k = k.unsqueeze(2)  # [B, H, 1, N, D_h]
        v = v.unsqueeze(2)
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)  # [B, H, S, N, D_h]
            v = torch.cat([past_v, v], dim=2)

        S = k.shape[2]   # S is the number of frames in the past (including the current frame)
        new_kv = (k, v)  # ([B, H, S, N, D_h], [B, H, S, N, D_h])
        a, b, c, d, e = k.shape
        k = k.reshape(a, b, c*d, e)  # [B, H, S*N, D_h]
        v = v.reshape(a, b, c*d, e)
        if pos_k is not None:
            pos_k = pos_k.repeat(1, c, 1)  # [B, S*N, 2]

    q, k = m.q_norm(q), m.k_norm(k)

    if m.rope is not None:
        q = m.rope(q, pos)
        k = m.rope(k, pos_k)

    # NOTE: By default, fused_attn is set to True in StreamVGGT model.
    # Yet here we force using naive attn instead fused attn to get attn maps
    q = q * m.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = m.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = m.proj(x)
    x = m.proj_drop(x)

    # sanity check
    if not torch.allclose(x.mean(), output[0].mean(), atol=1e-4):
    # if not torch.allclose(x, output, atol=1e-2):
        raise ValueError(
            f"Re-computed attention output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )
    
    # save attention maps and optionally q, k, v
    results[S-1].append(attn.detach().cpu())  # [B, num_heads, N, S*N]
    if q_list is not None:
        q_list[S-1].append(q.detach().cpu()) # NOTE: divided by sqrt(d_k) already
    if k_list is not None:
        k_list[S-1].append(k.detach().cpu())
    if v_list is not None:
        v_list[S-1].append(v.detach().cpu())


def get_output_hook(m: nn.Module, input_args, input_kwargs, output, results: list):
    """
    Hook function to extract output features from each DINO block in the patch embedding module.
    NOTE: This function should apply to different FF3R models, but not exhaustively tested yet.
    NOTE: Assume `with_kwargs=True` when registering the hook.

    Args:
        m (nn.Module): The MLP module in the patch embedding block.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (dict): A dictionary to store the output features.
    """
    # save output features
    results.append(output.detach().cpu())  # [B, N, C]


def get_block_internal_hook(
    m: nn.Module, input_args, input_kwargs, output, results: defaultdict
):
    """
    Hook function to extract internal features (e.g., features before/after attention or FFN) from each block.
    NOTE: compatible with VGGT and DA3

    Args:
        m (nn.Module): A block (e.g., global block) in FF3R models.
        input_args (tuple): The positional arguments to the module.
        input_kwargs (dict): The keyword arguments to the module.
        outputs (tuple): The outputs from the module.
        results (dict): A dictionary to store the extracted features.
    """
    def attn_residual_func(x, pos=None):
        return m.ls1(m.attn(m.norm1(x), pos=pos))

    def ffn_residual_func(x):
        return m.ls2(m.mlp(m.norm2(x)))


    x = input_args[0] # [B, N, C]
    pos = input_kwargs.get("pos", None) # [B, N, 2]
    
    # forward and save intermediate features
    results["attn_in"].append(x.detach().cpu())
    x = x + attn_residual_func(x, pos=pos)
    results["attn_out"].append(x.detach().cpu())
    x = x + ffn_residual_func(x)
    results["ffn_out"].append(x.detach().cpu())

    # sanity check
    if not torch.allclose(x.mean(), output.mean(), atol=1e-4):
    # if not torch.allclose(x, output, atol=1e-2):
        raise ValueError(
            f"Re-computed block output does not match the original output!\n"
            f"{x.mean()} vs. {output.mean()}"
        )
