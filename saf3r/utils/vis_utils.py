"""
Utility functions for visualization.
"""
import os
import math
from typing import List

import open3d as o3d
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import plotly.graph_objects as go

import torch
import torch.nn.functional as F


__all__ = [
    "plot_html",
    "save_ply",
    "plot_attn_img_overlay",
    "plot_attn_distribution",
    "plot_full_attn_map",
    "plot_multiheads_full_attn_map",
]



###############################################################################
# APIs for 3D plots
###############################################################################
def plot_html(scene_data, pred_data, save_path, keep_ratio=0.5, cmap_name="rainbow"):
    """
    Plot predicted camera poses and point clouds and save the result as an HTML file.

    Args:
        scene_data (Dict): A dictionary containing the scene data, including image files.
        pred_data (Dict): A dictionary containing the predictions from the VGGT model.
        save_path (str): The path to save the visualization HTML file.
        keep_ratio (float): The ratio of points to keep based on confidence scores for visualization. Default is 0.5 (keep top 50% points).
        cmap_name (str, optional): The name of the matplotlib colormap to use for different cameras.
    """
    from saf3r.utils.geometry_utils import unproject_depth_map_to_point_map

    # Convert extrinsics to 4x4 matrices if needed
    ext = pred_data.extrinsics
    out = ext
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
    pred_data.extrinsics = out

    # Unproject to 3d word points
    points = unproject_depth_map_to_point_map(
        pred_data.depth, pred_data.extrinsics, pred_data.intrinsics
    ).reshape(-1, 3) # numpy (S, H, W, 3) word
    conf_flat = pred_data.conf.reshape(-1)
    colors = pred_data.processed_images.reshape(-1, 3).astype(np.float32)

    # Filter points based on the predicted confidence maps
    num_keep = int(len(conf_flat) * keep_ratio)
    sorted_indices = np.argsort(conf_flat)[::-1]  
    keep_indices = sorted_indices[:num_keep]

    points = points[keep_indices]
    colors = colors[keep_indices]
    conf_flat = conf_flat[keep_indices]

    # Convert to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Plot
    rgb = Image.open(scene_data.image_files[0]).convert("RGB")
    orig_w, orig_h = rgb.size
    cameras_pred = [
        {
            "K": np.array(pred_data.intrinsics[i]),
            "c2w": np.linalg.inv(np.array(pred_data.extrinsics[i])),  # w2c -> c2w
            "H": orig_h, "W": orig_w,
        }
        for i in range(len(pred_data.extrinsics)) # for each camera in the selected list
    ]
    fig = plot_geom_with_axes_and_cameras(
        geom_pred=pcd,
        geom_gt=None,
        cameras_pred=cameras_pred,
        camera_line_width=2, camera_z_ratio=0.02,
        pred_cam_color="#1f77b4" if cmap_name is None else None,
        pred_cmap_name=cmap_name,
        show_geom_legend=True,
        pred_geom_name="Pred PCD",
    )

    # Save the visualization to an HTML file
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Visualization saved to: {save_path}")


def save_ply(
    pred_data, save_path, keep_ratio=0.5, voxel_size=None, max_points=200000,
    color_mode="auto", write_ascii=False, random_seed=0,
):
    """
    Save VGGT-style prediction as a colored PLY point cloud.

    Args:
        pred_data:
            VGGT prediction object/dict with:
                - depth
                - extrinsics
                - intrinsics
                - conf
                - processed_images
        save_path (str):
            Output .ply path.
        keep_ratio (float):
            Keep top-k confidence points by ratio. Example: 0.5 keeps top 50%.
            Set to 1.0 to keep all valid points before downsampling.
        voxel_size (float or None):
            If not None, apply Open3D voxel downsampling.
            Example: 0.01 for 1cm if your point cloud unit is meter.
        max_points (int or None):
            If not None, randomly sample at most this many points after voxel downsampling.
        color_mode (str):
            How to interpret pred_data.processed_images.
            Options:
                - "auto": infer from value range
                - "uint8": colors are already in [0, 255]
                - "float01": colors are in [0, 1]
                - "minus_one_one": colors are in [-1, 1]
                - "imagenet": ImageNet-normalized colors
        write_ascii (bool):
            If True, save ASCII PLY. If False, save binary little-endian PLY.
            Binary is recommended because it is much smaller.
        random_seed (int):
            Random seed for max_points sampling.

    Returns:
        points, colors_uint8
    """
    from saf3r.utils.geometry_utils import unproject_depth_map_to_point_map

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_attr_or_item(obj, name):
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    def images_to_colors_uint8(images, mode="auto"):
        """
        Convert processed_images to uint8 RGB colors in [0, 255].
        Supports:
            [S, H, W, 3]
            [S, 3, H, W]
            [1, S, H, W, 3]
            [1, S, 3, H, W]
        """
        images = to_numpy(images).astype(np.float32)

        # Remove batch dim if needed
        if images.ndim == 5 and images.shape[0] == 1:
            images = images[0]

        # [S, 3, H, W] -> [S, H, W, 3]
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))

        # [3, H, W] -> [H, W, 3]
        if images.ndim == 3 and images.shape[0] == 3:
            images = np.transpose(images, (1, 2, 0))

        colors = images.reshape(-1, 3)

        if mode == "uint8":
            colors = np.clip(colors, 0, 255)

        elif mode == "float01":
            colors = np.clip(colors, 0.0, 1.0) * 255.0

        elif mode == "minus_one_one":
            colors = (np.clip(colors, -1.0, 1.0) + 1.0) * 127.5

        elif mode == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            colors = colors * std + mean
            colors = np.clip(colors, 0.0, 1.0) * 255.0

        elif mode == "auto":
            cmin, cmax = float(np.nanmin(colors)), float(np.nanmax(colors))

            if cmin >= 0.0 and cmax <= 1.0:
                # [0, 1]
                colors = colors * 255.0
            elif cmin >= 0.0 and cmax <= 255.0:
                # [0, 255]
                colors = colors
            elif cmin >= -1.1 and cmax <= 1.1:
                # likely [-1, 1]
                colors = (np.clip(colors, -1.0, 1.0) + 1.0) * 127.5
            else:
                # likely ImageNet-normalized
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                colors = colors * std + mean
                colors = np.clip(colors, 0.0, 1.0) * 255.0

        else:
            raise ValueError(f"Unsupported color_mode: {mode}")

        return np.clip(colors, 0, 255).round().astype(np.uint8)

    def write_ply_uint8(path, points, colors_uint8, ascii=False):
        """
        Write PLY with explicit uchar RGB fields.
        points: (N, 3), float
        colors_uint8: (N, 3), uint8
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        n = points.shape[0]
        fmt = "ascii" if ascii else "binary_little_endian"

        header = (
            "ply\n"
            f"format {fmt} 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        if ascii:
            with open(path, "w") as f:
                f.write(header)
                for p, c in zip(points, colors_uint8):
                    f.write(
                        f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                    )
        else:
            vertex = np.empty(
                n,
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            vertex["x"] = points[:, 0].astype(np.float32)
            vertex["y"] = points[:, 1].astype(np.float32)
            vertex["z"] = points[:, 2].astype(np.float32)
            vertex["red"] = colors_uint8[:, 0]
            vertex["green"] = colors_uint8[:, 1]
            vertex["blue"] = colors_uint8[:, 2]

            with open(path, "wb") as f:
                f.write(header.encode("ascii"))
                vertex.tofile(f)


    # Convert extrinsics to 4x4 matrices if needed
    ext = pred_data.extrinsics
    out = ext
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
    pred_data.extrinsics = out

    # Unproject prediction
    depth = get_attr_or_item(pred_data, "depth")
    extrinsics = get_attr_or_item(pred_data, "extrinsics")
    intrinsics = get_attr_or_item(pred_data, "intrinsics")
    conf = get_attr_or_item(pred_data, "conf")
    processed_images = get_attr_or_item(pred_data, "processed_images")

    points = unproject_depth_map_to_point_map(
        depth,
        extrinsics,
        intrinsics,
    )
    points = to_numpy(points).reshape(-1, 3).astype(np.float32)

    conf_flat = to_numpy(conf).reshape(-1)

    colors_uint8 = images_to_colors_uint8(
        processed_images,
        mode=color_mode,
    )

    assert points.shape[0] == colors_uint8.shape[0], (
        f"Points and colors have different lengths: "
        f"{points.shape[0]} vs {colors_uint8.shape[0]}"
    )

    # Confidence filtering
    if keep_ratio is not None and keep_ratio < 1.0:
        keep_ratio = float(keep_ratio)
        num_keep = max(1, int(len(conf_flat) * keep_ratio))

        keep_indices = np.argpartition(-conf_flat, num_keep - 1)[:num_keep]

        points = points[keep_indices]
        colors_uint8 = colors_uint8[keep_indices]
        conf_flat = conf_flat[keep_indices]

    # Remove invalid points/colors
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors_uint8).all(axis=1)
    )
    points = points[valid]
    colors_uint8 = colors_uint8[valid]

    if len(points) == 0:
        print("No valid points to save.")
        return points, colors_uint8

    # print(f"After confidence filtering: {len(points)} points")

    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(
            colors_uint8.astype(np.float32) / 255.0
        )

        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

        points = np.asarray(pcd.points).astype(np.float32)
        colors_uint8 = (
            np.clip(np.asarray(pcd.colors), 0.0, 1.0) * 255.0
        ).round().astype(np.uint8)

        # print(f"After voxel downsampling: {len(points)} points")

    # Optional random max-point sampling
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(points), size=max_points, replace=False)

        points = points[idx]
        colors_uint8 = colors_uint8[idx]

        # print(f"After random sampling: {len(points)} points")

    # Save PLY with uchar RGB
    write_ply_uint8(
        save_path,
        points,
        colors_uint8,
        ascii=write_ascii,
    )

    print(f"Point cloud saved to: {save_path} ({len(points)} points)")

    return points, colors_uint8



###############################################################################
# Attention map visualization
###############################################################################
def plot_attn_per_head_per_layer(
    attn_list: List[torch.Tensor],
    layer_idx: int | List[int],
    head_idx: int | List[int] = None,
    title_prefix: str = "GlobalAttn",
):
    """
    Plot attention maps for a specific layer (or list of layers) and head (or list of heads) from a list of attention tensors.
    Always shows average over all heads first, then specific heads if provided.

    Args:
    - attn_list: List of attention map tensors. Each element corresponds to a layer with shape [B, num_heads, N, N].
    - layer_idx: The index (int) or list of indices (list) of the layer(s) to visualize.
    - head_idx: The index (int) or list of indices (list) of the head(s) to visualize. If None, only shows average over all heads.
    - title_prefix: Prefix string for the plot title.

    Usage:
        plot_attn_per_head_per_layer(flame_list, layer_idx=[0, 6, 12], head_idx=[0, 1, 2], title_prefix="VGGT_Flame")
    """

    # Normalize layer_idx to a list
    if isinstance(layer_idx, int):
        layer_indices = [layer_idx]
    else:
        layer_indices = layer_idx

    # Normalize head_idx to a list, always include None (average) at the beginning
    head_indices = [None]  # Start with average
    if head_idx is not None:
        if isinstance(head_idx, int):
            head_indices.append(head_idx)
        else:
            head_indices.extend(head_idx)

    # Get batch size from the first layer
    batch_size = attn_list[layer_indices[0]].shape[0]

    # Determine grid size: rows=layers, cols=batch_size * num_heads
    num_layers = len(layer_indices)
    num_heads = len(head_indices)
    num_cols = batch_size * num_heads

    # Create figure with subplots
    fig, axes = plt.subplots(num_layers, num_cols, figsize=(5 * num_cols, 4 * num_layers), squeeze=False)

    for row_idx, l_idx in enumerate(layer_indices):
        attn_tensor_batch = attn_list[l_idx].detach().cpu()
        
        col_idx = 0
        for h_idx in head_indices:
            # Handle the Head dimension
            if h_idx is not None:
                attn_maps = attn_tensor_batch[:, h_idx, :, :]
                head_label = f"Head- {h_idx}"
            else:
                attn_maps = torch.mean(attn_tensor_batch, dim=1)
                head_label = "Avg Heads"

            # Plot for each batch sample
            for b_i in range(batch_size):
                ax = axes[row_idx, col_idx]
                map_data = attn_maps[b_i].numpy()

                # cmap: viridis | plasma
                im = ax.imshow(map_data, cmap="plasma", aspect="equal")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Labels
                if col_idx == 0:
                    ax.set_ylabel(f"$\mathbf{{Layer- {l_idx}}}$" "\nQuery")
                
                ax.set_xlabel("Key")
                ax.set_title(f"$\mathbf{{{head_label}}}$ (B{b_i})")

                col_idx += 1

    # Plot title
    head_str = f"AvgHeads+Heads{head_idx}" if head_idx is not None else "AvgHeads"
    main_title = f"{title_prefix} - {head_str}"
    plt.suptitle(main_title, y=1.0, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_attn_img_overlay(
    images,                       # PIL.Image | np.ndarray | torch.Tensor | list of them
    *,
    attn_global=None,             # [L,1,H,Ng,Ng]
    attn_local=None,              # [L,S,H,Nl,Nl]
    mode="global",                # "global" | "local"
    layer=0,
    head="mean",                  # "mean" | int
    query_img_idx=0,              # which image contains the query token
    query_local_idx=0,            # local token index within that image: [0..N_local)
    grid_h=37,
    grid_w=21,
    num_special=5,
    special_token_arrange="first",# "first" | "last"
    scale=11,                     # pixel per patch (like your app)
    overlay_alpha=0.5,            # overlay: alpha*img + (1-alpha)*heatmap
    query_color="#FF0000",        # query highlight rectangle color
    cmap="cividis",               # cividis | viridis | inferno | magma | jet
    show_titles=True,
    save_path="",

    # -------- truncate display to global topk --------
    topk=0,                       # 0 disables truncation
    topk_include_special=False,   # whether topk candidates include special tokens
    topk_show_mode="transparent", # "transparent" | "hard"

    # -------- show topk rank numbers --------
    topk_show_rank=False,         # draw rank numbers on topk tokens
    topk_rank_max=20,             # only annotate first K ranks to reduce clutter
):
    """
    Jupyter-display (matplotlib) visualization:
      - left: composite image (special bar + gap + image) with query highlight
      - right: composite attention overlay (special bar colored + heatmap overlay)
    """
    # -----------------------
    # 0) normalize images -> list[PIL.Image]
    # -----------------------
    if not isinstance(images, (list, tuple)):
        images = [images]

    pil_imgs = []
    for im in images:
        if isinstance(im, Image.Image):
            pil = im.convert("RGB")
        else:
            if isinstance(im, torch.Tensor):
                x = im.detach().cpu()
                if x.ndim == 4:
                    assert x.shape[0] == 1, "batch size > 1 not supported"
                    x = x[0]
                if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # C,H,W
                    x = x.permute(1, 2, 0)
                x = x.numpy()
            else:
                x = np.asarray(im)
                if x.ndim == 4:
                    assert x.shape[0] == 1, "batch size > 1 not supported"
                    x = x[0]

            if x.ndim == 2:
                x = np.stack([x, x, x], -1)
            if x.dtype.kind in ("u", "i"):
                x = x.astype(np.uint8)
            else:
                xx = x.astype(np.float32)
                if xx.max() <= 1.5:
                    xx = xx * 255.0
                xx = np.clip(xx, 0, 255).astype(np.uint8)
                x = xx
            pil = Image.fromarray(x).convert("RGB")

        pil_imgs.append(pil)

    S = len(pil_imgs)
    N_local = num_special + grid_h * grid_w

    # -----------------------
    # 1) pick the query row vector
    # -----------------------
    attn_row = None
    head_str = "Avg" if head == "mean" else f"H{int(head)}"

    if mode == "global":
        assert attn_global is not None
        assert attn_global.ndim == 5
        L, one, Hh, Ng, Ng2 = attn_global.shape
        assert one == 1 and Ng == Ng2
        assert Ng == S * N_local
        assert 0 <= layer < L
        assert 0 <= query_img_idx < S
        assert 0 <= query_local_idx < N_local

        q_global = query_img_idx * N_local + query_local_idx
        if head == "mean":
            attn_row = attn_global[layer, 0, :, q_global, :].float().mean(0)
        else:
            attn_row = attn_global[layer, 0, int(head), q_global, :].float()

    elif mode == "local":
        assert attn_local is not None
        assert attn_local.ndim == 5
        L, S2, Hh, Nl, Nl2 = attn_local.shape
        assert S2 == S
        assert Nl == Nl2 == N_local
        assert 0 <= layer < L
        assert 0 <= query_img_idx < S
        assert 0 <= query_local_idx < N_local

        if head == "mean":
            attn_row = attn_local[layer, query_img_idx, :, query_local_idx, :].float().mean(0)
        else:
            attn_row = attn_local[layer, query_img_idx, int(head), query_local_idx, :].float()
    else:
        raise ValueError("mode must be 'global' or 'local'")

    attn_row = attn_row.detach().cpu()
    vec_full = attn_row.numpy()

    # -----------------------
    # 1.5) compute global topk indices + rank map
    # -----------------------
    topk_idx_set = None
    topk_rank_map = {}  # idx -> rank (0-based)

    if topk and int(topk) > 0:
        vec = vec_full

        if not topk_include_special and num_special > 0:
            if mode == "global":
                mask = np.ones_like(vec, dtype=bool)
                for ii in range(S):
                    base = ii * N_local
                    if special_token_arrange == "first":
                        mask[base : base + num_special] = False
                    else:
                        mask[base + (N_local - num_special) : base + N_local] = False
                cand = np.where(mask)[0]
            else:
                if special_token_arrange == "first":
                    cand = np.arange(num_special, N_local)
                else:
                    cand = np.arange(0, N_local - num_special)
        else:
            cand = np.arange(vec.size)

        k = min(int(topk), cand.size)
        if k > 0:
            cand_vals = vec[cand]
            picked = cand[np.argpartition(-cand_vals, k - 1)[:k]]
            picked = picked[np.argsort(-vec[picked])]  # sort by score desc

            topk_idx_set = set(int(x) for x in picked)
            for rnk, idx in enumerate(picked):
                topk_rank_map[int(idx)] = rnk
        else:
            topk_idx_set = set()

    # -----------------------
    # 2) build & plot per-image composites
    # -----------------------
    patch_size = int(scale)
    special_bar_w = patch_size
    gap_w = int(patch_size * 0.5)
    display_w = grid_w * patch_size
    display_h = grid_h * patch_size
    total_w = special_bar_w + gap_w + display_w
    total_h = display_h

    fig, axes = plt.subplots(S, 2, figsize=(12, 3.6 * S))
    if S == 1:
        axes = np.array([axes])

    for i in range(S):
        # ---- slice per-image token vector
        if mode == "global":
            start = i * N_local
            v = attn_row[start:start + N_local]
            base_global = start
        else:
            if i != query_img_idx:
                v = None
                base_global = None
            else:
                v = attn_row
                base_global = 0  # local vector indices

        # ---- LEFT
        img_disp = pil_imgs[i].resize((display_w, display_h), Image.Resampling.LANCZOS)
        comp_left = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        img_offset_x = special_bar_w + gap_w
        comp_left.paste(img_disp, (img_offset_x, 0))
        draw = ImageDraw.Draw(comp_left)

        for t in range(num_special):
            y0 = t * patch_size
            y1 = y0 + patch_size
            draw.rectangle([0, y0, special_bar_w, y1], fill="white", outline=(200, 200, 200))

        for x in range(grid_w + 1):
            px = x * patch_size + img_offset_x
            px = min(px, total_w - 1)
            draw.line([(px, 0), (px, display_h)], fill=(200, 200, 200), width=1)
        for y in range(grid_h + 1):
            py = y * patch_size
            py = min(py, display_h - 1)
            draw.line([(img_offset_x, py), (total_w, py)], fill=(200, 200, 200), width=1)

        if i == query_img_idx:
            if query_local_idx < num_special:
                y0 = query_local_idx * patch_size
                y1 = y0 + patch_size
                draw.rectangle([0, y0, special_bar_w, y1], outline=query_color, width=3)
            else:
                spatial_idx = query_local_idx - num_special
                r = spatial_idx // grid_w
                c = spatial_idx % grid_w
                x0 = c * patch_size + img_offset_x
                y0 = r * patch_size
                x1 = x0 + patch_size
                y1 = y0 + patch_size
                draw.rectangle([x0, y0, x1, y1], outline=query_color, width=3)

        axL = axes[i, 0]
        axL.imshow(comp_left)
        axL.axis("off")
        if show_titles:
            axL.set_title(f"Image {i}" + (" (QUERY)" if i == query_img_idx else ""))

        # ---- RIGHT
        axR = axes[i, 1]
        axR.axis("off")

        if v is None:
            axR.text(0.02, 0.5, "Local attn: no cross-image attention", fontsize=12, va="center", ha="left")
            if show_titles:
                axR.set_title(f"{mode.upper()} Attn {i}")
            continue

        v_np = v.numpy()

        if special_token_arrange == "first":
            attn_special = v_np[:num_special]
            attn_spatial = v_np[num_special:].reshape(grid_h, grid_w)
        else:
            attn_special = v_np[-num_special:]
            attn_spatial = v_np[:-num_special].reshape(grid_h, grid_w)

        vmax = float(v_np.max())
        vmin = float(v_np.min())

        if vmax > 0:
            spatial_norm = attn_spatial / vmax
            special_norm = attn_special / vmax
        else:
            spatial_norm = attn_spatial
            special_norm = attn_special

        # ---- truncate to topk (FIX: include special handling)
        annotate_rc_to_rank = {}       # (r,c) -> rank (1-based)
        annotate_special_to_rank = {}  # special_idx (0..num_special-1) -> rank (1-based)

        if topk_idx_set is not None:
            keep_spatial = np.zeros((grid_h, grid_w), dtype=bool)
            keep_special = np.zeros((num_special,), dtype=bool) if num_special > 0 else None

            if mode == "global":
                for gidx in topk_idx_set:
                    if base_global <= gidx < base_global + N_local:
                        lidx = gidx - base_global  # local token index in this image

                        # Map local token to special/spatial index based on arrange
                        if special_token_arrange == "first":
                            if lidx < num_special:
                                s = int(lidx)
                                keep_special[s] = True
                                if topk_show_rank and gidx in topk_rank_map:
                                    rank0 = topk_rank_map[gidx]
                                    if rank0 < int(topk_rank_max):
                                        annotate_special_to_rank[s] = rank0 + 1
                                continue
                            sidx = lidx - num_special
                        else:
                            if lidx >= N_local - num_special:
                                s = int(lidx - (N_local - num_special))
                                keep_special[s] = True
                                if topk_show_rank and gidx in topk_rank_map:
                                    rank0 = topk_rank_map[gidx]
                                    if rank0 < int(topk_rank_max):
                                        annotate_special_to_rank[s] = rank0 + 1
                                continue
                            sidx = lidx

                        r = int(sidx // grid_w)
                        c = int(sidx % grid_w)
                        keep_spatial[r, c] = True
                        if topk_show_rank and gidx in topk_rank_map:
                            rank0 = topk_rank_map[gidx]
                            if rank0 < int(topk_rank_max):
                                annotate_rc_to_rank[(r, c)] = rank0 + 1

            else:
                # local indices
                for lidx in topk_idx_set:
                    lidx = int(lidx)

                    if special_token_arrange == "first":
                        if lidx < num_special:
                            s = lidx
                            keep_special[s] = True
                            if topk_show_rank and lidx in topk_rank_map:
                                rank0 = topk_rank_map[lidx]
                                if rank0 < int(topk_rank_max):
                                    annotate_special_to_rank[s] = rank0 + 1
                            continue
                        sidx = lidx - num_special
                    else:
                        if lidx >= N_local - num_special:
                            s = int(lidx - (N_local - num_special))
                            keep_special[s] = True
                            if topk_show_rank and lidx in topk_rank_map:
                                rank0 = topk_rank_map[lidx]
                                if rank0 < int(topk_rank_max):
                                    annotate_special_to_rank[s] = rank0 + 1
                            continue
                        sidx = lidx

                    r = int(sidx // grid_w)
                    c = int(sidx % grid_w)
                    keep_spatial[r, c] = True
                    if topk_show_rank and lidx in topk_rank_map:
                        rank0 = topk_rank_map[lidx]
                        if rank0 < int(topk_rank_max):
                            annotate_rc_to_rank[(r, c)] = rank0 + 1

            # Apply truncation
            spatial_norm = spatial_norm.copy()
            spatial_norm[~keep_spatial] = 0.0

            # Also truncate special bar if we are including specials
            if topk_include_special and keep_special is not None:
                special_norm = special_norm.copy()
                special_norm[~keep_special] = 0.0

        # ---- heatmap + overlay
        heat = cv2.resize(spatial_norm, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        heat_rgb = plt.get_cmap(cmap)(heat)[:, :, :3]

        img_np = np.asarray(img_disp).astype(np.float32) / 255.0

        if topk_idx_set is not None and topk_show_mode == "transparent":
            mask = (heat > 0).astype(np.float32)[..., None]
            overlay = img_np * (1.0 - mask) + (overlay_alpha * img_np + (1.0 - overlay_alpha) * heat_rgb) * mask
        else:
            overlay = overlay_alpha * img_np + (1.0 - overlay_alpha) * heat_rgb

        overlay = np.clip(overlay, 0, 1)

        # ---- composite right
        comp_right = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw2 = ImageDraw.Draw(comp_right)

        for t in range(num_special):
            score = float(np.clip(special_norm[t], 0.0, 1.0))
            rgb = plt.get_cmap(cmap)(score)[:3]
            rgb255 = tuple(int(c * 255) for c in rgb)
            y0 = t * patch_size
            y1 = y0 + patch_size
            draw2.rectangle([0, y0, special_bar_w, y1], fill=rgb255, outline=(200, 200, 200))

            # FIX: draw rank number on special bar cell
            if topk_show_rank and (t in annotate_special_to_rank):
                lab = str(annotate_special_to_rank[t])
                tx = 2
                ty = y0 + 1
                # simple stroke: black offsets then white
                draw2.text((tx-1, ty), lab, fill=(0, 0, 0))
                draw2.text((tx+1, ty), lab, fill=(0, 0, 0))
                draw2.text((tx, ty-1), lab, fill=(0, 0, 0))
                draw2.text((tx, ty+1), lab, fill=(0, 0, 0))
                draw2.text((tx, ty),   lab, fill=(255, 255, 255))

        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

        # draw rank numbers on overlay image (spatial)
        if topk_show_rank and len(annotate_rc_to_rank) > 0:
            d = ImageDraw.Draw(overlay_img)
            for (r, c), lab in annotate_rc_to_rank.items():
                x = c * patch_size + 2
                y = r * patch_size + 1
                s = str(lab)
                d.text((x-1, y), s, fill=(0, 0, 0))
                d.text((x+1, y), s, fill=(0, 0, 0))
                d.text((x, y-1), s, fill=(0, 0, 0))
                d.text((x, y+1), s, fill=(0, 0, 0))
                d.text((x, y),   s, fill=(255, 255, 255))

        comp_right.paste(overlay_img, (img_offset_x, 0))

        im = axR.imshow(comp_right)

        # colorbar: if truncated, use [0, max(visible)]
        if topk_idx_set is not None:
            cb_vmin = 0.0
            cb_vmax = float(heat.max()) if float(heat.max()) > 0 else 1.0
        else:
            cb_vmin = vmin
            cb_vmax = vmax if vmax > vmin else (vmin + 1e-6)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cb_vmin, vmax=cb_vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=axR, fraction=0.046, pad=0.04)

        if show_titles:
            extra = f" | topk={topk}" if topk_idx_set is not None else ""
            axR.set_title(f"{mode.upper()} Attn {i} | L{layer}, {head_str} | max={vmax:.4f}, min={vmin:.4f}{extra}")

    if show_titles:
        fig.suptitle(
            f"{mode.upper()} overlay | query=(img {query_img_idx}, local {query_local_idx}) | "
            f"N_local={N_local} (special={num_special}, grid={grid_h}x{grid_w})"
            + (f" | topk={topk}" if topk_idx_set is not None else ""),
            y=0.94
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_attn_distribution(
    attn_probs,
    *,
    cur_idx=None,
    sort_desc=False,
    mode="pdf",               # "pdf" | "cdf" | "hist"
    tokens_per_imgs=None,
    topk=0,
    show_grid=True,
    draw_img_separators_when_sorted=False,
    save_path="",
):

    if isinstance(attn_probs, torch.Tensor):
        p = attn_probs.detach().cpu().numpy().reshape(-1)
    else:
        p = np.asarray(attn_probs).reshape(-1)

    if sort_desc:
        p_plot = np.sort(p)[::-1]
    else:
        p_plot = p

    N = p_plot.size
    x = np.arange(N)

    fig, ax = plt.subplots(figsize=(9, 3))
    mode = mode.lower()

    # =========================================================
    # PDF (index domain)
    # =========================================================
    if mode == "pdf":
        ax.plot(x, p_plot, linewidth=1.5)
        ax.set_ylabel("P")
        ax.set_xlabel("Token index" + (" (sorted)" if sort_desc else ""))

        if tokens_per_imgs is not None and (not sort_desc or draw_img_separators_when_sorted):
            tpi = int(tokens_per_imgs)
            for c in range(tpi, N, tpi):
                ax.axvline(
                    c - 0.5, linestyle="--", linewidth=1.0, alpha=0.7, color="gray"
                )


        ymin, ymax = ax.get_ylim()
        yrng = ymax - ymin
        marker_y = ymin + 0.02 * yrng
        if topk > 0:
            idxs = np.argpartition(-p_plot, topk - 1)[:topk]
            idxs = np.sort(idxs)

            ax.scatter(
                idxs,
                np.full_like(idxs, marker_y, dtype=float),
                marker="v",
                s=15,
                c="red",
                zorder=6,
                label=f"Top-{topk} Key Tokens",
            )
            ax.legend(loc="upper right", fontsize=8)

        if cur_idx is not None:
            ax.scatter(
                [cur_idx],
                [marker_y],
                marker="v",
                s=15,
                c="blue",
                zorder=7,
                label="Query Patch",
            )
            ax.legend(loc="upper right", fontsize=8)

    # =========================================================
    # CDF (index domain)
    # =========================================================
    elif mode == "cdf":
        cdf = np.cumsum(p_plot)
        ax.plot(x, cdf, linewidth=1.8)
        ax.set_ylabel("CDF mass")
        ax.set_xlabel("Token index" + (" (sorted)" if sort_desc else ""))

        if topk > 0:
            idx = topk - 1
            mass = float(cdf[idx])
            ax.axvline(idx, linestyle="--", linewidth=1.2, alpha=0.7, color="gray")
            ax.annotate(
                f"topk={topk}\nmass={mass*100:.2f}%",
                xy=(idx, mass),
                xytext=(idx + max(5, int(N * 0.02)), mass),
                ha="left",
                va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

    # =========================================================
    # HISTOGRAM (value domain)
    # =========================================================
    elif mode == "hist":
        bins = 50
        ax.hist(p_plot, bins=bins, alpha=0.7, edgecolor="none")
        ax.set_xlabel("Scores")
        ax.set_ylabel("Count")

        if topk > 0:
            topk_vals = np.partition(p_plot, -topk)[-topk:]

            ymin, ymax = ax.get_ylim()
            yrng = ymax - ymin
            marker_y = ymin - 0.03 * yrng
            ax.set_ylim(ymin - 0.12 * yrng, ymax)

            ax.scatter(
                topk_vals,
                np.full_like(topk_vals, marker_y),
                marker="v",
                s=35,
                c="red",
                zorder=6,
                label=f"Top-{topk} Values",
                clip_on=False,
            )
            ax.legend(loc="upper right")

    else:
        raise ValueError("mode must be 'pdf', 'cdf', or 'hist'")

    if show_grid:
        ax.grid(alpha=0.25)

    ax.set_title(f"Attention {mode.upper()}" + (" (sorted)" if sort_desc else ""))

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def plot_full_attn_map(
    attn_map,
    *,
    percentile=100,
    cmap="magma",
    tokens_per_imgs=None,
    grid_line_color="gray",
    cur_token=None,
    cur_token_line_color="cyan",
    normalize=False,         # whether to apply PowerNorm
    gamma=0.3,               # PowerNorm gamma (if normalize=True)
    subtitle="Attention Map",
    save_path="",
):
    """
    imshow attention map
    attn_map: [N, N] numpy array or torch tensor
    """
    # to numpy
    if hasattr(attn_map, "detach"):
        attn = attn_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attn_map)

    assert attn.ndim == 2 and attn.shape[0] == attn.shape[1], "attn_map must be [N,N]"
    N = attn.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if normalize:
        from matplotlib import colors
        # γ < 1 → amplifies differences among small values (enhances contrast)
        # γ > 1 → compresses small values and emphasizes large values
        # γ = 1 → equivalent to linear scaling / normal normalization
        norm = colors.PowerNorm(gamma=gamma)
        im = ax.imshow(attn, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{subtitle} (PowerNorm γ={gamma})")
    else:
        vmax = np.percentile(attn, percentile)
        im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{subtitle} (vmax at {percentile:.1f} percentile)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    if tokens_per_imgs is not None:
        tpi = int(tokens_per_imgs)
        for c in range(tpi, N, tpi):
            ax.axvline(c - 0.5, linestyle="--", color=grid_line_color, linewidth=0.5, alpha=0.8)
            ax.axhline(c - 0.5, linestyle="--", color=grid_line_color, linewidth=0.5, alpha=0.8)

    if cur_token is not None:
        ax.axhline(cur_token - 0.5, color=cur_token_line_color, linewidth=1.0, alpha=0.95)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_multiheads_full_attn_map(
    attn_map,
    *,
    ncols: int = 4,
    percentile: float = 100,
    cmap: str = "magma",
    tokens_per_imgs: int | None = None,
    grid_line_color: str = "gray",
    cur_token: int | None = None,
    cur_token_line_color: str = "cyan",
    normalize: bool = False,
    gamma: float = 0.3,
    head_titles: list[str] | None = None,
    figsize_per_cell: float = 4.0,
    subtitle="Attention Maps Across All Heads",
    save_path: str = "",
    dpi: int = 200,
):
    """
    Plot attention maps in a grid.

    Args:
        attn_map: must be [H, N, N]
        ncols: number of columns in grid
        percentile: vmax percentile (ignored if normalize=True)
        normalize: if True, use PowerNorm(gamma)
        tokens_per_imgs: draw dashed grid lines every tokens_per_img
        cur_token: highlight query token row (horizontal line)
        head_titles: optional titles per head (len==H)
        save_path: if provided, save figure and close
    """
    # ---------- to numpy ----------
    if hasattr(attn_map, "detach"):
        attn = attn_map.detach().float().cpu().numpy()
    else:
        attn = np.asarray(attn_map)

    # ---------- strict shape check ----------
    if attn.ndim != 3:
        raise ValueError(f"Expected attn_map shape [H, N, N], but got {attn.shape}")

    H, N, N2 = attn.shape
    if N != N2:
        raise ValueError(f"attn_map must be square in last 2 dims, got {attn.shape}")

    # optional: validate cur_token range early
    if cur_token is not None and not (0 <= int(cur_token) < N):
        raise ValueError(f"cur_token out of range: cur_token={cur_token}, valid=[0, {N-1}]")

    # ---------- layout ----------
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(H / ncols))
    figsize = (figsize_per_cell * ncols, figsize_per_cell * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # shared vmax across heads (better for comparing heads)
    if not normalize:
        vmax = np.percentile(attn, percentile) if percentile < 100 else float(attn.max())
        vmin = 0.0
        norm_obj = None
    else:
        from matplotlib import colors
        vmin, vmax = None, None
        norm_obj = colors.PowerNorm(gamma=gamma)

    # ---------- plot ----------
    last_im = None
    for h in range(nrows * ncols):
        ax = axes[h // ncols][h % ncols]
        if h >= H:
            ax.axis("off")
            continue

        m = attn[h]
        if normalize:
            im = ax.imshow(m, cmap=cmap, norm=norm_obj, interpolation="nearest")
        else:
            im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        last_im = im

        # title
        if head_titles is not None and h < len(head_titles):
            ax.set_title(head_titles[h])
        else:
            ax.set_title(f"Head {h}")

        # grid lines for frames/images
        if tokens_per_imgs is not None:
            tpi = int(tokens_per_imgs)
            if tpi <= 0:
                raise ValueError(f"tokens_per_img must be positive, got {tokens_per_imgs}")
            for c in range(tpi, N, tpi):
                ax.axvline(c - 0.5, linestyle="--", color=grid_line_color, linewidth=0.5, alpha=0.8)
                ax.axhline(c - 0.5, linestyle="--", color=grid_line_color, linewidth=0.5, alpha=0.8)

        # highlight current token row
        if cur_token is not None:
            ax.axhline(int(cur_token) - 0.5, color=cur_token_line_color, linewidth=1.0, alpha=0.95)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        subtitle + f" (PowerNorm γ={gamma})" if normalize else f"(vmax at {percentile:.1f} percentile)",
        fontsize=18,
        y=1.0
    )
    plt.tight_layout()

    # ---------- colorbar (one for all) ----------
    if last_im is not None:
        # [left, bottom, width, height] in figure coordinates
        # bottom controls the distance between the colorbar and the bottom of the figure
        # smaller bottom values place the colorbar closer to the bottom
        # height controls the thickness of the colorbar
        # left/width control the horizontal margins and length
        cax = fig.add_axes([0.15, -0.02, 0.70, 0.01])  # manual tuning needed 
        fig.colorbar(last_im, cax=cax, orientation="horizontal")

    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()




###############################################################################
# Plotly for 3D mesh/pcd visualization
###############################################################################
# =========================
# World axes
# =========================
def _make_axes_traces(
    axis_len=1.0,
    line_width=6,
    origin_color="white",
    show_text=False,
):
    """
    World axes: fixed RGB colors + optional arrowheads via Cone (not implemented here).
    """
    _AXIS_RGB = {
        "X": "#ff0000",  # red
        "Y": "#00ff00",  # green
        "Z": "#0000ff",  # blue
    }

    traces = []

    axes = [
        ("X", [0, axis_len], [0, 0],        [0, 0]),
        ("Y", [0, 0],        [0, axis_len], [0, 0]),
        ("Z", [0, 0],        [0, 0],        [0, axis_len]),
    ]

    for name, xs, ys, zs in axes:
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=line_width, color=_AXIS_RGB[name]),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ))

    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text" if show_text else "markers",
        text=["Origin"] if show_text else None,
        textposition="top center",
        marker=dict(size=6, color=origin_color),
        name="Origin",
        showlegend=False,
        hoverinfo="skip",
    ))

    return traces


# =========================
# Camera frustum (single color per camera)
# =========================
def _transform_points_c2w(P_cam, c2w):
    """
    P_cam: (N,3) points in camera coordinates
    c2w: (4,4) camera-to-world
    """
    P_cam = np.asarray(P_cam, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    P_h = np.concatenate([P_cam, np.ones((len(P_cam), 1), dtype=np.float32)], axis=1)  # (N,4)
    Pw_h = (c2w @ P_h.T).T
    return Pw_h[:, :3]


def make_camera_frustum_traces(
    K, W, H, c2w,
    z=1.0,
    name="cam",
    line_width=2,
    color="#ffffff",
    showlegend=True,         # controls whether THIS camera contributes a legend item (only used on first edge)
    legendgroup=None,        # "Pred" or "GT" so legend toggles the whole group
    legend_item_name=None,   # what to show in legend (e.g., "Pred"/"GT"); if None uses `name`
    *,
    show_name_text=True,
):
    """
    K: (3,3) intrinsics
    W,H: image width/height
    c2w: (4,4) camera-to-world pose
    z: frustum depth in camera coords (scale of the pyramid)
    color: single color used for all parts of THIS camera

    Legend grouping:
      - legendgroup: string; all traces from this camera will share it
        so clicking legend toggles the whole group when traces share
        the same legendgroup value (e.g., "Pred" / "GT").
    """
    K = np.asarray(K, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    corners_px = np.array(
        [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
        dtype=np.float32,
    )

    # OpenCV-like pinhole backprojection (Z forward)
    corners_cam = np.stack(
        [
            (corners_px[:, 0] - cx) / fx * z,
            (corners_px[:, 1] - cy) / fy * z,
            np.full(4, z, dtype=np.float32),
        ],
        axis=1,
    )

    origin_cam = np.array([[0, 0, 0]], dtype=np.float32)
    axis_cam = np.array([[0, 0, z * 1.2]], dtype=np.float32)

    corners_w = _transform_points_c2w(corners_cam, c2w)
    origin_w = _transform_points_c2w(origin_cam, c2w)[0]
    axis_w = _transform_points_c2w(axis_cam, c2w)[0]

    traces = []

    # Pyramid edges: origin -> each corner
    for i in range(4):
        traces.append(
            go.Scatter3d(
                x=[origin_w[0], corners_w[i, 0]],
                y=[origin_w[1], corners_w[i, 1]],
                z=[origin_w[2], corners_w[i, 2]],
                mode="lines",
                line=dict(width=line_width, color=color),
                name=(legend_item_name if legend_item_name is not None else name) if (i == 0 and showlegend) else None,
                showlegend=(i == 0 and showlegend),
                legendgroup=legendgroup,
                hoverinfo="skip",
            )
        )

    # Image plane rectangle
    rect = [0, 1, 2, 3, 0]
    traces.append(
        go.Scatter3d(
            x=corners_w[rect, 0],
            y=corners_w[rect, 1],
            z=corners_w[rect, 2],
            mode="lines",
            line=dict(width=line_width, color=color),
            name=None,
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    # Optical axis
    traces.append(
        go.Scatter3d(
            x=[origin_w[0], axis_w[0]],
            y=[origin_w[1], axis_w[1]],
            z=[origin_w[2], axis_w[2]],
            mode="lines",
            line=dict(width=line_width, color=color),
            name=None,
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    # Camera center marker (+ optional text)
    traces.append(
        go.Scatter3d(
            x=[origin_w[0]],
            y=[origin_w[1]],
            z=[origin_w[2]],
            mode=("markers+text" if show_name_text else "markers"),
            text=[name] if show_name_text else None,
            textposition="top center",
            marker=dict(size=line_width, color=color),
            name=None,
            showlegend=False,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )

    return traces


# =========================
# Geometry rendering (mesh / point cloud) + custom color
# =========================
def _extract_points_for_bbox(geom):
    """
    Returns (N,3) float32 points for bbox estimation, or None if unsupported.
    """
    if isinstance(geom, np.ndarray):
        pts = np.asarray(geom, dtype=np.float32)
        return pts if pts.ndim == 2 and pts.shape[1] == 3 else None

    try:
        import open3d as o3d
    except Exception:
        o3d = None

    if o3d is not None:
        if isinstance(geom, o3d.geometry.PointCloud):
            return np.asarray(geom.points, dtype=np.float32)
        if isinstance(geom, o3d.geometry.TriangleMesh):
            return np.asarray(geom.vertices, dtype=np.float32)

    return None


def _auto_axis_len_from_geom(geom, default=1.0, scale=0.3):
    """
    Convert (N,3) colors to Plotly "rgb(r,g,b)" strings.

    colors: (N,3) float in [0,1] or uint8 in [0,255]
    returns: list[str] like ["rgb(r,g,b)", ...] for Plotly
    """
    pts = _extract_points_for_bbox(geom)
    if pts is None or len(pts) == 0:
        return default
    vmin, vmax = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    return (scale * diag) if diag > 0 else default


def _maybe_downsample_points(pts, max_points=200_000, seed=0):
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    if n <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return pts[idx]


def _colors_to_plotly_rgb(colors) -> List[str]:
    c = np.asarray(colors)
    if c.dtype != np.uint8:
        c = np.clip(c, 0.0, 1.0)
        c = (c * 255.0).round().astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in c]


def geom_to_traces(
    geom,
    *,
    geom_name="geom",
    geom_color=None,
    geom_opacity=1.0,
    max_points=200_000,
    point_size=1,
    point_color=None,
    point_colors=None,
    mesh_color=None,
    use_vertex_colors_if_available=True,
    seed=0,
    showlegend=False,
    legendgroup=None,
):
    """
    Create Plotly traces for:
      - open3d.geometry.TriangleMesh  -> Mesh3d
      - open3d.geometry.PointCloud   -> Scatter3d
      - np.ndarray (N,3) points      -> Scatter3d
    """
    if geom_color is not None:
        point_color = point_color if point_color is not None else geom_color
        mesh_color = mesh_color if mesh_color is not None else geom_color

    # --- numpy points ---
    if isinstance(geom, np.ndarray):
        pts = _maybe_downsample_points(geom, max_points=max_points, seed=seed)
        marker = dict(size=point_size)
        if point_colors is not None:
            pc = np.asarray(point_colors)
            if len(pc) != len(geom):
                raise ValueError("point_colors length must match original points length")
            pc = _maybe_downsample_points(pc, max_points=max_points, seed=seed)
            marker["color"] = _colors_to_plotly_rgb(pc)
        elif point_color is not None:
            marker["color"] = point_color

        tr = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=marker,
            name=geom_name,
            showlegend=bool(showlegend),
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
        return [tr]

    # --- open3d geometry ---
    try:
        import open3d as o3d
    except Exception as e:
        raise TypeError("Open3D geometry provided but open3d is not importable.") from e

    # PointCloud
    if isinstance(geom, o3d.geometry.PointCloud):
        pts0 = np.asarray(geom.points, dtype=np.float32)
        pts = _maybe_downsample_points(pts0, max_points=max_points, seed=seed)

        marker = dict(size=point_size)
        if point_colors is not None:
            pc = np.asarray(point_colors)
            if len(pc) != len(pts0):
                raise ValueError("point_colors length must match pcd.points length")
            pc = _maybe_downsample_points(pc, max_points=max_points, seed=seed)
            marker["color"] = _colors_to_plotly_rgb(pc)
        elif use_vertex_colors_if_available and geom.has_colors():
            pc = np.asarray(geom.colors)
            pc = _maybe_downsample_points(pc, max_points=max_points, seed=seed)
            marker["color"] = _colors_to_plotly_rgb(pc)
        elif point_color is not None:
            marker["color"] = point_color

        tr = go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=marker,
            name=geom_name,
            showlegend=bool(showlegend),
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
        return [tr]

    # TriangleMesh
    if isinstance(geom, o3d.geometry.TriangleMesh):
        v = np.asarray(geom.vertices, dtype=np.float32)
        f = np.asarray(geom.triangles, dtype=np.int32)

        kwargs = dict(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            opacity=float(geom_opacity),
            name=geom_name,
            hoverinfo="skip",
            showlegend=bool(showlegend),
            legendgroup=legendgroup,
        )
        if use_vertex_colors_if_available and geom.has_vertex_colors():
            vc = np.asarray(geom.vertex_colors)
            kwargs["vertexcolor"] = _colors_to_plotly_rgb(vc)
        elif mesh_color is not None:
            kwargs["color"] = mesh_color

        return [go.Mesh3d(**kwargs)]

    raise TypeError(f"Unsupported geometry type: {type(geom)}")


# =========================
# Main: geometry + world axes + cameras (Pred vs GT group colors + legend on + group toggle)
# =========================
def plot_geom_with_axes_and_cameras(
    geom_pred=None,
    cameras_pred=None,
    geom_gt=None,
    cameras_gt=None,
    *,
    axis_len=None,
    pred_geom_name="Pred Geometry",
    gt_geom_name="GT Geometry",
    # Point cloud appearance
    pred_geom_color="#FFD700",   # yellow
    gt_geom_color="#1f77b4",     # blue
    pred_geom_opacity=1.0,
    gt_geom_opacity=1.0,
    max_points=200_000,
    pred_point_size=1,
    gt_point_size=1,
    # If you have per-point colors, pass them here (optional)
    pred_point_colors=None,
    gt_point_colors=None,
    # Mesh options (if you pass meshes)
    use_vertex_colors_if_available=True,
    pred_mesh_color=None,
    gt_mesh_color=None,
    # Camera appearance (same scheme as before)
    pred_cam_color="#FFD700",
    gt_cam_color="#1f77b4",
    pred_cmap_name=None,
    gt_cmap_name=None,
    camera_line_width=4,
    camera_z_ratio=0.2,
    show_camera_name_text=False,
    # Legend behavior
    show_geom_legend=False,          # keep off by default (avoid legend spam)
    legend_groupclick="togglegroup", # for camera group toggle,
    seed=0,
):
    """
    cameras_pred / cameras_gt: list of dict, each contains:
      - K, W, H, c2w, z(optional), name(optional)

    Legend:
      - Shows exactly two legend items: "Pred" and "GT"
      - Clicking "Pred"/"GT" toggles visibility of the entire group
    Coloring:
      - All pred cameras -> pred_color
      - All gt cameras   -> gt_color
    """
    data = []

    # Geometry traces (GT first, then Pred so pred sits on top visually)
    # GT geometry
    if geom_gt is not None:
        data += geom_to_traces(
            geom_gt,
            geom_name=gt_geom_name,
            geom_color=gt_geom_color,
            geom_opacity=gt_geom_opacity,
            max_points=max_points,
            point_size=gt_point_size,
            point_colors=gt_point_colors,
            mesh_color=gt_mesh_color,
            use_vertex_colors_if_available=use_vertex_colors_if_available,
            seed=seed,
            showlegend=show_geom_legend,
            legendgroup="GT_GEOM" if show_geom_legend else None,
        )
    # Pred geometry
    if geom_pred is not None:
        data += geom_to_traces(
            geom_pred,
            geom_name=pred_geom_name,
            geom_color=pred_geom_color,
            geom_opacity=pred_geom_opacity,
            max_points=max_points,
            point_size=pred_point_size,
            point_colors=pred_point_colors,
            mesh_color=pred_mesh_color,
            use_vertex_colors_if_available=use_vertex_colors_if_available,
            seed=seed + 1,  # different seed so downsample patterns differ slightly (optional)
            showlegend=show_geom_legend,
            legendgroup="PRED_GEOM" if show_geom_legend else None,
        )

    # Auto axis length
    if axis_len is None:
        axis_len = _auto_axis_len_from_geom(geom_pred, default=1.0, scale=0.2)

    # NOTE: turn on/off world axes
    # data += _make_axes_traces(axis_len=axis_len)

    cameras_gt = cameras_gt or []

    # --- Pred group ---
    num_pred = len(cameras_pred) if cameras_pred is not None else 0
    for i, cam in enumerate(cameras_pred or []):
        # Only the FIRST pred camera contributes a legend entry named "Pred"
        showlegend = (i == 0)

        cam_color = pred_cam_color
        if pred_cmap_name is not None:
            try:
                import matplotlib
                cmap = matplotlib.colormaps[pred_cmap_name]
            except Exception:
                cmap = cm.get_cmap(pred_cmap_name)
            val = i / max(1, num_pred - 1) if num_pred > 1 else 0.5
            cam_color = colors.to_hex(cmap(val))

        data += make_camera_frustum_traces(
            K=cam["K"], W=cam["W"], H=cam["H"],
            c2w=cam["c2w"],
            z=cam.get("z", axis_len * camera_z_ratio),
            name=cam.get("name", f"pred{i}"),
            color=cam_color,
            line_width=camera_line_width,
            showlegend=showlegend,
            legendgroup="Pred",
            legend_item_name="Pred Cams",
            show_name_text=show_camera_name_text,
        )

    # --- GT group ---
    num_gt = len(cameras_gt)
    for i, cam in enumerate(cameras_gt):
        showlegend = (i == 0)

        cam_color = gt_cam_color
        if gt_cmap_name is not None:
            try:
                import matplotlib
                cmap = matplotlib.colormaps[gt_cmap_name]
            except Exception:
                cmap = cm.get_cmap(gt_cmap_name)
            val = i / max(1, num_gt - 1) if num_gt > 1 else 0.5
            cam_color = colors.to_hex(cmap(val))

        data += make_camera_frustum_traces(
            K=cam["K"], W=cam["W"], H=cam["H"],
            c2w=cam["c2w"],
            z=cam.get("z", axis_len * camera_z_ratio),
            name=cam.get("name", f"gt{i}"),
            color=cam_color,
            line_width=camera_line_width,
            showlegend=showlegend,
            legendgroup="GT",
            legend_item_name="GT Cams",
            show_name_text=show_camera_name_text,
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            # NOTE: the following lines is used to turn on/off background display
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='', ticks=''),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='', ticks=''),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, title='', ticks=''),
            bgcolor="white"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.02, y=0.98, groupclick=legend_groupclick),
        scene_camera=dict(
            eye=dict(x=0, y=0, z=-1.2),
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
        )
    )
    return fig
