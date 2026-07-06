import os
import os.path as osp
import argparse
from datetime import datetime
from addict import Dict

import torch
from saf3r.utils.model_utils import build_model, infer_model
from saf3r.utils.vis_utils import plot_html, save_ply


# Pre-defined inference configurations for different base models
VGGT_CFG = Dict(
    # base model configs
    name="VGGT", dpt_only=True, ckpt_path="checkpoints/vggt/model_tracker_fixed_e20.pt",
    
    # SAF3R patching configs
    sparse_config_path = "configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json",
    patch_module = Dict(
        type = "headsparse",
        topk_mode = "token",
        lazy_dino_topk = True,
        topk = 4
    )
)

Pi3_CFG = Dict(
    # base model configs
    name="Pi3", ckpt_path="checkpoints/pi3",
    
    # SAF3R patching configs
    sparse_config_path = "configs/sparse_attn/pi3/eth3d-train-fltr-calib_cmpmse.json",
    patch_module = Dict(
        type = "headsparse",
        topk_mode = "token",
        lazy_dino_topk = True,
        topk = 4
    )
)

DA3_CFG = Dict(
    # base model configs
    name="DA3", ckpt_path="checkpoints/da3",
    
    # SAF3R patching configs
    sparse_config_path = "configs/sparse_attn/da3/adjusted_eth3d-train-fltr-calib_cmpmse.json",
    patch_module = Dict(
        type = "headsparse",
        topk_mode = "token",
        lazy_dino_topk = True,
        topk = 4
    )
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup test scene data
    scene_data = Dict(
        image_files = [
            osp.join(args.data_dir, img_name) for img_name in sorted(os.listdir(args.data_dir))
        ]
    )
    print("Number of images found:", len(scene_data.image_files))

    # Build model
    model_name = args.model.lower()
    print("Using base model:", model_name)

    model_cfg = {
        "vggt": VGGT_CFG, "pi3": Pi3_CFG, "da3": DA3_CFG
    }[model_name]
    model = build_model(model_cfg, device)

    # Run inference
    pred_data, stats = infer_model(model, model_cfg, scene_data)

    # Plot results (camera & points) as a HTML file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = f"tmp_plots/saf3r_{model_name}_{ts}.html"
    plot_html(scene_data, pred_data, html_path, keep_ratio=args.keep_ratio, cmap_name="rainbow")

    # Save the point clouds as a PLY file
    ply_path = f"tmp_plots/saf3r_{model_name}_{ts}.ply"
    save_ply(pred_data, ply_path, keep_ratio=args.keep_ratio-0.2, max_points=200000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAF3R inference demo.")
    parser.add_argument(
        "--model", type=str, default="vggt", choices=["vggt", "pi3", "da3"],
        help="Base model to use for inference"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/courthouse",
        help="Root directory containing the input images"
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=0.7,
        help="Ratio of points to keep for visualization and point cloud export"
    )
    args = parser.parse_args()
    main(args)
