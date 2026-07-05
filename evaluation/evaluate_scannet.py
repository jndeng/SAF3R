"""
Evaluate F3R models on the ScanNet dataset for camera pose estimation and point-cloud reconstruction.
"""
import os.path as osp
import argparse
import random
from tqdm import tqdm
from addict import Dict
from hydra import initialize, compose

import numpy as np
import torch

from benchmarks.vggt.fastvggt_utils import (
    get_all_scenes, load_poses, get_sorted_image_paths, build_frame_selection,
    to_homogeneous, evaluate_scene_and_save, compute_average_metrics_and_save,
)
from benchmarks.vggt.geometry import unproject_depth_map_to_point_map
from saf3r.utils.model_utils import build_model, infer_model


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruct_world_points_from_depth(pred_data, depth_conf_thresh):
    """
    Reconstruct world points from depth maps and camera parameters.

    NOTE: modified from eval_utils.infer_vggt_and_reconstruct() with model inference part removed.
    """
    
    proc_imgs = pred_data.processed_images # [S, H, W, 3]
    depth_np = pred_data.depth # [S, H, W]
    depth_conf_np = pred_data.conf # [S, H, W]
    extrinsic_np = pred_data.extrinsics # [S, 3, 4]
    intrinsic_np = pred_data.intrinsics # [S, 3, 3]

    # filter
    depth_mask = depth_conf_np >= depth_conf_thresh
    depth_np[~depth_mask] = np.nan

    # unproject
    world_points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np) # [S, H, W, 3]

    all_points = []
    all_colors = []
    for frame_idx in range(world_points.shape[0]):
        points = world_points[frame_idx].reshape(-1, 3) # [N, 3]
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1) # [N, 3]
        valid_points = points[valid_mask]

        if len(valid_points) > 0:
            all_points.append(valid_points)

            # Generate corresponding colors
            img_hwc = proc_imgs[frame_idx]  # [H, W, 3] uint8
            rgb_flat = img_hwc.reshape(-1, 3) # [N, 3]
            valid_colors = rgb_flat[valid_mask]
            all_colors.append(valid_colors)

    camera_poses = to_homogeneous(extrinsic_np)
    all_cam_to_world_mat = list(camera_poses)

    return all_points, all_colors, all_cam_to_world_mat


def main(cfg):
    if "scannet" not in cfg.selected_datasets:
        return

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Get dataset config for ScanNet-v2
    data_cfg = cfg.datasets.scannet

    # Load model
    model = build_model(cfg.model, device)

    # Set seed after model loading
    set_random_seeds(cfg.seed)

    # Scene sampling
    scannet_scenes = get_all_scenes(data_cfg.scannet_dir, data_cfg.scenes)
    print("Number of scenes to evaluate:", len(scannet_scenes))

    # Process each scene
    all_scenes_metrics = {"scenes": {}, "average": {}}
    for scene in tqdm(scannet_scenes):
        print(f"\nScene: {scene}")

        # Load scene data
        scene_dir   = osp.join(data_cfg.scannet_dir, f"{scene}")
        images_dir  = osp.join(scene_dir, "color")
        pose_path   = osp.join(scene_dir, "pose")
        image_paths = get_sorted_image_paths(images_dir)

        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if (poses_gt is None or first_gt_pose is None or available_pose_frame_ids is None):
            print(f"Skipping scene {scene}: no pose data")
            continue

        # Filter frames
        selected_frame_ids, selected_image_paths, selected_pose_indices = build_frame_selection(
            image_paths, available_pose_frame_ids,
            data_cfg.max_frames, data_cfg.sample_mode, data_cfg.step_size
        )
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
        print(f"Sampled ({data_cfg.sample_mode}) {len(selected_frame_ids)} frames for evaluation")

        try:
            # Construct scene data dict for data loading api
            scene_data = Dict(image_files=image_paths)

            # Infer
            pred_data, stats = infer_model(model, cfg.model, scene_data)

            # Reconstruct world points from depth and camera parameters
            world_points, point_colors, cam_to_world_mat = reconstruct_world_points_from_depth(
                pred_data, data_cfg.depth_conf_thresh
            )

            # Evaluate and save
            suffix = "_SAF3R" if cfg.model.get("patch_module", {}).get("type", "") == "headsparse" else ""
            metrics = evaluate_scene_and_save(
                scene, c2ws, first_gt_pose, selected_frame_ids, cam_to_world_mat,
                world_points, gt_ply_dir=data_cfg.scannet_dir,
                chamfer_max_dist=data_cfg.chamfer_max_dist, inference_time_s=stats.latency,
                plot_flag=data_cfg.plot,
                plot_path=osp.join(data_cfg.plot_dir, cfg.model.name + suffix, f"{scene}.png")
            )

            if metrics is not None:
                print("Scene metrics:")
                all_scenes_metrics["scenes"][scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key in ["ate", "are", "rpe_rot", "rpe_trans", "chamfer_distance", "inference_time_s"]
                }
                for metric_name, value in all_scenes_metrics["scenes"][scene].items():
                    print(f"{metric_name}: {value:.4f}")

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback

            traceback.print_exc()

    # Summarize average metrics
    average_metrics = compute_average_metrics_and_save(all_scenes_metrics)
    print("\nAverage metrics:")
    for metric_name, value in average_metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on the ScanNet-v2 dataset")
    parser.add_argument(
        "--config_path", type=str, default="../configs/evaluation",
        help="Path to the config directory"
    )
    parser.add_argument(
        "--config", type=str, default="vggt_evaluation.yaml",
        help="Name of the config file (with or without .yaml extension, default: vggt_evaluation)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config)

    main(cfg)