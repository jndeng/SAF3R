import os
import glob
import re
import random
import argparse
from addict import Dict
from hydra import initialize, compose

import pycolmap
import numpy as np
import torch

from saf3r.utils.model_utils import build_model, infer_model
from saf3r.patch import save_profile_results


# known problematic views
ETH3D_FILTER_KEYS = {
    "delivery_area": ["711.JPG", "712.JPG", "713.JPG", "714.JPG"],
    "electro": ["9289.JPG", "9290.JPG", "9291.JPG", "9292.JPG", "9293.JPG", "9298.JPG"],
    "playground": ["587.JPG", "588.JPG", "589.JPG", "590.JPG", "591.JPG", "592.JPG"],
    "relief": [
        "427.JPG", "428.JPG", "429.JPG", "430.JPG", "431.JPG", "432.JPG",
        "433.JPG", "434.JPG", "435.JPG", "436.JPG", "437.JPG", "438.JPG",
    ],
    "relief_2": [
        "458.JPG", "459.JPG", "460.JPG", "461.JPG", "462.JPG", "463.JPG",
        "464.JPG", "465.JPG", "466.JPG", "467.JPG", "468.JPG",
    ],
}


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


def sample_frames(image_files, max_frames, indices=None, sample_mode="uniform"):
    num_frames = len(image_files)

    # Sample a subset of images
    if indices is None:
        if num_frames <= max_frames:
            sampled_indices = list(range(num_frames))
        else:
            if sample_mode == "random":
                random.seed(42)
                indices = list(range(num_frames))
                random.shuffle(indices)
                sampled_indices = sorted(indices[:max_frames])
            elif sample_mode == "uniform":
                sampled_indices = np.linspace(
                    0, num_frames - 1, num=max_frames, dtype=np.int64
                ).tolist()
    else:
        sampled_indices = sorted(indices)

    # Return sampled image files
    return [image_files[i] for i in sampled_indices]


def load_scannet(cfg):
    """
    Load image paths from ScanNet.
    """
    scenes = []
    for scene in cfg.scenes:
        scene_dir = os.path.join(cfg.dataset_dir, "scans", scene, "color")
        image_files = glob.glob(os.path.join(scene_dir, "*.jpg"))
        image_files = sorted(image_files, key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]))

        # sample frames
        image_files = sample_frames(image_files, cfg.max_frames, cfg.indices, cfg.sample_mode)

        scene_data = Dict()
        scene_data.image_files = image_files
        scenes.append(scene_data)

    return scenes


def load_7scenes(cfg):
    """
    Load image paths from 7Scenes.
    """
    scenes = []
    for scene in cfg.scenes:
        scene_dir = os.path.join(cfg.dataset_dir, scene)
        image_files = sorted(glob.glob(os.path.join(scene_dir, "*.color.png")))

        # sample frames
        image_files = sample_frames(image_files, cfg.max_frames, cfg.indices, cfg.sample_mode)

        scene_data = Dict()
        scene_data.image_files = image_files
        scenes.append(scene_data)

    return scenes


def parse_eth3d_images_txt(filepath: str) -> dict:
    """
    Parse COLMAP-style images.txt file.

    Returns:
        Dict mapping image path to pose parameters
    """
    pose_dict = {}
    with open(filepath) as f:
        lines = f.readlines()
        for idx, line in enumerate(lines[4:]):  # Skip header
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Every other line contains pose info
            if idx % 2 == 0:
                parts = line.split()
                if len(parts) < 10:
                    continue
                # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                image_id = parts[0]
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                camera_id = parts[8]
                name = parts[9]
                pose_dict[name] = {
                    "image_id": image_id,
                    "quat": [qw, qx, qy, qz],
                    "trans": [tx, ty, tz],
                    "camera_id": camera_id,
                }
    return pose_dict


def load_eth3d(cfg):
    """
    Load image paths from ETH3D.
    """
    scenes = []
    for scene in cfg.scenes:
        if cfg.from_calib_file:
            calib_file = os.path.join(cfg.dataset_dir, scene, "dslr_calibration_jpg", "images.txt")
            pose_dict = parse_eth3d_images_txt(calib_file)
            image_files = list(pose_dict.keys())
            image_files = [
                os.path.join(cfg.dataset_dir, scene, "images", img_id)
                for img_id in pose_dict.keys()
            ]
        else:
            scene_dir = os.path.join(cfg.dataset_dir, scene, "images", cfg.image_type)
            image_files = sorted(glob.glob(os.path.join(scene_dir, "*.JPG")))

        # filter problematic views
        if cfg.filter_views and scene in ETH3D_FILTER_KEYS:
            filter_keys = ETH3D_FILTER_KEYS[scene]
            image_files = [
                f for f in image_files
                if not any(key in os.path.basename(f) for key in filter_keys)
            ]

        # sample frames
        image_files = sample_frames(image_files, cfg.max_frames, cfg.indices, cfg.sample_mode)

        scene_data = Dict()
        scene_data.image_files = image_files
        scenes.append(scene_data)

    return scenes


def load_scene_data(cfg):
    """
    Load data for profiling sparse attention patterns. Only images paths are loaded.
    """
    all_scenes = []
    for dataset_name, dataset_cfg in cfg.datasets.items():
        # load and sample image paths
        match dataset_name:
            case "eth3d":
                scenes = load_eth3d(dataset_cfg)
            case "7scenes":
                scenes = load_7scenes(dataset_cfg)
            case "scannet":
                scenes = load_scannet(dataset_cfg)
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

        all_scenes += scenes

    return all_scenes


def main(cfg):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model, device)

    # Set random seeds
    set_random_seeds(cfg.seed)

    # Load scenes
    scene_data_list = load_scene_data(cfg)

    # Iterate over scenes
    for scene_data in scene_data_list:
        # print("\n".join([os.path.basename(f) for f in scene_data.image_files]))
        infer_model(model, cfg.model, scene_data)

    # Save profile results
    save_profile_results(model, cfg.model.name.lower(), cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile sparse attention patterns for F3R models based on a calibration dataset.")
    parser.add_argument(
        "--config_path", type=str, default="../configs/profile",
        help="Path to the config directory"
    )
    parser.add_argument(
        "--config", type=str, default="profile_saf3r_vggt.yaml",
        help="Name of the config file (with or without .yaml extension, default: profile_saf3r_vggt)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config)

    main(cfg)
