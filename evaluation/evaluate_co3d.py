"""
Evaluate F3R models on the Co3D-v2 dataset for camera pose estimation.
"""
import os
import argparse
import gzip
import json
import random
from addict import Dict
from tqdm import tqdm
from hydra import initialize, compose

import numpy as np
import torch

from benchmarks.vggt.vggt_utils import *
from saf3r.utils.model_utils import build_model, infer_model

# Set computation precision
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False


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


def process_sequence(
    model, model_cfg, seq_name, seq_data, co3d_dir, min_num_images, num_frames,
    use_ba, disable_rand, device, seq_multiple=1
):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: F3R model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if not disable_rand and (len(seq_data) < min_num_images):
        return None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    # NOTE: frames are filtered in annotation file, so we use all frames here
    if disable_rand:
        ids = np.arange(len(metadata))
    # Random sample num_frames images
    else:
        ids = np.random.choice(len(metadata), num_frames, replace=False)
        print(f"[{seq_name}] sampled image ids {ids}")

    # NOTE: for LiteVGGT which requires sequence length to be a multiple of 8
    #       we force using the first N selected frames to satisfy this constraint
    if len(ids) % seq_multiple != 0:
        print(f"[Warning] Truncated sequence length from {len(ids)} to {len(ids)//seq_multiple*seq_multiple} to satisfy TE FP8 requirements.")
        ids = ids[:len(ids) // seq_multiple * seq_multiple]
        num_frames = len(ids)

    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    assert not use_ba, "BA not supported"

    scene_data = Dict(
        image_files=[os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    )

    # Infer
    pred_data, stats = infer_model(model, model_cfg, scene_data)
    pred_extrinsic = torch.from_numpy(pred_data.extrinsics).to(device)

    with torch.amp.autocast("cuda", dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        # NOTE: It's not a frame-to-frame error, but rather relative pose errors between all pairs of camera frames
        # which is a robust metric commonly used in SfM and Visual Odometry evaluation.
        # rel_rangle_deg: n_frame*(n_frame-1)/2, in degrees
        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

    return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def main(cfg):
    if "co3d" not in cfg.selected_datasets:
        return

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Get dataset config for Co3D-v2
    data_cfg = cfg.datasets.co3d

    # Load model
    model = build_model(cfg.model, device)

    # Set random seeds
    set_random_seeds(cfg.seed)

    # All eval categories
    SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]

    categories = SEEN_CATEGORIES
    if data_cfg.categories is not None:
        categories = [c for c in data_cfg.categories if c in SEEN_CATEGORIES]
        
    # Evaluate on each category
    per_category_results = {}
    for category in categories:
        annotation_file = os.path.join(data_cfg.co3d_anno_dir, f"{category}_{data_cfg.split}.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []

        # Sample sequences for fast evaluation
        seq_names = sorted(list(annotation.keys()))
        if data_cfg.fast_eval and len(seq_names) > 10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)

        # Process each sequence
        for seq_name in tqdm(seq_names, desc=f"Processing sequences for {category}"):
            seq_data = annotation[seq_name]
            if not os.path.exists(os.path.join(data_cfg.co3d_dir, category, seq_name)):
                raise ValueError(f"Cannot find sequence: {os.path.join(data_cfg.co3d_dir, category, seq_name)}")
            
            # NOTE: arrays with N*(N-1)/2 elements where N is the number of frames in the sequence, representing relative pose errors between all pairs of camera frames
            seq_rError, seq_tError = process_sequence(
                model, cfg.model, seq_name, seq_data, data_cfg.co3d_dir,
                data_cfg.min_num_images, data_cfg.num_frames, data_cfg.use_ba, 
                data_cfg.disable_rand, device,
                seq_multiple=(8 if cfg.model.name.lower() == "litevggt" else 1)  # force the sampled sequence length to be a multiple of seq_multiple (e.g., 8 for LiteVGGT) if specified
            )

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5,  _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3,  _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

        print("="*80)
        # Print results with colors
        GREEN = "\033[92m"
        RED   = "\033[91m"
        BLUE  = "\033[94m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"

        print(f"{BOLD}{BLUE}AUC of {category} test set:{RESET} {GREEN}{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), {Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3){RESET}")
        mean_AUC_30_by_now = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15_by_now = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5_by_now = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3_by_now = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print(f"{BOLD}{BLUE}Mean AUC of categories by now:{RESET} {RED}{mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3){RESET}")
        print("="*80)

    # Print summary results
    print("\nSummary of AUC results:")
    print("-"*50)
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30), {per_category_results[category]['Auc_15']:.4f} (AUC@15), {per_category_results[category]['Auc_5']:.4f} (AUC@5), {per_category_results[category]['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        mean_AUC_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        mean_AUC_15 = np.mean([per_category_results[category]["Auc_15"] for category in per_category_results])
        mean_AUC_5 = np.mean([per_category_results[category]["Auc_5"] for category in per_category_results])
        mean_AUC_3 = np.mean([per_category_results[category]["Auc_3"] for category in per_category_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate camera pose estimation on the Co3Dv2 dataset")
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
