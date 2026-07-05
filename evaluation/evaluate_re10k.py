"""
Evaluate F3R models on the RealEstate10K dataset for camera pose estimation.
"""
import os
import os.path as osp
import argparse
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


def get_re10k_sequences(re10k_dir, re10k_meta_path):
    """
    Get sequence names and corresponding frame IDs for a given category.

    Args:
        re10k_dir: RealEstate10K dataset directory
        re10k_seqid_path: Path to sequence ID mapping file (JSON)
        category: Category name
    Returns:
        List of tuples containing sequence names and frame IDs
    """
    with open(re10k_meta_path, "r") as f:
        seq_id_map = json.load(f)

    # load available sequence names for the directory
    vaild_seq_id_map = {}
    for seq_name in sorted(os.listdir(re10k_dir)):
        img_path = osp.join(re10k_dir, seq_name, "images")
        anno_path = osp.join(re10k_dir, seq_name, "annotations.json")
        if osp.exists(img_path) and osp.exists(anno_path) and seq_name in seq_id_map:
            vaild_seq_id_map[seq_name] = seq_id_map[seq_name]
    
    print(f"Found {len(vaild_seq_id_map)} valid sequences.")
    return vaild_seq_id_map


def sample_re10k_sequence(re10k_dir, annotation_path, img_ids, seq_multiple=1):
    """
    Load annotation data for a sequence from a JSON file, and then sample frames
    based on the provided frame IDs in the metadata.

    Args:
        annotation_path: Path to the annotation JSON file
    Returns:
       filepaths: List of file paths for the sampled frames
       intrinsics: Intrinsic matrices Nx3x3 for the sampled frames
       extrinsics: Extrinsic matrices Nx4x4 for the sampled frames
    """
    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    # NOTE: for LiteVGGT which requires sequence length to be a multiple of 8
    #       we force using the first N selected frames to satisfy this constraint
    if len(img_ids) % seq_multiple != 0:
        print(f"[Warning] Truncated sequence length from {len(img_ids)} to {len(img_ids)//seq_multiple*seq_multiple} to satisfy TE FP8 requirements.")
        img_ids = img_ids[:len(img_ids) // seq_multiple * seq_multiple]
    
    filepaths, intrinsics, extrinsics = [], [], []
    for img_id in img_ids:
        frame_data = annotation[img_id]
        assert frame_data["idx"] == img_id
        filepaths.append(osp.join(re10k_dir, frame_data["filepath"]))
        intrinsics.append(np.array(frame_data["intrinsics"])) # 3x3
        extrinsics.append(np.array(frame_data["extrinsics"])) # 4x4 homogeneous

    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)

    return filepaths, intrinsics, extrinsics


def process_sequence(
    model, model_cfg, filepaths, intrinsics, extrinsics, use_ba, device,
):
    """
    Process a single sequence and compute pose errors.

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    assert not use_ba, "BA not supported"

    scene_data = Dict(image_files=filepaths)
    num_frames = len(filepaths)

    # Infer
    pred_data, stats = infer_model(model, model_cfg, scene_data)
    pred_extrinsic = torch.from_numpy(pred_data.extrinsics).to(device)

    with torch.amp.autocast("cuda", dtype=torch.float64):
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)
        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.from_numpy(extrinsics).to(device)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

    return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def main(cfg):
    if "re10k" not in cfg.selected_datasets:
        return
    
    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Get dataset config for RealEstate10K
    data_cfg = cfg.datasets.re10k

    # Load model
    model = build_model(cfg.model, device)

    # Set random seeds
    set_random_seeds(cfg.seed)

    # Sequences to evaluate
    seq_id_map = get_re10k_sequences(data_cfg.re10k_dir, data_cfg.re10k_meta_path)

    per_seq_results = {}
    for seq_name, img_ids in tqdm(seq_id_map.items(), desc="Evaluating sequences"):
        seq_dir = osp.join(data_cfg.re10k_dir, seq_name)
        annotation_path = osp.join(seq_dir, "annotations.json")

        try:
            filepaths, intrinsics, extrinsics = sample_re10k_sequence(
                data_cfg.re10k_dir, annotation_path, img_ids,
                seq_multiple=(8 if cfg.model.name.lower() == "litevggt" else 1)  # force the sampled sequence length to be a multiple of seq_multiple (e.g., 8 for LiteVGGT) if specified
            )
        except FileNotFoundError:
            print(f"Sampling failed for {seq_name}, skipping.")
            continue

        rError = []
        tError = []

        seq_rError, seq_tError = process_sequence(
            model, cfg.model, filepaths, intrinsics, extrinsics, data_cfg.use_ba, device,
        )

        rError.extend(seq_rError)
        tError.extend(seq_tError)

        rError = np.array(rError)
        tError = np.array(tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5,  _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3,  _ = calculate_auc_np(rError, tError, max_threshold=3)

        per_seq_results[seq_name] = {
            "rError": rError,
            "tError": tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3
        }

    if per_seq_results:
        mean_AUC_30 = np.mean([per_seq_results[seq_name]["Auc_30"] for seq_name in per_seq_results])
        mean_AUC_15 = np.mean([per_seq_results[seq_name]["Auc_15"] for seq_name in per_seq_results])
        mean_AUC_5  = np.mean([per_seq_results[seq_name]["Auc_5"] for seq_name in per_seq_results])
        mean_AUC_3  = np.mean([per_seq_results[seq_name]["Auc_3"] for seq_name in per_seq_results])
        print("-"*50)
        print(f"Mean AUC: {mean_AUC_30:.4f} (AUC@30), {mean_AUC_15:.4f} (AUC@15), {mean_AUC_5:.4f} (AUC@5), {mean_AUC_3:.4f} (AUC@3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate camera pose estimation on the RealEstate10K dataset")
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
