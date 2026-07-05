"""
Evaluate F3R models on DA3-Bench across multiple tasks and datasets.

Supported evaluation tasks:
    - Camera pose estimation
    - Video depth estimation
    - 3D point-cloud reconstruction (unposed)

NOTE: Camera pose and video depth are evaluated as separate tasks, whereas 3D point cloud
reconstruction effectively combines both, as its performance depends on both pose and depth.
"""
import argparse
from addict import Dict
from hydra import initialize, compose

import pycolmap
import open3d as o3d
import numpy as np
import torch

from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.bench.utils import evaluate_3d_reconstruction, sample_points_from_mesh
from benchmarks.da3.bench.utils import compute_pose
from benchmarks.da3.da3bench_utils import sample_frames, fuse3d, evaluate_reconstruction_dtu
from benchmarks.da3.geometry import as_homogeneous
from benchmarks.pi3.eval_videodepth import evaluate_video_depth
from utils import get_indicator
from saf3r.utils.data_utils import load_depths
from saf3r.utils.model_utils import build_model, infer_model


AVAILABLE_DATASETS = ["7scenes", "eth3d", "scannetpp", "hiroom", "dtu64", "dtu"]


def evaluate(dataset, gt_data, pred_data, eval_tasks):
    all_metrics = Dict()

    # Evaluate camera pose estimation
    if "pose" in eval_tasks:
        metrics = compute_pose(
            torch.from_numpy(as_homogeneous(pred_data.extrinsics)),
            torch.from_numpy(as_homogeneous(gt_data.extrinsics)),
        )
        all_metrics.pose = metrics

    # Evaluate depth estimation
    if "depth" in eval_tasks:
        metrics = evaluate_video_depth(
            pred_data.depth, load_depths(gt_data), gt_data.dataset_name
        )
        # remove other metrics
        del metrics["Sq Rel"]
        del metrics["RMSE"]
        del metrics["Log RMSE"]
        del metrics["δ < 1."]
        del metrics["δ < 1.25^2"]
        del metrics["δ < 1.25^3"]
        del metrics["valid_pixels"]
        all_metrics.depth = metrics
 
    # Evaluate 3D reconstruction
    if "recon_unposed" in eval_tasks:
        pred_mesh, pred_pcd, aligned_intrinsics, aligned_extrinsics = fuse3d(
            dataset, gt_data.scene_name, gt_data, pred_data, mode="recon_unposed"
        )

        # Update intrinsics and extrinsics after global alignment
        pred_data.aligned_intrinsics = aligned_intrinsics
        pred_data.aligned_extrinsics = aligned_extrinsics
        pred_data.mesh = pred_mesh
        pred_data.pcd = pred_pcd

        # Compute metrics
        if gt_data.dataset_name == "dtu":
            # NOTE: DTU adopts a separate evaluation protocol
            result = evaluate_reconstruction_dtu(
                dataset, pred_data.pcd, gt_data.pcd,
                mask_file=gt_data.aux.mask_file, plane_file=gt_data.aux.plane_file,
                use_gpu=True
            )
            metrics = Dict({"comp": result[0], "acc": result[1], "overall": result[2]})
        else:
            metrics = evaluate_3d_reconstruction(
                pred_data.pcd,
                gt_data.pcd,
                threshold=dataset.eval_threshold,
                down_sample=dataset.down_sample,
            )
        all_metrics.recon_unposed = metrics


    return all_metrics


def load_scene_data(
    dataset, dataset_name, scene_name, max_frames, sample_mode="uniform",
    seq_multiple=1, seed=0,
):
    # load scene
    scene_data = dataset.get_data(scene_name)

    # sample frames
    frame_ids, scene_data = sample_frames(
        scene_data, scene_name, max_frames, indices=None, sample_mode=sample_mode,
        seq_multiple=seq_multiple,  # force the sampled sequence length to be a multiple of seq_multiple if specified
        seed=seed
    )

    # set GT meta for evaluation
    gt_data = Dict({
        "dataset_name": dataset_name,
        "scene_name": scene_name,
        "extrinsics": scene_data.extrinsics,
        "intrinsics": scene_data.intrinsics,
        "image_files": scene_data.image_files,
        "aux": Dict({
            "gt_mesh_path": scene_data.aux.gt_mesh_path,
            "gt_pcd_path": scene_data.aux.gt_pcd_path,
            "gt_depth_files": scene_data.aux.gt_depth_files,
            "aliasing_mask_files": scene_data.aux.aliasing_mask_files,
            "mask_file": scene_data.aux.mask_file,
            "plane_file": scene_data.aux.plane_file,
        }),
    })

    # get ground truth point-cloud for evaluation
    if gt_data.aux.gt_mesh_path: # return empty {} if not exists
        gt_data.mesh = o3d.io.read_triangle_mesh(gt_data.aux.gt_mesh_path)
        gt_data.pcd = sample_points_from_mesh(gt_data.mesh, dataset.sampling_number)
    elif gt_data.aux.gt_pcd_path: # return empty {} if not exists
        gt_data.mesh = None
        gt_data.pcd = o3d.io.read_point_cloud(gt_data.aux.gt_pcd_path)
    else:
        # datasets only support camera pose estimation (e.g., DTU-64)
        pass

    return scene_data, gt_data


def print_metrics(metrics, all_scenes, dataset_name):
    print(f"\nPer-scene metrics of dataset [{dataset_name}]")

    scene_width = 10
    metric_width = 16

    first_scene = all_scenes[0]

    for task, task_metrics in metrics[first_scene].items():
        header = (
            f"{task:<{metric_width}} "
            + " ".join(f"{scene:>{scene_width}}" for scene in all_scenes)
            + f" {'AVG':>{scene_width}}"
        )
        print(header)
        print("-" * len(header))

        for m in task_metrics:
            values = np.array(
                [metrics[scene][task][m] for scene in all_scenes],
                dtype=float,
            )
            avg = values.mean()

            metric_name = f"{m}({get_indicator(m)})"
            row = (
                f"{metric_name:<{metric_width}} "
                + " ".join(f"{v:>{scene_width}.4f}" for v in values)
                + f" {avg:>{scene_width}.4f}"
            )
            print(row)
        
        print()


def print_dataset_summary(metrics, all_scenes):
    ordered_metrics = [
        ("pose", "auc30", "AUC@30"),
        ("pose", "auc15", "AUC@15"),
        ("pose", "auc05", "AUC@5"),
        ("pose", "auc03", "AUC@3"),
        ("recon_unposed", "acc", "Acc"),
        ("recon_unposed", "comp", "Comp"),
        ("recon_unposed", "overall", "CD"),
        ("recon_unposed", "precision", "Prec"),
        ("recon_unposed", "recall", "Rec"),
        ("recon_unposed", "fscore", "F1"),
        ("depth", "Abs Rel", "AbsRel"),
        ("depth", "δ < 1.25", "δ<1.25"),
    ]

    results = []
    for task, metric_key, _ in ordered_metrics:
        values = []

        for scene in all_scenes:
            v = metrics.get(scene, {}).get(task, {}).get(metric_key, None)
            if v is not None:
                values.append(float(v))

        if len(values) > 0:
            avg = np.mean(values)
            results.append(f"{avg:.4f}")
        else:
            results.append("-")

    print("Dataset AVG")
    header = "\t".join(name for _, _, name in ordered_metrics)
    print(header)

    row = "\t".join(results)
    print(row)
    print()


def main(cfg):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model, device)

    # Iterate over datasets
    for dataset_name in cfg.selected_datasets:
        # Filter out unavailable datasets
        if dataset_name not in AVAILABLE_DATASETS:
            continue
        data_cfg = cfg.datasets[dataset_name]

        # Load dataset
        dataset = MV_REGISTRY.get(dataset_name.lower())()

        # Parse scenes
        all_scenes = data_cfg.scenes if data_cfg.scenes is not None else dataset.SCENES

        # Iterate over scenes
        metrics = Dict()
        for scene_name in all_scenes:
            # Load data
            scene_data, gt_data = load_scene_data(
                dataset, dataset_name, scene_name, data_cfg.max_frames, data_cfg.sample_mode,
                seq_multiple=(8 if cfg.model.name.lower() == "litevggt" else 1),
                seed=cfg.seed,
            )

            # Inference
            pred_data, stats = infer_model(model, cfg.model, scene_data)
            print(f"Latency: {stats.latency:.2f} (s)  |  Max Mem.: {stats.max_mem:.2f} (GB)")

            # Evaluate
            metrics[scene_name] = evaluate(dataset, gt_data, pred_data, data_cfg.eval_tasks)

        # print metrics summary
        print_metrics(metrics, all_scenes, dataset_name)
        print_dataset_summary(metrics, all_scenes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on DA3-Bench across multiple tasks and datasets.")
    parser.add_argument(
        "--config_path", type=str, default="../configs/evaluation",
        help="Path to the config directory"
    )
    parser.add_argument(
        "--config", type=str, default="vggt_eval.yaml",
        help="Name of the config file (with or without .yaml extension, default: vggt_eval)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config)

    main(cfg)
