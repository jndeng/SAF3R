import argparse
from tqdm import tqdm
from hydra import initialize, compose

import pycolmap
import numpy as np
import torch

from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.da3bench_utils import sample_frames
from saf3r.utils.model_utils import build_model, infer_model


def load_scene_data(
    dataset, dataset_name, scene_name, max_frames, sample_mode="uniform",
    seq_multiple=1, seed=0,
):
    # load scene
    scene_data = dataset.get_data(scene_name)

    # sample frames
    frame_ids, scene_data = sample_frames(
        scene_data, scene_name, max_frames, indices=None, sample_mode=sample_mode,
        seq_multiple=seq_multiple, seed=seed
    )

    return scene_data


def main(args, model_cfg):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model, device)

    # Load scene data
    dataset = MV_REGISTRY.get("7scenes")()
    scene_data = load_scene_data(dataset, "7scenes", "chess", args.num_images)
    print()

    # Warmup iterations
    for _ in tqdm(range(args.warmup_iter), desc="Warmup", leave=False):
        infer_model(model, model_cfg, scene_data)
    print()

    # Measure iterations
    latency, max_mem = [], []

    pbar = tqdm(range(args.measure_iter), desc="Benchmark")
    for i in pbar:
        _, stats = infer_model(model, model_cfg, scene_data)

        latency.append(stats.latency)
        max_mem.append(stats.max_mem)

        pbar.set_postfix({
            "latency": f"{stats.latency:.1f}s",
            "max_mem": f"{stats.max_mem:.1f}GB",
        })
    
    print("\n" + "=" * 50)
    print(f"Average Latency:  {np.mean(latency):.2f} (s)")
    print(f"Average Max Mem.: {np.mean(max_mem):.2f} (GB)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark F3R models.")
    parser.add_argument(
        "--config_path", type=str, default="../configs/evaluation",
        help="Path to the config directory"
    )
    parser.add_argument(
        "--config", type=str, default="vggt_eval.yaml",
        help="Name of the config file (with or without .yaml extension, default: vggt_eval)"
    )

    parser.add_argument(
        "--warmup_iter", type=int, default=5,
        help="Number of warmup iterations before measuring latency (default: 5)"
    )
    parser.add_argument(
        "--measure_iter", type=int, default=5,
        help="Number of iterations to measure latency (default: 5)"
    )
    parser.add_argument(
        "--num_images", type=int, default=500,
        help="Number of images to use for evaluation (default: 500)"
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config)

    main(args, cfg.model)
