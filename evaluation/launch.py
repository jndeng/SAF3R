"""
Unified launcher for evaluation scripts.
Automatically calls the appropriate evaluation script in a separate child process
based on the datasets specified in the config.
"""
import os
import sys
import subprocess
import argparse
from omegaconf import OmegaConf
from hydra import initialize, compose


AVAILABLE_DA3BENCH_DATASETS = {
    "7scenes", "eth3d", "scannetpp", "hiroom", "dtu64", "dtu"
}


def run_evaluation_script(script_name, args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    cmd = [
        sys.executable, script_path, 
        "--config_path", args.config_path, 
        "--config", args.config
    ]
    
    # Run the script and automatically raise an error if it fails
    subprocess.run(cmd, check=True)


def main(cfg, args):
    selected_datasets = set(cfg.selected_datasets)

    if "co3d" in selected_datasets:
        run_evaluation_script("evaluate_co3d.py", args)

    if "re10k" in selected_datasets:
        run_evaluation_script("evaluate_re10k.py", args)

    if selected_datasets & AVAILABLE_DA3BENCH_DATASETS:
        run_evaluation_script("evaluate_da3bench.py", args)

    if "scannet" in selected_datasets:
        run_evaluation_script("evaluate_scannet.py", args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch model evaluation across different datasets and tasks.")
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

    main(cfg, args)
