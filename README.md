<div align="center">

<h1>SAF3R: Dynamic Sparse Attention for Feed-Forward 3D Reconstruction Transformers</h1>

[![Paper](https://img.shields.io/static/v1?label=Paper&message=arXiv-xxxx.xxxxx&color=red&logo=arxiv)](https://arxiv.org/abs/xxxx.xxxxx)

</div>

SAF3R is a training-free dynamic sparse attention framework that accelerates existing feed-forward 3D reconstruction models, such as VGGT, by reducing the computational cost of global attention. By exploiting head-wise sparsity heterogeneity, SAF3R replaces each full global attention head with the sparse attention kernel that best matches its attention pattern.

### What's in this repo
* **Visualization tools** for analyzing head-wise global attention patterns in feed-forward 3D reconstruction models. Currently supported models include VGGT, Pi3, DA3, and StreamVGGT (streaming model).
* **SAF3R offline profiling code** that automatically assigns each global attention head to one of four predefined sparse attention patterns.
* **SAF3R inference patches** that enable dynamic sparse attention inference for VGGT, Pi3, and DA3.
* **Benchmark and evaluation tools** for evaluating model performance and efficiency. Currently supported benchmarks include DA3-Bench, Co3D-v2, RealEstate10K, and ScanNet.

## Table of Contents
- [Installation](#installation)
- [Analysis Tools](#analysis-tools)
- [Model Inference](#model-inference)
  - [Running Inference Demo](#running-inference-demo)
  - [Minimal Code Snippet (for SAF3R-VGGT)](#minimal-code-snippet-for-saf3r-vggt)
- [Evaluation Benchmarks](#evaluation-benchmarks)
  - [Supported Benchmarks & Datasets](#supported-benchmarks--datasets)
  - [Datasets Preparation](#datasets-preparation)
  - [Running Evaluation](#running-evaluation)
  - [Benchmarking Efficiency](#benchmarking-efficiency)
- [Profiling Attention Heads](#profiling-attention-heads)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


## Installation

```bash
conda create -n saf3r python=3.10 -y
conda activate saf3r
git clone https://github.com/jndeng/SAF3R
cd SAF3R
pip install -e .
```


## Analysis Tools

We provide visualization and analysis tools for frame/global attention patterns for each supported model under `tools/`. Each tool is implemented as a standalone Jupyter notebook and can be run independently.


## Model Inference

We provide inference example code for using SAF3R on different 3R models.

> [!NOTE]
> Checkpoints will be automatically downloaded to the local cache directory (`checkpoints/`) during the first run.
> They can also be manually downloaded from [VGGT](https://huggingface.co/facebook/VGGT_tracker_fixed), [Pi3](https://huggingface.co/yyfz233/Pi3), [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT), and [StreamVGGT](https://huggingface.co/lch01/StreamVGGT).

### Running Inference Demo

To run a demo inference script using a specific model on your image directories:
```bash
python scripts/inference_demo.py --model vggt --data_dir data/courthouse
```
Supported options for `--model` are `vggt`, `pi3`, and `da3`. By default, the predicted point clouds will be exported under `tmp_plots/` as `.ply` files and interactively visualizable `.html` files.

> [!NOTE]
> The first run may take longer due to JIT compilation of the custom Triton kernels. The compiled kernels are cached and reused in subsequent runs.

### Minimal Code Snippet (for SAF3R-VGGT)

<details>
<summary>Show code</summary>

```python
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model, infer_model

# 1. Configure the model and SAF3R patch settings
model_cfg = Dict(
    name="VGGT", 
    dpt_only=True, 
    ckpt_path="checkpoints/vggt/model_tracker_fixed_e20.pt",
    sparse_config_path="configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json",
    patch_module=Dict(
        type="headsparse",
        topk_mode="token",
        lazy_dino_topk=True,
        topk=4
    )
)

# 2. Build and automatically patch the model with SAF3R kernels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(model_cfg, device)

# 3. Setup input data (image file paths)
scene_data = Dict(
    image_files=[
        "data/courthouse/000000.png",
        "data/courthouse/000001.png",
        ...
    ]
)

# 4. Perform dynamic sparse attention inference
pred_data, stats = infer_model(model, model_cfg, scene_data)

# Extract unified predictions
extrinsics = pred_data.extrinsics     # [N, 3, 4]
intrinsics = pred_data.intrinsics     # [N, 3, 3]
depth      = pred_data.depth          # [N, H, W]
```
</details>


## Evaluation Benchmarks

We provide evaluation code for multiple tasks and benchmarks.

### Supported Benchmarks & Datasets
* DA3-Bench (7Scenes, ETH3D, ScanNet++, HiRoom, DTU64, DTU)
  - Camera pose estimation
  - Video depth estimation
  - 3D point-cloud reconstruction
* Co3D-v2
  - Camera pose estimation
* RealEstate10K
  - Camera pose estimation
* ScanNet (v2)
  - Camera pose estimation
  - 3D point-cloud reconstruction

### Datasets Preparation
Please follow the corresponding instructions to prepare each dataset.
* DA3-Bench (7Scenes, ETH3D, ScanNet++, HiRoom, DTU64, DTU)
    - Follow the [DA3-Bench dataset download instructions](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md#-dataset-download) to download the datasets. This should create the `workspace/benchmark_dataset` directory under the project root.
* Co3D (v2):
    - Follow the [VGGT preparation instructions](https://github.com/facebookresearch/vggt/tree/evaluation/evaluation#dataset-preparation) to prepare the dataset and place it under `workspace/benchmark_dataset`.
* RealEstate10K:
    - Follow the [Pi3 preparation scripts](https://github.com/yyfz/Pi3/blob/evaluation/datasets/preprocess/prepare_re10k.sh) to prepare the dataset and place it under `workspace/benchmark_dataset`.
* ScanNet (v2):
    - Follow the [ScanNet instructions](http://www.scan-net.org/ScanNet/) to download the dataset and place it under `workspace/benchmark_dataset`. The list of the 50 evaluation scenes can be found [here](https://github.com/mystorm16/FastVGGT/blob/main/eval/scannet_50.yaml).

The downloaded datasets should be organized under `workspace/benchmark_dataset/` as follows:

```text
workspace/benchmark_dataset/
├── 7scenes/
├── co3dv2/
├── dtu/
├── dtu64/
├── eth3d/
├── hiroom/
├── realestate10k/
├── scannetpp/
└── scannetv2/
```

### Running Evaluation
Run the unified launcher with the desired configuration file:
```bash
bash scripts/evaluate.sh eval_saf3r_vggt
```

### Benchmarking Efficiency
To measure inference latency and peak memory usage across different sequences:
```bash
bash scripts/benchmark_efficiency.sh eval_saf3r_vggt 300
```
This runs the efficiency benchmark using the specified configuration file and sequence length.


## Profiling Attention Heads

We provide profiled global-attention head configurations under `configs/sparse_attn/`. 

To generate these profiling results from scratch:

1. Download the calibration dataset (e.g., ETH3D) following the instructions in [DA3-bench](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md).
2. Run the profiling script using the desired configuration file under `configs/profile/`:
   ```bash
   bash scripts/profile.sh profile_saf3r_vggt.yaml
   ```


## Acknowledgements
This repository builds upon several excellent open-source projects, including [VGGT](https://github.com/facebookresearch/vggt), [Pi3](https://github.com/yyfz/Pi3), [DA3](https://github.com/ByteDance-Seed/depth-anything-3), [StreamVGGT](https://github.com/wzzheng/streamvggt), [FastVGGT](https://github.com/mystorm16/FastVGGT), [SparseVGGT](https://github.com/brianwang00001/sparse-vggt), and [Speed3R](https://github.com/Visual-AI/speed3r). We sincerely thank the authors and contributors for making their code publicly available.


## Citation
If you find SAF3R useful for your research or project, please consider citing:

```bibtex
```
