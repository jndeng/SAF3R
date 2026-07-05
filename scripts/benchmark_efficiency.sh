#!/bin/bash
set -euo pipefail

# default
config="${1:-eval_vggt.yaml}"
num_images="${2:-300}"

# run evaluation
echo "=> Benchmarking config: [$config]"
python evaluation/benchmark_efficiency.py --config "$config" --num_images "$num_images"
