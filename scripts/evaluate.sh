#!/bin/bash
set -euo pipefail

# default
config="${1:-eval_vggt.yaml}"

# run evaluation
echo "=> Start evaluation using config: [$config]"
python evaluation/launch.py --config "$config"
