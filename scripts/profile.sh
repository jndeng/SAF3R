#!/bin/bash
set -euo pipefail

# default
config="${1:-profile_saf3r_vggt.yaml}"

# run evaluation
echo "=> Start profiling using config: [$config]"
python saf3r/profile_models.py --config "$config"
