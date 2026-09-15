#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# Reuse a working workstation CUDA installation instead of replacing its driver stack.
python3 -m venv --system-site-packages .venv-depth
.venv-depth/bin/python -m pip install -r experiments/depth/requirements.txt
if ! .venv-depth/bin/python -c 'import torch' >/dev/null 2>&1; then
  .venv-depth/bin/python -m pip install 'torch>=2.2,<3'
fi
.venv-depth/bin/python depth_study.py doctor --config experiments/depth/configs/paper_depths.json
