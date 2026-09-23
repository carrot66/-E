#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Run after: conda create -n math python=3.11 -y && conda activate math
python -c 'import sys; assert sys.version_info[:2] == (3,11), "Please activate Python 3.11 environment math"'
python -m pip install --upgrade pip
# CUDA 12.4 wheels contain the CUDA runtime; a compatible NVIDIA driver is needed.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirement.txt
python -m pip check
python -m unittest -v test_alignment test_pipeline
python -m unittest -v test_model_interfaces
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "Available:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable; inspect nvidia-smi and the installed wheel"'
