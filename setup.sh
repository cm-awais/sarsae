#!/bin/bash
# One-time setup on the login node, inside your activated Python environment.
set -e
source "$(dirname "$0")/env.sh"
cd "$SARSAE_ROOT"; mkdir -p repos logs work
[ -d repos/SAR-JEPA ] || git clone -q https://github.com/waterdisappear/SAR-JEPA.git repos/SAR-JEPA
[ -d repos/SARATR-X ] || git clone -q https://github.com/waterdisappear/SARATR-X.git repos/SARATR-X
[ -d repos/DOFA ]     || git clone -q https://github.com/zhu-xlab/DOFA.git repos/DOFA
pip uninstall -y -q torchao 2>/dev/null || true          # conflicts with recent peft, not needed
pip install -q "timm>=1.0.20" peft easydict scikit-learn pandas huggingface_hub kagglehub
python - <<'PY'
import torch, timm, peft, sklearn
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| timm", timm.__version__, "| peft", peft.__version__)
PY
echo "Setup done."
