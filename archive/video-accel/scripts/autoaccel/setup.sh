#!/bin/bash
# AutoAccel Phase 0.5: Environment Setup
# Run on GPU server (ssh 5090)
#
# Prerequisites: PyTorch 2.10+, CUDA 12.8, diffusers 0.37+
#
# Usage:
#   bash scripts/autoaccel/setup.sh

set -e

echo "=== AutoAccel Phase 0.5 Setup ==="

# 1. SageAttention2
echo ""
echo "--- Installing SageAttention2 ---"
# RTX 5090 (Blackwell, sm_120) needs special build
# Option A: Blackwell prebuilt wheel
pip install sageattention==2.2.0 --no-build-isolation 2>/dev/null || {
    echo "Standard install failed, trying Blackwell wheel..."
    pip install sageattention --no-build-isolation \
        --find-links https://github.com/mobcat40/sageattention-blackwell/releases
}

# 2. TeaCache / First Block Cache
# Built into diffusers >= 0.35.0, no separate install needed
echo ""
echo "--- Checking diffusers version for First Block Cache ---"
python3 -c "
import diffusers
print(f'diffusers version: {diffusers.__version__}')
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
print('✅ First Block Cache available')
" || {
    echo "❌ diffusers too old, need >= 0.35.0"
    echo "pip install --upgrade diffusers"
    exit 1
}

# 3. Verify SageAttention
echo ""
echo "--- Verifying SageAttention2 ---"
python3 -c "
from sageattention import sageattn
import torch
# Quick test
q = torch.randn(1, 12, 128, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 12, 128, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 12, 128, 64, dtype=torch.bfloat16, device='cuda')
o = sageattn(q, k, v)
print(f'✅ SageAttention2 works, output shape: {o.shape}')
"

# 4. Verify Wan model access
echo ""
echo "--- Checking Wan 2.1 model ---"
python3 -c "
from diffusers import WanPipeline
print('✅ WanPipeline importable')
# Model will be downloaded on first run if not cached
"

echo ""
echo "=== Setup Complete ==="
echo "Run: python scripts/autoaccel/phase05_baseline.py --help"
