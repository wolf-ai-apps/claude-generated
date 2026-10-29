#!/usr/bin/env bash
# Lab 3a: inspect every layer of the GPU software stack, top to bottom.
# Run on a GPU host (or inside a GPU container) and read the comments next to each check.
set -u

section() { printf '\n== %s\n' "$1"; }

section "1. Kernel driver (installed on the HOST, never in the container)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
  # The "CUDA Version" in the nvidia-smi banner is the MAX CUDA runtime this driver supports,
  # not what is installed.
  nvidia-smi | grep -o "CUDA Version: [0-9.]*"
else
  echo "nvidia-smi not found: no driver, or you are in a container started without --gpus"
fi

section "2. GPU topology (how GPUs talk to each other: NV# = NVLink, PIX/PHB/SYS = PCIe paths)"
nvidia-smi topo -m 2>/dev/null || echo "n/a"

section "3. CUDA toolkit (only needed to COMPILE kernels; PyTorch wheels bundle their own runtime)"
command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "nvcc not installed (fine for serving)"

section "4. Container runtime hook (lets containers see GPUs)"
command -v nvidia-ctk >/dev/null && nvidia-ctk --version | head -1 || echo "nvidia-container-toolkit not found"

section "5. Framework view (PyTorch's bundled CUDA runtime, cuDNN and NCCL)"
python3 - <<'EOF' 2>/dev/null || echo "torch not installed"
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("cuDNN", torch.backends.cudnn.version())
try:
    print("NCCL", ".".join(map(str, torch.cuda.nccl.version())))
except Exception as e:
    print("NCCL n/a:", e)
EOF

section "6. Rule to remember"
echo "driver (host) >= what the CUDA runtime (in your image/wheel) requires."
echo "Newer driver + older runtime: OK.  Older driver + newer runtime: 'CUDA driver version is insufficient'."
