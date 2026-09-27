#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${ASSETPIPE_INSTALL_ROOT:-/workspace/.assetpipe}"
TRELLIS_DIR="$INSTALL_ROOT/TRELLIS"
TRELLIS_COMMIT="442aa1e1afb9014e80681d3bf604e8d728a86ee7"
CONDA_DIR="$INSTALL_ROOT/conda"
ENV_NAME="trellis"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU is visible. Start this script inside a GPU RunPod."
    exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
elif command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
else
    echo "Root access or sudo is required for the one-time setup."
    exit 1
fi

echo "[setup] Installing Blender and system build tools..."
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    blender build-essential ca-certificates curl git libgl1 libglib2.0-0 \
    libjpeg-dev sudo

mkdir -p "$INSTALL_ROOT"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "[setup] Installing Miniconda..."
    curl -fsSL -o /tmp/assetpipe-miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/assetpipe-miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/assetpipe-miniconda.sh
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

if [ ! -d "$TRELLIS_DIR/.git" ]; then
    echo "[setup] Downloading TRELLIS v1..."
    git clone https://github.com/microsoft/TRELLIS.git "$TRELLIS_DIR"
fi

cd "$TRELLIS_DIR"
git fetch --all --tags
git checkout "$TRELLIS_COMMIT"
git submodule update --init --recursive

INSTALL_MARKER="$TRELLIS_DIR/.assetpipe-installed-$TRELLIS_COMMIT"
if [ ! -f "$INSTALL_MARKER" ]; then
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "[setup] Creating the TRELLIS Python environment..."
        conda create -y --override-channels -c conda-forge \
            -n "$ENV_NAME" python=3.10 pip
    fi

    conda activate "$ENV_NAME"
    echo "[setup] Installing the CUDA 11.8 toolkit used by TRELLIS v1..."
    conda install -y --override-channels \
        -c nvidia/label/cuda-11.8.0 -c conda-forge cuda
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"

    echo "[setup] Installing the official TRELLIS v1 PyTorch stack..."
    conda install -y --override-channels \
        -c pytorch -c nvidia -c conda-forge \
        pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8

    echo "[setup] Installing build tools..."
    python -m pip install --upgrade setuptools wheel packaging ninja

    echo "[setup] Installing TRELLIS v1 dependencies. This is the long step."
    rm -rf /tmp/extensions
    export PIP_NO_BUILD_ISOLATION=1
    . ./setup.sh --basic --xformers --diffoctreerast \
        --spconv --mipgaussian --kaolin --nvdiffrast
    unset PIP_NO_BUILD_ISOLATION
    rm -rf /tmp/extensions
    touch "$INSTALL_MARKER"
fi

echo "[setup] Installing the asset pipeline..."
conda run -n "$ENV_NAME" python -m pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"

echo "[setup] Verifying TRELLIS v1 and GPU access..."
ATTN_BACKEND=xformers SPCONV_ALGO=native PYTHONPATH="$TRELLIS_DIR" \
    conda run -n "$ENV_NAME" python -c \
    "import torch; from trellis.pipelines import TrellisImageTo3DPipeline; assert torch.cuda.is_available(), 'PyTorch cannot see the RunPod GPU'; print('TRELLIS v1 environment OK:', torch.__version__, 'CUDA', torch.version.cuda)"

mkdir -p \
    /workspace/specs /workspace/sources /workspace/meshes \
    /workspace/renders /workspace/sprites /workspace/cache \
    /workspace/models /workspace/errors

PIXEL_LORA="/workspace/models/pixel-art.safetensors"
PIXEL_LORA_SHA256="4234637cb80c998f41e348e6a6cb6bc20d8d038b2b0f256b6129b3b5e353eef7"
if [ ! -f "$PIXEL_LORA" ]; then
    echo "[setup] Downloading the default SDXL pixel-art LoRA..."
    curl -fL --retry 3 \
        https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors \
        -o "$PIXEL_LORA.download"
    echo "$PIXEL_LORA_SHA256  $PIXEL_LORA.download" | sha256sum -c -
    mv "$PIXEL_LORA.download" "$PIXEL_LORA"
fi

echo "[setup] Removing installer caches to preserve workspace space..."
"$CONDA_DIR/bin/conda" clean --all -y
conda run -n "$ENV_NAME" python -m pip cache purge || true

echo
echo "Setup complete."
echo "Next: put your inputs in /workspace, then run:"
echo "  cd $REPO_DIR"
echo "  bash run_pipeline.sh all"
