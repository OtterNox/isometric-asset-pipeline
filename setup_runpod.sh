#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${ASSETPIPE_INSTALL_ROOT:-/workspace/.assetpipe}"
TRELLIS_DIR="$INSTALL_ROOT/TRELLIS.2"
TRELLIS_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
CONDA_DIR="$INSTALL_ROOT/conda"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU is visible. Start this script inside a GPU RunPod."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "The CUDA compiler (nvcc) is missing. Use a RunPod image with CUDA 12.4 development tools."
    exit 1
fi

if ! nvcc --version | grep -q "release 12.4"; then
    echo "CUDA 12.4 is required by the pinned TRELLIS environment."
    echo "Choose a RunPod image whose CUDA version is 12.4, then try again."
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
NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC_PATH")")}"

if [ ! -d "$TRELLIS_DIR/.git" ]; then
    echo "[setup] Downloading TRELLIS.2..."
    git clone https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
fi

cd "$TRELLIS_DIR"
git fetch --all --tags
git checkout "$TRELLIS_COMMIT"
git submodule update --init --recursive

INSTALL_MARKER="$TRELLIS_DIR/.assetpipe-installed-$TRELLIS_COMMIT"
if [ ! -f "$INSTALL_MARKER" ]; then
    echo "[setup] Compiling TRELLIS dependencies. This is the long step."
    if conda env list | awk '{print $1}' | grep -qx trellis2; then
        conda activate trellis2
        . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec \
            --cumesh --o-voxel --flexgemm
    else
        . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec \
            --cumesh --o-voxel --flexgemm
    fi
    touch "$INSTALL_MARKER"
fi

echo "[setup] Installing the asset pipeline..."
conda run -n trellis2 python -m pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"

mkdir -p \
    /workspace/specs /workspace/sources /workspace/meshes \
    /workspace/renders /workspace/sprites /workspace/cache \
    /workspace/models /workspace/errors

echo
echo "Setup complete."
echo "Next: put your inputs in /workspace, then run:"
echo "  cd $REPO_DIR"
echo "  bash run_pipeline.sh all"
