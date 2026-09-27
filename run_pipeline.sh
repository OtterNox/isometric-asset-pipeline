#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-}"
DATA_ROOT="${ASSETPIPE_DATA_ROOT:-/workspace}"

case "$COMMAND" in
    source|all)
        DEFAULT_INPUT="$DATA_ROOT/outputs/specs/assets.jsonl"
        ;;
    build)
        DEFAULT_INPUT="$DATA_ROOT/outputs/sources"
        ;;
    *)
        echo "Usage: bash run_pipeline.sh source|build|all [--force]"
        echo
        echo "  source  prompts -> source PNGs"
        echo "  build   source PNGs -> meshes -> renders -> sprites"
        echo "  all     prompts -> source PNGs -> meshes -> renders -> sprites"
        exit 2
        ;;
esac

shift
INPUT_PATH="${ASSETPIPE_INPUT:-$DEFAULT_INPUT}"
CONFIG_PATH="${ASSETPIPE_CONFIG:-$REPO_DIR/config.yaml}"
INSTALL_ROOT="${ASSETPIPE_INSTALL_ROOT:-/opt/assetpipe}"
PYTHON_BIN="$INSTALL_ROOT/conda/envs/trellis/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Pipeline environment not found. Run: bash setup_runpod.sh"
    exit 1
fi

if [ ! -e "$INPUT_PATH" ]; then
    echo "Input not found: $INPUT_PATH"
    exit 1
fi

if [ "$COMMAND" != "source" ] && [ ! -f "$DATA_ROOT/loras/pixel-art.safetensors" ]; then
    echo "Warning: $DATA_ROOT/loras/pixel-art.safetensors is missing."
    echo "Meshes and renders can still be generated, but the pixel stage will record failures."
fi

export TRELLIS_DIR="${TRELLIS_DIR:-$INSTALL_ROOT/TRELLIS}"
cd "$REPO_DIR"
export PYTHONPATH="$TRELLIS_DIR:${PYTHONPATH:-}"
export ASSETPIPE_DATA_ROOT="$DATA_ROOT"
export HF_HOME="${HF_HOME:-$DATA_ROOT/models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/models/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache}"
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPCONV_ALGO="${SPCONV_ALGO:-native}"

exec "$PYTHON_BIN" -m assetpipe "$COMMAND" "$INPUT_PATH" \
    --config "$CONFIG_PATH" "$@"
