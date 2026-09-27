#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${ASSETPIPE_APP_DIR:-/root/app}"
DATA_ROOT="${ASSETPIPE_DATA_ROOT:-/workspace}"
REPO_URL="${ASSETPIPE_REPO_URL:-https://github.com/OtterNox/isometric-asset-pipeline.git}"
GIT_REF="${ASSETPIPE_GIT_REF:-main}"

case "$APP_DIR" in
    /|/root|/opt|"$DATA_ROOT"|"$DATA_ROOT"/*)
        echo "Refusing unsafe ASSETPIPE_APP_DIR: $APP_DIR" >&2
        exit 2
        ;;
esac

mkdir -p \
    "$DATA_ROOT/models" "$DATA_ROOT/loras" "$DATA_ROOT/cache" \
    "$DATA_ROOT/outputs/specs" "$DATA_ROOT/outputs/sources" \
    "$DATA_ROOT/outputs/meshes" "$DATA_ROOT/outputs/renders" \
    "$DATA_ROOT/outputs/sprites" "$DATA_ROOT/outputs/errors"

if [ -d "$APP_DIR/.git" ] && \
   [ "$(git -C "$APP_DIR" remote get-url origin)" = "$REPO_URL" ]; then
    echo "[startup] Updating application in $APP_DIR..."
    git -C "$APP_DIR" fetch --depth 1 origin "$GIT_REF"
    git -C "$APP_DIR" reset --hard FETCH_HEAD
else
    echo "[startup] Cloning application into $APP_DIR..."
    rm -rf "$APP_DIR"
    git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" "$APP_DIR"
fi

export ASSETPIPE_DATA_ROOT="$DATA_ROOT"
export ASSETPIPE_CONFIG="${ASSETPIPE_CONFIG:-$APP_DIR/config.yaml}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$DATA_ROOT/models/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache}"
export PYTHONPATH="${TRELLIS_DIR:-/opt/TRELLIS}:$APP_DIR:${PYTHONPATH:-}"

if [ "$#" -eq 0 ]; then
    set -- "${ASSETPIPE_COMMAND:-all}"
fi

case "$1" in
    source|build|all)
        exec bash "$APP_DIR/run_pipeline.sh" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
