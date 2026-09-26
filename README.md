# AI Asset Pipeline

A containerized batch CLI for building consistent isometric pixel-art assets. The
pipeline uses JSONL manifests, YAML configuration, and the filesystem as its
processing state.

This repository currently implements **Milestone 1 only**: project setup, CLI
parsing, configuration loading, manifest loading, source discovery, and workspace
directory creation. The generation and rendering stages are intentionally not yet
implemented.

## Manifest format

Each non-empty line in a manifest is one JSON asset specification:

```json
{"id":"chair_001","prompt":"A chunky cozy fantasy oak chair","seed":1001}
```

`id` and `prompt` are required. `seed`, `type`, and `animation` are optional.

## CLI

The Milestone 1 commands validate their input and prepare configured directories:

```bash
python -m assetpipe source /workspace/specs/assets.jsonl
python -m assetpipe build /workspace/sources
python -m assetpipe all /workspace/specs/assets.jsonl
```

Every command accepts:

```bash
--config /workspace/config.yaml
--force
```

`--force` is parsed now and will control regeneration when the processing stages
are added in later milestones.

## Test data

The `testdata` directory contains a small manifest covering default fields and
the future-facing animation schema:

```bash
python -m assetpipe source testdata/assets.jsonl --config testdata/config.yaml
python -m assetpipe build testdata/sources --config testdata/config.yaml
```

## Docker

Build the single CUDA image:

```bash
docker build -t asset-pipeline .
```

Run it with an NVIDIA GPU and a mounted workspace:

```bash
docker run --rm --gpus all \
  -v "$PWD/workspace:/workspace" \
  asset-pipeline source /workspace/specs/assets.jsonl \
  --config /app/config.yaml
```

Model weights are not included in the image. Hugging Face downloads are cached
under `/workspace/cache/huggingface`.
