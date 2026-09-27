# AI Asset Pipeline

A containerized batch CLI for building consistent isometric pixel-art assets. The
pipeline uses JSONL manifests, YAML configuration, and the filesystem as its
processing state.

This repository implements **Milestones 1 through 8**: project setup,
CLI parsing, configuration loading, manifest loading, FLUX source generation,
PNG-to-GLB conversion with TRELLIS v1, fixed four-view Blender rendering, SDXL
ControlNet pixel conversion, build orchestration, and full pipeline orchestration.
Each stage supports resumable execution, isolated failure reporting, summaries,
and basic GPU cost estimates.

## Manifest format

Each non-empty line in a manifest is one JSON asset specification:

```json
{"id":"chair_001","prompt":"A chunky cozy fantasy oak chair","seed":1001}
```

`id` and `prompt` are required. `seed`, `type`, and `animation` are optional.

## CLI

The CLI exposes the three final command names:

```bash
python -m assetpipe source /workspace/specs/assets.jsonl
python -m assetpipe build /workspace/sources
python -m assetpipe all /workspace/specs/assets.jsonl
```

`source` generates source images, `build` converts existing source PNGs through
all three build phases, and `all` runs both pipelines.

Every command accepts:

```bash
--config /workspace/config.yaml
--force
```

`--force` is passed through every implemented stage.

By default, completed outputs are skipped. Set `runtime.skip_existing: false` or
pass `--force` to regenerate them. With `runtime.stop_on_error: false`, one bad
item is recorded under `/workspace/errors/<stage>.jsonl` and the batch continues.

## Source stage

Generate the configured FLUX source images from a JSONL manifest:

```bash
python -m assetpipe source /workspace/specs/assets.jsonl \
  --config /app/config.yaml
```

The model is loaded once for the batch. Explicit asset seeds are honored; missing
seeds use `seed_offset` plus the asset's manifest position. The configured prompt
suffix is appended to every prompt, and the optional source LoRA is loaded once.
Existing source PNGs are preserved unless `--force` is supplied.

## Build and full pipelines

Build every source PNG in a directory through TRELLIS, Blender, and pixel
conversion:

```bash
python -m assetpipe build /workspace/sources --config /app/config.yaml
```

Generate sources from a manifest and then build them:

```bash
python -m assetpipe all /workspace/specs/assets.jsonl --config /app/config.yaml
```

GPU-heavy work is phase-batched across the complete input: all meshes are created
with one TRELLIS load, Blender then renders all meshes, and all sprites are created
with one SDXL/ControlNet load.

## Blender stage

The Blender stage renders every GLB in a mesh directory using the camera and view
settings from `config.yaml`:

```python
from assetpipe.blender_stage import render_meshes
from assetpipe.config import load_config

config = load_config("config.yaml")
render_meshes(config["paths"]["meshes"], config)
```

Each mesh produces `ne.png`, `nw.png`, `sw.png`, and `se.png` under its asset
directory in the configured render root. Existing views are preserved unless
`force=True`.

## TRELLIS stage

The TRELLIS stage loads the configured model once and converts a batch of source
PNGs into GLB meshes:

```python
from assetpipe.blender_stage import render_meshes
from assetpipe.config import load_config
from assetpipe.manifest import discover_sources
from assetpipe.trellis import generate_meshes

config = load_config("config.yaml")
sources = discover_sources(config["paths"]["sources"])
generate_meshes(sources, config)
render_meshes(config["paths"]["meshes"], config)
```

Model weights are downloaded through Hugging Face on first use and cached under
the configured `HF_HOME`. TRELLIS v1 uses DINOv2 for image conditioning, so it
does not require access approval for the gated DINOv3 repository used by
TRELLIS.2. DINOv2 is cached under `/workspace/cache/torch`. Existing GLBs are
preserved unless `force=True`.

## Pixel stage

The pixel stage loads SDXL, Canny ControlNet, and the configured pixel-art LoRA
once, then converts every directional render into a final sprite:

```python
from assetpipe.config import load_config
from assetpipe.pixel import generate_sprites

config = load_config("config.yaml")
generate_sprites(config["paths"]["renders"], config)
```

The render supplies the initial image and alpha silhouette, while its Canny image
supplies structural control. The result is resized to `final_size` with nearest-
neighbor sampling and the original transparent silhouette is restored. Existing
sprite views are preserved unless `force=True`.

## Test data

The `testdata` directory contains a small manifest covering default fields and
the future-facing animation schema. Source generation requires the CUDA image and
downloads FLUX weights on first use:

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

## RunPod quick start

### Pull-and-run setup

For the simplest interactive RunPod workflow, start a GPU Pod using an Ubuntu
22.04 image. The setup installs TRELLIS v1's isolated CUDA 11.8 environment, so
the container's displayed CUDA version can be newer. Use an NVIDIA GPU with at
least 16 GB VRAM. Open its terminal and run:

```bash
cd /workspace
git clone https://github.com/OtterNox/isometric-asset-pipeline.git
cd isometric-asset-pipeline
bash setup_runpod.sh
```

The setup script installs Blender, Miniconda, original TRELLIS v1, and the
pipeline's Python packages. It uses xFormers rather than FlashAttention and
creates the standard input and output directories. The first setup takes a
while because TRELLIS compiles CUDA extensions; subsequent runs reuse the
environment and model cache stored under `/workspace`.

If this repository was already set up for TRELLIS.2, update it and run setup
again. The new TRELLIS v1 environment is installed alongside the old one, and
your existing source PNGs are kept:

```bash
cd /workspace/isometric-asset-pipeline
git pull
bash setup_runpod.sh
bash run_pipeline.sh build
```

After placing the inputs described below in `/workspace`, start a pipeline with
one short command:

```bash
bash run_pipeline.sh source
bash run_pipeline.sh build
bash run_pipeline.sh all
```

Add `--force` to regenerate completed files. Normally, simply repeat the same
command and the pipeline resumes from missing outputs.

### Prebuilt-image setup

Build and publish the image to a container registry, then create a RunPod Pod
from that image with an NVIDIA GPU that has at least 24 GB VRAM. Attach a
persistent volume at `/workspace`; this preserves inputs, outputs, LoRAs, and
downloaded Hugging Face model weights between Pods.

Place these inputs on the volume before the run:

```text
/workspace/specs/assets.jsonl
```

The source-style LoRA is optional while `source.lora.enabled` is `false`. The
RunPod setup script downloads a default SDXL pixel-art LoRA to
`/workspace/models/pixel-art.safetensors`. Replace that file if you prefer a
different compatible style. Copy `config.yaml` to `/workspace/config.yaml`
when you want to customize settings, then use one of these container commands:

```bash
all /workspace/specs/assets.jsonl --config /workspace/config.yaml
build /workspace/sources --config /workspace/config.yaml
source /workspace/specs/assets.jsonl --config /workspace/config.yaml
```

Use `all` for prompt-to-sprite, `build` for existing PNG inputs, or `source` for
prompt-to-source-PNG only. For `build`, place PNGs directly in
`/workspace/sources`; each filename stem becomes the asset ID. Add `--force` only
when you intentionally want to overwrite completed results.
