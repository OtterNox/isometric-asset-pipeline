import logging
import os
import time
from pathlib import Path

from .metrics import elapsed, estimate_cost, print_stage_summary, start_timer
from .utils import append_error, asset_id_from_path, clear_cuda


LOGGER = logging.getLogger(__name__)


def _load_runtime(model_id: str):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPCONV_ALGO", "native")

    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_id)
    pipeline.cuda()
    return pipeline, Image, postprocessing_utils


def _export_mesh(outputs, output_path: Path, trellis_config: dict, postprocessing_utils) -> None:
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=trellis_config["simplify"],
        texture_size=trellis_config["texture_size"],
    )
    glb.export(str(output_path))


def generate_meshes(
    source_paths: list[str],
    config: dict,
    force: bool = False,
) -> None:
    stage_started = start_timer()
    trellis_config = config["trellis"]
    mesh_root = Path(config["paths"]["meshes"])
    error_file = str(Path(config["paths"]["errors"]) / "trellis.jsonl")
    stop_on_error = config["runtime"]["stop_on_error"]
    regenerate = force or not config["runtime"]["skip_existing"]
    mesh_root.mkdir(parents=True, exist_ok=True)
    success = 0
    skipped = 0
    failed = 0

    pending: list[tuple[Path, Path, str]] = []
    for source_value in source_paths:
        source_path = Path(source_value)
        if not source_path.is_file():
            exc = FileNotFoundError(f"Source image not found: {source_path}")
            asset_id = asset_id_from_path(str(source_path))
            append_error(
                error_file,
                {"asset_id": asset_id, "stage": "trellis", "error": str(exc)},
            )
            LOGGER.error("[trellis] %s failed: %s", asset_id, exc)
            failed += 1
            if stop_on_error:
                seconds = elapsed(stage_started)
                print_stage_summary(
                    "trellis", len(source_paths), success, skipped, failed, seconds, "item"
                )
                print(f"[trellis] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
                raise exc
            continue

        asset_id = asset_id_from_path(str(source_path))
        output_path = mesh_root / f"{asset_id}.glb"
        if output_path.is_file() and not regenerate:
            LOGGER.info("[trellis] %s skipped", asset_id)
            skipped += 1
            continue
        pending.append((source_path, output_path, asset_id))

    if not pending:
        seconds = elapsed(stage_started)
        print_stage_summary(
            "trellis", len(source_paths), success, skipped, failed, seconds, "item"
        )
        print(f"[trellis] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
        return

    LOGGER.info("[trellis] loading %s", trellis_config["model"])
    pipeline = None
    try:
        pipeline, image_module, postprocessing_utils = _load_runtime(
            trellis_config["model"]
        )
    except Exception as exc:
        for _, _, asset_id in pending:
            append_error(
                error_file,
                {"asset_id": asset_id, "stage": "trellis", "error": str(exc)},
            )
            LOGGER.error("[trellis] %s failed: %s", asset_id, exc)
        failed += len(pending)
        if pipeline is not None:
            del pipeline
        clear_cuda()
        seconds = elapsed(stage_started)
        print_stage_summary(
            "trellis", len(source_paths), success, skipped, failed, seconds, "item"
        )
        print(f"[trellis] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
        if stop_on_error:
            raise
        return

    fatal_error = None
    try:
        for source_path, output_path, asset_id in pending:
            started = time.perf_counter()
            temporary_path = output_path.with_suffix(".tmp.glb")
            outputs = None
            try:
                LOGGER.info("[trellis] %s generating", asset_id)
                with image_module.open(source_path) as image:
                    outputs = pipeline.run(
                        image,
                        seed=trellis_config["seed"],
                    )
                _export_mesh(
                    outputs,
                    temporary_path,
                    trellis_config,
                    postprocessing_utils,
                )
                if not temporary_path.is_file():
                    raise RuntimeError(f"TRELLIS did not create output: {output_path}")
                temporary_path.replace(output_path)
                success += 1
                LOGGER.info(
                    "[trellis] %s completed in %.1fs",
                    asset_id,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                failed += 1
                append_error(
                    error_file,
                    {"asset_id": asset_id, "stage": "trellis", "error": str(exc)},
                )
                LOGGER.error("[trellis] %s failed: %s", asset_id, exc)
                if stop_on_error:
                    fatal_error = exc
                    break
            finally:
                if outputs is not None:
                    del outputs
    finally:
        del pipeline
        clear_cuda()

    seconds = elapsed(stage_started)
    print_stage_summary(
        "trellis", len(source_paths), success, skipped, failed, seconds, "item"
    )
    print(f"[trellis] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
    if fatal_error is not None:
        raise fatal_error
