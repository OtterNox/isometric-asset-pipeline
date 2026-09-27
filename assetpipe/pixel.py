import logging
import time
from pathlib import Path

from .metrics import elapsed, estimate_cost, print_stage_summary, start_timer
from .utils import append_error, clear_cuda


LOGGER = logging.getLogger(__name__)
LORA_ADAPTER_NAME = "pixel_style"
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200


def _load_runtime(pixel_config: dict):
    import cv2
    import numpy as np
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
    from PIL import Image

    controlnet = ControlNetModel.from_pretrained(
        pixel_config["controlnet"],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        pixel_config["model"],
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline, Image, np, cv2


def _load_lora(pipeline, lora_config: dict) -> None:
    lora_path = Path(lora_config["path"])
    if not lora_path.is_file():
        raise FileNotFoundError(f"Pixel LoRA not found: {lora_path}")

    pipeline.load_lora_weights(
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name=LORA_ADAPTER_NAME,
    )
    pipeline.set_adapters(
        LORA_ADAPTER_NAME,
        adapter_weights=lora_config["weight"],
    )


def _discover_renders(render_root: Path) -> list[tuple[str, Path]]:
    renders: list[tuple[str, Path]] = []
    for asset_dir in sorted(path for path in render_root.iterdir() if path.is_dir()):
        renders.extend(
            (asset_dir.name, path)
            for path in sorted(asset_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".png"
        )
    return renders


def _prepare_images(render_path: Path, image_module, np, cv2):
    with image_module.open(render_path) as opened:
        render = opened.convert("RGBA")

    background = image_module.new("RGBA", render.size, (255, 255, 255, 255))
    initial_image = image_module.alpha_composite(background, render).convert("RGB")
    grayscale = cv2.cvtColor(np.array(initial_image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grayscale, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    control_image = image_module.fromarray(np.stack([edges] * 3, axis=-1))
    return render, initial_image, control_image


def _save_sprite(generated, alpha, output_path: Path, final_size: int, image_module) -> None:
    size = (final_size, final_size)
    sprite = generated.convert("RGB").resize(size, image_module.Resampling.NEAREST)
    sprite.putalpha(alpha.resize(size, image_module.Resampling.NEAREST))
    sprite.save(output_path, format="PNG")


def generate_sprites(
    render_root: str,
    config: dict,
    force: bool = False,
) -> None:
    stage_started = start_timer()
    render_directory = Path(render_root)
    if not render_directory.is_dir():
        raise NotADirectoryError(f"Render directory not found: {render_directory}")

    pixel_config = config["pixel"]
    sprite_root = Path(config["paths"]["sprites"])
    error_file = str(Path(config["paths"]["errors"]) / "pixel.jsonl")
    stop_on_error = config["runtime"]["stop_on_error"]
    regenerate = force or not config["runtime"]["skip_existing"]
    success = 0
    skipped = 0
    failed = 0
    pending: list[tuple[str, Path, Path]] = []
    renders = _discover_renders(render_directory)

    for asset_id, render_path in renders:
        output_path = sprite_root / asset_id / render_path.name
        if output_path.is_file() and not regenerate:
            LOGGER.info("[pixel] %s/%s skipped", asset_id, render_path.stem)
            skipped += 1
            continue
        pending.append((asset_id, render_path, output_path))

    if not pending:
        seconds = elapsed(stage_started)
        print_stage_summary(
            "pixel", len(renders), success, skipped, failed, seconds, "view"
        )
        print(
            f"[pixel] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}"
        )
        return

    LOGGER.info("[pixel] loading %s", pixel_config["model"])
    pipeline = None
    try:
        pipeline, image_module, np, cv2 = _load_runtime(pixel_config)
        _load_lora(pipeline, pixel_config["lora"])
    except Exception as exc:
        for asset_id, render_path, _ in pending:
            item_id = f"{asset_id}/{render_path.stem}"
            append_error(
                error_file,
                {"asset_id": item_id, "stage": "pixel", "error": str(exc)},
            )
            LOGGER.error("[pixel] %s failed: %s", item_id, exc)
        failed = len(pending)
        if pipeline is not None:
            del pipeline
        clear_cuda()
        seconds = elapsed(stage_started)
        print_stage_summary(
            "pixel", len(renders), success, skipped, failed, seconds, "view"
        )
        print(
            f"[pixel] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}"
        )
        if stop_on_error:
            raise
        return

    fatal_error = None
    try:
        for asset_id, render_path, output_path in pending:
            started = time.perf_counter()
            temporary_path = output_path.with_suffix(".tmp.png")
            item_id = f"{asset_id}/{render_path.stem}"
            render = initial_image = control_image = generated = None
            try:
                LOGGER.info("[pixel] %s generating", item_id)
                render, initial_image, control_image = _prepare_images(
                    render_path,
                    image_module,
                    np,
                    cv2,
                )
                generated = pipeline(
                    prompt=pixel_config["prompt"],
                    negative_prompt=pixel_config["negative_prompt"],
                    image=initial_image,
                    control_image=control_image,
                    strength=pixel_config["denoise"],
                    controlnet_conditioning_scale=pixel_config["control_strength"],
                    num_inference_steps=pixel_config["steps"],
                    guidance_scale=pixel_config["guidance"],
                ).images[0]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _save_sprite(
                    generated,
                    render.getchannel("A"),
                    temporary_path,
                    pixel_config["final_size"],
                    image_module,
                )
                if not temporary_path.is_file():
                    raise RuntimeError(
                        f"Pixel stage did not create output: {output_path}"
                    )
                temporary_path.replace(output_path)
                success += 1
                LOGGER.info(
                    "[pixel] %s completed in %.1fs",
                    item_id,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                failed += 1
                append_error(
                    error_file,
                    {"asset_id": item_id, "stage": "pixel", "error": str(exc)},
                )
                LOGGER.error("[pixel] %s failed: %s", item_id, exc)
                if stop_on_error:
                    fatal_error = exc
                    break
            finally:
                del render, initial_image, control_image, generated
    finally:
        del pipeline
        clear_cuda()

    seconds = elapsed(stage_started)
    print_stage_summary(
        "pixel", len(renders), success, skipped, failed, seconds, "view"
    )
    print(
        f"[pixel] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}"
    )
    if fatal_error is not None:
        raise fatal_error
