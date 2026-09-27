import logging
import time
from pathlib import Path

from .manifest import AssetSpec
from .metrics import elapsed, estimate_cost, print_stage_summary, start_timer
from .utils import append_error, clear_cuda


LOGGER = logging.getLogger(__name__)
LORA_ADAPTER_NAME = "source_style"


def _load_runtime(model_id: str):
    import torch
    from diffusers import FluxPipeline

    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    return pipeline, torch


def _load_lora(pipeline, lora_config: dict) -> None:
    if not lora_config["enabled"]:
        return

    lora_path = Path(lora_config["path"])
    if not lora_path.is_file():
        raise FileNotFoundError(f"Source LoRA not found: {lora_path}")

    pipeline.load_lora_weights(
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name=LORA_ADAPTER_NAME,
    )
    pipeline.set_adapters(
        LORA_ADAPTER_NAME,
        adapter_weights=lora_config["weight"],
    )


def _prompt_for(spec: AssetSpec, suffix: str) -> str:
    return ", ".join(part.strip() for part in (spec.prompt, suffix) if part.strip())


def generate_sources(
    specs: list[AssetSpec],
    config: dict,
    force: bool = False,
) -> None:
    stage_started = start_timer()
    source_config = config["source"]
    source_root = Path(config["paths"]["sources"])
    error_file = str(Path(config["paths"]["errors"]) / "source.jsonl")
    stop_on_error = config["runtime"]["stop_on_error"]
    regenerate = force or not config["runtime"]["skip_existing"]
    source_root.mkdir(parents=True, exist_ok=True)
    success = 0
    skipped = 0
    failed = 0

    pending: list[tuple[AssetSpec, Path, int]] = []
    for index, spec in enumerate(specs):
        output_path = source_root / f"{spec.id}.png"
        if output_path.is_file() and not regenerate:
            LOGGER.info("[source] %s skipped", spec.id)
            skipped += 1
            continue
        seed = spec.seed if spec.seed is not None else source_config["seed_offset"] + index
        pending.append((spec, output_path, seed))

    if not pending:
        seconds = elapsed(stage_started)
        print_stage_summary("source", len(specs), success, skipped, failed, seconds, "item")
        print(f"[source] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
        return

    LOGGER.info("[source] loading %s", source_config["model"])
    pipeline = None
    try:
        pipeline, torch = _load_runtime(source_config["model"])
        _load_lora(pipeline, source_config["lora"])
    except Exception as exc:
        for spec, _, _ in pending:
            append_error(
                error_file,
                {"asset_id": spec.id, "stage": "source", "error": str(exc)},
            )
            LOGGER.error("[source] %s failed: %s", spec.id, exc)
        failed = len(pending)
        if pipeline is not None:
            del pipeline
        clear_cuda()
        seconds = elapsed(stage_started)
        print_stage_summary("source", len(specs), success, skipped, failed, seconds, "item")
        print(f"[source] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
        if stop_on_error:
            raise
        return

    fatal_error = None
    try:
        for spec, output_path, seed in pending:
            started = time.perf_counter()
            temporary_path = output_path.with_suffix(".tmp.png")
            try:
                LOGGER.info("[source] %s generating", spec.id)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                image = pipeline(
                    prompt=_prompt_for(spec, source_config["prompt_suffix"]),
                    width=source_config["width"],
                    height=source_config["height"],
                    num_inference_steps=source_config["steps"],
                    guidance_scale=source_config["guidance"],
                    generator=generator,
                ).images[0]
                image.save(temporary_path, format="PNG")
                if not temporary_path.is_file():
                    raise RuntimeError(f"Source model did not create output: {output_path}")
                temporary_path.replace(output_path)
                del image
                success += 1
                LOGGER.info(
                    "[source] %s completed in %.1fs",
                    spec.id,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                failed += 1
                append_error(
                    error_file,
                    {"asset_id": spec.id, "stage": "source", "error": str(exc)},
                )
                LOGGER.error("[source] %s failed: %s", spec.id, exc)
                if stop_on_error:
                    fatal_error = exc
                    break
    finally:
        del pipeline
        clear_cuda()

    seconds = elapsed(stage_started)
    print_stage_summary("source", len(specs), success, skipped, failed, seconds, "item")
    print(f"[source] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
    if fatal_error is not None:
        raise fatal_error
