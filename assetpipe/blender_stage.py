import json
import logging
import subprocess
import time
from pathlib import Path

from .metrics import elapsed, estimate_cost, print_stage_summary, start_timer
from .utils import append_error, asset_id_from_path


LOGGER = logging.getLogger(__name__)


def _blender_script_path() -> Path:
    return Path(__file__).resolve().parent.parent / "blender" / "render_asset.py"


def _mesh_paths(mesh_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in mesh_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".glb"
    )


def render_meshes(mesh_dir: str, config: dict, force: bool = False) -> None:
    stage_started = start_timer()
    directory = Path(mesh_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Mesh directory not found: {directory}")

    blender_config = config["blender"]
    render_root = Path(config["paths"]["renders"])
    error_file = str(Path(config["paths"]["errors"]) / "blender.jsonl")
    stop_on_error = config["runtime"]["stop_on_error"]
    regenerate = force or not config["runtime"]["skip_existing"]
    views = blender_config["views"]
    script_path = _blender_script_path()

    if not script_path.is_file():
        raise FileNotFoundError(f"Blender render script not found: {script_path}")

    mesh_paths = _mesh_paths(directory)
    success = 0
    skipped = 0
    failed = 0
    fatal_error = None

    for mesh_path in mesh_paths:
        asset_id = asset_id_from_path(str(mesh_path))
        output_dir = render_root / asset_id
        expected_outputs = [output_dir / f"{name}.png" for name in views]

        if not regenerate and all(path.is_file() for path in expected_outputs):
            LOGGER.info("[blender] %s skipped", asset_id)
            skipped += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(blender_config["executable"]),
            "--background",
            "--python",
            str(script_path),
            "--",
            "--input",
            str(mesh_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--render-size",
            str(blender_config["render_size"]),
            "--engine",
            str(blender_config["engine"]),
            "--elevation",
            str(blender_config["camera"]["elevation_deg"]),
            "--views-json",
            json.dumps(views),
            "--transparent",
            str(bool(blender_config["transparent"])).lower(),
        ]
        if regenerate:
            command.append("--force")

        LOGGER.info("[blender] %s rendering", asset_id)
        started = time.perf_counter()
        try:
            subprocess.run(command, check=True)
            missing = [path.name for path in expected_outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    f"Blender did not create expected view(s): {', '.join(missing)}"
                )
            success += 1
            LOGGER.info(
                "[blender] %s completed in %.1fs",
                asset_id,
                time.perf_counter() - started,
            )
        except Exception as exc:
            failed += 1
            append_error(
                error_file,
                {"asset_id": asset_id, "stage": "blender", "error": str(exc)},
            )
            LOGGER.error("[blender] %s failed: %s", asset_id, exc)
            if stop_on_error:
                fatal_error = exc
                break

    seconds = elapsed(stage_started)
    print_stage_summary(
        "blender", len(mesh_paths), success, skipped, failed, seconds, "asset"
    )
    print(f"[blender] estimated_cost=${estimate_cost(seconds, config['runtime']['gpu_hourly_cost']):.4f}")
    if fatal_error is not None:
        raise fatal_error
