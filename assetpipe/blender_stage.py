import json
import logging
import subprocess
import time
from pathlib import Path

from .utils import asset_id_from_path


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
    directory = Path(mesh_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Mesh directory not found: {directory}")

    blender_config = config["blender"]
    render_root = Path(config["paths"]["renders"])
    views = blender_config["views"]
    script_path = _blender_script_path()

    if not script_path.is_file():
        raise FileNotFoundError(f"Blender render script not found: {script_path}")

    for mesh_path in _mesh_paths(directory):
        asset_id = asset_id_from_path(str(mesh_path))
        output_dir = render_root / asset_id
        expected_outputs = [output_dir / f"{name}.png" for name in views]

        if not force and all(path.is_file() for path in expected_outputs):
            LOGGER.info("[blender] %s skipped", asset_id)
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
        if force:
            command.append("--force")

        LOGGER.info("[blender] %s rendering", asset_id)
        started = time.perf_counter()
        subprocess.run(command, check=True)
        LOGGER.info(
            "[blender] %s completed in %.1fs",
            asset_id,
            time.perf_counter() - started,
        )
