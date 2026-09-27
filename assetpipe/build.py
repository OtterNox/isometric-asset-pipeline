import logging

from .blender_stage import render_meshes
from .manifest import discover_sources
from .pixel import generate_sprites
from .trellis import generate_meshes


LOGGER = logging.getLogger(__name__)


def build_assets(source_dir: str, config: dict, force: bool = False) -> None:
    source_paths = discover_sources(source_dir)
    LOGGER.info("[build] discovered %d source PNG(s)", len(source_paths))

    LOGGER.info("[build] starting TRELLIS phase")
    generate_meshes(source_paths, config, force)

    LOGGER.info("[build] starting Blender phase")
    render_meshes(config["paths"]["meshes"], config, force)

    LOGGER.info("[build] starting pixel phase")
    generate_sprites(config["paths"]["renders"], config, force)
