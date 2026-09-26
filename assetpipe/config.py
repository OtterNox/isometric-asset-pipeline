from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = ("source", "trellis", "blender", "pixel", "paths", "runtime")
REQUIRED_PATHS = (
    "workspace",
    "specs",
    "sources",
    "meshes",
    "renders",
    "sprites",
    "cache",
    "models",
    "errors",
)


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a YAML mapping")

    missing_sections = [key for key in REQUIRED_SECTIONS if key not in config]
    if missing_sections:
        raise ValueError(
            f"Config is missing required section(s): {', '.join(missing_sections)}"
        )

    paths = config["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Config section 'paths' must be a mapping")

    missing_paths = [key for key in REQUIRED_PATHS if not paths.get(key)]
    if missing_paths:
        raise ValueError(
            f"Config is missing required path(s): {', '.join(missing_paths)}"
        )

    return config


def ensure_dirs(config: dict) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Config section 'paths' must be a mapping")

    for name in REQUIRED_PATHS:
        value = paths.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config path '{name}' must be a non-empty string")
        Path(value).mkdir(parents=True, exist_ok=True)
