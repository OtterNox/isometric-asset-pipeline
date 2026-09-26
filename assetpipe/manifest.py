import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AnimationSpec:
    enabled: bool = False
    kind: Optional[str] = None
    preset: Optional[str] = None
    clips: Optional[list[str]] = None
    fps: int = 8
    frames: Optional[int] = None
    directions: str = "four"


@dataclass
class AssetSpec:
    id: str
    prompt: str
    seed: Optional[int] = None
    type: str = "prop"
    animation: AnimationSpec = field(default_factory=AnimationSpec)


def _parse_spec(payload: object, line_number: int) -> AssetSpec:
    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number}: asset spec must be a JSON object")

    for required in ("id", "prompt"):
        value = payload.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Line {line_number}: '{required}' must be a non-empty string"
            )

    animation_payload = payload.get("animation")
    if animation_payload is None:
        animation = AnimationSpec()
    elif isinstance(animation_payload, dict):
        try:
            animation = AnimationSpec(**animation_payload)
        except TypeError as exc:
            raise ValueError(f"Line {line_number}: invalid animation spec: {exc}") from exc
    else:
        raise ValueError(f"Line {line_number}: 'animation' must be an object")

    spec_payload = dict(payload)
    spec_payload["animation"] = animation
    try:
        return AssetSpec(**spec_payload)
    except TypeError as exc:
        raise ValueError(f"Line {line_number}: invalid asset spec: {exc}") from exc


def load_specs(path: str) -> list[AssetSpec]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    specs: list[AssetSpec] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            specs.append(_parse_spec(payload, line_number))
    return specs


def discover_sources(source_dir: str) -> list[str]:
    directory = Path(source_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Source directory not found: {directory}")
    return sorted(
        str(path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
