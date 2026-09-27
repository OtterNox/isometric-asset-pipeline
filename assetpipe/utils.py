import gc
import json
from pathlib import Path


def clear_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def asset_id_from_path(path: str) -> str:
    return Path(path).stem


def append_error(error_file: str, payload: dict) -> None:
    path = Path(error_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
