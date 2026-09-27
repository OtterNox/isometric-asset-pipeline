import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assetpipe.blender_stage import render_meshes
from assetpipe.manifest import AssetSpec
from assetpipe.pixel import generate_sprites
from assetpipe.source import generate_sources
from assetpipe.trellis import generate_meshes


class _SavedImage:
    def save(self, path, format=None):
        Path(path).write_bytes(b"image")


class _Result:
    images = [_SavedImage()]


class _Generator:
    def manual_seed(self, seed):
        return self


class _Torch:
    class Generator:
        def __new__(cls, device=None):
            return _Generator()


class _SourcePipeline:
    def __call__(self, **kwargs):
        if kwargs["prompt"].startswith("bad"):
            raise RuntimeError("source failure")
        return _Result()


class _SuccessfulSourcePipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return _Result()


class _OpenedImage:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _ImageModule:
    @staticmethod
    def open(path):
        return _OpenedImage()


class _MeshPipeline:
    def run(self, image, pipeline_type):
        return [object()]


class _Render:
    @staticmethod
    def getchannel(name):
        return object()


class _PixelPipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("pixel failure")
        return _Result()


def _config(root: Path) -> dict:
    paths = {
        name: str(root / name)
        for name in (
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
    }
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return {
        "source": {
            "model": "source-model",
            "width": 8,
            "height": 8,
            "steps": 1,
            "guidance": 0.0,
            "seed_offset": 1,
            "prompt_suffix": "suffix",
            "lora": {"enabled": False, "path": "", "weight": 1.0},
        },
        "trellis": {
            "model": "mesh-model",
            "resolution": 512,
            "texture_size": 8,
            "decimation_target": 10,
            "remesh": True,
        },
        "blender": {
            "executable": "blender",
            "render_size": 8,
            "transparent": True,
            "engine": "BLENDER_EEVEE_NEXT",
            "camera": {"elevation_deg": 35.264},
            "views": {"ne": 45, "nw": 135, "sw": 225, "se": 315},
        },
        "pixel": {
            "model": "pixel-model",
            "controlnet": "control-model",
            "prompt": "pixel",
            "negative_prompt": "bad",
            "lora": {"path": "unused", "weight": 1.0},
            "denoise": 0.25,
            "control_strength": 0.9,
            "steps": 1,
            "guidance": 1.0,
            "final_size": 8,
        },
        "paths": paths,
        "runtime": {
            "gpu_hourly_cost": 1.0,
            "stop_on_error": False,
            "skip_existing": True,
        },
    }


class ReliabilityTests(unittest.TestCase):
    def test_source_resumes_and_records_failure(self):
        with tempfile.TemporaryDirectory() as value:
            config = _config(Path(value))
            sources = Path(config["paths"]["sources"])
            (sources / "done.png").write_bytes(b"done")
            specs = [
                AssetSpec("done", "done"),
                AssetSpec("good", "good"),
                AssetSpec("bad", "bad"),
            ]
            output = io.StringIO()
            with (
                patch("assetpipe.source._load_runtime", return_value=(_SourcePipeline(), _Torch)),
                patch("assetpipe.source.clear_cuda"),
                contextlib.redirect_stdout(output),
            ):
                generate_sources(specs, config)

            self.assertTrue((sources / "good.png").is_file())
            self.assertFalse((sources / "bad.png").exists())
            self.assertFalse((sources / "bad.tmp.png").exists())
            error = json.loads(
                (Path(config["paths"]["errors"]) / "source.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(error["asset_id"], "bad")
            self.assertIn("success=1 skipped=1 failed=1", output.getvalue())

            resumed = _SuccessfulSourcePipeline()
            with (
                patch(
                    "assetpipe.source._load_runtime",
                    return_value=(resumed, _Torch),
                ),
                patch("assetpipe.source.clear_cuda"),
            ):
                generate_sources(specs, config)
            self.assertEqual(resumed.calls, 1)
            self.assertTrue((sources / "bad.png").is_file())

    def test_trellis_continues_after_missing_input(self):
        with tempfile.TemporaryDirectory() as value:
            config = _config(Path(value))
            source = Path(config["paths"]["sources"]) / "good.png"
            source.write_bytes(b"png")

            def export(mesh, path, trellis_config, o_voxel):
                path.write_bytes(b"glb")

            with (
                patch(
                    "assetpipe.trellis._load_runtime",
                    return_value=(_MeshPipeline(), _ImageModule, object()),
                ),
                patch("assetpipe.trellis._export_mesh", side_effect=export),
                patch("assetpipe.trellis.clear_cuda"),
            ):
                generate_meshes([str(source), str(source.with_name("missing.png"))], config)

            self.assertTrue((Path(config["paths"]["meshes"]) / "good.glb").is_file())
            errors = (Path(config["paths"]["errors"]) / "trellis.jsonl").read_text()
            self.assertIn('"asset_id": "missing"', errors)

    def test_blender_and_pixel_isolate_item_failures(self):
        with tempfile.TemporaryDirectory() as value:
            config = _config(Path(value))
            meshes = Path(config["paths"]["meshes"])
            (meshes / "good.glb").write_bytes(b"glb")
            (meshes / "bad.glb").write_bytes(b"glb")

            def run(command, check):
                mesh_name = Path(command[command.index("--input") + 1]).stem
                if mesh_name == "bad":
                    raise subprocess.CalledProcessError(1, command)
                output_dir = Path(command[command.index("--output-dir") + 1])
                for view in config["blender"]["views"]:
                    (output_dir / f"{view}.png").write_bytes(b"png")

            with patch("assetpipe.blender_stage.subprocess.run", side_effect=run):
                render_meshes(str(meshes), config)

            renders = Path(config["paths"]["renders"])
            self.assertTrue((renders / "good" / "ne.png").is_file())
            self.assertIn(
                '"asset_id": "bad"',
                (Path(config["paths"]["errors"]) / "blender.jsonl").read_text(),
            )

            pixel_pipeline = _PixelPipeline()

            def save_sprite(generated, alpha, path, final_size, image_module):
                path.write_bytes(b"sprite")

            with (
                patch(
                    "assetpipe.pixel._load_runtime",
                    return_value=(pixel_pipeline, object(), object(), object()),
                ),
                patch("assetpipe.pixel._load_lora"),
                patch(
                    "assetpipe.pixel._prepare_images",
                    return_value=(_Render(), object(), object()),
                ),
                patch("assetpipe.pixel._save_sprite", side_effect=save_sprite),
                patch("assetpipe.pixel.clear_cuda"),
            ):
                generate_sprites(str(renders), config)

            sprites = Path(config["paths"]["sprites"]) / "good"
            self.assertTrue((sprites / "ne.png").is_file())
            self.assertTrue(any(sprites.glob("*.png")))
            self.assertIn(
                '"stage": "pixel"',
                (Path(config["paths"]["errors"]) / "pixel.jsonl").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
