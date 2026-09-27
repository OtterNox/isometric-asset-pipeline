import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render one GLB from fixed views.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-size", required=True, type=int)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--elevation", required=True, type=float)
    parser.add_argument("--views-json", required=True)
    parser.add_argument("--transparent", required=True, type=parse_bool)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(arguments)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_glb(path: Path) -> list[bpy.types.Object]:
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = list(bpy.context.scene.objects)
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    return imported


def mesh_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        if obj.type == "MESH"
        for corner in obj.bound_box
    ]
    if not points:
        raise RuntimeError("Imported GLB contains no mesh objects")

    minimum = Vector(tuple(min(point[index] for point in points) for index in range(3)))
    maximum = Vector(tuple(max(point[index] for point in points) for index in range(3)))
    return minimum, maximum


def center_asset(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    minimum, maximum = mesh_bounds(objects)
    offset = -((minimum + maximum) * 0.5)
    object_set = set(objects)
    roots = [obj for obj in objects if obj.parent not in object_set]
    for obj in roots:
        obj.location += offset
    bpy.context.view_layer.update()
    return mesh_bounds(objects)


def point_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def create_area_light(
    name: str,
    location: Vector,
    energy: float,
    size: float,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    point_at(light, Vector((0.0, 0.0, 0.0)))


def setup_lighting(asset_size: float) -> None:
    world = bpy.data.worlds.new("Asset World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.08, 0.08, 0.08, 1.0)
    background.inputs["Strength"].default_value = 0.8
    bpy.context.scene.world = world

    distance = max(asset_size * 2.5, 2.0)
    light_size = max(asset_size * 1.5, 1.0)
    create_area_light(
        "Key Light",
        Vector((distance, -distance, distance * 1.5)),
        1000.0,
        light_size,
    )
    create_area_light(
        "Fill Light",
        Vector((-distance, distance * 0.5, distance)),
        500.0,
        light_size,
    )


def setup_camera(asset_diagonal: float) -> bpy.types.Object:
    data = bpy.data.cameras.new("Isometric Camera")
    data.type = "ORTHO"
    data.ortho_scale = max(asset_diagonal * 1.15, 0.1)
    data.clip_start = 0.01
    data.clip_end = max(asset_diagonal * 20.0, 100.0)
    camera = bpy.data.objects.new("Isometric Camera", data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def set_render_engine(scene: bpy.types.Scene, engine: str) -> None:
    try:
        scene.render.engine = engine
    except TypeError:
        if engine != "BLENDER_EEVEE_NEXT":
            raise
        scene.render.engine = "BLENDER_EEVEE"
        print("BLENDER_EEVEE_NEXT unavailable; using BLENDER_EEVEE")


def configure_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    set_render_engine(scene, args.engine)
    scene.render.resolution_x = args.render_size
    scene.render.resolution_y = args.render_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = args.transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"


def render_views(
    camera: bpy.types.Object,
    views: dict[str, float],
    elevation_deg: float,
    distance: float,
    output_dir: Path,
    force: bool,
) -> None:
    elevation = math.radians(elevation_deg)
    for name, azimuth_deg in views.items():
        output_path = output_dir / f"{name}.png"
        if output_path.is_file() and not force:
            print(f"[blender] {name} skipped")
            continue

        azimuth = math.radians(float(azimuth_deg))
        camera.location = Vector(
            (
                distance * math.cos(elevation) * math.cos(azimuth),
                distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
            )
        )
        point_at(camera, Vector((0.0, 0.0, 0.0)))
        temporary_path = output_dir / f"{name}.tmp.png"
        bpy.context.scene.render.filepath = str(temporary_path)
        bpy.ops.render.render(write_still=True)
        if not temporary_path.is_file():
            raise RuntimeError(f"Blender did not create output: {output_path}")
        temporary_path.replace(output_path)
        print(f"[blender] {name} completed")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input GLB not found: {input_path}")

    views = json.loads(args.views_json)
    if not isinstance(views, dict) or not views:
        raise ValueError("--views-json must contain a non-empty object")

    output_dir.mkdir(parents=True, exist_ok=True)
    clear_scene()
    objects = import_glb(input_path)
    minimum, maximum = center_asset(objects)
    dimensions = maximum - minimum
    diagonal = max(dimensions.length, 0.1)

    configure_render(args)
    setup_lighting(max(dimensions))
    camera = setup_camera(diagonal)
    render_views(
        camera,
        views,
        args.elevation,
        max(diagonal * 2.5, 2.0),
        output_dir,
        args.force,
    )


if __name__ == "__main__":
    main()
