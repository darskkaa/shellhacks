"""Prepare verified local glTF assets and load real textured meshes into PyBullet."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pybullet as p
import trimesh

ASSETS = {
    "person": (
        "Soldier.glb",
        "dfb230fc1f942f259dd00281a1186953ad602fc5d69067ce63e24b2aa439736b",
    ),
    "car": (
        "ToyCar.glb",
        "01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c",
    ),
}


def prepare_assets(asset_dir: Path) -> None:
    """Convert already downloaded, hash-checked GLBs; never access the network."""
    manifest = {}
    for name, (filename, expected_hash) in ASSETS.items():
        source = asset_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}; follow simulator/ASSETS.md")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Unexpected SHA256 for {source}: {digest}")
        scene = trimesh.load_scene(source)
        meshes = []
        for node in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node]
            if name == "car" and geometry_name == "Fabric":
                continue
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
            )
            # TinyRenderer uses OBJ diffuse materials, not glTF PBR extensions.
            mesh.visual.material = mesh.visual.material.to_simple()
            meshes.append(mesh)
        converted = trimesh.Scene(meshes)
        bounds = converted.bounds
        size = bounds[1] - bounds[0]
        scale = 1.7 / size[2] if name == "person" else 3.0 / max(size[:2])
        converted.apply_translation(
            [-bounds[:, 0].mean(), -bounds[:, 1].mean(), -bounds[0, 2]]
        )
        converted.apply_scale(scale)
        if name == "person":
            converted.apply_transform(
                trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 0, 1])
            )
        destination = asset_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        obj, resources = trimesh.exchange.obj.export_obj(converted, return_texture=True)
        (destination / "model.obj").write_text(obj)
        for resource_name, content in resources.items():
            (destination / resource_name).write_bytes(content)
        manifest[name] = {
            "source": filename,
            "sha256": digest,
            "bounds_m": converted.bounds.tolist(),
            "vertices": sum(len(mesh.vertices) for mesh in meshes),
            "notes": "Static bind pose; no skeletal animation"
            if name == "person"
            else "Fabric backdrop removed; PBR simplified",
        }
    (asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_scene(asset_dir: Path) -> dict[str, int]:
    """Load person at (4,0,0), car at (6,-2,0) into the current physics client."""
    asset_dir = asset_dir.resolve()
    if not all((asset_dir / name / "model.obj").is_file() for name in ASSETS):
        prepare_assets(asset_dir)
    objects = {}
    for name, position in (("person", [4, 0, 0]), ("car", [6, -2, 0])):
        path = asset_dir / name / "model.obj"
        visual = p.createVisualShape(
            p.GEOM_MESH, fileName=str(path), rgbaColor=[1, 1, 1, 1]
        )
        collision = p.createCollisionShape(
            p.GEOM_MESH, fileName=str(path), flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(
                [0, 0, math.pi if name == "person" else 0]
            ),
        )
        if body < 0:
            raise RuntimeError(f"Could not load {path}")
        objects[name] = body
    return objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path, default=Path(__file__).parent / "artifacts/assets"
    )
    args = parser.parse_args()
    prepare_assets(args.assets)
    print(json.dumps(json.loads((args.assets / "manifest.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
