# Simulator scene assets

The scene contains real textured triangle meshes rendered by PyBullet. It uses no photo billboards or detector labels derived from scene object IDs. The person is 1.7 m tall at `(4, 0, 0)`, facing the starting camera; the car is 3 m long at `(6, -2, 0)`.

## Sources and attribution

- **Soldier** from the [three.js example asset](https://github.com/mrdoob/three.js/blob/dev/examples/models/gltf/Soldier.glb). The [official example](https://threejs.org/examples/webgl_animation_skinning_blending.html) credits **Mixamo**. [Adobe's Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html) permits royalty-free character/animation use in personal, commercial, and nonprofit projects, including rendered art, films, and games. This character is used locally in the rendered demo under those asset terms; the three.js repository's MIT license is not asserted to relicense the character. Raw/converted character files are not distributed in this repository. Changes: baked node transforms, converted glTF to OBJ/diffuse texture, rotated to Z-up and scaled. This is the static T-pose; skinning and animation are not evaluated.
- **Toy Car**, © 2020 Public, licensed **CC0**; initial car model by Guido Odendahl, extensions and scene composition by Eric Chadwick. [Source and attribution](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/ToyCar/README.md). Changes: removed the fabric backdrop, baked node transforms, rotated/scaled, simplified PBR materials to OBJ diffuse materials. Glass/transmission/clearcoat rendering differs from the source.

## Fetch once, prepare locally

From the repository root, fetch the two data files into the ignored cache. These commands have timeouts and 12 MB response limits; neither executes downloaded code. Downloads are approximately 2.05 MB and 5.17 MB. Do not retry a throttled request in a loop.

```sh
mkdir -p simulator/artifacts/assets
curl --fail --location --connect-timeout 10 --max-time 60 --max-filesize 12000000 \
  -o simulator/artifacts/assets/Soldier.glb \
  https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb
curl --fail --location --connect-timeout 10 --max-time 60 --max-filesize 12000000 \
  -o simulator/artifacts/assets/ToyCar.glb \
  https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/ToyCar/glTF-Binary/ToyCar.glb
.venv/bin/python simulator/scene_assets.py
```

`prepare_assets(asset_dir)` checks the GLBs against the following SHA256 pins before conversion. If upstream data changes, conversion fails rather than silently accepting different assets.

| Input | SHA256 |
| --- | --- |
| `Soldier.glb` | `dfb230fc1f942f259dd00281a1186953ad602fc5d69067ce63e24b2aa439736b` |
| `ToyCar.glb` | `01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c` |

Outputs: `person/model.obj`, `car/model.obj`, their material/texture files, and `manifest.json` with source hashes, dimensions, and conversion notes. Nothing downloads during preparation or scene loading. Generated meshes and source binaries stay in the ignored cache, outside Git.

`load_scene(asset_dir: Path) -> dict[str, int]` loads the models into the current PyBullet connection and returns `person` and `car` body IDs. Missing converted files trigger local preparation; missing GLBs produce an actionable error. Existing converted files are treated as a local trusted cache. Delete their directories or rerun preparation after changing conversion code.

Rendering validation used PyBullet DIRECT/TinyRenderer, with both meshes visible from camera `(0.2, 0, 0.8)` looking along +X. The person receives a further 180° Z rotation at load time to face the camera. See [VISION.md](VISION.md) and `verification-vision.json` for actual detector results. Detection remains sensitive to viewpoint and asset appearance; no accuracy or generalization claim follows from this scene.
