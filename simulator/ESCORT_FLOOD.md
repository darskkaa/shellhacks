# Flooded sidewalk, new path

`simulator/escort_flood_video.py` renders a guide-dog escort in which the planned sidewalk is under water.
The dog sees the water with its own camera, replans the whole walking route with that water marked
impassable, and leads the rider round the building to the Waymo.

Streets, curbs, crossings and hazard locations are real data; the dog, rider, cars and anything tagged illustrative are simulated.
The dog's depth and water sensing come from the simulator's ground truth; a trained YOLO model reads the walk signal.

```
.venv/bin/python -m simulator.escort_flood_video --offline
# -> simulator/artifacts/escort_flood/escort_flood.{webm,mp4}, report.json, mid_frame.png
```

It makes no network calls. `--offline` is accepted to match the other sims; this video never calls Gemini.
A full render takes about 25 minutes on a laptop CPU (PyBullet tiny renderer, 1,781 dog steps) and produces a 54 s video. The dog walks 264 m, 138 m more than the 126 m plan.

## What happens

1. The rider starts on the west sidewalk of NE 2nd Ave, 20 m north of 2100 NE 2nd Ave. The phone's
   walking route over the OSM grid (`gemini_waymo_sim.global_route` logic) runs south along NE 2nd Ave,
   then west along NE 21st St to the Waymo, 126 m in all. That route passes through the flooded frontage.
2. At about 5 m, the dog's segmentation labels the standing water and `DogMap` marks it in the dog's map.
   When seen water sits on the route within 12 m ahead, the dog stops and sweeps its camera across the
   sidewalk so the map covers the water's full width.
3. It replans the global route (`plan()` in this file, a copy of `global_route` plus a mask). Seen water
   cells, inflated by 0.8 m, are impassable. No crosswalk over NE 2nd Ave is mapped here, and the
   building runs from the flooded frontage all the way to its west end. So the only dry walking route
   goes back north, west along the building's north side, south down the flush service alley past its
   west end, and east along NE 21st St. That is 257 m from the stop point, against 115 m through the
   water.
4. The dog says "The sidewalk ahead is flooded. Taking another way; about 140 metres longer." The dog
   goes round its rider (the rider is an obstacle in the local planner, and the dog filters its own
   body, harness and rider out of its depth map). It then follows the new route and calls out each
   corner with the remaining distance.
5. It stops beside the Waymo's door and gives the handle side.

The route map inset shows the planned route as a grey dashed line, the current route in green, and
the walked trail and seen water in blue. The dog's local map shows only what its camera has seen.
Long straight stretches are fast-forwarded 4x and labelled on screen. Events play at normal speed.

## Real vs illustrative

| Element | Source |
|---|---|
| Streets, lane counts, curbs, sidewalks, service alleys, buildings (footprints and heights) | OpenStreetMap (`real_block.py` cache) |
| Water location: curb drain in front of 2100 NE 2nd Ave | Miami-Dade 311 ticket 23-10236820, "drain clogged / cleaning" (SafeRoute layer) |
| King tide on Sep 27 2026 (2.8 ft at 10:08), NWS Coastal Flood Statement | SafeRoute conditions cached in `data/pickup.json` |
| Waymo curb on NE 21st St | `pickup_choice.py`: risk 4.1 vs 25.1 at the requested pin, 0 flood reports within 60 m |
| Other 311 drain puddles (1.6 m) | `waymo_scene.build_scene` |
| **Extent of the standing water** | **Illustrative.** The report is a point. The pool is drawn from the curb drain to the building wall, 3 m straight plus rounded ends (about 8.5 m long), so it closes the sidewalk. It is labelled on screen. |
| **Rider start** | **Illustrative**: 20 m north of the 311 pin on the same sidewalk (pickup.json's start is inside the pool) |
| Robot dog and rider | Simulated (PyBullet) |

## Why the pickup stays on NE 21st St

The pickup is still the curb `pickup_choice.py` chose, since the new route ends there anyway. It
approaches from the west along the sidewalk, crossing only the flush service driveway at x = -186 m.
That curb scored lowest of all 66 curbs (no floods, crashes or work zones within the SafeRoute radii).
The requested pin on NE 2nd Ave sits next to the very drain that flooded. `build_scene` places the
car; the stop point is 1.6 m in from the door. The dog stops 0.6 m past it in its direction of travel,
so the rider ends up at the handle.

## Self-checks (asserts at the end of the run)

- The dog never steps off a curb: max single-step ground drop < 0.05 m.
- Neither the dog nor the rider is ever on the roadway (0 steps).
- The rider never steps into water: the rider's closest approach to any water shape (the pool, other
  drain puddles) is > 0.2 m. The run prints the actual value, about 5 m, because the dog turns back
  before the water.
- The dog reaches the Waymo.
- The final walk differs from the first plan around the flooded stretch: the replan is more than 5 m
  longer, and the walked trail diverges more than 10 m from the initial route (about 76 m in practice).
- Before walking, the script also asserts that the initial phone route really runs through the water.

`report.json` records these numbers and every spoken event.

## Implementation notes

- It reuses `DogMap`, `astar`, `Trail`, `dog_camera`, `detections`, `dilate` and `draw_local_map` from
  `gemini_waymo_sim.py`, and `RealBlock`, `build_scene` and the actor builders from `waymo_scene.py`.
  No shared module is edited. `DogMap` and `cell()` read the map extent from `gemini_waymo_sim`'s module
  globals, so this script widens them (`gws.X0/Y0/NX/NY`) to its larger window.
- Needed shared change (optional): `gemini_waymo_sim.DogMap` could take the window as an argument
  instead of module globals.
- The chase camera holds a behind-left oblique view. It drops to a steep view from just behind when a
  building would block the oblique camera. Heading and blend are smoothed, so the U-turn does not whip-pan.
