# Obstacle course to the Waymo

`simulator/escort_obstacles_video.py` renders a ~1 min demo video: the simulated robot guide dog leads a
blind rider down the west sidewalk of Biscayne Blvd (Edgewater, Miami) from NE 22nd St to NE 21st St,
steers around the obstacles on that sidewalk, turns onto NE 21st St and stops the rider at the door of a
Waymo parked at the curb.

```
.venv/bin/python -m simulator.escort_obstacles_video --offline
# -> simulator/artifacts/escort_obstacles/escort_obstacles.{webm,mp4}, mission_report.json, last_frame.png
```

No network is used (the Gemini narrator is not used; speech is onboard). Runtime is a few minutes on CPU.

## What is real

- Street geometry: OpenStreetMap sidewalks, curbs (roads 0.15 m below the sidewalk), lane counts and
  buildings, from the cached block in `simulator/data/` (`real_block.py`, `waymo_scene.py`).
- Bus stop sign: the OSM bus-stop node on this sidewalk (25.7978, -80.1892).
- FDOT work zones: the three work-zone points of the SR 5/US-1 Biscayne Blvd traffic-signal project inside
  the view (SafeRoute Miami layer). Cones are drawn at each real point, which lies in the Biscayne roadway.
- Pickup curb: chosen with `pickup_choice.score` / `curb_candidates` / `walk_to` over every real curb within
  60 m of the requested pin (the Biscayne Blvd & NE 21st St signal), walking from the rider's start, with the
  live conditions cached in `simulator/data/pickup.json` (king tide, flood multiplier x2). The Biscayne curb
  scores high (an FDOT work zone within 30 m, 5 lanes); the NE 21st St north curb scores lowest and gets
  the Waymo. The numbers are printed on the end card and in `mission_report.json`.
- Perception and planning: the dog sees only through its own depth/segmentation camera and plans with the
  `gemini_waymo_sim.py` stack (height map, obstacle and drop-off detection, local A*), with two local
  changes: a 0.6 m obstacle buffer (rider clearance) and line-of-sight smoothing of the A* path so the dog
  returns to the sidewalk centreline after each detour.

## What is illustrative (labelled on screen)

- Work-zone barricades: the two mid-block FDOT points get a barricade with cones on the curb half of the
  west sidewalk, at the same latitude as the real point. The real data is a point; the barricade shape and
  its spill onto the sidewalk are illustrative.
- E-scooter: one parked e-scooter mid-block; it is in no dataset.
- The walking route is the centreline of the mapped sidewalk (checked walkable on the OSM raster), not a
  phone-routing answer.
- Pace: straight walking is shown at about 3x the dog's pace, passes near obstacles at about real pace.

## Self-checks (asserts at the end of the run)

- The dog never steps off a curb: largest single-stride drop < 0.05 m.
- The dog reaches the goal beside the Waymo door.
- At least three obstacles are announced and passed.
- Dog centre and rider centre both stay > 0.25 m from every obstacle surface (the dog plans with a 0.6 m buffer, not the default 0.4 m, so the rider on its right clears too).
- The chosen pickup curb scores lower than the requested Biscayne curb.
