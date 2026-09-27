# Guide-dog crossing at a real Miami crosswalk

`real_crosswalk_demo.py` rebuilds the guide-dog crossing from `escort_demo.py` at a real
crosswalk in Edgewater, Miami. The crosswalk is the one over **Biscayne Boulevard (US 1) on
the north side of NE 19th Street**. The geometry and hazard data are real. The robot dog,
its handler and the cars are simulated.

The dog is driven only by the unchanged `blind_escort_yolo` model. It starts after 3
consecutive frames vote WALK, and only if the time left in the pedestrian phase covers the
crossing. The simulator's own signal state is used only to grade it.

## Why this crossing, not NE 2nd Ave & NE 20th St

The NE 2nd Ave crosswalks at NE 20th St don't have what this demo needs. In the cached OSM
data they are a single 45 m `footway=crossing` way that covers three legs. It is tagged only
`crossing=marked`, and no end is a lowered-kerb node: the nearest is 6.6 m away, on the
service-road crossing.

The Biscayne crossing has everything:

- `crossing=traffic_signals` and `crossing:signals=yes`
- a lowered kerb with tactile paving at both ends
- a `highway=traffic_signals` node 13 m away
- the FDOT "traffic signals" work zone 18 m away

The script picks it by its OSM node (`TARGET_LATLON`) and checks those tags before building.

## What is real (source)

All data comes from `simulator/real_block.py`, which reads the cached OpenStreetMap
Overpass extract and SafeRoute Miami layers in `simulator/data/`.

| Element | Value | Source |
|---|---|---|
| Crossing polyline, length | 18.07 m, walked end to end by the dog | OSM `footway=crossing` way |
| Crossing tags | `crossing=traffic_signals`, `crossing:markings=lines`, `crossing:signals=yes` | OSM |
| Lowered kerbs + tactile paving | both crossing ends, plus every other lowered kerb in the 40 x 40 m window | OSM `kerb=lowered`, `tactile_paving=yes` |
| Lanes | 5: 3 southbound + 2 northbound; the direction split comes from the raw OSM tags | OSM `lanes`, `lanes:forward/backward` |
| Kerb-to-kerb width | each side's pavement edge passes through the road's crossing-end kerb nodes | OSM kerb nodes |
| Speed limit (car speed, amber) | 30 mph | OSM `maxspeed` (FDOT GIS) |
| Vehicle signal position | hung from a span wire over the `highway=traffic_signals` node | OSM node |
| Buildings | footprints extruded to the OSM height (one in the window, 13.2 m) | OSM `building`, `height` / `building:levels` |
| Street furniture | none mapped in the window; trees, poles, bus stops and benches would be drawn | OSM |
| Flood patches | none of the 311 reports falls in the window; any that did would be drawn as standing water | Miami-Dade 311 via SafeRoute |
| Crashes within 40 m of the crossing | 3 (0 pedestrian), 2018-2019 | FDOT crash records via SafeRoute |
| Work zones within 30 m | "SR 5/US-1/BISCAYNE BLVD AT VARIOUS INTERSECTIONS - TRAFFIC S IGNALS" (typo as in the data), 17.6 m and 20.3 m | FDOT work-zone points via SafeRoute |
| Dog camera | 1920x1080 at 58° diagonal FOV (30.4° vertical); only its aim is tilted toward the head | Logitech Brio 105 spec (the repo's camera) |

## Perception: map-guided signal ROI

Real autonomous-vehicle stacks use the HD map to find traffic-light regions of interest.
The dog does the same with the signal head it knows from the map:

1. Each frame, it projects the head's 3D position into the 1920x1080 image.
2. It crops a square around that point. The side is 9 times the projected face height,
   and at least 96 px.
3. It runs the unchanged model on the crop. `predict()` upscales the crop to 640.

Pedestrian-signal classes come only from the ROI pass. A full-frame pass still supplies
every other class, including the `conflict_vehicle_cyclist` veto. Using the ROI for
signals also removes the false WALK (0.78) that a passing car triggered in a full-frame
run. The camera panel shows the ROI as a cyan box, with the crop as an inset.

Tuning and calibration:

- **Face size:** the face is the real size of about **0.45 m**, so escort_demo's oversized
  0.8 m face was not needed.
- **Self-lit face:** the signal face is an LED that lights itself, but TinyRenderer only
  shades surfaces by the sun. A second, unshaded render supplies the face's pixels through
  the segmentation mask; the rest of the scene stays shaded. Without this, the model never
  read WALK at 18.8 m.
- **The 9x margin is tuned on this scene.** Sweeping the margin from 2 to 16, 9 was the
  only value that read WALK at every tested kerb position (0.50-0.74) without ever reading
  WALK on the DON'T WALK face (0.78-0.87). At 10-11, WALK reads drop out.
- **Calibration warning:** as with escort_demo's `--walk-conf`, recalibrate the margin on
  real footage before trusting it.

## Pace and start rule

The escort pace is a **production-speed escort; the MechDog prototype tops out at 0.42 m/s
(tools/calc_robot_speed.py) and could not finish this 18.1 m crossing in time**:

- The simulated escort walks at 0.9 m/s, a blind cane user's pace set by the handler. It
  needs 20.1 s to cross.
- At 0.42 m/s the MechDog would need 43.0 s, against 23.9 s of WALK plus clearance.

The start rule (`TimedEscort`):

- **Budget:** the dog assumes the MUTCD-standard budget for the map's crossing length,
  7 s + 18.07 / 1.07 = 23.9 s.
- **Timing WALK:** it counts from the last non-WALK frame it saw, which is the earliest the
  WALK could have begun.
- **When it goes:** only if the remaining budget is at least the crossing length divided by
  0.9 m/s.
- **When it waits:** if WALK was already showing when it started watching, the start time
  is unknown and it waits for the next cycle.
- **Grading:** the report checks the start against the simulator's true remaining time.

## What is standard or approximated

- **Signal timing is not measured,** because no Miami-Dade timing data is available. The
  MUTCD standard gives a 7 s WALK, and the pedestrian clearance is 18.07 m ÷ 1.07 m/s
  (3.5 ft/s) = 16.9 s. Amber is 3.2 s from the ITE formula (1 s + v/2·3.05 m/s² at 30 mph).
  All-red is 2 s. The Biscayne green is 20 s, an arbitrary value that only sets how long the
  dog waits. The face shows a steady hand during clearance, where a real head flashes and
  counts down.
- **Pedestrian head position:** OSM puts signal nodes on the roadway centreline, not at the
  heads. The head therefore stands on the far corner, 0.8 m behind the far kerb and 2 m
  toward the intersection. Its face centre is 2.6 m up (MUTCD puts the bottom 7-10 ft above
  the sidewalk).
- **Road geometry:**
  - 3.3 m lanes, because OSM has no `width` tag
  - 0.15 m curbs
  - curb ramps at a 1:12 slope, 1.5 m wide, with a 0.6 m detectable-warning pad
  - crosswalk lines 3 m apart
  - stop bar 1.2 m ahead of the crosswalk
  - square corners
  - sidewalks: everything off the pavement, because OSM sidewalks are centrelines without width
- **Cars:** one car in each of the two curbside lanes per direction on Biscayne, at the speed
  limit. They yield to anyone in the crosswalk. There is no NE 19th traffic and no turning
  vehicles.
- **Rendering and motion:** frames come from PyBullet's TinyRenderer. Motion is a planar
  proxy, and the handler is a static mesh.

## Run

```sh
uv pip install --python .venv/bin/python --only-binary=:all: onnxruntime==1.30.0 opencv-python-headless==4.11.0.86
.venv/bin/python -m simulator.real_crosswalk_demo
```

`--assets` must point at a folder with the converted car and person meshes. Build them first
as described in [ASSETS.md](ASSETS.md); that writes them to the default,
`simulator/artifacts/assets`, so `--assets` can be omitted.

Options: `--seconds` (5-120, default 40), `--model`, `--walk-conf` (default 0.4),
`--output`.

A headless run takes about 4 minutes and writes these files to
`simulator/artifacts/real_crosswalk/`:

- `escort.gif`: every other frame. Left: overhead ground truth. Right: the dog camera with
  model boxes and the ROI. Bottom: a band with the data and the pace label.
- `first.png`, `last.png`
- `report.json`: checks, site facts, timing, camera and ROI settings, the start estimate,
  and a per-frame trace.

The process exits 1 if any check fails.

### Video

```sh
.venv/bin/python -m simulator.crossing_video < /dev/null
```

`crossing_video.py` runs the same demo with the same options and checks, and writes the same
gif and report. Through the demo's `frame_hook`, it also renders a chase camera every 0.1 s of
simulated time and writes a 1280x720, 20 fps video, `crossing.webm` and `crossing.mp4`, to the
same folder.

- **Playback:** real time, with short holds on the WALK votes, the go decision and the arrival.
  The video ends 2 s after arrival.
- **On screen:** the simulated signal state, the dog camera with its model boxes and cyan ROI,
  the ROI crop with the WALK and DON'T WALK confidences and the 3-frame confirmation, and the
  true WALK + clearance time left against the time the pair still needs at 0.9 m/s.
- **Captions:** the caption bar shows the dog's state, a guidance phrase and a line of real
  data. The spoken phrases are captions only; the demo has no speech output.
- **Cards:** a title card separates real, standard and simulated elements. The end card shows
  the outcome numbers from `report.json`.

For a quick framing cut, add `--video-seconds 14 --seconds 14.5 --output <scratch dir>`. That run
fails the arrival checks and exits 1, but still writes the video.

## Checks (`report.json`)

- `no_walk_votes_on_dont_walk`
- `crossing_started_on_true_walk`
- `never_started_without_enough_time`: the true remaining WALK + clearance at the start is at
  least the crossing time at 0.9 m/s
- `in_road_only_while_cars_held_on_red`
- `dog_reached_far_curb`: the dog stops 1.2 m past the far kerb, on the ramp
- `handler_reached_far_curb`
- `handler_stayed_beside_dog`: within 0.8 m

## Result (default run)

All 7 checks passed.

- **While waiting:** all 47 DON'T WALK frames voted DON'T WALK, with the ROI stop confidence
  up to 0.86. WALK was confirmed 0.6 s after it appeared, at 0.65 confidence, on 3 frames.
- **Start:** the dog started at t = 9.8 s with 23.3 s of WALK + clearance left (its own
  estimate was also 23.3 s), against 20.1 s needed.
- **Clearing the road:** the dog left the road at t = 30.3 s. Clearance ended at 33.1 s.
- **Traffic:** the dog spent 0 s in the road while cars had green or amber. The closest car
  centre while the pair was in the road was 4.0 m.

## Limits

- The signal face is synthetic, and the ROI margin was tuned on this one scene. This shows
  the control loop and map-guided perception at a real intersection's geometry. It is not
  evidence of real-street accuracy or safety.
- The start rule's WALK timing assumes the model reads the signal continuously. If WALK is
  missed for several frames, the onset estimate comes late and the dog overestimates the
  time it has. The ground-truth check exists to catch this.
