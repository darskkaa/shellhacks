# Waymo pickup reroute

`simulator/waymo_pickup_sim.py` animates a Waymo driving from Wynwood Walls to pick up a blind rider who asked
for a pickup at 2100 NE 2nd Ave. Halfway there, the car moves the pickup to a safer curb on NE 21st St and
reroutes to it. Waymax simulates the last 80 m. Output goes to `simulator/artifacts/waymo_pickup/`:
`waymo_pickup.gif`, `first_frame.png`, `last_frame.png` and `report.json`.

1. Our router plans the trip from Wynwood Walls to the requested curb.
   - Fastest: 2:05, 1.74 km, risk 80 (crash 20, flood 60).
   - Safest: 2:25, 1.71 km, risk 24 (crash 4, flood 20).

   The car takes the safest route.
2. At the intersection past the halfway point (929 m, NE 27th St & NE 2nd Ave), the pickup check in
   `simulator/data/pickup.json` flags the requested NE 2nd Ave curb: 4 lanes, two 311 flood reports within
   60 m, 1 crash, risk 25.1. It moves the pickup to the north curb of NE 21st St, which scores risk 4.1 with
   no hazards. The car reroutes from that intersection.
   - Fastest: 0:55, risk 28.
   - Safest: 1:13, 0.90 km, risk 23.
3. For the last 80 m, Waymax rolls the car out on NE 21st St. It stops 0.4 m off the north curb, with the
   curb on the car's right.
4. The last frame shows the numbers behind the decision.

## What's ours

- `simulator/road_network.py` builds the road graph. It covers every drivable OSM way in lat 25.790–25.806,
  lon -80.203 to -80.185: 1664 nodes and 3296 directed edges, cached in `simulator/data/drive_osm.json`.
  - Edges respect `oneway` and `lanes`.
  - Speeds come from `maxspeed`, or a default per highway class when the tag is missing (residential
    25 mph, primary 35 mph and so on).
- Hazards come from SafeRoute's own layers (`saferoute/data/{crashes,flooding,construction}.geojson`) and
  use the server's radii and weights:
  - Crash within 40 m: 1 point, plus 3 per serious injury and 10 per death.
  - 311 flood report within 60 m: 5 points times the flood multiplier.
  - FDOT work zone within 30 m: 15 points per project.

  The flood multiplier is SafeRoute's rule, applied per point (`pickup_choice.flood_multiplier`): x4 inside
  an NWS flood-alert polygon, x2 on a king tide, otherwise x1. It is x2 here.
- Each hazard is charged to its nearest road segment, so a route pays for it once.
- The router runs Dijkstra.
  - Edge cost = free-flow time + 10 s per risk point, so the car will drive about 10 s extra to avoid one
    crash record and about 100 s to avoid a flood report.
  - The fastest route is the same search with the weight set to 0.
- The pull-in plan, also ours: the last 80 m of the safest route, in the rightmost lane. Over the last 25 m it
  eases to 0.4 m off the curb. Speed is capped by the speed limit, 2 m/s² lateral and 1.5 m/s² braking.
- The Waymax roadgraph is built from the same OSM ways (lane count x 3.3 m). Road-edge points that fall inside
  a crossing street are dropped.

## What's Waymax

`simulator/waymax_pullin.py` builds a Waymax `SimulatorState` itself, without the Waymo Open Motion Dataset.
- Roadgraph points: road edges and lane centres.
- The Waymo is the SDC (4.7 x 1.9 m), and its planned path is the SDC path.
- There are no other agents. We have no real traffic data for the street, so none are invented.

Waymax's expert actor turns the plan into bicycle-model controls at every 0.1 s step, and
`InvertibleBicycleModel` integrates them. The car's drawn poses come from that rollout. Waymax's metrics
over the 147 steps:

| Metric | Result |
|---|---|
| offroad | 0 |
| overlap | 0 |
| wrong-way | 0 m |
| off-route | 0 m |
| kinematic infeasibility | 0 |
| divergence from plan | ≤ 0.07 m |
| route progression | 1.0 |
| stop error | 0.01 m |

## Run

    .venv/bin/python -m simulator.waymo_pickup_sim           # offline: OSM, hazards and pickup.json are cached
    .venv/bin/python -m simulator.road_network --refresh     # re-download the OSM drive graph (Overpass)

The venv needs `jax[cpu]`, `tensorflow-cpu`, `chex`, `flax`, `dm_env`, `dm-tree` and `immutabledict`.
Waymax itself is installed with
`uv pip install --no-deps git+https://github.com/waymo-research/waymax.git`.
Its `setup.py` pins the full `tensorflow` package; the sim only needs `tensorflow-cpu`, which Waymax
imports but does not use on this code path.

The script asserts three things:
- The chosen curb's risk is no higher than the requested curb's.
- The safest reroute either avoids the requested curb's road segment or has lower hazard cost than the
  fastest. Here it drives past that segment on NE 2nd Ave, with risk 23 against 28.
- The Waymax rollout never goes offroad and stops within 1 m of the planned pose.

## Approximations

- Travel times are free-flow at the speed limit, with no signals, turn delays or traffic.
- Between the pickup check and the Waymax segment, the car moves along the road centreline at a constant
  animation rate.
- The pickup check fires at the first intersection past half the trip, not at a time taken from data.
- OSM has no kerb-return geometry, so intersection corners are square. The Waymax segment therefore starts
  just past the last turn: a square inside corner would read as offroad even for a turn a real car makes.
  The car enters at cornering speed, √(2 m/s² × 6 m) ≈ 3.5 m/s.
- The Waymax actor replays our plan through bicycle dynamics. It is not a learned or IDM planner: IDM needs a
  lead vehicle to stop, and inventing one would add a fake agent.
- Our risk leaves out the parts of SafeRoute's score that need live services: low-elevation share, rain,
  surge zones and live incidents.

## Licenses

- Map data © OpenStreetMap contributors, ODbL 1.0.
- SafeRoute hazard layers: Miami-Dade 311 and FDOT open data, as used by the SafeRoute server.
- The final approach was made using the Waymax Licensed Materials, provided by Waymo LLC under the Waymax
  License Agreement for Non-Commercial Use, available at
  https://github.com/waymo-research/waymax/blob/main/LICENSE. Your access and use of the Waymax Licensed
  Materials are governed by the terms and conditions contained therein.
- The license allows research, teaching and personal experimentation only. It forbids use for any
  real-world vehicle operation, for simulating driving scenarios for commercial purposes, and in production
  systems. Anyone who receives the GIF or the code is bound by the same terms, and the notice above must go
  with them.
- Cite Waymax as: Gulino et al., "Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous
  Driving Research", NeurIPS 2023 Datasets and Benchmarks.
