# Pawthfinder: Devpost draft

**Tagline:** A robot guide dog and a safety-first map that bring accessible streets to people before the city gets there.

**Links:** live map https://saferoute-miami.vercel.app · simulation demos https://saferoute-sim.vercel.app

**Tracks:** Best Overall, Waymo. Only add MongoDB Atlas or Gemini API if `MONGODB_URI` / `GEMINI_API_KEY` are set on the Vercel project before judging: the live site uses neither without them.

## Inspiration

About a million Americans are blind, and millions more use wheelchairs, canes or walkers (CDC). For them a city is an obstacle course: curbs without ramps, crosswalks without audio signals, scooters on the sidewalk and, in Miami, streets that flood on a sunny day. Cities fix this one corner at a time. A talking crosswalk adds about $3,600 per intersection (U.S. Access Board) and one curb ramp costs about $2,500 to fix (Baltimore DOT). Most places aren't there yet.

So we asked: what if the safety check traveled with the person instead of waiting for the city?

## What it does

Pawthfinder has two halves.

**1. A robot guide dog (simulated).** A MechDog robot leads a blind rider on real Miami streets. It plans from its own camera: depth tells it where curbs drop off and where obstacles stand, and it never leads the rider off a curb. Three demos on real Edgewater, Miami blocks:

- **Obstacle course:** 112 m down Biscayne Blvd around real bus stops and FDOT work zones to the door of a waiting Waymo.
- **Flooded sidewalk:** its planned route runs through a real Miami-Dade 311 flood report. It sees the water, stops, replans around the block, and gets the rider there dry.
- **Crossing with traffic:** at a real five-lane crossing, a trained YOLO model reads the walk signal from the dog's camera. The dog waits through DON'T WALK and only starts when there's enough time left to finish at a cane user's pace.

**2. SafeRoute, a safety-first map.** Map apps pick the fastest route. SafeRoute scores every alternative on 34,817 real hazards and picks the safest, then uses time and the car's battery only to break ties:

- 23,411 injury and fatal crashes (FDOT) and 5,449 flooding reports (Miami-Dade 311), plus road elevation.
- Active FDOT construction, live police-reported crashes, NOAA tide and rain, National Weather Service alerts, and hurricane surge zones.
- A frontier chart shows every option, safety against time, and a card explains the pick in one line ("Avoids 8 flood reports and 19 crashes…").
- It also picks the **safest pickup curb** for a ride: it moved a Waymo pickup off a flooded four-lane curb to a quiet side street, and Waymo's open-source Waymax simulator checks the car can pull in there cleanly.

## How we built it

- **Simulation:** PyBullet worlds built straight from OpenStreetMap (streets at their real lane counts, curbs, crossings, curb ramps, buildings) plus SafeRoute's hazard data. The dog turns its depth camera into a height map, finds curbs, holes and water, and replans with A* every step. The crossing uses our custom YOLO11 model with a map-guided crop so it can read a real-size signal from the far curb.
- **Car pull-in:** our own route planner on the OpenStreetMap road graph, weighted by the same hazards, with the last 80 m driven in Waymo's Waymax.
- **SafeRoute:** a Node.js server with no framework on Vercel, using the Google Routes, Elevation and Maps JavaScript APIs. Hazards are loaded into an in-memory grid index. A Pareto ranking with an electric-car energy model breaks ties. A recorded-demo mode keeps the example trips working if Google's daily limit is hit, and says clearly when a result is recorded.
- **Accessibility:** keyboard and screen-reader support, 44 px touch targets and a text version of every chart. axe reports 0 violations and Lighthouse gives 100 for accessibility.

## Challenges

- Low cameras can't see over a curb, so a drop-off shows up as missing ground. We treat "ground we should see but can't" as a possible drop.
- The robot vision model couldn't read a real-size walk signal from 18 m away. A crop guided by the signal's mapped position fixed it without retraining.
- Today's $450 MechDog walks 0.42 m/s, too slow to finish that crossing in time. The demo walks at a cane user's 0.9 m/s and says so.
- Miami's 311 and FDOT data needed cleaning, and Google's daily limits pushed us to add caching and recorded demos.

## Accomplishments

- Every scene is real data: real streets, real flood reports, real crash history.
- The dog decides from what it sees, not from a script.
- An honest demo: anything illustrative is labeled on screen.

## What we learned

The safest route is often only seconds slower, and nobody shows people that choice today. And accessibility isn't one feature: curbs, signals, water and the ride itself all have to work together.

## What's next

- Put the planner on the real MechDog, with a faster walking gait.
- Audio guidance and haptic feedback for the rider.
- Wheelchair routing that avoids missing ramps and steep slopes.
- Fresher crash data and more cities.

## Built with

pybullet · python · numpy · onnxruntime · yolo11 · openstreetmap · waymax · jax · node.js · google-maps · google-routes-api · vercel · noaa · fdot · miami-dade-open-data

_Simulated: the robot, the rider, the cars and anything tagged illustrative. Real: the streets, curbs, crossings and hazard locations. Waymax is used under its non-commercial license._
