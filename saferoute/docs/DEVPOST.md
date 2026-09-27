# SafeRoute Miami: Devpost draft

**Tracks:** Waymo, Best Overall, MongoDB Atlas (MLH), Gemini API (MLH), State Farm, Microsoft

## Inspiration

Navigation apps optimize for minutes. In Miami, the fastest route can run through a crash corridor or a street that floods every king tide. Waymo's goal is to be the most trusted driver, and trust starts with choosing the route. We wanted any driver to be able to see that tradeoff and pick the safer road.

## What it does

Enter a trip. SafeRoute pulls Google's alternative routes and scores each one on:

- **Crash history:** 23,411 injury and fatal crashes in Miami-Dade (FDOT, 2018-2019), weighted by severity.
- **Flooding:** 5,449 flooding and standing-water reports from Miami-Dade 311 (2021-2023), plus the share of the route below 1 m elevation from the Google Elevation API.
- **Construction:** 152 active FDOT construction projects in Miami-Dade. Work zones are where lanes shift, signs change, and maps go out of date, which makes them hard for human and autonomous drivers alike.
- **Hurricanes and surge:** active Atlantic storms and forecast cones from the National Hurricane Center, Miami-Dade's storm-surge evacuation zones A-E, and NOAA SLOSH surge depth. When a cone covers Miami (or in the demo's Category 1-5 simulator), routes through active zones are penalized per mile and per foot of surge.
- **Rain:** NOAA MRMS radar rainfall over the last 24 hours along each route, and the WPC rain forecast for the next 24 hours.
- **Live conditions:** today's high tide from NOAA's Virginia Key gauge and active National Weather Service alerts. On king-tide days flood risk counts double. Routes that enter an NWS flood or tropical alert area (tested against the alert polygons) count it four times.

The map shows each route and every crash and flood report along the selected one; tap any dot for the underlying record. A flood simulator raises a water level against Google road elevation and shows, on the map and an elevation chart, exactly which stretches of each route go under water. Gemini sums up the recommendation in plain English.

Example: for FIU to Brickell on Sep 26 (a king tide plus an NWS Coastal Flood Statement), it picked FL-836: the fastest route at 19 min, with 369 nearby crashes and zero flood reports, versus 442 crashes and 12 flood reports on the SW 24th St alternative.

## How we built it

- **Node.js** server with no framework. **Google Routes API** for alternatives, **Elevation API** for low-lying segments, **Maps JavaScript API** for the map.
- **MongoDB Atlas** stores 34,817 point hazards and 1,547 evacuation-zone polygons with `2dsphere` indexes. The server loads the point hazards into an in-memory grid index at startup (local GeoJSON when Atlas is not configured), and each trip makes one `$geoIntersects` query for zones the route lines cross, then computes exact distances and point-in-polygon exposure in JS.
- **Gemini** (`gemini-flash-lite-latest`, ~1 s versus ~15 s for the larger flash model) turns the scores into a short recommendation. It receives only the computed numbers, so it has no facts to invent.
- Public data: FDOT ArcGIS crash and active-construction services, Miami-Dade Open Data 311 and schools, NOAA CO-OPS, and api.weather.gov.

## Challenges

- FDOT's public crash layer stops at 2019, and Miami-Dade's 311 layers ship with null geometry, so we rebuilt points from the latitude and longitude columns.
- FEMA flood zones turned out to cover most of Miami, so they barely tell routes apart. Actual 311 flooding reports and elevation do.
- FDOT splits one construction project into several segment records, so we count distinct projects to avoid inflating work zones.
- NOAA's official surge maps are tile-only, so surge depth comes from a public county mirror of NOAA's national SLOSH rasters, sampled at 60 points per route in one multipoint call.
- Live storm alerts don't show up on command, so the demo has a clearly labeled "simulate a flood warning" switch.

## What we learned

The safest route is often only a few minutes slower, and drivers never see that option today.

## What's next

- Add fresher crash data (Signal Four Analytics).
- Weight hazards by time of day.
- Pedestrian and cyclist modes.
- Feed the scores into fleet routing for autonomous vehicles.
