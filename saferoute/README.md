# SafeRoute Miami

Ranks Google driving routes across Miami-Dade by crash history, flood reports, low elevation, and live tide and storm alerts, so drivers can trade a few minutes for a safer trip.

## Run

```sh
cp -n .env.example ../.env            # keys live in the repo-root .env (git-ignored); -n never overwrites
git config core.hooksPath .githooks   # blocks committing .env or API keys
npm install
npm run fetch-data                    # downloads hazard data into data/
npm run load-mongo                    # optional: loads data/ into MongoDB Atlas with a 2dsphere index
npm start                             # http://localhost:3000
npm run check                         # geometry self-check
```

Requires Node 20.11+. The Google Cloud project needs **Maps JavaScript API**, **Routes API**, and **Elevation API** enabled.

## Environment

| Variable                  | Required    | Use                                                                                                                    |
| ------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| `GOOGLE_MAPS_API_KEY`     | yes         | Server-side Routes and Elevation. API-restrict it; do not add HTTP-referrer restrictions (server calls would fail).    |
| `GOOGLE_MAPS_BROWSER_KEY` | recommended | Maps JavaScript key sent to the browser; restrict it by HTTP referrer. Falls back to `GOOGLE_MAPS_API_KEY` when unset. |
| `GEMINI_API_KEY`          | no          | plain-English route recommendation                                                                                     |
| `MONGODB_URI`             | no          | when set, hazards are queried from Atlas (`saferoute.hazards`); otherwise from `data/`                                 |
| `NWS_CONTACT`             | no          | `User-Agent` contact sent to api.weather.gov                                                                           |
| `PORT`                    | no          | default 3000                                                                                                           |

## Data

| Source                                                           | What                                                                                                                | Loaded                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| FDOT Crashes_All layer 4                                         | Miami-Dade injury/fatal crashes, 2018-2019 (latest public)                                                          | `data/crashes.geojson`                        |
| Miami-Dade 311                                                   | flooding, standing water, clogged drain reports 2021-2023                                                           | `data/flooding.geojson`                       |
| Miami-Dade Public Schools                                        | 451 school sites                                                                                                    | `data/schools.geojson`                        |
| FDOT Active Construction Projects                                | Miami-Dade work segments not yet past their estimated end date (152 on Sep 26 2026), densified to points every 40 m | `data/construction.geojson`                   |
| NOAA CO-OPS station 8723214                                      | today's high tides (king tide flag)                                                                                 | live, cached 10 min                           |
| NWS alerts                                                       | Miami-Dade flood/tropical alerts (storm mode)                                                                       | live, cached 10 min                           |
| Google Elevation                                                 | share of route below 1 m                                                                                            | live, cached 24 h per route                   |
| Miami-Dade Hurricane Evacuation Zones                            | 1,547 storm-surge zone polygons A-E (A floods first)                                                                | `data/evac-zones.geojson`, Atlas `evac_zones` |
| NHC CurrentStorms + NOAA tropical MapServer                      | active Atlantic storms and 5-day forecast cones; a cone over Miami activates surge zones                            | live, cached 30 min                           |
| NWS forecast zones                                               | polygons of the zones a Miami flood alert covers, so only routes inside get the storm multiplier                    | live, cached 24 h per zone                    |
| NOAA MRMS QPE (mapservices.weather.noaa.gov)                     | radar/gauge rain in the last 24 h along each route                                                                  | live, cached 10 min per route                 |
| NOAA WPC QPF                                                     | day-1 rain forecast at central Miami                                                                                | live, cached 1 h                              |
| NOAA SLOSH MOM inundation (via Delaware County PA public mirror) | worst-case surge depth along the route for the active hurricane category                                            | live, cached 24 h per route and category      |
| Miami-Dade Police traffic feed (traffic.mdpd.com/api/)           | crashes happening now, deduped                                                                                      | live, cached 2 min                            |

## How scoring works

1. Google Routes returns up to three alternatives (cached 5 min per origin/destination). When it returns fewer than three, SafeRoute asks for detours forced through a pass-through point 1.2 km either side of the trip midpoint and labels them "SafeRoute detour". In hazard mode (an NWS flood/tropical alert, an active surge category, or a simulator water level) it always fans out detours at 1.2, 2.5 and 4.5 km on both sides, so an inland path is among the candidates (up to 9 routes, cached 5 min).
2. One MongoDB `$geoWithin` query pulls every point hazard in the routes' bounding box, and one `$geoIntersects` query pulls the active evacuation zones the routes cross; `geo.mjs` keeps those within 40 m (crashes), 60 m (flood reports), 30 m (construction), or 150 m (schools) of each route.
3. Crash risk = crashes + 3 × serious injuries + 10 × deaths + 15 × distinct active work zones + 25 × live police-reported crashes within 60 m. Flood risk = (5 × flood reports + % of route below 1 m) × tide/alert multiplier × rain factor + surge. The multiplier is 2 on king-tide days and 4 when the route enters an NWS flood or tropical alert area. The rain factor is 1 + MRMS 24 h rain / 50 mm, capped at 3. Surge = 120 per km inside active evacuation zones (scaled by how far below the active category the zone triggers) + 60 per foot of worst-case SLOSH surge. When the request carries `waterFt`, each mile of road at or below that level (from the elevation profile) adds 250, so flooded roads outweigh crash history.
4. Gemini writes a three-sentence recommendation from the numbers only.

Every upstream call has a timeout, is logged (path and status, never keys), and is not retried: a 429 returns a clear error to the page instead of re-probing.

## Frontend

`public/index.html` is one static page (no build step). Beyond the route cards it has:

- **Flood simulator**: a 0-12 ft water-level slider. Each route's `elevationProfile` (100 Google Elevation samples) is compared with the level; submerged stretches turn blue on the map, a liquid gauge and each card show the share under water, and an SVG elevation chart shows ground versus water. Hovering the chart moves a marker along the route. It is a bathtub model, not a flood forecast.
- **Clickable hazards**: every dot on the map opens a one-line description (crash year, road and injuries; 311 report; FDOT project; live police call).
- Icons are a local Lucide sprite (`public/icons.svg`, ISC). Card spotlight and liquid gauge follow React Bits' SpotlightCard and SloshGauge; the border beam, shimmer button and count-up follow Magic UI. All are re-written as plain CSS/JS, and all motion is off under `prefers-reduced-motion`.

## Performance and debugging

- Each `/api/routes` response carries a `Server-Timing` header (inputs, detours, atlas, score), visible in DevTools > Network > Timing, and the same line is logged. A new trip takes about 1-1.6 s on a laptop; repeats are served from cache in well under a second.
- The Gemini summary is not on the critical path: the response returns an `explainId` and the page fetches `GET /api/explain?id=...` afterwards.
- Point hazards load from Atlas into memory once at startup; evacuation zones are still queried per trip with `$geoIntersects`.
- JSON and static files are gzipped; static assets cache for a day; fonts load without blocking first paint.
- `npm run lint` runs eslint and prettier checks.

## API

`GET /api/explain?id=<explainId>` returns `{explanation}` once Gemini answers (404 for unknown ids).

`POST /api/routes` with `{"origin": "...", "destination": "...", "simulate": "storm"?}` returns `{conditions, routes, explanation}`. Routes are sorted safest first, each with crash, flood, elevation, work-zone, and school-zone counts, a risk score, an `elevationProfile` of `[lat, lng, meters]` samples, and `hazards` arrays of `[lat, lng, severity, detail]` for the map. `waterFt` (0-12, optional) reroutes around a flat water level. `simulate` accepts `"storm"` (simulated flood warning) or `"hurricane-1"` ... `"hurricane-5"` (simulated hurricane of that category, which activates evacuation zones A through the matching letter and SLOSH surge depth). The response also carries `surgeZones` (active zone polygons along the routes) and `conditions.storms`, `conditions.alertAreas`, `conditions.rainForecast`, `conditions.surgeCategory`.
