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
npm run check                         # geometry, ranking, snapshot and page self-checks
npm run record                        # optional: record demo snapshots from a running server (see below)
```

Requires Node 20.11+. The Google Cloud project needs **Maps JavaScript API**, **Routes API**, and **Elevation API** enabled.

## Deploy to Vercel

`server.mjs` default-exports its request handler; `node server.mjs` still listens on `PORT`, and on Vercel `api/index.mjs` re-exports the handler as one Node function. `vercel.json` routes every path to it (so the page gets its Maps key substituted), bundles `data/` and `public/`, and sets `maxDuration` to 60 s.

1. Create a Vercel project from this repo with **Root Directory** `saferoute` (framework preset "Other", no build command).
2. In Project Settings > Environment Variables set `GOOGLE_MAPS_API_KEY` and `GOOGLE_MAPS_BROWSER_KEY`, plus optional `GEMINI_API_KEY`, `MONGODB_URI`, `NWS_CONTACT` (see below). Do not set `PORT`.
3. In Google Cloud Console add the Vercel domain (`https://<project>.vercel.app/*`) to the browser key's HTTP-referrer restrictions.
4. Deploy from this folder:

```sh
npx vercel link    # pick the project; creates .vercel/ (git-ignored by the CLI)
npx vercel --prod
```

Hazard GeoJSON in `data/` is committed, so no build step is needed. With `MONGODB_URI`, Atlas must allow Vercel's dynamic IPs (network access `0.0.0.0/0`); if Atlas is unreachable the function falls back to `data/`. Gemini summaries live in instance memory, so a `/api/explain` call that lands on a different instance gets a 404 and the page just hides the summary.

## Environment

| Variable                  | Required  | Use                                                                                                                                                                        |
| ------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GOOGLE_MAPS_API_KEY`     | yes       | Server-side Routes and Elevation. API-restrict it; do not add HTTP-referrer restrictions (server calls would fail).                                                        |
| `GOOGLE_MAPS_BROWSER_KEY` | on Vercel | Maps JavaScript key sent to the browser; restrict it by HTTP referrer. Required on Vercel (the page is public); locally it falls back to `GOOGLE_MAPS_API_KEY` when unset. |
| `GEMINI_API_KEY`          | no        | plain-English route recommendation                                                                                                                                         |
| `MONGODB_URI`             | no        | when set, hazards are queried from Atlas (`saferoute.hazards`); otherwise from `data/`                                                                                     |
| `NWS_CONTACT`             | no        | `User-Agent` contact sent to api.weather.gov                                                                                                                               |
| `PORT`                    | no        | default 3000; unused on Vercel                                                                                                                                             |
| `SNAPSHOT_MODE`           | no        | `off` (default), `fallback` or `always`; serves recorded demo answers from `data/snapshots/` (see Recorded demo mode)                                                      |

## Recorded demo mode

So a live demo survives Google's 150/day Routes cap and overnight changes in conditions, the server can answer example trips from recordings, always labeled as such.

- **Record** (spends Routes quota on the recording server): start the server with real keys and `SNAPSHOT_MODE=off`, then `npm run record` (or `npm run record -- https://<host> live,storm` for another server or a subset of scenarios). It reads the example chips and the Conditions select from `public/index.html` and POSTs every trip x scenario (live, flood warning, hurricanes 1-5; `waterFt` unset) one at a time, skipping and reporting failures. Each answer lands in `data/snapshots/<trip>--<scenario>-<hash>.json` as `{key, recordedAt, explanation, body}`, where `key` is the same normalized key as the 15-minute response cache and `body` is the exact `/api/routes` response. The recorder only talks to the server URL; it never reads `.env`.
- **Serve**: `SNAPSHOT_MODE=fallback` answers from a recording only when the live request fails (Google 429 or other error, or a timeout); `always` answers every recorded trip from its recording without calling Google, and other trips go live. A snapshot is served only for its exact key (trip, scenario, water level), never a neighbor. The startup log names the mode and how many recordings loaded. On Vercel: `npx vercel env add SNAPSHOT_MODE production`, then redeploy (`data/**` is already bundled).
- **Honesty label**: a served snapshot carries `snapshot: {recordedAt}` and `Server-Timing: snapshot;desc=<mode>`. The page then swaps "Live · 14 public data sources" for "Recorded <local date and time> · not live", titles the conditions panel "Conditions as recorded <time>", shows the recording time in its clock tile, and adds "Recorded <time>, not live data" to the screen-reader status.
- **Terms**: Google Maps Platform terms restrict caching or storing Routes API results. Recordings are for the hackathon demo only; delete them afterwards and do not ship them in a production deployment.

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
2. Point hazards are loaded into an in-memory grid index once at startup (from MongoDB Atlas when `MONGODB_URI` is set, else the local GeoJSON in `data/`), and each trip reads the cells along its routes. Active evacuation zones the routes cross come from one `$geoIntersects` query on Atlas, or a bounding-box filter of the local polygons; `geo.mjs` keeps those within 40 m (crashes), 60 m (flood reports), 30 m (construction), or 150 m (schools) of each route.
3. Crash risk = crashes + 3 × serious injuries + 10 × deaths + 15 × distinct active work zones + 25 × live police-reported crashes within 60 m. Flood risk = (5 × flood reports + % of route below 1 m) × tide/alert multiplier × rain factor + surge. The multiplier is 2 on king-tide days and 4 when the route enters an NWS flood or tropical alert area. The rain factor is 1 + MRMS 24 h rain / 50 mm, capped at 3. Surge = 120 per km inside active evacuation zones (scaled by how far below the active category the zone triggers) + 60 per foot of worst-case SLOSH surge. When the request carries `waterFt`, each mile of road at or below that level (from the elevation profile) adds 250, so flooded roads outweigh crash history.
4. Energy: `rank.mjs` estimates the Waymo Jaguar I-PACE's battery use per route from a small physics model: rolling resistance (2,200 kg, Crr 0.011), aero drag at the route's average speed (Cd 0.29, 2.6 m²), a stop-and-go term (60 / average km/h stops per km, each re-accelerating to at least 30 mph with 60% regen), elevation gain and loss from the elevation profile (60% regen on descents), 90% drivetrain efficiency, and a flat 1.5 kW for A/C and Waymo's sensors and compute. A 1.6 km downtown trip comes out around 0.3 kWh.
5. Ranking is Pareto over (risk, time, energy), safety first. Among routes whose risk is within `riskTie` = max(3, 5% of the lowest risk) of the safest, the fastest is recommended; if times are within 30 s, the one using the least energy. A route is on the frontier when no other route is at least as good on all three and better on one. Routes are ordered recommended first, then the other frontier routes by risk, then dominated routes by risk.
6. Gemini writes a three-sentence recommendation from the numbers only, including energy and the ranking rule.

Every upstream call has a timeout, is logged (path and status, never keys), and is not retried: a 429 returns a clear error to the page instead of re-probing (for Google Routes, that the daily demo limit resets at midnight Pacific). A failed live feed (police, NHC, WPC, NOAA, NWS) is remembered for 2 minutes so later requests skip it instead of waiting out its timeout; Google calls are retried on the next request unless Google rate limited us.

## Frontend

`public/index.html` is one static page (no build step). Beyond the route cards it has:

- **Why card**: names the concrete difference between the recommended route and Google's fastest (e.g. "Avoids 8 flood reports and 19 crashes for +8 s"), the flood exposure of the route we avoided, and the `ranking.rule` sentence.
- **Badges and frontier chart**: cards carry Recommended / Safest / Fastest / Most efficient badges; dominated routes are dashed with "Beaten by Route X". An SVG chart plots risk against minutes (bubble size = kWh) with the Pareto frontier drawn (when one route beats all others, the region it dominates is shaded instead of a line); points are keyboard-focusable and a visually hidden table carries the same data for screen readers.
- **Flood pins**: 311 flood reports within about 500 m of each other are grouped into numbered pins on every route, including the ones we avoided; low-lying stretches (below 1 m, or below the simulator level) are blue.
- **Flood demo chip**: "Design District → Legion Park" is the featured example and the trip the page searches on load; Google's fastest there passes 8 flood reports that the recommended route avoids for about 8 s.
- **Accessibility**: 44 px touch targets, visible focus, `aria-live` loading and results, a skip link, light and dark themes. Checked with axe-core (0 violations, desktop and phone) and Lighthouse (accessibility 100).

- **Flood simulator**: a 0-12 ft water-level slider (feet above mean sea level). Each route's `elevationProfile` (100 Google Elevation samples) is compared with the level; submerged stretches turn blue on the map, a liquid gauge and each card show the share under water, and an SVG elevation chart shows ground versus water. Hovering the chart moves a marker along the route. The panel lists each route's lowest point; while the level floods no route (on the demo trips nothing goes under until about 9.7 ft) it says so and "Find a drier route" is disabled, and when every route floods equally it warns that a drier route may not exist among these candidates. It is a bathtub model, not a flood forecast.
- **Clickable hazards**: every dot on the map opens a one-line description (crash year, road and injuries; 311 report; FDOT project; live police call).
- Icons are a local Lucide sprite (`public/icons.svg`, ISC). Card spotlight and liquid gauge follow React Bits' SpotlightCard and SloshGauge; the border beam, shimmer button and count-up follow Magic UI. All are re-written as plain CSS/JS, and all motion is off under `prefers-reduced-motion`.

## Performance and debugging

- Each `/api/routes` response carries a `Server-Timing` header (`cache;desc=miss` plus inputs, detours, atlas, score), visible in DevTools > Network > Timing, and the same line is logged. A new trip takes about 1-1.6 s on a laptop.
- Successful `/api/routes` responses are cached in memory for 15 minutes (100 trips, least recently used evicted), keyed by origin and destination (trimmed, lowercased, whitespace collapsed), scenario and `waterFt`. A hit returns instantly with `Server-Timing: cache;desc=hit` and spends no Google quota, so page loads and example chips are free after the first search. Trade-off: tide, alerts and live police crashes in a cached answer can be up to 15 minutes old. A response built while a live feed was down is cached for only 2 minutes. Each Vercel instance has its own cache.
- The Gemini summary is not on the critical path: the response returns an `explainId` and the page fetches `GET /api/explain?id=...` afterwards.
- Point hazards load into memory once at startup (Atlas when `MONGODB_URI` is set, local GeoJSON otherwise); evacuation zones are still queried per trip with `$geoIntersects` on Atlas.
- JSON and static files are gzipped; static assets cache for a day; fonts load without blocking first paint.
- `npm run lint` runs eslint and prettier checks.

## API

`GET /api/explain?id=<explainId>` returns `{explanation}` once Gemini answers (404 for unknown ids).

`POST /api/routes` with `{"origin": "...", "destination": "...", "simulate": "storm"?}` returns `{conditions, routes, ranking, explainId}`. Routes follow the ranking above (`ranking` is `{rule, riskTie}`), each with `energyKwh`, `frontier`, `dominatedBy` (letter of a dominating route, or null), `badges` (any of `safest`, `fastest`, `efficient`), `recommended` (exactly one), `extraMinutes` versus the fastest, crash, flood, elevation, work-zone, and school-zone counts, a risk score, an `elevationProfile` of `[lat, lng, meters]` samples, and `hazards` arrays of `[lat, lng, severity, detail]` for the map. `waterFt` (0-12, optional) reroutes around a flat water level. `simulate` accepts `"storm"` (simulated flood warning) or `"hurricane-1"` ... `"hurricane-5"` (simulated hurricane of that category, which activates evacuation zones A through the matching letter and SLOSH surge depth). The response also carries `surgeZones` (active zone polygons along the routes) and `conditions.storms`, `conditions.alertAreas`, `conditions.rainForecast`, `conditions.surgeCategory`. When the answer comes from a recording (`SNAPSHOT_MODE`), it also carries `snapshot: {recordedAt}` (ISO time).
