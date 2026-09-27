// SafeRoute Miami: scores Google route alternatives by crash, flood, and school-zone exposure.
// Run: npm start (reads keys from the repo-root .env). On Vercel, api/index.mjs re-exports the request handler.
import { createServer } from "node:http";
import { gzipSync } from "node:zlib";
import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import { MongoClient } from "mongodb";
import {
  boundsOf,
  buildIndex,
  decodePolyline,
  densify,
  distanceM,
  exposure,
  hazardsNear,
  loadIndex,
  pointInPolygon,
  polygonBounds,
} from "./geo.mjs";
import { rankRoutes, routeEnergyKwh } from "./rank.mjs";

const PORT = Number(process.env.PORT ?? 3000);
const MAPS_KEY = process.env.GOOGLE_MAPS_API_KEY;
// Browser-only key for Maps JS; restrict it by HTTP referrer. Falls back to the server key for local dev only:
// the page embeds this key, so on a public Vercel deploy the fallback would publish the unrestricted server key.
const BROWSER_MAPS_KEY = process.env.GOOGLE_MAPS_BROWSER_KEY || (process.env.VERCEL ? undefined : MAPS_KEY);
const GEMINI_KEY = process.env.GEMINI_API_KEY;
const NWS_CONTACT = process.env.NWS_CONTACT ?? "saferoute-miami hackathon demo";
if (!MAPS_KEY) throw new Error("GOOGLE_MAPS_API_KEY missing (repo-root .env locally, project env vars on Vercel)");
if (!BROWSER_MAPS_KEY)
  throw new Error("GOOGLE_MAPS_BROWSER_KEY missing: on Vercel it must be a separate, referrer-restricted key");

const CRASH_RADIUS_M = 40;
const FLOOD_RADIUS_M = 60;
const SCHOOL_RADIUS_M = 150;
// Construction points are densified every 40 m along FDOT work segments; 30 m catches the route driving through them.
const CONSTRUCTION_RADIUS_M = 30;
// ponytail: flat weight per active work zone (~15 historical crashes); calibrate against FDOT work-zone crash rates if time allows.
const WORK_ZONE_WEIGHT = 15;
const LIVE_INCIDENT_RADIUS_M = 60;
// ponytail: an active crash on the route outweighs a historical one; flat weight, no decay by age.
const LIVE_INCIDENT_WEIGHT = 25;
// Miami streets start ponding around 1 m (~3 ft) above sea level during king tides.
const LOW_ELEVATION_M = 1;
// ponytail: fixed threshold near Virginia Key MHHW (~2.0 ft MLLW); use NOAA flood thresholds if it matters.
const KING_TIDE_FT = 2.6;
const ELEVATION_SAMPLES = 100;
// Route points every 50 m for area tests (evacuation zones, NWS alert areas).
const AREA_SAMPLE_M = 50;
// ponytail: flat surge penalty per km inside an active evacuation zone, scaled by how deep the zone is under the
// active category (zone A under Cat 3 counts 3x); swap for NHC inundation depth when a storm actually threatens.
// A road inside an active surge zone is likely impassable, so it outweighs crash history (~1-3 points per crash).
const SURGE_WEIGHT_PER_KM = 120;
const MIAMI = [25.7617, -80.1918];
const MRMS_QPE = "https://mapservices.weather.noaa.gov/raster/rest/services/obs/mrms_qpe/ImageServer/getSamples";
const WPC_QPF = "https://mapservices.weather.noaa.gov/vector/rest/services/precip/wpc_qpf/MapServer/identify";
// NOAA's SLOSH Maximum-of-Maximums inundation rasters (layer 0 = Cat 5 ... 4 = Cat 1). NOAA's own copy is tiles-only,
// so identify runs against Delaware County PA's public mirror of the national GeoTIFFs. No SLA: failures are non-fatal.
const SLOSH_IDENTIFY = "https://gis.delcopa.gov/arcgis/rest/services/NOAA_Storm_Surge_Risk/MapServer/identify";
const RAIN_SAMPLES = 60;
// ponytail: 50 mm (~2 in) of rain in 24 h doubles flood weight, capped at 3x; tune against 311 flood reports by rain day.
const RAIN_DOUBLING_MM = 50;
// ponytail: flat penalty per foot of worst-case surge along the route.
const SURGE_DEPTH_WEIGHT_PER_FT = 60;
// ponytail: flat penalty per mile below the simulator's water level; a flooded mile is treated as near-impassable.
const UNDERWATER_WEIGHT_PER_MI = 250;
const M_TO_FT = 3.28084;
const NHC_MAPSERVER =
  "https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer";

// Literal new URL(..., import.meta.url) paths: Vercel's file tracer bundles exactly these files with the function,
// and they resolve from the module, not the working directory.
const KIND_FILES = {
  crash: new URL("./data/crashes.geojson", import.meta.url),
  flood: new URL("./data/flooding.geojson", import.meta.url),
  school: new URL("./data/schools.geojson", import.meta.url),
  construction: new URL("./data/construction.geojson", import.meta.url),
};

// Atlas is the hazard store when MONGODB_URI is set (npm run load-mongo); local GeoJSON otherwise.
const db = process.env.MONGODB_URI
  ? await new MongoClient(process.env.MONGODB_URI, { serverSelectionTimeoutMS: 5_000, connectTimeoutMS: 5_000 })
      .connect()
      .then((client) => client.db("saferoute"))
      .catch((err) => {
        console.error(`MongoDB Atlas unreachable (${err.message}); using local data/ instead`);
        return null;
      })
  : null;
const hazardsCollection = db?.collection("hazards") ?? null;
const evacCollection = db?.collection("evac_zones") ?? null;
const localEvacZones = db
  ? null
  : JSON.parse(await readFile(new URL("./data/evac-zones.geojson", import.meta.url), "utf8")).features.map((f) => ({
      geometry: f.geometry,
      category: f.properties.CATEGORY,
      zone: f.properties.ZONEID,
    }));
// Point hazards are static (35k points, ~10 MB), so they are read once at startup into the in-memory grid index:
// from Atlas when configured, else from data/. A per-request bbox query cost 0.45-1 s for wide detour sets.
const hazardIndexesAll = hazardsCollection
  ? await hazardsCollection
      .find({}, { projection: { _id: 0 } })
      .toArray()
      .then((docs) =>
        Object.fromEntries(
          Object.keys(KIND_FILES).map((kind) => [kind, buildIndex(docs.filter((d) => d.kind === kind))]),
        ),
      )
  : Object.fromEntries(
      await Promise.all(Object.entries(KIND_FILES).map(async ([kind, file]) => [kind, await loadIndex(file)])),
    );
console.log(
  `hazards: ${hazardsCollection ? "MongoDB Atlas" : "local GeoJSON"}, ${Object.values(hazardIndexesAll).reduce((n, x) => n + x.size, 0)} points in memory`,
);

// Evacuation zone polygons crossed by any of the paths and triggered at or below the given hurricane category.
async function evacZonesAlong(paths, category) {
  if (category < 1) return [];
  if (evacCollection) {
    const line = { type: "MultiLineString", coordinates: paths.map((p) => p.map(([lat, lng]) => [lng, lat])) };
    return evacCollection
      .find(
        { category: { $lte: category }, geometry: { $geoIntersects: { $geometry: line } } },
        { projection: { _id: 0 } },
      )
      .toArray();
  }
  // Bounding-box overlap keeps zones that enclose the whole route too; exposure() does the exact test later.
  const [s, w, n, e] = boundsOf(paths, 0);
  return localEvacZones.filter((z) => {
    if (z.category > category) return false;
    const [zs, zw, zn, ze] = polygonBounds(z.geometry);
    return zs <= n && zn >= s && zw <= e && ze >= w;
  });
}

// ---------- live conditions ----------

// Every upstream call goes through here: explicit timeout, one log line per call (path only, so the
// Elevation key in the query string never reaches logs), and no automatic retries. This server answers
// a person waiting on a page, so a 429 fails fast with a clear message instead of stalling or re-probing.
async function fetchJson(url, init = {}, timeoutMs = 10_000) {
  const { host, pathname } = new URL(url);
  const started = Date.now();
  let res;
  try {
    res = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    console.log(`api ${host}${pathname} FAILED ${err.name} ${Date.now() - started}ms`);
    throw Object.assign(new Error(`${host} unreachable (${err.name})`, { cause: err }), { host });
  }
  console.log(`api ${host}${pathname} ${res.status} ${Date.now() - started}ms`);
  const body = await res.json().catch(() => null);
  if (res.status === 429)
    throw Object.assign(
      new Error(
        host === "routes.googleapis.com"
          ? "Google's daily route limit for this demo was reached; it resets at midnight Pacific. Try an example trip already searched in the last 15 minutes."
          : `${host} rate limited us; wait a few minutes before retrying`,
      ),
      { host, rateLimited: true },
    );
  if (!res.ok) {
    // Upstream error text stays in server logs; the browser only sees host and status.
    console.error(`api ${host}${pathname} error: ${body?.error?.message ?? "(no message)"}`);
    throw Object.assign(new Error(`${host} HTTP ${res.status}`), { upstreamStatus: res.status, host });
  }
  if (body === null) throw Object.assign(new Error(`${host} returned non-JSON`), { host });
  return body;
}

// In-memory TTL cache of promises: repeat demo queries cost zero upstream calls, and concurrent identical
// requests share one call. A failed call is kept for NEGATIVE_TTL_MS so a dead feed fails fast instead of stalling
// every request for its full timeout. Google calls (routes, elevation, Gemini) are the exception: a transient failure
// is retried on the next request, and only a rate limit is kept, so a quota error does not re-spend quota.
// ponytail: unbounded-ish Map capped by insertion order; swap for an LRU if traffic ever matters.
const CACHE_MAX = 500;
const cache = new Map();
function cached(key, ttlMs, fn) {
  const hit = cache.get(key);
  if (hit && hit.expires > Date.now()) return hit.value;
  const value = fn().catch((err) => {
    if (err.rateLimited || !/googleapis\.com$/.test(err.host ?? ""))
      cache.set(key, { value, expires: Date.now() + NEGATIVE_TTL_MS });
    else cache.delete(key);
    throw err;
  });
  cache.set(key, { value, expires: Date.now() + ttlMs });
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
  return value;
}
const MINUTE = 60_000;
const NEGATIVE_TTL_MS = 2 * MINUTE;

// Tides and alerts change slowly; one fetch per 10 minutes covers every request in between.
const getConditions = () => cached("conditions", 10 * MINUTE, fetchConditions);

async function fetchConditions() {
  const [tide, alerts] = await Promise.allSettled([
    fetchJson(
      "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?station=8723214&product=predictions" +
        "&datum=MLLW&units=english&time_zone=lst_ldt&interval=hilo&date=today&format=json",
    ),
    fetchJson("https://api.weather.gov/alerts/active?area=FL", {
      headers: { "User-Agent": NWS_CONTACT, Accept: "application/geo+json" },
    }),
  ]);
  const highs = tide.status === "fulfilled" ? (tide.value.predictions ?? []).filter((p) => p.type === "H") : [];
  const peak = highs.reduce((m, p) => (Number(p.v) > Number(m?.v ?? -Infinity) ? p : m), null);
  const localFeatures =
    alerts.status === "fulfilled"
      ? (Array.isArray(alerts.value?.features) ? alerts.value.features : []).filter((f) =>
          /Miami-Dade|Biscayne Bay|Metro Miami/i.test(f.properties.areaDesc),
        )
      : [];
  const localAlerts = localFeatures.map(({ properties: p }) => ({
    event: p.event,
    headline: p.headline,
    severity: p.severity,
  }));
  const stormMode = localAlerts.some((a) => /Flood|Hurricane|Tropical|Storm Surge/i.test(a.event));
  // Where each flood-type alert applies: its own polygon when NWS drew one, else its forecast zones' polygons.
  const alertAreas = [];
  for (const f of localFeatures) {
    if (!/Flood|Hurricane|Tropical|Storm Surge/i.test(f.properties.event)) continue;
    if (f.geometry) {
      alertAreas.push({ label: f.properties.event, geometry: f.geometry });
      continue;
    }
    const zones = await Promise.allSettled((f.properties.affectedZones ?? []).map(getNwsZone));
    for (const z of zones) {
      if (z.status === "fulfilled" && z.value) alertAreas.push({ label: f.properties.event, geometry: z.value });
    }
  }
  return {
    highTide: peak ? { time: peak.t, feet: Number(peak.v), kingTide: Number(peak.v) >= KING_TIDE_FT } : null,
    alerts: localAlerts,
    alertAreas,
    stormMode,
    unavailable: [tide, alerts].flatMap((r, i) => (r.status === "rejected" ? [["tides", "alerts"][i]] : [])),
  };
}

// NWS forecast/county zone outlines rarely change; cache each for a day. Only zones named by Miami alerts are fetched.
function getNwsZone(url) {
  if (!/^https:\/\/api\.weather\.gov\/zones\//.test(url)) return Promise.resolve(null);
  return cached(`zone|${url}`, 24 * 60 * MINUTE, async () => {
    const zone = await fetchJson(url, { headers: { "User-Agent": NWS_CONTACT, Accept: "application/geo+json" } });
    return /Polygon/.test(zone.geometry?.type ?? "") ? zone.geometry : null;
  });
}

// Evenly spaced subset of points, as [lng, lat], for one multipoint raster query.
function samplePoints(points, count) {
  const step = Math.max(1, Math.ceil(points.length / count));
  return points.filter((_, i) => i % step === 0).map(([lat, lng]) => [Number(lng.toFixed(5)), Number(lat.toFixed(5))]);
}
const multipoint = (pts) => JSON.stringify({ points: pts, spatialReference: { wkid: 4326 } });

// Observed rain over the last 24 h along the route from NOAA MRMS (~1 km radar/gauge mosaic, updated hourly).
function rainAlong(points, polyline) {
  return cached(`rain|${polyline}`, 10 * MINUTE, async () => {
    const params = new URLSearchParams({
      geometry: multipoint(samplePoints(points, RAIN_SAMPLES)),
      geometryType: "esriGeometryMultipoint",
      renderingRule: JSON.stringify({ rasterFunction: "rft_24hr" }),
      returnFirstValueOnly: "true",
      f: "json",
    });
    const body = await fetchJson(MRMS_QPE, { method: "POST", body: params });
    const mm = (body.samples ?? []).map((x) => Number(x.value)).filter((v) => Number.isFinite(v) && v >= 0);
    if (!mm.length) throw new Error("MRMS returned no samples");
    return { maxMm: Math.max(...mm), meanMm: mm.reduce((a, b) => a + b, 0) / mm.length };
  });
}

// WPC day-1 rain forecast (inches) at central Miami; isohyets are city-scale, so one point stands in for the county.
const getRainForecast = () =>
  cached("qpf", 60 * MINUTE, async () => {
    const params = new URLSearchParams({
      geometry: JSON.stringify({ x: MIAMI[1], y: MIAMI[0], spatialReference: { wkid: 4326 } }),
      geometryType: "esriGeometryPoint",
      sr: "4326",
      tolerance: "1",
      mapExtent: "-80.45,25.55,-80.12,25.98",
      imageDisplay: "800,800,96",
      layers: "all:1",
      returnGeometry: "false",
      f: "json",
    });
    const hit = (await fetchJson(`${WPC_QPF}?${params}`)).results?.find((r) => r.layerName === "QPF 24 Hour Day 1");
    // No polygon at the point means the forecast is below the lowest isohyet, i.e. effectively dry.
    return { inches: hit ? Number(hit.attributes.QPF) : 0, valid: hit?.attributes["Valid Time"] ?? null };
  });

// Worst-case surge depth (ft above ground) along the route for a hurricane category, from SLOSH MOM classes like
// "02 to 03 feet above ground"; the upper bound of the class is used.
function surgeDepthAlong(points, polyline, category) {
  return cached(`slosh|${category}|${polyline}`, 24 * 60 * MINUTE, async () => {
    const params = new URLSearchParams({
      geometry: multipoint(samplePoints(points, RAIN_SAMPLES)),
      geometryType: "esriGeometryMultipoint",
      sr: "4326",
      tolerance: "1",
      mapExtent: "-80.45,25.55,-80.12,25.98",
      imageDisplay: "2000,2000,96",
      layers: `all:${5 - category}`,
      returnGeometry: "false",
      f: "json",
    });
    const body = await fetchJson(SLOSH_IDENTIFY, { method: "POST", body: params }, 15_000);
    // Take the largest number in each class label so open-ended top classes ("... or greater") still count.
    const depths = (body.results ?? [])
      .map((r) => (r.attributes?.["Raster.data_range"] ?? "").match(/\d+/g))
      .filter(Boolean)
      .map((nums) => Math.max(...nums.map(Number)));
    return {
      maxFt: depths.length ? Math.max(...depths) : 0,
      flooded: depths.length,
      sampled: body.results?.length ?? 0,
    };
  });
}

// Saffir-Simpson category from sustained wind in knots; 0 means below hurricane strength.
const saffirSimpson = (kt) => (kt >= 137 ? 5 : kt >= 113 ? 4 : kt >= 96 ? 3 : kt >= 83 ? 2 : kt >= 64 ? 1 : 0);

// Active Atlantic storms from NHC with their 5-day forecast cone (NOAA ArcGIS service). Advisories come every
// 6 hours, so a 30-minute cache is plenty.
const getStorms = () => cached("storms", 30 * MINUTE, fetchStorms);

async function fetchStorms() {
  const current = await fetchJson("https://www.nhc.noaa.gov/CurrentStorms.json");
  const atlantic = (current.activeStorms ?? []).filter((st) => /^AT\d$/.test(st.binNumber));
  if (!atlantic.length) return [];
  // Layer ids shift between seasons, so resolve "<bin> Forecast Cone" by name.
  const layers = (await fetchJson(`${NHC_MAPSERVER}?f=json`)).layers ?? [];
  // One storm's cone failing should not blank out the others.
  const settled = await Promise.allSettled(
    atlantic.map(async (st) => {
      const coneLayer = layers.find((l) => l.name === `${st.binNumber} Forecast Cone`);
      const cone = coneLayer
        ? ((await fetchJson(`${NHC_MAPSERVER}/${coneLayer.id}/query?where=1%3D1&outFields=advisnum&f=geojson`))
            .features?.[0]?.geometry ?? null)
        : null;
      const knots = Number(st.intensity);
      const center = [Number(st.latitudeNumeric), Number(st.longitudeNumeric)];
      return {
        name: st.name,
        classification: st.classification,
        knots,
        category: saffirSimpson(knots),
        milesFromMiami: Math.round(distanceM(MIAMI, center) / 1609.34),
        threatensMiami: cone ? pointInPolygon(MIAMI, cone) : false,
        cone,
      };
    }),
  );
  for (const r of settled) if (r.status === "rejected") console.error("nhc storm:", r.reason.message);
  return settled.filter((r) => r.status === "fulfilled").map((r) => r.value);
}

// Miami-Dade Police active traffic incidents (public JSON, no key, no documented limits).
// Cached 2 minutes so page traffic never becomes more than one upstream call per 2 minutes.
const getLiveIncidents = () => cached("incidents", 2 * MINUTE, fetchLiveIncidents);

async function fetchLiveIncidents() {
  const rows = await fetchJson("https://traffic.mdpd.com/api/");
  if (!Array.isArray(rows)) throw new Error("MDPD traffic feed shape changed: expected an array");
  // The feed repeats the same incident several times; key on address + time + type.
  const unique = new Map();
  for (const r of rows) {
    const lat = Number(r.latitude),
      lng = Number(r.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;
    unique.set(`${r.address}|${r.createTime}|${r.signal}`, {
      type: "Feature",
      geometry: { type: "Point", coordinates: [lng, lat] },
      properties: {
        signal: String(r.signal ?? ""),
        address: String(r.address ?? ""),
        createTime: String(r.createTime ?? ""),
      },
    });
  }
  return [...unique.values()];
}

// ---------- google ----------

// wide = hazards are active, so fan out extra detours on both sides even when Google already offers alternatives.
function getRoutes(origin, destination, wide = false) {
  const key = `routes|${wide}|${JSON.stringify([origin.trim().toLowerCase(), destination.trim().toLowerCase()])}`;
  return cached(key, 5 * MINUTE, async () => {
    const base = wide
      ? structuredClone(await getRoutes(origin, destination))
      : await computeRoutes(origin, destination);
    const offsets = wide ? WIDE_DETOUR_OFFSETS_M : base.length < MIN_ROUTES ? [DETOUR_OFFSET_M] : [];
    return addDetours(origin, destination, base, offsets);
  });
}

async function computeRoutes(origin, destination, intermediates) {
  const body = await fetchJson("https://routes.googleapis.com/directions/v2:computeRoutes", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Goog-Api-Key": MAPS_KEY,
      "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.description",
    },
    body: JSON.stringify({
      origin: { address: origin },
      destination: { address: destination },
      travelMode: "DRIVE",
      routingPreference: "TRAFFIC_AWARE",
      computeAlternativeRoutes: !intermediates,
      ...(intermediates && { intermediates }),
    }),
  });
  const routes = body.routes ?? [];
  for (const r of routes) {
    if (typeof r.polyline?.encodedPolyline !== "string" || typeof r.duration !== "string") {
      throw new Error("Routes API response shape changed: missing polyline or duration");
    }
  }
  return routes;
}

// Google often returns a single route at quiet hours, and never plans around floods. SafeRoute adds its own candidates:
// routes forced through a pass-through point offset either side of the trip's midpoint.
const MIN_ROUTES = 3;
const DETOUR_OFFSET_M = 1200;
// Under flood or hurricane conditions, fan out further so an inland path is among the candidates.
const WIDE_DETOUR_OFFSETS_M = [1200, 2500, 4500];

async function addDetours(origin, destination, routes, offsets) {
  if (!routes.length || !offsets.length) return routes;
  const path = decodePolyline(routes[0].polyline.encodedPolyline);
  const [a, b, mid] = [path[0], path.at(-1), path[Math.floor(path.length / 2)]];
  // Unit vector perpendicular to the origin->destination line, in meters, converted back to degrees at the midpoint.
  const dx = (b[1] - a[1]) * Math.cos((mid[0] * Math.PI) / 180),
    dy = b[0] - a[0];
  const len = Math.hypot(dx, dy) || 1;
  const perp = [dx / len, -dy / len];
  const detours = await Promise.allSettled(
    offsets.flatMap((offset) =>
      [1, -1].map((side) => {
        const lat = mid[0] + (side * perp[0] * offset) / 111_320;
        const lng = mid[1] + (side * perp[1] * offset) / (111_320 * Math.cos((mid[0] * Math.PI) / 180));
        return computeRoutes(origin, destination, [
          { location: { latLng: { latitude: lat, longitude: lng } }, via: true },
        ]);
      }),
    ),
  );
  for (const d of detours) {
    const detour = d.status === "fulfilled" ? d.value[0] : null;
    // Skip detours that collapse onto a route we already have (same length within 2%) or wander far (> 1.6x length).
    if (!detour || detour.distanceMeters > 1.6 * routes[0].distanceMeters) continue;
    const mid = (r) => {
      const p = decodePolyline(r.polyline.encodedPolyline);
      return p[Math.floor(p.length / 2)];
    };
    const detourMid = mid(detour);
    const duplicate = routes.some(
      (r) =>
        Math.abs(r.distanceMeters - detour.distanceMeters) < 0.02 * r.distanceMeters &&
        distanceM(mid(r), detourMid) < 400,
    );
    if (duplicate) continue;
    routes.push({ ...detour, description: `${detour.description || "local roads"} (SafeRoute detour)` });
  }
  return routes;
}

// Fraction of evenly spaced samples along the route that sit below LOW_ELEVATION_M.
// Elevation is static, so cache per route shape for a day.
function lowElevationShare(path, polyline) {
  return cached(`elev|${polyline}`, 24 * 60 * MINUTE, () => fetchLowElevationShare(path));
}

async function fetchLowElevationShare(path) {
  const stride = Math.max(1, Math.ceil(path.length / 40));
  const anchors = path.filter((_, i) => i % stride === 0 || i === path.length - 1);
  const q = anchors.map(([lat, lng]) => `${lat.toFixed(5)},${lng.toFixed(5)}`).join("|");
  const body = await fetchJson(
    `https://maps.googleapis.com/maps/api/elevation/json?path=${q}&samples=${ELEVATION_SAMPLES}&key=${MAPS_KEY}`,
  );
  if (body.status !== "OK") throw new Error(`Elevation ${body.status}: ${body.error_message ?? ""}`);
  const samples = body.results.filter((r) => Number.isFinite(r.elevation) && r.location);
  if (!samples.length) throw new Error("Elevation returned no samples");
  const elev = samples.map((r) => r.elevation);
  return {
    share: elev.filter((e) => e < LOW_ELEVATION_M).length / elev.length,
    minM: Math.min(...elev),
    // Evenly spaced [lat, lng, meters] along the route, for the client flood simulator and elevation chart.
    profile: samples.map((r) => [
      Number(r.location.lat.toFixed(5)),
      Number(r.location.lng.toFixed(5)),
      Number(r.elevation.toFixed(2)),
    ]),
  };
}

function explain(summary) {
  if (!GEMINI_KEY) return Promise.resolve(null);
  return cached(`gemini|${JSON.stringify(summary)}`, 10 * MINUTE, () => fetchExplanation(summary)).catch((err) => {
    console.error("gemini:", err.message);
    return null;
  });
}

async function fetchExplanation(summary) {
  const prompt =
    // Route A is already the recommended route (rank.mjs); the model explains that choice, it does not re-rank.
    "You are a driving-safety assistant for Waymo rides in Miami. Route A is the recommended route under rankingRule. " +
    "In at most 3 short sentences and under 60 words, explain why Route A beats the alternatives on safety first, then " +
    "time and the Waymo's battery energy (waymoEnergyKwh), and state any extra minutes it costs, citing the numbers. " +
    "Crash data is 2018-2019 injury/fatal crashes; flood data " +
    "is 311 flooding reports 2021-2023; work zones are active FDOT construction projects; " +
    "activeIncidentsOnRouteNow are live police-reported crashes happening now; " +
    "rainLast24hMm is NOAA radar rainfall; worstCaseSurgeFeet is NOAA SLOSH surge depth for the active category; " +
    "milesUnderSimulatedWater is road below the user's chosen flood level; " +
    "milesInActiveHurricaneEvacuationZones is distance through storm-surge evacuation zones (zone A floods first). Use only the numbers given; never " +
    "mention traffic, congestion, or causes not in the data. If there is only one route, say Google returned one " +
    "route and summarize its hazards.\n\n" +
    JSON.stringify(summary);
  const body = await fetchJson(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent",
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY },
      body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
    },
    20_000,
  );
  return body.candidates?.[0]?.content?.parts?.map((p) => p.text).join("") ?? null;
}

// ---------- scoring ----------

async function scoreRoute(route, path, indexes, conditions, liveIndex, surge, waterFt) {
  const nearCrashes = hazardsNear(indexes.crash, path, CRASH_RADIUS_M);
  const nearFloods = hazardsNear(indexes.flood, path, FLOOD_RADIUS_M);
  const nearSchools = hazardsNear(indexes.school, path, SCHOOL_RADIUS_M);
  const nearWork = hazardsNear(indexes.construction, path, CONSTRUCTION_RADIUS_M);
  const nearLive = hazardsNear(liveIndex, path, LIVE_INCIDENT_RADIUS_M);
  // FDOT splits one project into several segment records, so count distinct descriptions.
  const workZones = [...new Map(nearWork.map((h) => [h.props.description, h.props])).values()];
  let elevation = null;
  try {
    elevation = await lowElevationShare(path, route.polyline.encodedPolyline);
  } catch (err) {
    console.error("elevation:", err.message);
  }

  const killed = nearCrashes.reduce((s, h) => s + Number(h.props.NUMBER_OF_KILLED ?? 0), 0);
  const serious = nearCrashes.reduce((s, h) => s + Number(h.props.NUMBER_OF_SERIOUS_INJURIES ?? 0), 0);
  const crashRisk =
    nearCrashes.length +
    3 * serious +
    10 * killed +
    WORK_ZONE_WEIGHT * workZones.length +
    LIVE_INCIDENT_WEIGHT * nearLive.length;
  const points = densify(path, AREA_SAMPLE_M);
  const [rain, surgeDepth] = await Promise.all([
    rainAlong(points, route.polyline.encodedPolyline).catch((err) => {
      console.error("mrms:", err.message);
      return null;
    }),
    surge.category
      ? surgeDepthAlong(points, route.polyline.encodedPolyline, surge.category).catch((err) => {
          console.error("slosh:", err.message);
          return null;
        })
      : null,
  ]);
  const rainFactor = 1 + Math.min((rain?.maxMm ?? 0) / RAIN_DOUBLING_MM, 2);
  // A flood alert only amplifies routes that actually enter its area; simulated alerts and alerts whose area
  // could not be resolved cover the whole county.
  const alertShare = conditions.alertAreas.length ? exposure(points, conditions.alertAreas, AREA_SAMPLE_M).share : 0;
  const inAlertArea = conditions.stormMode && (conditions.simulated || !conditions.alertAreas.length || alertShare > 0);
  // Flood exposure only matters when water is likely: amplify under storm alerts or king tides.
  const floodMultiplier = inAlertArea ? 4 : conditions.highTide?.kingTide ? 2 : 1;
  const surgeExposure = exposure(
    points,
    surge.zones.map((z) => ({ label: z.zone, geometry: z.geometry, category: z.category })),
    AREA_SAMPLE_M,
  );
  const surgeKm = Object.fromEntries(
    Object.entries(surgeExposure.metersByLabel).map(([zone, m]) => [zone, Number((m / 1000).toFixed(1))]),
  );
  const zoneCategory = Object.fromEntries(surge.zones.map((z) => [z.zone, z.category]));
  const surgeRisk = Object.entries(surgeKm).reduce(
    (sum, [zone, km]) => sum + SURGE_WEIGHT_PER_KM * km * (surge.category - zoneCategory[zone] + 1),
    0,
  );
  const profile = elevation?.profile ?? [];
  const underwaterMi =
    waterFt && profile.length
      ? (profile.filter((p) => p[2] * M_TO_FT <= waterFt).length / profile.length) * (route.distanceMeters / 1609.34)
      : 0;
  const floodRisk =
    UNDERWATER_WEIGHT_PER_MI * underwaterMi +
    floodMultiplier * rainFactor * (nearFloods.length * 5 + (elevation?.share ?? 0) * 100) +
    surgeRisk +
    SURGE_DEPTH_WEIGHT_PER_FT * (surgeDepth?.maxFt ?? 0);

  const durationSec = parseInt(route.duration, 10);
  return {
    description: route.description,
    durationSec,
    distanceM: route.distanceMeters,
    energyKwh: routeEnergyKwh({ distanceM: route.distanceMeters, durationSec, elevationProfile: elevation?.profile }),
    polyline: route.polyline.encodedPolyline,
    crashes: { count: nearCrashes.length, killed, serious },
    floodReports: nearFloods.length,
    lowElevationPct: elevation ? Math.round(elevation.share * 100) : null,
    elevationProfile: elevation?.profile ?? null,
    inFloodAlertArea: inAlertArea,
    surgeZoneKm: surgeKm,
    underwaterMi: Number(underwaterMi.toFixed(1)),
    rain24hMm: rain ? Math.round(rain.maxMm) : null,
    maxSurgeFt: surgeDepth ? surgeDepth.maxFt : null,
    schoolsNearby: nearSchools.map((h) => h.props.NAME),
    liveIncidents: nearLive.map((h) => h.props),
    workZones: workZones.map((p) => ({ description: p.description, estEnd: p.estEnd })),
    risk: { crash: Math.round(crashRisk), flood: Math.round(floodRisk), total: Math.round(crashRisk + floodRisk) },
    hazards: {
      // [lat, lng, severity, detail] where detail is the short text shown when the dot is clicked.
      crashes: nearCrashes.map((h) => {
        h = { ...h, lat: round5(h.lat), lng: round5(h.lng) };
        const p = h.props,
          killed = Number(p.NUMBER_OF_KILLED ?? 0),
          serious = Number(p.NUMBER_OF_SERIOUS_INJURIES ?? 0);
        const hurt = killed
          ? `${killed} killed`
          : serious
            ? `${serious} seriously injured`
            : `${Number(p.NUMBER_OF_INJURED ?? 0)} injured`;
        return [
          h.lat,
          h.lng,
          killed ? 2 : serious ? 1 : 0,
          `${p.CALENDAR_YEAR} crash on ${p.ON_ROADWAY_NAME ?? "this road"}: ${hurt}`,
        ];
      }),
      floods: nearFloods.map((h) => [
        round5(h.lat),
        round5(h.lng),
        1,
        `311 report: ${h.props.issue_type.toLowerCase()} at ${h.props.street_address} (${new Date(h.props.ticket_created_date_time).getFullYear()})`,
      ]),
      construction: nearWork.map((h) => [round5(h.lat), round5(h.lng), 1, `FDOT work zone: ${h.props.description}`]),
      live: nearLive.map((h) => [h.lat, h.lng, 1, `Now: ${h.props.signal.toLowerCase()} at ${h.props.address}`]),
    },
  };
}

async function handleRoutes(req, res) {
  let input;
  try {
    input = JSON.parse(await readBody(req));
  } catch (err) {
    if (err.status === 413) {
      // Answer first, then drop the connection so the rest of the oversized upload is never read.
      res.on("finish", () => req.destroy());
      return sendJson(res, 413, { error: "Request body too large." }, { Connection: "close" });
    }
    return sendJson(res, 400, { error: "Body must be JSON" });
  }
  const { origin, destination, simulate } = input ?? {};
  // Snapped to the slider's 0.5 ft step so arbitrary decimals can't fill the response cache.
  const waterFt = Number.isFinite(input?.waterFt) ? Math.round(Math.min(Math.max(input.waterFt, 0), 12) * 2) / 2 : 0;
  const simulatedCategory = /^hurricane-[1-5]$/.test(simulate ?? "") ? Number(simulate.slice(-1)) : 0;
  const valid = (s) => typeof s === "string" && s.trim().length > 0 && s.length <= 200;
  if (!valid(origin) || !valid(destination))
    return sendJson(res, 400, { error: "Enter both a start and a destination." });
  if (origin.trim().toLowerCase() === destination.trim().toLowerCase())
    return sendJson(res, 400, { error: "Start and destination are the same place." });

  // Whole-response cache: page loads and example chips repeat the same few trips, and each miss spends Google quota.
  const norm = (s) => s.trim().toLowerCase().replace(/\s+/g, " ");
  const scenario = simulate === "storm" || simulatedCategory ? simulate : "";
  const responseKey = JSON.stringify([norm(origin), norm(destination), scenario, waterFt]);
  const hit = responses.get(responseKey);
  if (hit && hit.expires > Date.now()) {
    // Re-insert so eviction drops the least recently used trip.
    responses.delete(responseKey);
    responses.set(responseKey, hit);
    console.log("routes cache hit");
    return sendJson(res, 200, hit.body, { "Server-Timing": "cache;desc=hit" });
  }
  responses.delete(responseKey);

  // Stage timings go out as a Server-Timing header (visible in DevTools > Network > Timing) and to the log.
  const timings = [];
  let mark = performance.now();
  const lap = (name) => {
    const now = performance.now();
    timings.push([name, now - mark]);
    mark = now;
  };
  const [baseRoutes, cachedConditions, incidents, storms, rainForecast] = await Promise.all([
    getRoutes(origin, destination),
    getConditions(),
    getLiveIncidents().catch((err) => {
      console.error("mdpd:", err.message);
      return null;
    }),
    getStorms().catch((err) => {
      console.error("nhc:", err.message);
      return null;
    }),
    getRainForecast().catch((err) => {
      console.error("wpc:", err.message);
      return null;
    }),
  ]);
  lap("inputs");
  const conditions = structuredClone(cachedConditions);
  conditions.liveIncidents = incidents?.length ?? null;
  if (!incidents) conditions.unavailable.push("live incidents");
  const liveIndex = buildIndex(incidents ?? []);
  // Demo switch: judges rarely see a live hurricane warning, so the UI can inject one (clearly labeled as simulated).
  if (simulate === "storm") {
    conditions.stormMode = true;
    conditions.simulated = true;
    conditions.alerts.unshift({
      event: "Flood Warning (simulated)",
      headline: "Simulated flood warning for Miami-Dade",
      severity: "Severe",
    });
  }
  if (simulatedCategory) {
    conditions.stormMode = true;
    conditions.simulated = true;
    conditions.alerts.unshift({
      event: `Hurricane Warning (simulated Category ${simulatedCategory})`,
      headline: "Simulated hurricane for Miami-Dade",
      severity: "Extreme",
    });
  }
  conditions.storms = storms;
  conditions.rainForecast = rainForecast;
  if (!storms) conditions.unavailable.push("hurricane center");
  // Surge zones activate from, in order: the demo simulator, a real NHC cone over Miami (at least zone A, even for a
  // tropical storm), or an NWS hurricane / storm surge warning (ponytail: assumes zones A-B without a category).
  const threatening = (storms ?? []).filter((st) => st.threatensMiami);
  const surgeCategory =
    simulatedCategory ||
    (threatening.length ? Math.max(1, ...threatening.map((st) => st.category)) : 0) ||
    (conditions.alerts.some((a) => /Hurricane Warning|Storm Surge Warning/i.test(a.event)) ? 2 : 0);
  conditions.surgeCategory = surgeCategory;
  conditions.waterFt = waterFt;
  // Hazard mode: plan around the water, not just rank Google's picks.
  conditions.hazardRouting = Boolean(surgeCategory || waterFt || conditions.stormMode);
  const routes = conditions.hazardRouting ? await getRoutes(origin, destination, true) : baseRoutes;
  lap("detours");
  if (!routes.length)
    return sendJson(res, 404, {
      error: "Google could not find a driving route. Check the addresses or pick an example trip.",
    });
  // Geocoded to the same spot (e.g. two spellings of one place): nothing to compare.
  if (routes[0].distanceMeters < 200) return sendJson(res, 400, { error: "Start and destination are the same place." });
  const paths = routes.map((r) => decodePolyline(r.polyline.encodedPolyline));
  const indexes = hazardIndexesAll;
  const surgeZones = await evacZonesAlong(paths, surgeCategory);
  lap("atlas");
  const surge = { category: surgeCategory, zones: surgeZones };
  const { routes: scored, ranking } = rankRoutes(
    await Promise.all(routes.map((r, i) => scoreRoute(r, paths[i], indexes, conditions, liveIndex, surge, waterFt))),
  );
  lap("score");
  const fastest = Math.min(...scored.map((r) => r.durationSec));
  for (const r of scored) r.extraMinutes = Math.round((r.durationSec - fastest) / 60);

  // Polygons stay out of the model prompt; it only needs the conditions in words and numbers.
  const { alertAreas, storms: stormList, ...conditionsText } = conditions;
  const summary = {
    conditions: {
      ...conditionsText,
      activeAtlanticStorms: (stormList ?? []).map(({ cone, ...st }) => st),
    },
    rankingRule: ranking.rule,
    // Minutes and a trimmed shape so the model quotes numbers a driver would say out loud.
    routes: scored.map((r, i) => ({
      route: String.fromCharCode(65 + i),
      via: r.description,
      minutes: Math.round(r.durationSec / 60),
      extraMinutes: r.extraMinutes,
      miles: Number((r.distanceM / 1609.34).toFixed(1)),
      crashesNearby: r.crashes.count,
      fatalCrashes: r.crashes.killed,
      floodReports: r.floodReports,
      activeWorkZones: r.workZones.map((w) => w.description),
      activeIncidentsOnRouteNow: r.liveIncidents.map((i) => `${i.signal} at ${i.address}`),
      percentBelow1m: r.lowElevationPct,
      milesInActiveHurricaneEvacuationZones: Object.fromEntries(
        Object.entries(r.surgeZoneKm).map(([z, km]) => [z, Number((km / 1.609).toFixed(1))]),
      ),
      insideFloodAlertArea: r.inFloodAlertArea,
      milesUnderSimulatedWater: r.underwaterMi,
      rainLast24hMm: r.rain24hMm,
      worstCaseSurgeFeet: r.maxSurgeFt,
      riskScore: r.risk.total,
      waymoEnergyKwh: r.energyKwh,
      badges: r.badges,
      paretoOptimal: r.frontier,
      dominatedByRoute: r.dominatedBy,
    })),
  };
  const explainId = createHash("sha1").update(JSON.stringify(summary)).digest("hex").slice(0, 16);
  explanations.set(explainId, explain(summary));
  if (explanations.size > 200) explanations.delete(explanations.keys().next().value);
  const serverTiming = ["cache;desc=miss", ...timings.map(([name, ms]) => `${name};dur=${ms.toFixed(0)}`)].join(", ");
  console.log(`routes ${routes.length} | ${serverTiming}`);
  // The map draws alert areas only for real storm mode and cones only when they cover Miami.
  const wireConditions = {
    ...conditions,
    alertAreas:
      conditions.stormMode && !conditions.simulated
        ? conditions.alertAreas.map((a) => ({ ...a, geometry: slimGeometry(a.geometry) }))
        : [],
    storms: conditions.storms?.map((st) => ({ ...st, cone: st.threatensMiami ? slimGeometry(st.cone) : null })) ?? null,
  };
  const body = {
    conditions: wireConditions,
    surgeZones: surgeZones.map((z) => ({ zone: z.zone, category: z.category, geometry: slimGeometry(z.geometry) })),
    routes: scored,
    ranking,
    explainId,
  };
  // A response built while a live feed was down is kept briefly: long enough that reloads skip the dead feed and
  // spend no Routes quota, short enough that the full view returns soon after the feed recovers.
  const ttl = conditions.unavailable.length ? DEGRADED_RESPONSE_TTL_MS : RESPONSE_TTL_MS;
  responses.set(responseKey, { body, expires: Date.now() + ttl });
  if (responses.size > RESPONSE_CACHE_MAX) responses.delete(responses.keys().next().value);
  sendJson(res, 200, body, { "Server-Timing": serverTiming });
}

// Successful /api/routes bodies by normalized (origin, destination, scenario, waterFt). Live conditions inside
// (tide, alerts, police crashes) can be up to 15 minutes stale on a hit; Map order doubles as LRU order.
const RESPONSE_TTL_MS = 15 * MINUTE;
const DEGRADED_RESPONSE_TTL_MS = 2 * MINUTE;
const RESPONSE_CACHE_MAX = 100;
const responses = new Map();

// Gemini summaries in flight or done, by id; the route response returns before the model does.
const explanations = new Map();

async function handleExplain(req, res) {
  const id = new URL(req.url, "http://localhost").searchParams.get("id") ?? "";
  const pending = explanations.get(id);
  if (!pending) return sendJson(res, 404, { error: "unknown or expired explanation id" });
  sendJson(res, 200, { explanation: await pending });
}

// ---------- http ----------

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
};
// The only files served; an allowlist also rules out path traversal.
const INDEX_HTML = new URL("./public/index.html", import.meta.url);
const PUBLIC_FILES = {
  "/": INDEX_HTML,
  "/index.html": INDEX_HTML,
  "/icons.svg": new URL("./public/icons.svg", import.meta.url),
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    const onData = (chunk) => {
      data += chunk;
      if (data.length > 10_000) {
        req.off("data", onData);
        req.pause();
        reject(Object.assign(new Error("body too large"), { status: 413 }));
      }
    };
    req.on("data", onData);
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

// Round coordinates to 5 decimals (~1 m) for the wire.
const round5 = (n) => Math.round(n * 1e5) / 1e5;
const roundCoords = (c) => (typeof c[0] === "number" ? c.map(round5) : c.map(roundCoords));
const slimGeometry = (g) => g && { type: g.type, coordinates: roundCoords(g.coordinates) };

// JSON responses are gzipped when the browser accepts it: route payloads are hundreds of KB of coordinates.
function sendJson(res, status, body, headers = {}) {
  const json = JSON.stringify(body);
  if (/\bgzip\b/.test(res.req?.headers["accept-encoding"] ?? "") && json.length > 1024) {
    res.writeHead(status, {
      "Content-Type": "application/json",
      "Content-Encoding": "gzip",
      Vary: "Accept-Encoding",
      ...headers,
    });
    res.end(gzipSync(json));
    return;
  }
  res.writeHead(status, { "Content-Type": "application/json", ...headers });
  res.end(json);
}

async function serveStatic(req, res) {
  const url = new URL(req.url, "http://localhost");
  const fileUrl = PUBLIC_FILES[url.pathname];
  if (!fileUrl) return sendJson(res, 404, { error: "not found" });
  const file = fileUrl.pathname;
  try {
    let content = await readFile(fileUrl);
    // The Maps JS key is public by design; protect it with HTTP-referrer restrictions in Cloud Console.
    if (file.endsWith("index.html")) content = content.toString().replace("__MAPS_KEY__", () => BROWSER_MAPS_KEY);
    // HTML revalidates every load (it carries the key and changes during development); other assets cache for a day.
    const headers = {
      "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
      "Cache-Control": file.endsWith(".html") ? "no-cache" : "public, max-age=86400",
    };
    if (/\bgzip\b/.test(req.headers["accept-encoding"] ?? "") && /\.(html|svg|js|css|json)$/.test(file)) {
      res.writeHead(200, { ...headers, "Content-Encoding": "gzip", Vary: "Accept-Encoding" });
      res.end(gzipSync(content));
      return;
    }
    res.writeHead(200, headers);
    res.end(content);
  } catch {
    sendJson(res, 404, { error: "not found" });
  }
}

// Default export so Vercel can run this module as a Node function (api/index.mjs); `node server.mjs` also listens.
export default async function handler(req, res) {
  // 8. Route on the path only, so query strings do not change which handler runs.
  const path = new URL(req.url, "http://localhost").pathname;
  try {
    if (req.method === "POST" && path === "/api/routes") return await handleRoutes(req, res);
    if (req.method === "GET" && path === "/api/explain") return await handleExplain(req, res);
    if (req.method === "GET") return await serveStatic(req, res);
    sendJson(res, 405, { error: "method not allowed" });
  } catch (err) {
    console.error(err);
    if (err.host === "routes.googleapis.com" && err.upstreamStatus === 400) {
      return sendJson(res, 400, {
        error: "Google could not read one of those addresses. Try a fuller address or an example trip.",
      });
    }
    sendJson(res, 502, { error: err.message });
  }
}

// realpath: argv[1] keeps symlinks while import.meta.filename resolves them.
// An unresolvable argv[1] (as on some serverless runtimes) means this module was imported, not run.
const runDirectly = (() => {
  try {
    return Boolean(process.argv[1]) && realpathSync(process.argv[1]) === import.meta.filename;
  } catch {
    return false;
  }
})();
if (runDirectly) {
  createServer(handler).listen(PORT, () => console.log(`SafeRoute Miami on http://localhost:${PORT}`));
}
