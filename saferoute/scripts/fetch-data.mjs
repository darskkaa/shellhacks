// Downloads Miami-Dade hazard points from public ArcGIS services into data/*.geojson.
import { writeFile } from "node:fs/promises";
import { densify } from "../geo.mjs";

const FDOT_INJURY_CRASHES = "https://gis.fdot.gov/arcgis/rest/services/Crashes_All/MapServer/4/query";
const MDC_311 = (year) =>
  `https://services.arcgis.com/8Pc9XBTAsYuxx9Ny/arcgis/rest/services/data_311_${year}/FeatureServer/0/query`;
const FLOOD_ISSUES = [
  "FLOODING / STANDING WATER - LOCALIZED",
  "RER DRAINAGE CANALS FLOOD COMPLAINT",
  "DRAIN CLOGGED / CLEANING",
];
const PAGE_SIZE = 1000;
// Polite pause between pages; ~60 pages total, so this adds well under a minute.
const PAGE_PAUSE_MS = 250;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Offset paging until the server returns an empty page (a short page can still mean
// "server max is below PAGE_SIZE", so it is not treated as the end). Deduped by idField.
// No retries: a 429 or error stops the run and prints where it stopped.
async function fetchAll(url, where, outFields, idField) {
  const features = [];
  const seen = new Set();
  for (let offset = 0; ; offset += PAGE_SIZE) {
    const params = new URLSearchParams({
      where,
      outFields,
      outSR: "4326",
      resultOffset: String(offset),
      resultRecordCount: String(PAGE_SIZE),
      orderByFields: idField,
      f: "geojson",
    });
    const res = await fetch(url, { method: "POST", body: params, signal: AbortSignal.timeout(60_000) });
    if (!res.ok) throw new Error(`${url} HTTP ${res.status} at offset ${offset}; re-run to resume from scratch`);
    const page = await res.json();
    if (page.error) throw new Error(`${url} ${JSON.stringify(page.error)} at offset ${offset}`);
    if (!Array.isArray(page.features)) throw new Error(`${url} unexpected response shape at offset ${offset}`);
    if (!page.features.length) break;
    for (const f of page.features) {
      // 311 layers ship null geometry; coordinates live in attributes instead.
      const { latitude: lat, longitude: lon } = f.properties;
      f.geometry ??= Number.isFinite(lat) && Number.isFinite(lon) ? { type: "Point", coordinates: [lon, lat] } : null;
      const id = f.properties[idField];
      if (f.geometry && !seen.has(id)) {
        seen.add(id);
        features.push(f);
      }
    }
    process.stdout.write(`\r${url.split("/services/")[1]} ${features.length}`);
    await sleep(PAGE_PAUSE_MS);
  }
  process.stdout.write("\n");
  return features;
}

const crashes = await fetchAll(
  FDOT_INJURY_CRASHES,
  "COUNTY_TXT='MIAMI-DADE' AND CALENDAR_YEAR BETWEEN 2018 AND 2019",
  "CRASH_NUMBER,CALENDAR_YEAR,ON_ROADWAY_NAME,NUMBER_OF_KILLED,NUMBER_OF_SERIOUS_INJURIES,NUMBER_OF_INJURED,PEDESTRIAN_RELATED_IND,BICYCLIST_RELATED_IND",
  "CRASH_NUMBER",
);
await writeFile("data/crashes.geojson", JSON.stringify({ type: "FeatureCollection", features: crashes }));

const flooding = [];
for (const year of [2021, 2022, 2023]) {
  const where = `issue_type IN (${FLOOD_ISSUES.map((t) => `'${t}'`).join(",")})`;
  flooding.push(
    ...(await fetchAll(
      MDC_311(year),
      where,
      "ticket_id,issue_type,street_address,ticket_created_date_time,latitude,longitude",
      "ticket_id",
    )),
  );
}
await writeFile("data/flooding.geojson", JSON.stringify({ type: "FeatureCollection", features: flooding }));

console.log(`crashes: ${crashes.length}, flooding reports: ${flooding.length}`);

const SCHOOLS =
  "https://services.arcgis.com/8Pc9XBTAsYuxx9Ny/arcgis/rest/services/SchoolSite_gdb/FeatureServer/0/query";
const schools = await fetchAll(SCHOOLS, "1=1", "OBJECTID,NAME,ADDRESS,GRADES,ENROLLMNT", "OBJECTID");
await writeFile("data/schools.geojson", JSON.stringify({ type: "FeatureCollection", features: schools }));
console.log(`schools: ${schools.length}`);

// FDOT work segments are polylines; densify to points every ~40 m so the same
// point-to-route distance check used for crashes finds overlap with a route.
const FDOT_CONSTRUCTION =
  "https://gis.fdot.gov/arcgis/rest/services/Active_Construction_Projects/FeatureServer/1/query";
const DENSIFY_M = 40;
const projects = await fetchAll(
  FDOT_CONSTRUCTION,
  `County='Miami-Dade' AND (EstEndDate IS NULL OR EstEndDate >= DATE '${new Date().toISOString().slice(0, 10)}')`,
  "OBJECTID,Description,StartDate,EstEndDate,Website",
  "OBJECTID",
);
const constructionPoints = [];
for (const p of projects) {
  const lines = p.geometry.type === "MultiLineString" ? p.geometry.coordinates : [p.geometry.coordinates];
  const props = {
    project: p.properties.OBJECTID,
    description: p.properties.Description?.replace(/\s+/g, " ").trim(),
    estEnd: p.properties.EstEndDate,
  };
  for (const line of lines) {
    for (const [lat, lng] of densify(
      line.map(([x, y]) => [y, x]),
      DENSIFY_M,
    )) {
      constructionPoints.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: [lng, lat] },
        properties: props,
      });
    }
  }
}
await writeFile(
  "data/construction.geojson",
  JSON.stringify({ type: "FeatureCollection", features: constructionPoints }),
);
console.log(`construction: ${projects.length} active projects, ${constructionPoints.length} points`);

// Storm-surge evacuation zones A-E (CATEGORY 1-5 = lowest hurricane category that triggers the zone).
const EVAC_ZONES =
  "https://services.arcgis.com/8Pc9XBTAsYuxx9Ny/arcgis/rest/services/HurricaneEvacZone_gdb/FeatureServer/0/query";
const zones = await fetchAll(
  EVAC_ZONES,
  "ZONEID IS NOT NULL AND CATEGORY BETWEEN 1 AND 5",
  "OBJECTID,CATEGORY,ZONEID",
  "OBJECTID",
);
await writeFile("data/evac-zones.geojson", JSON.stringify({ type: "FeatureCollection", features: zones }));
console.log(`evacuation zone polygons: ${zones.length}`);
