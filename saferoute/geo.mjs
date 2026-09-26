// Polyline decoding and grid-indexed hazard lookup along a route.
import { readFile } from "node:fs/promises";

const M_PER_DEG_LAT = 111_320;
const mPerDegLon = (lat) => M_PER_DEG_LAT * Math.cos((lat * Math.PI) / 180);

export function decodePolyline(str) {
  const points = [];
  let i = 0,
    lat = 0,
    lng = 0;
  while (i < str.length) {
    for (const axis of [0, 1]) {
      let shift = 0,
        result = 0,
        byte;
      do {
        byte = str.charCodeAt(i++) - 63;
        result |= (byte & 0x1f) << shift;
        shift += 5;
      } while (byte >= 0x20);
      const delta = result & 1 ? ~(result >> 1) : result >> 1;
      if (axis === 0) lat += delta;
      else lng += delta;
    }
    points.push([lat / 1e5, lng / 1e5]);
  }
  return points;
}

// Distance in meters from point p to segment ab, all [lat, lng], using a local flat projection.
export function distToSegmentM(p, a, b) {
  const kx = mPerDegLon(p[0]);
  const ax = (a[1] - p[1]) * kx,
    ay = (a[0] - p[0]) * M_PER_DEG_LAT;
  const bx = (b[1] - p[1]) * kx,
    by = (b[0] - p[0]) * M_PER_DEG_LAT;
  const dx = bx - ax,
    dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2)) : 0;
  return Math.hypot(ax + t * dx, ay + t * dy);
}

// ~550 m cells; big enough that any radius above fits in a 3x3 neighborhood.
const CELL_DEG = 0.005;
const cellKey = (lat, lng) => `${Math.floor(lat / CELL_DEG)},${Math.floor(lng / CELL_DEG)}`;

export function buildIndex(features) {
  const grid = new Map();
  for (const f of features) {
    const [lng, lat] = f.geometry.coordinates;
    const item = { lat, lng, props: f.properties };
    const key = cellKey(lat, lng);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(item);
  }
  return { grid, size: features.length };
}

export async function loadIndex(file) {
  return buildIndex(JSON.parse(await readFile(file, "utf8")).features);
}

// Bounding box [south, west, north, east] around one or more paths, padded by padM meters.
export function boundsOf(paths, padM) {
  let s = 90,
    w = 180,
    n = -90,
    e = -180;
  for (const [lat, lng] of paths.flat()) {
    s = Math.min(s, lat);
    n = Math.max(n, lat);
    w = Math.min(w, lng);
    e = Math.max(e, lng);
  }
  const dLat = padM / M_PER_DEG_LAT,
    dLng = padM / mPerDegLon((s + n) / 2);
  return [s - dLat, w - dLng, n + dLat, e + dLng];
}

// Every hazard within radiusM of the polyline, each counted once.
export function hazardsNear(index, path, radiusM) {
  const found = new Set();
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i],
      b = path[i + 1];
    const cells = new Set();
    const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / CELL_DEG));
    for (let s = 0; s <= steps; s++) {
      const lat = a[0] + ((b[0] - a[0]) * s) / steps;
      const lng = a[1] + ((b[1] - a[1]) * s) / steps;
      const r = Math.floor(lat / CELL_DEG),
        c = Math.floor(lng / CELL_DEG);
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) cells.add(`${r + dr},${c + dc}`);
    }
    for (const key of cells) {
      for (const h of index.grid.get(key) ?? []) {
        if (!found.has(h) && distToSegmentM([h.lat, h.lng], a, b) <= radiusM) found.add(h);
      }
    }
  }
  return [...found];
}

// Ray-casting test against a GeoJSON Polygon or MultiPolygon; p is [lat, lng]. Holes are respected.
export function pointInPolygon([lat, lng], geometry) {
  const polygons = geometry.type === "MultiPolygon" ? geometry.coordinates : [geometry.coordinates];
  return polygons.some((rings) => {
    let inside = false;
    for (const ring of rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i],
          [xj, yj] = ring[j];
        if (yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
      }
    }
    return inside;
  });
}

// Points every stepM meters along a path of [lat, lng], so length-in-area can be estimated by counting.
export function densify(path, stepM) {
  const out = [];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i],
      b = path[i + 1];
    const meters = Math.hypot((b[1] - a[1]) * mPerDegLon(a[0]), (b[0] - a[0]) * M_PER_DEG_LAT);
    const steps = Math.max(1, Math.ceil(meters / stepM));
    for (let s = 0; s < steps; s++) out.push([a[0] + ((b[0] - a[0]) * s) / steps, a[1] + ((b[1] - a[1]) * s) / steps]);
  }
  if (path.length) out.push(path.at(-1));
  return out;
}

// Great-circle distance in meters between [lat, lng] points.
export function distanceM(a, b) {
  const rad = Math.PI / 180;
  const dLat = (b[0] - a[0]) * rad,
    dLng = (b[1] - a[1]) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * rad) * Math.cos(b[0] * rad) * Math.sin(dLng / 2) ** 2;
  return 2 * 6_371_000 * Math.asin(Math.sqrt(h));
}

// [south, west, north, east] of a GeoJSON Polygon or MultiPolygon, for cheap rejection before pointInPolygon.
export function polygonBounds(geometry) {
  let s = 90,
    w = 180,
    n = -90,
    e = -180;
  const rings = geometry.type === "MultiPolygon" ? geometry.coordinates.flat() : geometry.coordinates;
  for (const [lng, lat] of rings.flat()) {
    s = Math.min(s, lat);
    n = Math.max(n, lat);
    w = Math.min(w, lng);
    e = Math.max(e, lng);
  }
  return [s, w, n, e];
}

// Share of points inside any of the areas, and meters inside each area's label, given pointSpacingM between points.
export function exposure(points, areas, pointSpacingM) {
  const prepared = areas.map((a) => ({ ...a, bounds: polygonBounds(a.geometry) }));
  const metersByLabel = {};
  let inside = 0;
  for (const p of points) {
    const hit = prepared.find(
      (a) =>
        p[0] >= a.bounds[0] &&
        p[0] <= a.bounds[2] &&
        p[1] >= a.bounds[1] &&
        p[1] <= a.bounds[3] &&
        pointInPolygon(p, a.geometry),
    );
    if (!hit) continue;
    inside++;
    metersByLabel[hit.label] = (metersByLabel[hit.label] ?? 0) + pointSpacingM;
  }
  return { share: points.length ? inside / points.length : 0, metersByLabel };
}
