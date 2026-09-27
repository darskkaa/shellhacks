// Recorded-demo mode: /api/routes bodies saved by scripts/record-snapshots.mjs, served when live routing is off or down.
// Pure helpers shared by server.mjs, the recorder and check-snapshot.mjs.
import { readdir, readFile } from "node:fs/promises";

export const SNAPSHOT_MODES = ["off", "fallback", "always"];

export function parseSnapshotMode(value) {
  const mode = value || "off";
  if (!SNAPSHOT_MODES.includes(mode))
    throw new Error(`SNAPSHOT_MODE must be one of ${SNAPSHOT_MODES.join(", ")} (got "${value}")`);
  return mode;
}

// The /api/routes response-cache key: normalized trip, scenario ("" when live) and the snapped water level.
export function routesKey({ origin, destination, simulate, waterFt = 0 }) {
  const norm = (s) => s.trim().toLowerCase().replace(/\s+/g, " ");
  const scenario = simulate === "storm" || /^hurricane-[1-5]$/.test(simulate ?? "") ? simulate : "";
  return JSON.stringify([norm(origin), norm(destination), scenario, waterFt]);
}

// The snapshot to serve for this request, or null. "always" serves whenever the key has one; "fallback" only after
// the live request failed. The stored key is re-checked so a mis-indexed file can never answer another trip.
export function pickSnapshot({ mode, key, snapshots, liveFailed = false }) {
  const snap = snapshots.get(key);
  if (!snap || snap.key !== key) return null;
  return mode === "always" || (mode === "fallback" && liveFailed) ? snap : null;
}

// Every valid *.json in dir, by its stored key. A missing dir means no recordings yet.
export async function loadSnapshots(dir) {
  const snapshots = new Map();
  let names = [];
  try {
    names = (await readdir(dir)).filter((n) => n.endsWith(".json"));
  } catch (err) {
    if (err.code !== "ENOENT") throw err;
  }
  for (const name of names) {
    const snap = JSON.parse(await readFile(new URL(name, dir), "utf8"));
    if (typeof snap.key !== "string" || !Date.parse(snap.recordedAt) || !Array.isArray(snap.body?.routes)) {
      console.error(`snapshot ${name}: missing key, recordedAt or body.routes; skipped`);
      continue;
    }
    snapshots.set(snap.key, snap);
  }
  return snapshots;
}
