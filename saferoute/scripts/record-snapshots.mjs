// Records /api/routes answers for every example chip x Conditions scenario into data/snapshots/ for SNAPSHOT_MODE.
// Run against a live server: npm run record -- [serverUrl] [scenario,...]   (default http://localhost:3000, all)
// Talks only to that server; it holds the keys. Each new trip costs Google Routes quota there.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { routesKey } from "../snapshot.mjs";

const server = new URL(process.argv[2] ?? "http://localhost:3000");
const only = process.argv[3]?.split(",");
const DELAY_MS = 1500;
const dir = new URL("../data/snapshots/", import.meta.url);

const html = await readFile(new URL("../public/index.html", import.meta.url), "utf8");
const attr = (tag, name) => tag.match(new RegExp(`${name}="([^"]*)"`))?.[1];
const trips = [...html.matchAll(/<button\b[^>]*\bdata-o="[^>]*>/g)].map(([tag]) => ({
  origin: attr(tag, "data-o"),
  destination: attr(tag, "data-d"),
}));
const select = html.match(/<select id="scenario">([\s\S]*?)<\/select>/)?.[1] ?? "";
const scenarios = [...select.matchAll(/<option value="([^"]*)"/g)].map((m) => m[1]);
if (!trips.length || trips.some((t) => !t.origin || !t.destination) || !scenarios.includes(""))
  throw new Error("could not read example chips or the Conditions select from public/index.html");

const slug = (s) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
await mkdir(dir, { recursive: true });
let failed = 0;
for (const trip of trips) {
  for (const simulate of scenarios.filter((s) => !only || only.includes(s || "live"))) {
    const label = `${trip.origin} -> ${trip.destination} [${simulate || "live"}]`;
    try {
      const res = await fetch(new URL("/api/routes", server), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...trip, simulate: simulate || undefined }),
        signal: AbortSignal.timeout(90_000),
      });
      const body = await res.json().catch(() => null);
      if (!res.ok || !Array.isArray(body?.routes) || !body.routes.length)
        throw new Error(body?.error ?? `HTTP ${res.status}`);
      // Never re-record a recording: a server in SNAPSHOT_MODE would hand back an old snapshot as new.
      if (body.snapshot) throw new Error("server answered from a snapshot; run it with SNAPSHOT_MODE=off");
      const explanation = await fetch(new URL(`/api/explain?id=${encodeURIComponent(body.explainId)}`, server))
        .then((r) => (r.ok ? r.json() : null))
        .then((r) => r?.explanation ?? null)
        .catch(() => null);
      const key = routesKey({ ...trip, simulate });
      const snap = JSON.stringify({ key, recordedAt: new Date().toISOString(), explanation, body });
      const hash = createHash("sha1").update(key).digest("hex").slice(0, 8);
      const name = `${slug(trip.origin)}--${slug(trip.destination)}--${simulate || "live"}-${hash}.json`;
      await writeFile(new URL(name, dir), snap);
      const rec = body.routes.find((r) => r.recommended) ?? body.routes[0];
      console.log(
        `ok   ${label}: ${body.routes.length} routes, recommended via ${rec.description || "local roads"}, ${snap.length} bytes`,
      );
    } catch (err) {
      failed++;
      console.log(`FAIL ${label}: ${err.message}`);
    }
    await new Promise((r) => setTimeout(r, DELAY_MS));
  }
}
console.log(failed ? `${failed} failed; rerun with just those scenarios` : "all recorded");
process.exitCode = failed ? 1 : 0;
