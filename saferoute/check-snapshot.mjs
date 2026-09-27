// Self-check for snapshot.mjs: node check-snapshot.mjs
import assert from "node:assert/strict";
import { parseSnapshotMode, pickSnapshot, routesKey } from "./snapshot.mjs";

// The key normalizes like the response cache: case, outer and inner whitespace; unknown scenarios count as live.
const trip = { origin: "Coconut Grove, Miami, FL", destination: "Wynwood Walls, Miami, FL" };
const live = routesKey(trip);
assert.equal(live, routesKey({ origin: "  coconut  grove, miami, fl ", destination: "WYNWOOD WALLS, Miami, FL" }));
assert.equal(live, routesKey({ ...trip, simulate: "bogus", waterFt: 0 }));
const storm = routesKey({ ...trip, simulate: "storm" });
assert.notEqual(storm, live);
assert.notEqual(routesKey({ ...trip, simulate: "hurricane-3" }), storm);
assert.notEqual(routesKey({ ...trip, waterFt: 2.5 }), live);
assert.notEqual(routesKey({ origin: trip.destination, destination: trip.origin }), live);

const snap = { key: storm, recordedAt: "2026-09-27T20:00:00.000Z", body: { routes: [{}] } };
const snapshots = new Map([[storm, snap]]);
const pick = (mode, key, liveFailed) => pickSnapshot({ mode, key, snapshots, liveFailed });

assert.equal(pick("off", storm, false), null);
assert.equal(pick("off", storm, true), null);
assert.equal(pick("always", storm, false), snap);
assert.equal(pick("fallback", storm, false), null);
assert.equal(pick("fallback", storm, true), snap);
// No cross-key serving: other scenario, water level, or trip goes live (or errors) in every mode.
for (const mode of ["off", "fallback", "always"])
  for (const key of [live, routesKey({ ...trip, simulate: "storm", waterFt: 1 })])
    assert.equal(pick(mode, key, true), null, `${mode} ${key}`);
// A file indexed under the wrong key is never served.
assert.equal(pickSnapshot({ mode: "always", key: live, snapshots: new Map([[live, snap]]) }), null);

assert.equal(parseSnapshotMode(undefined), "off");
assert.equal(parseSnapshotMode(""), "off");
assert.equal(parseSnapshotMode("always"), "always");
assert.throws(() => parseSnapshotMode("on"));
console.log("snapshot checks passed");
