// Self-check for rank.mjs: node check-rank.mjs
import assert from "node:assert/strict";
import { rankRoutes, routeEnergyKwh } from "./rank.mjs";

// A ~1.6 km downtown trip at ~27 km/h lands in the plausible 0.2-0.4 kWh band.
const downtown = routeEnergyKwh({ distanceM: 1600, durationSec: 215, elevationProfile: null });
assert(downtown >= 0.2 && downtown <= 0.4, `downtown ${downtown}`);

// Same average speed, longer trip: more energy.
const at = (km) => routeEnergyKwh({ distanceM: km * 1000, durationSec: km * 120, elevationProfile: null });
assert(at(1) < at(2) && at(2) < at(5) && at(5) < at(20));

// A 20 m climb and descent costs more than the flat route (regen recovers only part of it).
const flat = [...Array(11)].map((_, i) => [25.76 + i * 0.001, -80.19, 2]);
const hill = flat.map(([lat, lng], i) => [lat, lng, 2 + 20 - Math.abs(i - 5) * 4]);
const trip = { distanceM: 1100, durationSec: 150 };
assert(routeEnergyKwh({ ...trip, elevationProfile: hill }) > routeEnergyKwh({ ...trip, elevationProfile: flat }));

const route = (name, total, durationSec, energyKwh) => ({ name, risk: { total }, durationSec, energyKwh });
const names = (res) => res.routes.map((r) => r.name).join();

// Clear safety gap: the safest route wins even though it is slowest; D is dominated by the safest route.
let res = rankRoutes([
  route("fast", 80, 600, 0.9),
  route("safe", 40, 700, 1.0),
  route("dom", 45, 720, 1.1),
  route("mid", 60, 650, 0.8),
]);
assert.equal(names(res), "safe,mid,fast,dom");
assert.deepEqual(
  res.routes.map((r) => [r.recommended, r.frontier, r.dominatedBy]),
  [
    [true, true, null],
    [false, true, null],
    [false, true, null],
    [false, false, "A"],
  ],
);
assert.deepEqual(
  res.routes.map((r) => r.badges),
  [["safest"], ["efficient"], ["fastest"], []],
);
assert.equal(res.ranking.riskTie, 3);

// Risks within the tie (max(3, 5% of 100) = 5): the faster route wins over the marginally safer one.
res = rankRoutes([route("safer", 100, 900, 1.5), route("quicker", 104, 700, 1.4)]);
assert.equal(names(res), "quicker,safer");
assert.equal(res.ranking.riskTie, 5);
assert.deepEqual(res.routes[0].badges, ["fastest", "efficient"]);

// Risk and time both tied: least energy wins.
res = rankRoutes([route("hungry", 50, 600, 1.2), route("frugal", 51, 620, 0.9)]);
assert.equal(names(res), "frugal,hungry");
assert.equal(res.routes.filter((r) => r.recommended).length, 1);

// Identical routes: neither dominates the other, both stay on the frontier.
res = rankRoutes([route("x", 10, 300, 0.5), route("y", 10, 300, 0.5)]);
assert(res.routes.every((r) => r.frontier && r.badges.length === 3));

console.log("rank checks passed");
