// Self-check for geo.mjs: node check-geo.mjs
import assert from "node:assert/strict";
import { decodePolyline, densify, distanceM, distToSegmentM, exposure, hazardsNear, pointInPolygon } from "./geo.mjs";

// Example from Google's encoded polyline algorithm docs.
assert.deepEqual(decodePolyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@"), [
  [38.5, -120.2],
  [40.7, -120.95],
  [43.252, -126.453],
]);

// 0.001 deg lat is ~111 m; a point beside a segment's midpoint vs. past its end.
const a = [25.76, -80.2],
  b = [25.76, -80.19];
assert(Math.abs(distToSegmentM([25.761, -80.195], a, b) - 111.3) < 1);
assert(Math.abs(distToSegmentM([25.76, -80.18], a, b) - 1003) < 5);

// Hazards within the radius count once even when several segments pass them; far ones don't count.
const grid = new Map();
const near = { lat: 25.7602, lng: -80.195, props: {} };
const far = { lat: 25.77, lng: -80.195, props: {} };
for (const h of [near, far]) {
  const key = `${Math.floor(h.lat / 0.005)},${Math.floor(h.lng / 0.005)}`;
  grid.set(key, [...(grid.get(key) ?? []), h]);
}
const path = [a, [25.76, -80.195], b];
assert.deepEqual(hazardsNear({ grid }, path, 40), [near]);

// Square with a square hole: inside ring counts, hole and outside do not; MultiPolygon checks every part.
const square = (x, y, d) => [
  [x, y],
  [x + d, y],
  [x + d, y + d],
  [x, y + d],
  [x, y],
];
const donut = { type: "Polygon", coordinates: [square(-80.2, 25.7, 0.1), square(-80.17, 25.73, 0.04)] };
assert.equal(pointInPolygon([25.71, -80.19], donut), true);
assert.equal(pointInPolygon([25.75, -80.15], donut), false);
assert.equal(pointInPolygon([25.9, -80.19], donut), false);
assert.equal(
  pointInPolygon([25.95, -80.45], {
    type: "MultiPolygon",
    coordinates: [[square(-80.2, 25.7, 0.1)], [square(-80.5, 25.9, 0.1)]],
  }),
  true,
);

// ~1113 m segment densified every 100 m gives 12 steps + endpoint.
assert.equal(
  densify(
    [
      [25.76, -80.2],
      [25.77, -80.2],
    ],
    100,
  ).length,
  13,
);

// Miami to Orlando is ~330 km.
assert(Math.abs(distanceM([25.7617, -80.1918], [28.5384, -81.3789]) - 330_000) < 10_000);

// Path half inside the donut ring, half outside: exposure counts only the inside points, per label.
const line = densify(
  [
    [25.71, -80.15],
    [25.71, -80.25],
  ],
  100,
);
const exp = exposure(line, [{ label: "A", geometry: donut }], 100);
assert(exp.share > 0.4 && exp.share < 0.6, `share ${exp.share}`);
assert.equal(Object.keys(exp.metersByLabel).join(), "A");

console.log("geo checks passed");
