"""
Choose where the Waymo should pick up a blind rider, from real data only.

Every curb in the Edgewater block (both sides of each OSM street, sampled every 10 m) is scored
with SafeRoute Miami's own hazard radii and weights, applied to the curb instead of a drive:
  crash risk = crashes + 3 x serious/fatal + 15 x FDOT work-zone points   (40 m / 30 m radii)
  flood risk = flood multiplier x 5 x 311 flood reports                    (60 m radius)
The flood multiplier is SafeRoute's: 4 inside an active flood alert, 2 on a king tide, else 1; it is
read live from the running SafeRoute server when available. The rider's walking distance over the
real sidewalk network (curbs across a street with no mapped crosswalk are unreachable) and the
curb's lane count (boarding beside a wide arterial is harder for a blind rider) break ties.

    python -m simulator.pickup_choice          # writes simulator/data/pickup.json
"""

import json
import math
import urllib.request

import numpy as np

from simulator.real_block import DATA_DIR, load, to_latlon, to_xy
from simulator.waymo_scene import RealBlock

REQUESTED = {"label": "2100 NE 2nd Ave (rider's requested pin)", "address": "2100 NE 2nd Ave, Miami, FL 33137",
             "latlon": (25.79741, -80.19115)}
SAFEROUTE = "http://localhost:3000/api/routes"
CAR_START = "Wynwood Walls, Miami, FL"
PICKUP_JSON = DATA_DIR / "pickup.json"

CRASH_R, FLOOD_R, WORK_R = 40.0, 60.0, 30.0
LANE_W = 3.3
MAX_WALK_M = 260.0
WALK_RES = 0.4  # must be a multiple of the 0.2 m raster


def live_conditions(destination):
    """SafeRoute's live conditions (tide, alerts) plus its ranked routes, or None if the server is down."""
    req = urllib.request.Request(SAFEROUTE, data=json.dumps({"origin": CAR_START, "destination": destination}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except OSError as e:
        print(f"SafeRoute server unavailable ({e}); flood multiplier 1")
        return None


def _in_ring(lon, lat, ring):
    inside = False
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        if (y0 > lat) != (y1 > lat) and lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _in_geometry(lon, lat, geometry):
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return any(_in_ring(lon, lat, poly[0]) and not any(_in_ring(lon, lat, hole) for hole in poly[1:])
               for poly in polygons)


def flood_multiplier(conditions, xy):
    """SafeRoute's rule, applied to the curb: x4 inside an active flood-alert polygon, x2 on a king tide."""
    if not conditions:
        return 1
    c = conditions["conditions"]
    lat, lon = to_latlon(*xy)
    if c.get("stormMode") and any(_in_geometry(lon, lat, a["geometry"]) for a in c.get("alertAreas") or []):
        return 4
    return 2 if (c.get("highTide") or {}).get("kingTide") else 1


def curb_candidates(block):
    out = []
    for road in block["roads"]:
        if road["highway"] == "service":
            continue
        half = road["lanes"] * LANE_W / 2 + 0.3
        for (x0, y0), (x1, y1) in zip(road["pts"], road["pts"][1:]):
            seg = math.dist((x0, y0), (x1, y1))
            if seg < 1:
                continue
            nx, ny = -(y1 - y0) / seg, (x1 - x0) / seg
            for t in [i * 10.0 / seg for i in range(int(seg // 10) + 1)]:
                cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                for side in (1, -1):
                    out.append({"xy": (cx + side * half * nx, cy + side * half * ny), "road": road["name"],
                                "lanes": road["lanes"], "heading": math.atan2(y1 - y0, x1 - x0),
                                "side_normal": (side * nx, side * ny)})
    return out


def rider_start(world):
    """The rider waits on the OSM sidewalk in front of their building (the requested address)."""
    q = world.nearest_sidewalk(to_xy(*REQUESTED["latlon"]))
    ok = world.walkable(WALK_RES)
    ii, jj = np.nonzero(ok)
    cx, cy = world.x0 + (ii + 0.5) * WALK_RES, world.y0 + (jj + 0.5) * WALK_RES
    k = int(np.argmin(np.hypot(cx - q[0], cy - q[1])))
    return float(cx[k]), float(cy[k])


def walk_to(dist, world, xy):
    """Walking distance to the closest reachable sidewalk cell within 2 m of a curb point."""
    i, j = int((xy[0] - world.x0) / WALK_RES), int((xy[1] - world.y0) / WALK_RES)
    r = int(2.0 / WALK_RES)
    sub = dist[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
    return float(sub.min()) if sub.size else math.inf


def score(c, block, conditions, walk):
    mult = flood_multiplier(conditions, c["xy"])
    near = lambda items, r: [h for h in items if math.dist(h["xy"], c["xy"]) <= r]  # noqa: E731
    crashes = near(block["crashes"], CRASH_R)
    floods = near(block["flooding"], FLOOD_R)
    work = near(block["construction"], WORK_R)
    crash_risk = len(crashes) + 3 * sum(h["serious"] for h in crashes) + 15 * len(work)
    flood_risk = mult * 5 * len(floods)
    total = crash_risk + flood_risk + walk / 25 + 2 * max(0, c["lanes"] - 2)
    return {**c, "flood_multiplier": mult, "crash_risk": crash_risk, "flood_risk": flood_risk, "walk_m": round(walk), "total": round(total, 1),
            "crashes": len(crashes), "ped_crashes": sum(h["pedestrian"] for h in crashes), "floods": len(floods),
            "work_zones": len(work), "flood_tickets": [h["address"] for h in floods],
            "work_descriptions": sorted({h["description"] for h in work})}


def choose():
    block = load()
    bx, by = to_xy(*REQUESTED["latlon"])
    world = RealBlock(block, (bx - MAX_WALK_M, by - MAX_WALK_M, bx + MAX_WALK_M, by + MAX_WALK_M))
    start = rider_start(world)
    dist = world.walk_distances(start, WALK_RES)
    conditions = live_conditions(REQUESTED["address"])
    scored = []
    for c in curb_candidates(block):
        walk = walk_to(dist, world, c["xy"]) if world.inside(c["xy"]) else math.inf
        if walk <= MAX_WALK_M:
            scored.append(score(c, block, conditions, walk))
    # The requested pickup is the reachable curb closest to the rider's building.
    requested = min(scored, key=lambda c: c["walk_m"])
    best = min(scored, key=lambda c: c["total"])
    result = {
        "flood_multiplier": requested["flood_multiplier"],
        "conditions": conditions["conditions"] if conditions else None,
        "requested": {**requested, "label": REQUESTED["label"]},
        "chosen": {**best, "latlon": to_latlon(*best["xy"])},
        "rider_start": start,
        "candidates_scored": len(scored),
    }
    PICKUP_JSON.write_text(json.dumps(result, indent=2, default=list))
    return result


if __name__ == "__main__":
    r = choose()
    for k in ("requested", "chosen"):
        c = r[k]
        print(f"{k:9s} {c['road']:28s} xy=({c['xy'][0]:.0f},{c['xy'][1]:.0f}) total={c['total']} crash={c['crash_risk']} "
              f"flood={c['flood_risk']} walk={c['walk_m']} m floods={c['floods']} crashes={c['crashes']} "
              f"work={c['work_zones']}")
    print(f"flood multiplier {r['flood_multiplier']}, {r['candidates_scored']} curbs scored")
    assert r["chosen"]["total"] <= r["requested"]["total"]
