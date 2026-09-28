"""
Our own hazard-aware driving router for Wynwood -> Edgewater, Miami (replaces Google routing in the pickup sim).

Graph: every drivable OpenStreetMap way in BBOX (Overpass, ODbL), one directed edge per consecutive node pair,
respecting oneway, lanes and maxspeed (DEFAULT_MPH by highway class when maxspeed is missing).

Hazards: SafeRoute Miami's own layers (saferoute/data/*.geojson, the files its server scores routes with), with
the server's radii and weights (saferoute/server.mjs):
    crash      within 40 m: 1 + 3 x serious injuries + 10 x killed
    311 flood  within 60 m: 5 x flood multiplier (x4 inside an NWS flood-alert polygon, x2 on a king tide, else 1)
    work zone  within 30 m: 15 per distinct FDOT project
Each hazard is charged to the one road segment nearest to it (if that segment is within the radius), so a route
pays for it once, the way SafeRoute counts hazards along a whole route.

Edge cost = travel time (s) + RISK_S_PER_POINT x risk. RISK_S_PER_POINT = 10 means the router will spend up to
10 s of extra driving to avoid one point of SafeRoute risk: ~10 s to avoid a crash record, ~100 s to avoid a
311 flood report on a king tide. The fastest route is the same search with the weight set to 0.

    python -m simulator.road_network --refresh    # re-download the OSM road graph
"""

import argparse
import heapq
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from simulator.pickup_choice import CRASH_R, FLOOD_R, WORK_R, flood_multiplier
from simulator.real_block import DATA_DIR, overpass_json, to_xy

ROOT = Path(__file__).resolve().parents[1]
DRIVE_CACHE = DATA_DIR / "drive_osm.json"
SAFEROUTE_DATA = ROOT / "saferoute" / "data"

SOUTH, WEST, NORTH, EAST = 25.790, -80.203, 25.806, -80.185
DEFAULT_MPH = {"motorway": 55, "trunk": 45, "primary": 35, "secondary": 30, "tertiary": 30, "unclassified": 25,
               "residential": 25, "living_street": 15, "motorway_link": 35, "trunk_link": 30, "primary_link": 30,
               "secondary_link": 25, "tertiary_link": 25}
DEFAULT_LANES = {"motorway": 3, "trunk": 2, "primary": 4, "secondary": 2, "tertiary": 2}
MPH = 0.44704
LANE_W = 3.3
CRASH_W, SERIOUS_W, KILLED_W, FLOOD_W, WORK_W = 1, 3, 10, 5, 15
RISK_S_PER_POINT = 10.0


def fetch():
    hw = "|".join(DEFAULT_MPH)
    b = f"({SOUTH},{WEST},{NORTH},{EAST})"
    query = f'[out:json][timeout:90];(way["highway"~"^({hw})$"]{b};nwr["name"="Wynwood Walls"]{b};);out body;>;out skel qt;'
    osm = overpass_json(query)
    DRIVE_CACHE.write_text(json.dumps(osm))
    return osm


def load_hazards(conditions, pad=0.002):
    """SafeRoute's crash, flood and work-zone points inside the bbox (plus a margin), in local metres."""
    out = []
    for name in ("crashes", "flooding", "construction"):
        for f in json.loads((SAFEROUTE_DATA / f"{name}.geojson").read_text())["features"]:
            lon, lat = f["geometry"]["coordinates"][:2]
            if not (SOUTH - pad < lat < NORTH + pad and WEST - pad < lon < EAST + pad):
                continue
            p, xy = f["properties"], to_xy(lat, lon)
            if name == "crashes":
                serious, killed = int(p.get("NUMBER_OF_SERIOUS_INJURIES") or 0), int(p.get("NUMBER_OF_KILLED") or 0)
                out.append({"kind": "crash", "xy": xy, "radius": CRASH_R,
                            "risk": CRASH_W + SERIOUS_W * serious + KILLED_W * killed,
                            "detail": f"{p.get('CALENDAR_YEAR')} crash on {p.get('ON_ROADWAY_NAME')}"})
            elif name == "flooding":
                mult = flood_multiplier({"conditions": conditions}, xy)
                out.append({"kind": "flood", "xy": xy, "radius": FLOOD_R, "risk": FLOOD_W * mult, "mult": mult,
                            "detail": f"311 {p.get('issue_type', '').lower()} at {p.get('street_address')}"})
            else:
                out.append({"kind": "work", "xy": xy, "radius": WORK_R, "risk": WORK_W,
                            "project": p.get("description"), "detail": f"FDOT work zone: {p.get('description')}"})
    return out


def _speed_mps(tags):
    m = (tags.get("maxspeed") or "").split()
    if m and m[0].isdigit():
        return int(m[0]) * (MPH if "mph" in tags["maxspeed"] or len(m) == 1 else 1 / 3.6)
    return DEFAULT_MPH[tags["highway"]] * MPH


def _seg_dist(p, a, b):
    """Distance from points p (N, 2) to segments a->b (M, 2) -> (N, M), and the projection parameter."""
    ab = b - a
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - proj, axis=-1), t


class RoadNetwork:
    def __init__(self, conditions, refresh=False):
        osm = fetch() if refresh or not DRIVE_CACHE.exists() else json.loads(DRIVE_CACHE.read_text())
        nodes = {e["id"]: to_xy(e["lat"], e["lon"]) for e in osm["elements"] if e["type"] == "node"}
        self.xy = nodes
        self.edges = []    # directed: u, v, name, lanes, oneway, length, time, seg (undirected segment index)
        self.out = {}
        self.segments = []
        self.landmarks = {}
        for e in osm["elements"]:
            tags = e.get("tags", {})
            # Two OSM ways carry the name; the attraction is the walled lot on NW 2nd Ave.
            if tags.get("name") == "Wynwood Walls" and tags.get("tourism") == "attraction":
                pts = [nodes[n] for n in e.get("nodes", []) if n in nodes] or [to_xy(e["lat"], e["lon"])]
                self.landmarks["Wynwood Walls"] = tuple(np.mean(pts, axis=0))
            if e["type"] != "way" or tags.get("highway") not in DEFAULT_MPH or tags.get("area") == "yes":
                continue
            ow = tags.get("oneway", "yes" if tags["highway"].startswith("motorway") or
                          tags.get("junction") == "roundabout" else "no")
            lanes = int(str(tags.get("lanes", DEFAULT_LANES.get(tags["highway"], 2))).split(";")[0])
            speed = _speed_mps(tags)
            name = tags.get("name", tags.get("ref", tags["highway"]))
            ids = [n for n in e["nodes"] if n in nodes]
            for a, b in pairwise(ids):
                seg = len(self.segments)
                length = math.dist(nodes[a], nodes[b])
                self.segments.append({"a": a, "b": b, "name": name, "lanes": lanes, "hazards": []})
                dirs = [(a, b)] if ow in ("yes", "true", "1") else [(b, a)] if ow == "-1" else [(a, b), (b, a)]
                for u, v in dirs:
                    self.out.setdefault(u, []).append(len(self.edges))
                    self.edges.append({"u": u, "v": v, "name": name, "lanes": lanes, "oneway": len(dirs) == 1,
                                       "length": length, "time": length / speed, "seg": seg})
        self.hazards = load_hazards(conditions)
        a = np.array([self.xy[s["a"]] for s in self.segments])
        b = np.array([self.xy[s["b"]] for s in self.segments])
        d, _ = _seg_dist(np.array([h["xy"] for h in self.hazards]), a, b)
        for i, h in enumerate(self.hazards):
            j = int(np.argmin(d[i]))
            if d[i, j] <= h["radius"]:
                self.segments[j]["hazards"].append(i)
        for s in self.segments:
            s["risk"] = sum(self.hazards[i]["risk"] for i in s["hazards"])

    def nearest_node(self, xy):
        ids = list(self.out)
        pts = np.array([self.xy[n] for n in ids])
        return ids[int(np.argmin(np.linalg.norm(pts - np.asarray(xy), axis=1)))]

    def curb_edge(self, curb):
        """Directed edge on the curb's road, nearest the curb, that has the curb on the car's right."""
        best = None
        for i, e in enumerate(self.edges):
            if e["name"] != curb["road"]:
                continue
            (x0, y0), (x1, y1) = self.xy[e["u"]], self.xy[e["v"]]
            if (y1 - y0) * curb["side_normal"][0] - (x1 - x0) * curb["side_normal"][1] <= 0:
                continue  # right-hand normal (dy, -dx) must point at the curb
            d, t = _seg_dist(np.array([curb["xy"]]), np.array([[x0, y0]]), np.array([[x1, y1]]))
            if best is None or d[0, 0] < best[0]:
                best = (d[0, 0], i, float(t[0, 0]))
        assert best, f"no {curb['road']} edge runs with the curb on its right"
        return best[1], best[2]

    def route(self, start, goal_edge, weight):
        """Least-cost edge list from node `start` to the start of `goal_edge` plus that edge; weight in s/point."""
        goal_u = self.edges[goal_edge]["u"]
        dist, prev, seen = {start: 0.0}, {}, set()
        heap = [(0.0, start)]
        while heap:
            c, u = heapq.heappop(heap)
            if u in seen:
                continue
            seen.add(u)
            if u == goal_u:
                break
            for ei in self.out.get(u, []):
                e = self.edges[ei]
                nc = c + e["time"] + weight * self.segments[e["seg"]]["risk"]
                if nc < dist.get(e["v"], math.inf):
                    dist[e["v"]], prev[e["v"]] = nc, ei
                    heapq.heappush(heap, (nc, e["v"]))
        if goal_u not in seen:
            raise ValueError("goal not reachable in the OSM road graph; widen the bbox")
        path, n = [goal_edge], goal_u
        while n != start:
            path.append(prev[n])
            n = self.edges[prev[n]]["u"]
        return path[::-1]

    def summary(self, edge_ids, end_t=1.0):
        """Polyline, time, length and SafeRoute-style risk for an edge list whose last edge is driven to end_t.

        lanes and speed are per polyline vertex (vertex i > 0 ends edge i - 1)."""
        pts = [self.xy[self.edges[edge_ids[0]]["u"]]]
        time = length = 0.0
        for k, ei in enumerate(edge_ids):
            e = self.edges[ei]
            f = end_t if k == len(edge_ids) - 1 else 1.0
            (x0, y0), (x1, y1) = self.xy[e["u"]], self.xy[e["v"]]
            pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
            time += f * e["time"]
            length += f * e["length"]
        hz = sorted({h for ei in edge_ids for h in self.segments[self.edges[ei]["seg"]]["hazards"]})
        items = [self.hazards[i] for i in hz]
        crash = sum(h["risk"] for h in items if h["kind"] == "crash")
        work = WORK_W * len({h["project"] for h in items if h["kind"] == "work"})
        flood = sum(h["risk"] for h in items if h["kind"] == "flood")
        names = []
        for ei in edge_ids:
            n = self.edges[ei]["name"]
            if not names or names[-1] != n:
                names.append(n)
        lanes = [self.edges[ei]["lanes"] for ei in edge_ids]
        speed = [self.edges[ei]["length"] / self.edges[ei]["time"] for ei in edge_ids]
        return {"edges": edge_ids, "pts": pts, "time_s": round(time), "length_m": round(length),
                "lanes": lanes[:1] + lanes, "speed": speed[:1] + speed,
                "risk": {"crash": crash + work, "flood": flood, "total": crash + work + flood},
                "hazards": hz, "via": names}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    pickup = json.loads((DATA_DIR / "pickup.json").read_text())
    net = RoadNetwork(pickup["conditions"], refresh=args.refresh)
    print(f"{len(net.xy)} nodes, {len(net.edges)} directed edges, {len(net.hazards)} hazards, "
          f"{sum(bool(s['hazards']) for s in net.segments)} segments with hazards, landmarks {net.landmarks}")
    assert net.route(net.nearest_node(net.landmarks["Wynwood Walls"]), 0, 0), "graph search failed"
