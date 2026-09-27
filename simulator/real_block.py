"""
Real-world data for the Waymo escort sims: one block of Edgewater / Arts & Entertainment District, Miami
(Biscayne Blvd to NE 2nd Ave, NE 19th St to NE 23rd St).

Two sources, both cached in simulator/data/ so the sims run offline and reproducibly:
- OpenStreetMap (Overpass API, ODbL): roads with lane counts, sidewalks, crosswalks, lowered kerbs
  (curb ramps), traffic signals, buildings and street furniture.
- SafeRoute Miami's hazard layers (saferoute/data/*.geojson): Miami-Dade 311 flooding reports,
  FDOT work-zone points and FDOT crash records.

Coordinates are projected to local metres: x east, y north, origin at ORIGIN.

    python -m simulator.real_block --refresh    # re-download OSM and re-cut the hazard layers
"""

import argparse
import json
import math
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "simulator" / "data"
OSM_CACHE = DATA_DIR / "edgewater_osm.json"
HAZARD_CACHE = DATA_DIR / "edgewater_hazards.json"
SAFEROUTE_DATA = ROOT / "saferoute" / "data"

SOUTH, WEST, NORTH, EAST = 25.7935, -80.1935, 25.8005, -80.1865
ORIGIN = (25.7970, -80.1900)  # lat, lon of local (0, 0)
# The main instance often answers 504 under load; the mirror serves the same data.
OVERPASS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter")

EARTH_R = 6371008.8


def ssl_context():
    """NixOS keeps its CA bundle outside the path Python's default context looks in."""
    bundle = "/etc/ssl/certs/ca-bundle.crt"
    return ssl.create_default_context(cafile=bundle) if os.path.exists(bundle) else ssl.create_default_context()


def to_xy(lat, lon):
    """Equirectangular projection; error is millimetres over a few hundred metres."""
    x = math.radians(lon - ORIGIN[1]) * EARTH_R * math.cos(math.radians(ORIGIN[0]))
    y = math.radians(lat - ORIGIN[0]) * EARTH_R
    return x, y


def to_latlon(x, y):
    lat = ORIGIN[0] + math.degrees(y / EARTH_R)
    lon = ORIGIN[1] + math.degrees(x / (EARTH_R * math.cos(math.radians(ORIGIN[0]))))
    return lat, lon


def _overpass_query():
    b = f"({SOUTH},{WEST},{NORTH},{EAST})"
    return f"""[out:json][timeout:90];
(
  way["highway"]{b};
  way["building"]{b};
  node["kerb"]{b};
  node["highway"~"crossing|traffic_signals|street_lamp|bus_stop"]{b};
  node["amenity"~"bench|waste_basket|bicycle_parking|parking_entrance"]{b};
  node["natural"="tree"]{b};
  node["emergency"="fire_hydrant"]{b};
  node["barrier"~"bollard|kerb"]{b};
  node["power"~"pole"]{b};
  node["man_made"="utility_pole"]{b};
);
out body;
>;
out skel qt;"""


def overpass_json(query):
    """Run an Overpass query, falling back to the mirror when the main instance is overloaded or unreachable."""
    body = urllib.parse.urlencode({"data": query}).encode()
    for i, url in enumerate(OVERPASS):
        # Overpass rejects requests without a descriptive User-Agent (HTTP 406).
        req = urllib.request.Request(url, data=body, headers={"User-Agent": "shellhack-guide-dog-sim/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120, context=ssl_context()) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)  # HTTPError has a status; network failures don't
            if (code is not None and code not in (429, 502, 503, 504)) or i == len(OVERPASS) - 1:
                raise
            print(f"{url} failed ({code or e}); trying the mirror")


def fetch_osm():
    return overpass_json(_overpass_query())


def cut_hazards():
    """Subset SafeRoute's layers to the block (plus a 150 m margin for route scoring)."""
    pad = 0.0015
    out = {}
    for name in ("flooding", "construction", "crashes"):
        feats = json.loads((SAFEROUTE_DATA / f"{name}.geojson").read_text())["features"]
        out[name] = [f for f in feats
                     if SOUTH - pad < f["geometry"]["coordinates"][1] < NORTH + pad
                     and WEST - pad < f["geometry"]["coordinates"][0] < EAST + pad]
    return out


def refresh():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    osm = fetch_osm()
    OSM_CACHE.write_text(json.dumps(osm))
    hazards = cut_hazards()
    HAZARD_CACHE.write_text(json.dumps(hazards))
    return osm, hazards


def load():
    """Parsed block: roads, sidewalks, crossings, kerbs, signals, buildings, furniture, hazards (all in metres)."""
    if not OSM_CACHE.exists() or not HAZARD_CACHE.exists():
        refresh()
    osm = json.loads(OSM_CACHE.read_text())
    hazards = json.loads(HAZARD_CACHE.read_text())
    nodes = {}
    for e in osm["elements"]:
        # Tagged nodes come first, then untagged skeleton copies of the same ids; keep the tagged one.
        if e["type"] == "node" and ("tags" in e or e["id"] not in nodes):
            nodes[e["id"]] = e
    block = {"roads": [], "sidewalks": [], "crossings": [], "buildings": [], "kerbs": [], "signals": [],
             "furniture": [], "flooding": [], "construction": [], "crashes": []}

    for e in osm["elements"]:
        if e["type"] != "way" or "tags" not in e:
            continue
        pts = [to_xy(nodes[n]["lat"], nodes[n]["lon"]) for n in e["nodes"] if n in nodes]
        tags = e["tags"]
        hw = tags.get("highway")
        if "building" in tags:
            levels = float(tags.get("building:levels", "3").split(";")[0] or 3)
            block["buildings"].append({"pts": pts, "height": float(tags.get("height", levels * 3.5))})
        elif hw == "footway" and tags.get("footway") == "crossing":
            block["crossings"].append({"pts": pts, "tags": tags})
        elif hw == "footway" or tags.get("footway") == "sidewalk":
            block["sidewalks"].append({"pts": pts, "tags": tags})
        elif hw in ("primary", "secondary", "tertiary", "residential", "unclassified", "service"):
            lanes = int(str(tags.get("lanes", "1" if hw == "service" else "2")).split(";")[0])
            block["roads"].append({"pts": pts, "name": tags.get("name", hw), "highway": hw, "lanes": lanes,
                                   "oneway": tags.get("oneway") == "yes", "maxspeed": tags.get("maxspeed")})

    for e in nodes.values():
        tags = e.get("tags")
        if not tags:
            continue
        xy = to_xy(e["lat"], e["lon"])
        if "kerb" in tags:
            block["kerbs"].append({"xy": xy, "kerb": tags["kerb"], "tactile": tags.get("tactile_paving")})
        if tags.get("highway") == "traffic_signals" or tags.get("crossing") == "traffic_signals":
            block["signals"].append({"xy": xy, "tags": tags})
        kind = (tags.get("amenity") or tags.get("natural") or tags.get("emergency") or tags.get("barrier")
                or tags.get("man_made") or tags.get("power")
                or (tags.get("highway") if tags.get("highway") in ("street_lamp", "bus_stop") else None))
        if kind in ("bench", "waste_basket", "bicycle_parking", "tree", "fire_hydrant", "bollard", "utility_pole",
                    "pole", "street_lamp", "bus_stop"):
            block["furniture"].append({"xy": xy, "kind": kind})

    for f in hazards["flooding"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        block["flooding"].append({"xy": to_xy(lat, lon), "address": p.get("street_address"),
                                  "issue": p.get("issue_type"), "ticket": p.get("ticket_id")})
    for f in hazards["construction"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        block["construction"].append({"xy": to_xy(lat, lon), "description": f["properties"].get("description")})
    for f in hazards["crashes"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        p = f["properties"]
        block["crashes"].append({"xy": to_xy(lat, lon), "road": p.get("ON_ROADWAY_NAME"), "year": p.get("CALENDAR_YEAR"),
                                 "pedestrian": p.get("PEDESTRIAN_RELATED_IND") == "Y",
                                 "serious": (p.get("NUMBER_OF_SERIOUS_INJURIES") or 0) + (p.get("NUMBER_OF_KILLED") or 0)})
    return block


def summary(block):
    return {k: len(v) for k, v in block.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        refresh()
    b = load()
    print(json.dumps(summary(b)))
    x, y = to_xy(*to_latlon(12.3, -45.6))
    assert abs(x - 12.3) < 1e-6 and abs(y + 45.6) < 1e-6, "projection round-trip failed"
