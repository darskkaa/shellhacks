"""
Waymo pickup reroute: a Waymo drives to pick up a blind rider in Edgewater, Miami, and moves the pickup
to a safer curb mid-trip.

What is real:
- Street map: OpenStreetMap roads (drawn at their lane count), sidewalks, crossings and buildings of the
  shared Edgewater block (simulator/real_block.py).
- Routes: our own router (simulator/road_network.py) over the OSM drive network from Wynwood Walls to the
  block (oneway, lanes, maxspeed), with SafeRoute Miami's own hazard layers and weights: FDOT crash records,
  Miami-Dade 311 flooding reports and FDOT work zones. Fastest = least travel time; safest = least
  travel time + RISK_S_PER_POINT seconds per point of risk. Every hazard marker is one charged to a drawn route.
- Pickup decision: simulator/data/pickup.json from simulator/pickup_choice.py (every curb within walking
  distance of the rider scored with SafeRoute's hazard radii and weights), including the king tide and NWS
  alert conditions it was scored under.
- Final approach: simulated in Waymax (simulator/waymax_pullin.py) on a roadgraph built from the same OSM data.

What is approximated: travel times are free-flow at the speed limit (no signals or traffic); the car moves along
the road centreline at a constant animation rate until the Waymax segment; the pickup check fires as the car
reaches the intersection nearest the halfway point of the trip.

    python -m simulator.waymo_pickup_sim
"""

import argparse
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator import waymax_pullin  # noqa: E402
from simulator.pickup_choice import FLOOD_R, PICKUP_JSON  # noqa: E402
from simulator.real_block import EAST, NORTH, SOUTH, WEST, load, to_xy  # noqa: E402
from simulator.road_network import RISK_S_PER_POINT, RoadNetwork  # noqa: E402

OUT_DIR = ROOT / "simulator" / "artifacts" / "waymo_pickup"
REROUTE_AT = 0.5   # fraction of the trip after which the pickup check runs

MAP_W, PANEL_W, H = 600, 360, 600
LANE_W = 3.3
DRIVE_FRAMES_PER_KM = 50
WAYMAX_STRIDE = 4  # one GIF frame per 0.4 s of Waymax time
COLORS = {
    "bg": (236, 233, 226), "block": (244, 242, 237), "building": (214, 207, 196), "building_edge": (190, 182, 170),
    "road": (255, 255, 255), "road_edge": (196, 191, 182), "sidewalk": (205, 200, 190), "crossing": (150, 150, 150),
    "safest": (26, 152, 80), "fastest": (242, 142, 43), "reroute": (33, 102, 172), "stale": (170, 170, 170),
    "waymax": (136, 65, 157), "crash": (215, 48, 39), "flood": (44, 123, 229), "work": (253, 141, 60),
    "text": (30, 30, 30), "muted": (105, 105, 105), "panel": (250, 250, 248), "alert": (190, 30, 30),
}


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


F_TITLE, F_BODY, F_SMALL, F_LABEL = font(16, True), font(13), font(11), font(11, True)


class Path2D:
    """Polyline in metres with arc-length lookup."""

    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for a, b in pairwise(pts):
            self.cum.append(self.cum[-1] + math.dist(a, b))

    @property
    def length(self):
        return self.cum[-1]

    def at(self, s):
        s = min(max(s, 0.0), self.length)
        for i in range(len(self.pts) - 1):
            if s <= self.cum[i + 1] or i == len(self.pts) - 2:
                seg = self.cum[i + 1] - self.cum[i] or 1.0
                t = (s - self.cum[i]) / seg
                (x0, y0), (x1, y1) = self.pts[i], self.pts[i + 1]
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0)), math.atan2(y1 - y0, x1 - x0)

    def upto(self, s):
        return [p for p, c in zip(self.pts, self.cum) if c < s] + [self.at(s)[0]]


def fit_view(pts, pad=90.0):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * pad
    return cx, cy, MAP_W / span


def lerp_view(a, b, t):
    t = t * t * (3 - 2 * t)
    # Interpolate zoom geometrically so the zoom rate looks steady.
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] * (b[2] / a[2]) ** t


def minutes(sec):
    return f"{sec // 60}:{sec % 60:02d}"


def short(name):
    return name.replace("Northeast", "NE").replace("Northwest", "NW").replace("North ", "N ")


def via(r, n=3):
    names = [short(v) for v in r["via"]]
    return ", ".join(names[:n]) + (" ..." if len(names) > n else "")


def route_rows(tag, r, color):
    return [(f"{tag}: {via(r)}", color, F_SMALL),
            (f"   {minutes(r['time_s'])} min, {r['length_m'] / 1000:.2f} km, risk {r['risk']['total']}"
             f" (crash {r['risk']['crash']}, flood {r['risk']['flood']})")]


class Renderer:
    def __init__(self, block, hazards):
        self.block = block
        self.hazards = hazards
        self.bbox = [to_xy(SOUTH, WEST), to_xy(NORTH, EAST)]

    def draw(self, view, layers, panel):
        cx, cy, k = view
        img = Image.new("RGB", (MAP_W + PANEL_W, H), COLORS["bg"])
        d = ImageDraw.Draw(img)

        def px(p):
            return MAP_W / 2 + (p[0] - cx) * k, H / 2 - (p[1] - cy) * k

        (x0, y0), (x1, y1) = self.bbox
        d.rectangle([px((x0, y1)), px((x1, y0))], fill=COLORS["block"], outline=COLORS["muted"])
        d.text((px((x0, y1))[0] + 4, px((x0, y1))[1] - 14), "OpenStreetMap block", fill=COLORS["muted"], font=F_SMALL)
        for b in self.block["buildings"]:
            if len(b["pts"]) > 2:
                d.polygon([px(p) for p in b["pts"]], fill=COLORS["building"], outline=COLORS["building_edge"])
        for s in self.block["sidewalks"]:
            d.line([px(p) for p in s["pts"]], fill=COLORS["sidewalk"], width=max(1, round(2 * k)))
        for r in self.block["roads"]:
            w = r["lanes"] * LANE_W * k
            d.line([px(p) for p in r["pts"]], fill=COLORS["road_edge"], width=max(2, round(w + 2)), joint="curve")
        for r in self.block["roads"]:
            w = r["lanes"] * LANE_W * k
            d.line([px(p) for p in r["pts"]], fill=COLORS["road"], width=max(1, round(w)), joint="curve")
        for c in self.block["crossings"]:
            d.line([px(p) for p in c["pts"]], fill=COLORS["crossing"], width=max(1, round(k)))

        for pts, color, width in layers["lines"]:
            if len(pts) > 1:
                d.line([px(p) for p in pts], fill=color, width=width, joint="curve")

        for (x, y), r, color in layers.get("rings", []):
            px0, py0 = px((x - r, y + r))
            px1, py1 = px((x + r, y - r))
            d.ellipse([px0, py0, px1, py1], outline=color, width=2)
        for xy, label, color, crossed in layers["pins"]:
            x, y = px(xy)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color, outline="white", width=2)
            if crossed:
                d.line([x - 5, y - 5, x + 5, y + 5], fill="white", width=2)
                d.line([x - 5, y + 5, x + 5, y - 5], fill="white", width=2)

        # Hazards go over the pins: the requested curb sits within metres of the 311 flood reports.
        for h in self.hazards:
            x, y = px(h["xy"])
            r = 6 if h["kind"] == "flood" or h["risk"] > 1 else 4
            if h["kind"] == "work":
                d.rectangle([x - r, y - r, x + r, y + r], fill=COLORS["work"], outline="white")
            else:
                d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[h["kind"]], outline="white")
        for xy, label, color, _ in layers["pins"]:
            x, y = px(xy)
            w = d.textlength(label, font=F_LABEL)
            tx = x + 10 if x + 10 + w < MAP_W - 4 else x - 10 - w
            d.text((tx, y - 20), label, fill=color, font=F_LABEL, stroke_width=2, stroke_fill="white")

        car_xy, heading = layers["car"]
        self.draw_car(d, px(car_xy), heading, k)
        if layers.get("caption"):
            d.text((10, H - 22), layers["caption"], fill=COLORS["waymax"], font=F_LABEL, stroke_width=2,
                   stroke_fill="white")
        self.draw_panel(d, panel)
        return img

    @staticmethod
    def draw_car(d, c, heading, k):
        # Real footprint where the zoom allows, never smaller than a legible icon.
        length, width = max(waymax_pullin.CAR_L * k, 18), max(waymax_pullin.CAR_W * k, 9)
        cos_h, sin_h = math.cos(heading), -math.sin(heading)  # screen y points down
        corners = [(sx * length / 2, sy * width / 2) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        pts = [(c[0] + u * cos_h - v * sin_h, c[1] + u * sin_h + v * cos_h) for u, v in corners]
        d.polygon(pts, fill="white", outline=(20, 20, 20), width=2)
        d.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], fill=(40, 40, 40))
        nose = (c[0] + cos_h * length / 2, c[1] + sin_h * length / 2)
        d.line([c, nose], fill=(40, 40, 40), width=2)
        d.text((c[0] + 12, c[1] + 8), "Waymo", fill=(20, 20, 20), font=F_LABEL, stroke_width=2, stroke_fill="white")

    @staticmethod
    def draw_panel(d, panel):
        x0 = MAP_W
        d.rectangle([x0, 0, x0 + PANEL_W, H], fill=COLORS["panel"])
        d.line([x0, 0, x0, H], fill=COLORS["muted"])
        y = 12
        d.text((x0 + 12, y), panel["title"], fill=panel.get("title_color", COLORS["text"]), font=F_TITLE)
        y += 26
        for line in panel["lines"]:
            text, color, f = (line, COLORS["text"], F_BODY) if isinstance(line, str) else line
            d.text((x0 + 12, y), text, fill=color, font=f)
            y += 16 if f is F_BODY else 14
        legend = [("flood", "311 flood report"), ("crash", "FDOT crash record"), ("work", "FDOT work zone")]
        y = H - 20 * len(legend) - 34
        d.text((x0 + 12, y), "Hazards charged to these routes", fill=COLORS["muted"], font=F_SMALL)
        y += 18
        for kind, label in legend:
            d.ellipse([x0 + 14, y + 2, x0 + 24, y + 12], fill=COLORS[kind])
            d.text((x0 + 32, y), label, fill=COLORS["text"], font=F_SMALL)
            y += 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    block = load()
    pickup = json.loads(PICKUP_JSON.read_text())
    req, chosen = pickup["requested"], pickup["chosen"]
    rider = tuple(pickup["rider_start"])
    side = "north" if chosen["side_normal"][1] > 0 else "south"
    cond = pickup["conditions"]

    net = RoadNetwork(cond)
    wynwood = net.landmarks["Wynwood Walls"]
    start = net.nearest_node(wynwood)
    req_edge, req_t = net.curb_edge(req)
    cho_edge, cho_t = net.curb_edge(chosen)
    fast1 = net.summary(net.route(start, req_edge, 0.0), req_t)
    safe1 = net.summary(net.route(start, req_edge, RISK_S_PER_POINT), req_t)

    # The check fires at the first intersection past the halfway point; the car reroutes from that node.
    drive1_len = 0.0
    for ei in safe1["edges"]:
        drive1_len += net.edges[ei]["length"]
        if drive1_len >= REROUTE_AT * safe1["length_m"]:
            break
    reroute_node = net.edges[ei]["v"]
    fast2 = net.summary(net.route(reroute_node, cho_edge, 0.0), cho_t)
    safe2 = net.summary(net.route(reroute_node, cho_edge, RISK_S_PER_POINT), cho_t)

    ref = waymax_pullin.reference(safe2["pts"], safe2["lanes"], safe2["speed"])
    rg = waymax_pullin.roadgraph(net.segments, net.xy, (ref["x"][-1], ref["y"][-1]))
    sim, m = waymax_pullin.simulate(ref, rg)
    stop_err = math.dist((sim["x"][-1], sim["y"][-1]), (ref["x"][-1], ref["y"][-1]))

    drive1 = Path2D(safe1["pts"])
    drive2 = Path2D(safe2["pts"])
    handover = drive2.length - waymax_pullin.PULLIN_M
    shown = {i for r in (fast1, safe1, fast2, safe2) for i in r["hazards"]}
    renderer = Renderer(block, [net.hazards[i] for i in sorted(shown)])

    view_wide = fit_view(fast1["pts"] + safe1["pts"] + [rider, wynwood])
    view_reroute = fit_view(fast2["pts"] + safe2["pts"] + [rider])
    view_pickup = fit_view([rider, tuple(chosen["xy"]), tuple(req["xy"]), (ref["x"][0], ref["y"][0])], pad=40)

    mult = pickup["flood_multiplier"]
    tide = cond.get("highTide") or {}
    king = " (king tide)" if tide.get("kingTide") else ""
    alert = cond["alerts"][0] if cond.get("alerts") else None
    cond_lines = [(f"High tide {tide.get('feet', 0):.2f} ft at {tide.get('time', '?')[-5:]}{king}", COLORS["muted"],
                   F_SMALL)]
    if alert:
        cond_lines.append((f"NWS: {alert['event']} ({alert['severity']})", COLORS["muted"], F_SMALL))
    cond_lines.append((f"Flood weight here: x{mult}", COLORS["muted"], F_SMALL))
    cost_line = (f"cost = drive time + {RISK_S_PER_POINT:.0f} s per risk point", COLORS["muted"], F_SMALL)

    frames, durations = [], []

    def emit(view, lines, pins, rings, car, panel, ms, caption=None):
        frames.append(renderer.draw(view, {"lines": lines, "pins": pins, "rings": rings, "car": car,
                                           "caption": caption}, panel))
        durations.append(ms)

    req_pin = (tuple(req["xy"]), "requested pickup", COLORS["alert"], False)
    start_pin = (wynwood, "Wynwood Walls", COLORS["muted"], False)
    trip1_lines = [(fast1["pts"], COLORS["fastest"], 3), (safe1["pts"], COLORS["safest"], 5)]
    panel1 = {"title": "1. Trip to the rider's pin",
              "lines": [("Wynwood Walls -> 2100 NE 2nd Ave", COLORS["muted"], F_BODY),
                        "Our router: OSM roads + SafeRoute hazards",
                        *route_rows("Fastest", fast1, COLORS["fastest"]),
                        *route_rows("Safest", safe1, COLORS["safest"]),
                        cost_line, "",
                        f"Driving the safest: +{safe1['time_s'] - fast1['time_s']} s,",
                        f"  risk {fast1['risk']['total']} -> {safe1['risk']['total']}", "", *cond_lines]}

    n1 = max(20, round(drive1_len / 1000 * DRIVE_FRAMES_PER_KM))
    for _ in range(12):
        emit(view_wide, trip1_lines, [start_pin, req_pin], [], drive1.at(0), panel1, 250)
    for i in range(n1 + 1):
        s = drive1_len * i / n1
        emit(view_wide, trip1_lines + [(drive1.upto(s), (20, 90, 50), 6)], [start_pin, req_pin], [], drive1.at(s),
             panel1, 90)

    check = {"title": "2. Pickup safety check: MOVE PICKUP", "title_color": COLORS["alert"],
             "lines": [("Requested curb: " + short(req["road"]), COLORS["alert"], F_BODY),
                       f"  {req['lanes']} lanes, {req['floods']} flood reports <= {FLOOD_R:.0f} m (311):",
                       *[(f"    {t}", COLORS["muted"], F_SMALL) for t in req["flood_tickets"]],
                       f"  {req['crashes']} crash, {req['work_zones']} work zones, risk {req['total']}",
                       (f"Chosen curb: {short(chosen['road'])} ({side})", COLORS["safest"], F_BODY),
                       (f"  {chosen['lanes']} lanes, {chosen['floods']} floods, {chosen['crashes']} crashes, "
                        f"{chosen['work_zones']} work zones"),
                       f"  risk {chosen['total']}, rider walks {chosen['walk_m']} m",
                       (f"  best of {pickup['candidates_scored']} curbs scored", COLORS["muted"], F_SMALL),
                       "", ("Reroute from the car's intersection:", COLORS["reroute"], F_BODY),
                       *route_rows("Fastest", fast2, COLORS["fastest"]),
                       *route_rows("Safest", safe2, COLORS["reroute"]), cost_line]}
    ring = [(tuple(req["xy"]), FLOOD_R, COLORS["flood"])]
    stale = [(p, COLORS["stale"], 3) for p, _, _ in trip1_lines] + [(drive1.upto(drive1_len), (20, 90, 50), 6)]
    new_lines = stale + [(fast2["pts"], COLORS["fastest"], 3), (drive2.pts, COLORS["reroute"], 5)]
    pins = [(tuple(req["xy"]), "requested pickup", COLORS["alert"], True),
            (tuple(chosen["xy"]), "new pickup", COLORS["safest"], False)]
    car_at_reroute = drive1.at(drive1_len)
    for i in range(24):
        view = lerp_view(view_wide, view_reroute, min(1.0, i / 16))
        emit(view, new_lines if i >= 6 else trip1_lines, pins if i >= 6 else [req_pin], ring, car_at_reroute,
             check, 1600 if i in (0, 23) else 120)

    n2 = max(15, round(handover / 1000 * DRIVE_FRAMES_PER_KM * 1.5))
    for i in range(n2 + 1):
        s = handover * i / n2
        view = lerp_view(view_reroute, view_pickup, max(0.0, 1 - (handover - s) / 250))
        emit(view, new_lines + [(drive2.upto(s), (15, 60, 120), 6)], pins, ring, drive2.at(s), check, 90)

    waymax_panel = {"title": "3. Final approach in Waymax", "title_color": COLORS["waymax"],
                    "lines": [(f"Last {waymax_pullin.PULLIN_M:.0f} m: Waymax bicycle-model rollout", COLORS["waymax"],
                               F_BODY),
                              ("  expert actor tracks our lane-level plan;", COLORS["muted"], F_SMALL),
                              ("  roadgraph from OSM lanes; no other agents", COLORS["muted"], F_SMALL),
                              ("  (no real traffic data for this street)", COLORS["muted"], F_SMALL),
                              "", "Waymax metrics (max over rollout):",
                              (f"  offroad {m['offroad']['max']:.0f}, overlap {m['overlap']['max']:.0f}, "
                               f"wrong-way {m['sdc_wrongway']['max']:.1f} m"),
                              f"  kinematic infeasibility {m['kinematic_infeasibility']['max']:.0f}",
                              f"  divergence from plan {m['log_divergence']['max']:.2f} m",
                              f"  route progression {m['sdc_progression']['final']:.2f}",
                              "", f"Stops {waymax_pullin.CURB_GAP_M} m off the {side} curb,",
                              "  curb on the rider's door side"]}
    trail = new_lines + [(drive2.upto(handover), (15, 60, 120), 6)]
    for i in list(range(0, len(sim["x"]), WAYMAX_STRIDE)) + [len(sim["x"]) - 1]:
        pose = ((float(sim["x"][i]), float(sim["y"][i])), float(sim["yaw"][i]))
        path = [(float(x), float(y)) for x, y in zip(sim["x"][: i + 1], sim["y"][: i + 1])]
        caption = f"final approach simulated in Waymax   t={i * waymax_pullin.DT:.1f} s  {sim['speed'][i]:.1f} m/s"
        emit(view_pickup, trail + [(path, COLORS["waymax"], 4)], pins, ring, pose, waymax_panel,
             WAYMAX_STRIDE * 100, caption)

    final = {"title": "4. Parked at the safer curb", "title_color": COLORS["safest"],
             "lines": [(f"{short(chosen['road'])}, {side} curb", COLORS["safest"], F_BODY),
                       f"Pickup risk {req['total']} -> {chosen['total']} (flood x{mult})",
                       f"Flood reports at curb {req['floods']} -> {chosen['floods']}",
                       f"Crashes at curb {req['crashes']} -> {chosen['crashes']}",
                       f"Lanes beside the door {req['lanes']} -> {chosen['lanes']}",
                       f"Rider walk {req['walk_m']} m -> {chosen['walk_m']} m (with escort)",
                       "",
                       f"Drive: {drive1_len / 1000:.2f} km toward the pin, then",
                       f"  {safe2['length_m'] / 1000:.2f} km reroute, {minutes(safe2['time_s'])} min,",
                       f"  route risk {safe2['risk']['total']} (fastest {fast2['risk']['total']})",
                       f"Waymax: offroad {m['offroad']['max']:.0f}, stop error {stop_err:.2f} m",
                       "",
                       ("Final approach made using the Waymax Licensed", COLORS["muted"], F_SMALL),
                       ("Materials, provided by Waymo LLC under the Waymax", COLORS["muted"], F_SMALL),
                       ("License Agreement for Non-Commercial Use.", COLORS["muted"], F_SMALL)]}
    final_pins = [(pins[0][0], "requested pickup (rider waits here)", *pins[0][2:]), pins[1]]
    last = ((float(sim["x"][-1]), float(sim["y"][-1])), float(sim["yaw"][-1]))
    done = trail + [([(float(x), float(y)) for x, y in zip(sim["x"], sim["y"])], COLORS["waymax"], 4)]
    for i in range(20):
        emit(view_pickup, done, final_pins, ring, last, final, 4000 if i == 19 else 150)

    gif = args.output / "waymo_pickup.gif"
    frames[0].save(args.output / "first_frame.png")
    frames[-1].save(args.output / "last_frame.png")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)

    def brief(r):
        return {k: r[k] for k in ("time_s", "length_m", "risk", "via")}

    req_seg = net.edges[req_edge]["seg"]
    uses_req_curb = req_seg in {net.edges[e]["seg"] for e in safe2["edges"]}
    report = {
        "router": {"graph": {"nodes": len(net.xy), "directed_edges": len(net.edges), "hazards": len(net.hazards)},
                   "risk_s_per_point": RISK_S_PER_POINT},
        "trip_to_requested": {"fastest": brief(fast1), "safest": brief(safe1), "driven": "safest"},
        "reroute": {"at_m": round(drive1_len), "from_xy": [round(c, 1) for c in net.xy[reroute_node]],
                    "fastest": brief(fast2), "safest": brief(safe2), "safest_uses_requested_curb_segment": uses_req_curb},
        "pickup": {"requested_risk": req["total"], "chosen_risk": chosen["total"], "flood_multiplier": mult,
                   "chosen_curb": f"{chosen['road']} ({side})", "chosen_xy": chosen["xy"], "walk_m": chosen["walk_m"]},
        "waymax": {"pullin_m": waymax_pullin.PULLIN_M, "steps": len(sim["x"]), "dt_s": waymax_pullin.DT,
                   "roadgraph_points": len(rg[0]), "metrics": m, "stop_error_m": round(stop_err, 3),
                   "final_speed_mps": round(float(sim["speed"][-1]), 3)},
        "hazard_markers": len(shown), "frames": len(frames),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"trip 1: fastest {fast1['time_s']} s risk {fast1['risk']['total']}, safest {safe1['time_s']} s risk "
          f"{safe1['risk']['total']}; reroute at {drive1_len:.0f} m: fastest {fast2['time_s']} s risk "
          f"{fast2['risk']['total']}, safest {safe2['time_s']} s risk {safe2['risk']['total']}; pickup risk "
          f"{req['total']} -> {chosen['total']}; waymax offroad {m['offroad']['max']:.0f} overlap "
          f"{m['overlap']['max']:.0f} divergence {m['log_divergence']['max']:.2f} m stop error {stop_err:.2f} m; "
          f"frames={len(frames)} -> {gif}")
    assert chosen["total"] <= req["total"], "chosen pickup is not safer than requested"
    assert safe2["risk"]["total"] < fast2["risk"]["total"] or not uses_req_curb, \
        "reroute neither avoids the flooded requested curb nor lowers hazard cost"
    assert m["offroad"]["max"] == 0 and stop_err < 1.0, "Waymax pull-in left the road or missed the curb"


if __name__ == "__main__":
    main()
