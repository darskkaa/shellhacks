"""
Obstacle course to the Waymo: the guide dog leads a blind rider down the west sidewalk of Biscayne Blvd
(Edgewater, Miami) from NE 22nd St to NE 21st St, steering around what is on that sidewalk, and stops at
the door of a Waymo parked on the NE 21st St curb.

Real (simulator/real_block.py, cached OpenStreetMap + SafeRoute Miami layers):
- sidewalks, curbs, lanes, buildings, the bus stop at the OSM node on this sidewalk
- the FDOT work-zone points of the Biscayne Blvd traffic-signal project (cones drawn at each real point)
- the pickup curb, picked by simulator/pickup_choice.py's scoring (crashes, 311 floods, FDOT work zones,
  lanes, walk) over the real curbs within 60 m of the requested pin, with the cached live conditions
Illustrative (labelled on screen): the barricade shapes and their spill onto the sidewalk beside each
real work-zone point, and one e-scooter. Only the robot dog is simulated (PyBullet); it perceives with
the depth/segmentation stack in gemini_waymo_sim.py.

    .venv/bin/python -m simulator.escort_obstacles_video --offline
"""

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simulator.gemini_waymo_sim as gws  # noqa: E402
from simulator.gemini_waymo_sim import (  # noqa: E402
    HANDLER_LAG_M, HANDLER_RIGHT_M, LOOKAHEAD_M, MAX_TURN, STEP_M, DogMap, Trail, astar, center, dog_camera,
    detections, draw_local_map,
)
from simulator.pickup_choice import PICKUP_JSON, WALK_RES, curb_candidates, score, walk_to  # noqa: E402
from simulator.real_block import load, to_latlon  # noqa: E402
from simulator.video import write_video  # noqa: E402
from simulator.waymo_scene import (  # noqa: E402
    RealBlock, build_handle, build_person, build_robot_dog, build_scene, create_box, create_cylinder,
)

WINDOW = (30.0, 4.0, 120.0, 124.0)
# The dog's map module keys its grid off these; point it at this block.
gws.X0, gws.Y0 = WINDOW[0], WINDOW[1]
gws.NX, gws.NY = int((WINDOW[2] - WINDOW[0]) / gws.RES), int((WINDOW[3] - WINDOW[1]) / gws.RES)
# 0.6 m buffer (default 0.4) so the rider, walking on the dog's right, also clears what the dog passes.
gws.OBSTACLE_INFLATE = 6

SIDEWALK_X = 82.6            # centreline of the west Biscayne sidewalk (walkable x 79.2..85.8 in the raster)
START = (SIDEWALK_X, 113.0)  # in front of the building at the NE 22nd St corner
CORNER_Y = 19.8              # centreline of the north NE 21st St sidewalk (walkable y 17.4..21.8)
PIN_NEAR = (95.0, 15.0)      # requested pin: the Biscayne Blvd & NE 21st St intersection
PIN_RADIUS = 60.0
W, H = 1280, 720
MAIN_W, MAIN_H = 860, 560
FPS_MS = 50                  # one frame per dog step on straight sidewalk (x3 real pace)
NEAR_MS = 100                # one frame per step within 5 m of an obstacle (about real pace)
EVENT_MS = 1400
MIN_CLEAR = 0.25             # dog/handler centre to obstacle surface, m (body half-width ~0.18 m)

ORANGE, WHITE = [1.0, 0.45, 0.05, 1], [0.97, 0.97, 0.97, 1]


def font(size):
    return ImageFont.load_default(size=size)


class LazyFrame:
    """JPEG-compressed frame; keeps ~900 720p frames in memory for write_video."""

    def __init__(self, img):
        self.size = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=93)
        self.data = buf.getvalue()

    def convert(self, mode):
        return Image.open(io.BytesIO(self.data)).convert(mode)


# ---- obstacles --------------------------------------------------------------------------------------------

def cone(x, y, z):
    return [create_cylinder([x, y, z + 0.35], 0.14, 0.7, ORANGE),
            create_cylinder([x, y, z + 0.45], 0.1, 0.1, WHITE),
            create_box([x, y, z + 0.015], [0.2, 0.2, 0.015], [0.1, 0.1, 0.1, 1])]


def barricade(world, x0, x1, y):
    """Type III style barricade across the sidewalk from x0 to x1 at y, with cones at both ends."""
    z = world.ground_height((x0 + x1) / 2, y)
    ids = [create_box([x, y, z + 0.5], [0.04, 0.04, 0.5], WHITE) for x in (x0 + 0.1, x1 - 0.1)]
    n = max(2, int((x1 - x0) / 0.3))
    for k in range(n):
        xa = x0 + (x1 - x0) * (k + 0.5) / n
        for bz in (0.55, 0.85):
            ids.append(create_box([xa, y, z + bz], [(x1 - x0) / n / 2, 0.03, 0.09], ORANGE if k % 2 else WHITE))
    ids.append(create_cylinder([x0 + 0.3, y, z + 1.12], 0.07, 0.1, [1, 0.8, 0.1, 1]))  # warning lamp
    for x in (x0 - 0.35, x1 + 0.35):
        ids += cone(x, y, z)
    return ids, [(np.array([x, y]), 0.1) for x in np.arange(x0 - 0.35, x1 + 0.36, 0.15)]


def scooter(world, x, y, yaw):
    """Parked stand-up e-scooter on its kickstand."""
    z = world.ground_height(x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    ori = p.getQuaternionFromEuler([0, 0, yaw])
    green, black = [0.15, 0.75, 0.35, 1], [0.08, 0.08, 0.08, 1]
    at = lambda lx, lz: [x + lx * c, y + lx * s, z + lz]  # noqa: E731
    ids = [create_box(at(0, 0.12), [0.45, 0.08, 0.03], green, ori),
           create_box(at(0.45, 0.62), [0.025, 0.025, 0.5], black, ori),
           create_box(at(0.45, 1.1), [0.03, 0.28, 0.025], black, ori)]
    ids += [create_box(at(lx, 0.1), [0.1, 0.03, 0.1], black, ori) for lx in (-0.45, 0.5)]
    return ids, [(np.array([x + lx * c, y + lx * s]), 0.12) for lx in np.arange(-0.55, 0.61, 0.15)]


def build_obstacles(world, block):
    """Real FDOT points + OSM bus stop on this sidewalk, plus illustrative barricade shapes and a scooter."""
    obstacles, labels, road_cones = [], {}, []
    for wz in block["construction"]:
        wx, wy = wz["xy"]
        if not (WINDOW[0] < wx < WINDOW[2] and WINDOW[1] + 4 < wy < WINDOW[3] - 4):
            continue
        # Cones at the real point (in the Biscayne roadway, where the signal work is).
        ids = [b for dx, dy in ((-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)) for b in
               cone(wx + dx, wy + dy, world.ground_height(wx + dx, wy + dy))]
        labels.update({i: "traffic cone" for i in ids})
        road_cones.append({"xy": (wx, wy), "description": wz["description"]})
        if wy > 30:  # mid-block points: the work spills onto the curb half of the west sidewalk
            ids, foot = barricade(world, SIDEWALK_X - 0.6, 85.6, wy)
            labels.update({i: "work-zone barricade" for i in ids})
            obstacles.append({"name": "work-zone barricade", "xy": (SIDEWALK_X + 0.9, wy), "foot": foot,
                              "real": f"FDOT work zone: real point {wx - SIDEWALK_X - 0.9:.0f} m east "
                                      "(Biscayne Blvd signal project)",
                              "tag": "FDOT work zone (real location; barrier shape illustrative)",
                              "say": "Work zone barricade ahead. Going around it on the {side}."})
    for f in block["furniture"]:
        fx, fy = f["xy"]
        if f["kind"] == "bus_stop" and 79 < fx < 86 and 30 < fy < 112:
            obstacles.append({"name": "bus stop sign", "xy": (fx, fy), "foot": [(np.array([fx, fy]), 0.05)],
                              "real": f"Bus stop: OpenStreetMap node at {to_latlon(fx, fy)[0]:.5f}, "
                                      f"{to_latlon(fx, fy)[1]:.5f}",
                              "tag": "Bus stop (OSM, real)",
                              "say": "Bus stop ahead. Going around it on the {side}."})
    sx, sy = 80.7, 42.0
    ids, foot = scooter(world, sx, sy, math.radians(35))
    labels.update({i: "e-scooter" for i in ids})
    obstacles.append({"name": "e-scooter", "xy": (sx, sy), "foot": foot,
                      "real": "E-scooter is illustrative (not in any dataset)",
                      "tag": "E-scooter (illustrative)",
                      "say": "Scooter parked on the sidewalk. Going around it on the {side}."})
    for o in obstacles:
        o.update(announced=False, cleared=False, clearance=math.inf, handler_clearance=math.inf, side=None)
    obstacles.sort(key=lambda o: -o["xy"][1])
    return obstacles, labels, road_cones


def clearance(o, xy):
    return min(math.dist(c, xy) - r for c, r in o["foot"])


# ---- pickup -------------------------------------------------------------------------------------------------

def choose_pickup(block, world):
    """pickup_choice's scoring over the real curbs near the requested pin, walking from the rider's start."""
    cached = json.loads(PICKUP_JSON.read_text())
    conditions = {"conditions": cached["conditions"]} if cached.get("conditions") else None
    dist = world.walk_distances(START, WALK_RES)
    pin = min(block["signals"], key=lambda s: math.dist(s["xy"], PIN_NEAR))["xy"]
    near = [(c, walk_to(dist, world, c["xy"])) for c in curb_candidates(block)
            if world.inside(c["xy"], margin=3) and math.dist(c["xy"], pin) <= PIN_RADIUS]
    # Curbs across a street with no mapped crossing are unreachable (inf), as in pickup_choice.
    scored = [score(c, block, conditions, walk) for c, walk in near if math.isfinite(walk)]
    requested = min((c for c in scored if "Biscayne" in c["road"]), key=lambda c: math.dist(c["xy"], pin))
    chosen = min(scored, key=lambda c: c["total"])
    return {"requested": requested, "chosen": chosen, "scored": len(scored), "pin": pin,
            "flood_multiplier": chosen["flood_multiplier"]}


# ---- cameras and drawing ------------------------------------------------------------------------------------

def chase_view(world, focus, yaw, lift=0.0):
    """Behind, above and to the road side of the dog, so the sidewalk ahead and the buildings stay in view.
    lift (0..1) raises the camera and swings it further out while the route turns; the rider walks on the
    dog's right, so a lagging camera would otherwise look past the rider onto the dog."""
    c, s = math.cos(yaw), math.sin(yaw)
    gz = world.ground_height(*focus)
    back, side = 5.4 - 2.4 * lift, 3.4 + 2.6 * lift
    eye = [focus[0] - back * c - side * s, focus[1] - back * s + side * c, gz + 7.6 + 5.0 * lift]
    target = [focus[0] + 3.8 * c - 0.4 * s, focus[1] + 3.8 * s + 0.4 * c, gz]
    view = p.computeViewMatrix(eye, target, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(58, MAIN_W / MAIN_H, 0.1, 150)
    _, _, rgba, _, _ = p.getCameraImage(MAIN_W, MAIN_H, view, proj, renderer=p.ER_TINY_RENDERER)
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(MAIN_H, MAIN_W, 4)).convert("RGB")
    m = np.asarray(proj).reshape(4, 4, order="F") @ np.asarray(view).reshape(4, 4, order="F")

    def project(pt):
        v = m @ np.array([*pt, 1.0])
        if v[3] <= 0.1:
            return None
        return (v[0] / v[3] + 1) / 2 * MAIN_W, (1 - v[1] / v[3]) / 2 * MAIN_H
    return img, project


def tag(d, xy, text, col, f):
    x, y = xy
    tw = d.textlength(text, font=f)
    x = min(max(x - tw / 2, 6), MAIN_W - tw - 6)
    y = min(max(y, 40), MAIN_H - 240)
    d.rectangle([x - 5, y - 3, x + tw + 5, y + 21], fill=(12, 14, 18), outline=col, width=2)
    d.text((x, y), text, fill=col, font=f)


def route_map(world, route, trail, obstacles, road_cones, car_xy, dog):
    """North-up map of the route corridor (x 58..100, y 8..120)."""
    x0, x1, y0, y1 = 58.0, 100.0, 8.0, 120.0
    tex = world.texture.crop((int((x0 - world.x0) / 0.2), int((world.y1 - y1) / 0.2),
                              int((x1 - world.x0) / 0.2), int((world.y1 - y0) / 0.2)))
    k = 232 / tex.height
    img = tex.resize((int(tex.width * k), 232))
    d = ImageDraw.Draw(img)
    px = lambda q: ((q[0] - x0) / 0.2 * k, (y1 - q[1]) / 0.2 * k)  # noqa: E731
    d.line([px(q) for q in route], fill=(40, 190, 90), width=2)
    d.line([px(q) for q in trail.pts[1:]], fill=(40, 120, 255), width=3)
    for wz in road_cones:
        x, y = px(wz["xy"])
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(255, 120, 0))
    for o in obstacles:
        x, y = px(o["xy"])
        col = (120, 120, 120) if "illustrative" in o["tag"] else (255, 120, 0) if "FDOT" in o["tag"] else (40, 90, 230)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col, outline=(0, 0, 0))
    x, y = px(car_xy)
    d.rectangle([x - 6, y - 3, x + 6, y + 3], fill=(250, 250, 250), outline=(0, 160, 160))
    x, y = px(dog)
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    return img


def compose(chase, cam, dets, local_map, rmap, obstacles, action, speech, context, progress, hazard, street):
    out = Image.new("RGB", (W, H), (16, 20, 24))
    out.paste(chase, (0, 0))
    d = ImageDraw.Draw(out)
    f13, f15, f17, f20, f24 = font(13), font(15), font(17), font(20), font(24)
    d.rectangle([0, 0, MAIN_W, 30], fill=(10, 12, 15))
    d.text((10, 6), f"{street}, Edgewater, Miami  |  OpenStreetMap + FDOT + SafeRoute data",
           fill=(200, 225, 255), font=f17)
    # Dog's own map, bottom-left of the main view.
    lx, ly = 10, MAIN_H - local_map.height - 10
    d.rectangle([lx - 3, ly - 20, lx + local_map.width + 3, ly + local_map.height + 3], fill=(10, 12, 15))
    d.text((lx, ly - 18), "DOG'S MAP (what it has seen)", fill=(200, 210, 220), font=f13)
    out.paste(local_map, (lx, ly))

    cw, ch = W - MAIN_W, 315
    out.paste(cam.resize((cw, ch)), (MAIN_W, 0))
    k = cw / gws.CAM_W
    for label, (bx0, by0, bx1, by1), dist, _ in dets:
        col = (60, 230, 120) if label.startswith("Waymo") else (255, 150, 40) if label in (
            "traffic cone", "work-zone barricade") else (255, 80, 60)
        d.rectangle([MAIN_W + bx0 * k, by0 * k, MAIN_W + bx1 * k, by1 * k], outline=col, width=2)
        d.text((MAIN_W + bx0 * k + 3, max(by0 * k - 17, 30)), f"{label} {dist:.1f} m", fill=col, font=f15)
    d.rectangle([MAIN_W, 0, W, 26], fill=(10, 12, 15))
    d.text((MAIN_W + 8, 4), "ROBOT DOG CAMERA: segmentation + depth", fill=(255, 255, 255), font=f15)

    # Route map + obstacle checklist.
    y0 = ch
    d.rectangle([MAIN_W, y0, W, MAIN_H], fill=(20, 24, 30))
    out.paste(rmap, (MAIN_W + 6, y0 + 8 + 12))
    d.text((MAIN_W + 6, y0 + 3), "ROUTE (OSM)", fill=(200, 210, 220), font=f13)
    tx = MAIN_W + rmap.width + 16
    d.text((tx, y0 + 6), "ON THIS SIDEWALK", fill=(200, 210, 220), font=f15)
    for i, o in enumerate(obstacles):
        yy = y0 + 32 + i * 50
        state = "cleared" if o["cleared"] else "avoiding" if o["announced"] else "ahead"
        col = (60, 230, 120) if o["cleared"] else (255, 190, 60) if o["announced"] else (170, 180, 190)
        d.text((tx, yy), f"{o['name']}  [{state}]", fill=col, font=f15)
        src = "illustrative" if "illustrative" in o["tag"] and "FDOT" not in o["tag"] else \
            "real FDOT point, shape illustr." if "FDOT" in o["tag"] else "real (OSM)"
        extra = f", passed {o['clearance']:.1f} m" if o["cleared"] else ""
        d.text((tx, yy + 19), f"{src}{extra}", fill=(140, 150, 160), font=f13)

    d.rectangle([0, MAIN_H, W, H], fill=(44, 18, 14) if hazard else (22, 28, 36))
    col = (255, 190, 60) if action.startswith(("TURN", "AVOID")) else (60, 230, 120)
    d.text((16, MAIN_H + 10), f"PLANNER: [{action}]", fill=col, font=f20)
    d.text((16, MAIN_H + 42), f"Dog says: \"{speech}\"", fill=(255, 255, 255), font=f24)
    d.text((16, MAIN_H + 82), context, fill=(255, 205, 120), font=f17)
    d.text((16, MAIN_H + 112), progress, fill=(150, 170, 190), font=f15)
    d.text((16, MAIN_H + 134), "Simulated: the robot dog (PyBullet). Street, furniture, hazards and pickup "
           "scoring: real data. Items marked illustrative are not in any dataset.", fill=(110, 125, 140), font=f13)
    return out


def breakdown(c):
    """pickup_choice.score's terms, spelled out."""
    parts = [f"{c['work_zones']} FDOT work zone x15" if c["work_zones"] else "",
             f"crashes {c['crash_risk'] - 15 * c['work_zones']}" if c["crash_risk"] > 15 * c["work_zones"] else "",
             f"floods {c['flood_risk']}" if c["flood_risk"] else "",
             f"{c['lanes']} lanes +{2 * max(0, c['lanes'] - 2)}" if c["lanes"] > 2 else "",
             f"walk {c['walk_m']} m/25"]
    return " + ".join(q for q in parts if q)


def card(bg, lines):
    img = Image.eval(bg.filter(ImageFilter.GaussianBlur(5)), lambda v: v // 6)
    d = ImageDraw.Draw(img)
    y = 150
    for text, size, col in lines:
        f = font(size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, fill=col, font=f)
        y += size + 16
    return img


# ---- main ---------------------------------------------------------------------------------------------------

def line_of_sight(path, blocked, reach=30):
    """Farthest path cell (within reach) the dog can walk to in a straight line. Grid A* paths put
    their diagonal moves anywhere among equal-cost choices, often all at the end, so following the
    raw path keeps a dog that was pushed aside by an obstacle off the route centreline for good."""
    a = np.array(path[0], float)
    for k in range(min(reach, len(path) - 1), 1, -1):
        b = np.array(path[k], float)
        n = int(np.abs(b - a).max() * 2) + 1
        cells = np.rint(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int)
        if not blocked[cells[1:, 0], cells[1:, 1]].any():
            return path[k]
    return path[min(1, len(path) - 1)]


def densify(pts, step=0.4):
    out = []
    for a, b in zip(pts, pts[1:]):
        n = max(1, int(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in range(n)]
    return out + [pts[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="accepted for parity; this video never uses the network")
    ap.add_argument("--output", type=Path, default=ROOT / "simulator/artifacts/escort_obstacles")
    ap.add_argument("--max-steps", type=int, default=1600)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    block = load()
    p.connect(p.DIRECT)
    world = RealBlock(block, WINDOW)
    pick = choose_pickup(block, world)
    req, cho = pick["requested"], pick["chosen"]
    print(f"requested {req['road']} total {req['total']} -> chosen {cho['road']} {cho['xy']} total {cho['total']} "
          f"({pick['scored']} curbs scored)")
    assert cho["total"] < req["total"], "pickup scoring should move the car off the work-zone curb"

    labels, water_ids, (hx_door, hy_door), (car_x, car_y, car_yaw), n = build_scene(world, {"chosen": cho})
    obstacles, obs_labels, road_cones = build_obstacles(world, block)
    labels.update(obs_labels)
    assert len(obstacles) >= 3, "need at least three obstacles on the route"

    fwd = (math.cos(car_yaw), math.sin(car_yaw))
    route_dir = 1 if fwd[0] * (hx_door - SIDEWALK_X) + fwd[1] * (hy_door - CORNER_Y) > 0 else -1
    goal_xy = (hx_door + n[0] * 1.6 + route_dir * fwd[0] * HANDLER_LAG_M,
               hy_door + n[1] * 1.6 + route_dir * fwd[1] * HANDLER_LAG_M)
    route = densify([START, (SIDEWALK_X, CORNER_Y + 1.0), (SIDEWALK_X - 1.5, CORNER_Y), (goal_xy[0] + 3, CORNER_Y),
                     goal_xy])
    ok = world.walkable(0.2)
    for q in route[:-3]:
        assert ok[int((q[0] - world.x0) / 0.2), int((q[1] - world.y0) / 0.2)], f"route leaves the sidewalk at {q}"
    route_cum = np.concatenate([[0.0], np.cumsum([math.dist(a, b) for a, b in zip(route, route[1:])])])
    # Route arc length where the Biscayne leg ends and the NE 21st St leg begins (the corner vertex).
    corner_s = float(route_cum[int(np.argmin([math.dist(q, (SIDEWALK_X - 1.5, CORNER_Y)) for q in route]))])
    print(f"route {route_cum[-1]:.0f} m, goal {goal_xy[0]:.1f},{goal_xy[1]:.1f}, {len(obstacles)} obstacles")

    dog_map = DogMap()
    x, y = START
    yaw = -math.pi / 2
    trail = Trail(START, yaw)
    cam_focus, cam_yaw, cam_lift = np.array(START), yaw, 0.0
    actors, frames, durations, events = [], [], [], []
    speech = f"Harness on. Your Waymo is waiting on NE 21st Street, {route_cum[-1]:.0f} metres. Let's go."
    context = pickup_note = (f"Pickup moved off Biscayne Blvd (risk {req['total']}: {req['work_zones']} FDOT work zone(s) within 30 m, "
               f"{req['lanes']} lanes) to NE 21st St (risk {cho['total']})")
    arrived, max_drop, last_z, stuck, hazard_frames = False, 0.0, world.ground_height(x, y), 0, 0
    said_intersection = said_waymo = False
    step = 0

    for step in range(args.max_steps):
        cam, depth, seg, eye, target = dog_camera(world, x, y, yaw)
        dog_map.integrate(eye, target, depth, seg, water_ids, (x, y), yaw)
        patch = dog_map.patch((x, y))
        i0, _, j0, _ = patch
        blocked = dog_map.blocked(patch)
        si, sj = gws.cell(x, y)
        blocked[si - i0, sj - j0] = False
        k = int(np.argmin([math.dist(q, (x, y)) for q in route]))
        kw = min(int(np.searchsorted(route_cum, route_cum[k] + LOOKAHEAD_M)), len(route) - 1)
        # A waypoint inside an obstacle is unreachable; slide it along the route past the obstacle.
        while kw < len(route) - 1 and route_cum[kw] < route_cum[k] + 1.6 * LOOKAHEAD_M:
            ci, cj = gws.cell(*route[kw])
            if not (0 <= ci - i0 < blocked.shape[0] and 0 <= cj - j0 < blocked.shape[1]) or \
                    not blocked[ci - i0, cj - j0]:
                break
            kw += 1
        wp = goal_xy if math.dist((x, y), goal_xy) < LOOKAHEAD_M else route[kw]
        wi, wj = gws.cell(*wp)
        wi, wj = wi - i0, wj - j0
        path = astar(blocked, (si - i0, sj - j0), lambda c: math.hypot(c[0] - wi, c[1] - wj) <= 3,
                     lambda c: math.hypot(c[0] - wi, c[1] - wj))
        dets = detections(seg, depth, labels)
        seen = {label: dist for label, _, dist, _ in dets}

        new_event = None
        for o in obstacles:
            ahead = (o["xy"][0] - x) * math.cos(yaw) + (o["xy"][1] - y) * math.sin(yaw)
            if not o["announced"] and 0 < ahead < 3.6 and o["name"] in seen and path and len(path) > 5:
                # Which side the planned path passes on: the path cell nearest the obstacle, left or right of it.
                pts = [center(i0 + a, j0 + b) for a, b in path]
                q = min(pts, key=lambda c: math.dist(c, o["xy"]))
                side = "left" if math.cos(yaw) * (q[1] - o["xy"][1]) - math.sin(yaw) * (q[0] - o["xy"][0]) > 0 \
                    else "right"
                o.update(announced=True, side=side)
                new_event = ("OBSTACLE", f"{o['name']} ahead {seen[o['name']]:.1f} m; passing on the {side}",
                             o["say"].format(side=side), o["real"])
                hazard_frames = 14
                break
        if new_event is None and not said_intersection and y < 30:
            said_intersection = True
            wz = min(road_cones, key=lambda w: math.dist(w["xy"], (x, y)))
            new_event = ("INFO", "work zone in the intersection, off our path",
                         "Work zone out in the intersection. We stay on the sidewalk and turn right.",
                         f"FDOT work zone: real point in the Biscayne Blvd roadway, {math.dist(wz['xy'], (x, y)):.0f} m "
                         "away (cones mark the real point)")
        if new_event is None and not said_waymo and seen.get("Waymo", 99) < 9:
            said_waymo = True
            new_event = ("WAYMO", "Waymo in view at the curb", "I can see your Waymo at the curb just ahead.", pickup_note)

        if math.dist((x, y), goal_xy) < 0.35:
            arrived, action = True, "ARRIVED"
        elif path is None or len(path) < 2:
            action, stuck = "STOP", stuck + 1
            yaw += MAX_TURN
        else:
            stuck = 0
            ahead_cell = line_of_sight(path, blocked)
            tgt = center(i0 + ahead_cell[0], j0 + ahead_cell[1])
            want = math.atan2(tgt[1] - y, tgt[0] - x)
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += max(-MAX_TURN, min(MAX_TURN, err))
            avoiding = any(o["announced"] and not o["cleared"] for o in obstacles)
            action = ("AVOID " if avoiding and abs(err) >= 0.05 else "") + (
                "FORWARD" if abs(err) < 0.05 else "TURN_LEFT" if err > 0 else "TURN_RIGHT")
            if abs(err) < 0.9:
                near = tgt
                dd = math.dist(near, (x, y)) or 1.0
                nx_, ny_ = x + STEP_M * (near[0] - x) / dd, y + STEP_M * (near[1] - y) / dd
                here, there = dog_map.height[gws.cell(x, y)], dog_map.height[gws.cell(nx_, ny_)]
                if np.isnan(here) or np.isnan(there) or abs(there - here) <= 0.05:
                    x, y = nx_, ny_
                    trail.add((x, y))

        if arrived:
            hx, hy = trail.at(trail.cum[-1] - HANDLER_LAG_M)
            hyaw = trail.heading(trail.cum[-1] - HANDLER_LAG_M)
            side = "right" if math.sin(math.atan2(hy_door - hy, hx_door - hx) - hyaw) < 0 else "left"
            new_event = ("ARRIVED", f"arrived; door handle on the passenger's {side}",
                         f"We're at your Waymo. The door is on your {side}, handle at waist height.", pickup_note)

        for body in actors:
            p.removeBody(body)
        dz = world.ground_height(x, y)
        max_drop = max(max_drop, last_z - dz)
        last_z = dz
        s_h = trail.cum[-1] - HANDLER_LAG_M
        hx, hy = trail.at(s_h)
        hyaw = trail.heading(s_h)
        hx, hy = hx + HANDLER_RIGHT_M * math.sin(hyaw), hy - HANDLER_RIGHT_M * math.cos(hyaw)
        hz = world.ground_height(hx, hy)
        hand = [hx + 0.4 * math.cos(hyaw) - 0.28 * math.sin(hyaw), hy + 0.4 * math.sin(hyaw) + 0.28 * math.cos(hyaw),
                hz + 0.85]
        back = [x - 0.15 * math.cos(yaw), y - 0.15 * math.sin(yaw), dz + 0.34]
        actors[:] = build_robot_dog(x, y, dz, yaw) + build_person(hx, hy, hz, hyaw) + build_handle(back, hand)
        if step % 5 == 0:
            create_box([x, y, dz + 0.01], [0.05, 0.05, 0.003], [0.25, 0.55, 1.0, 1])

        near_any = False
        for o in obstacles:
            c = clearance(o, (x, y))
            o["clearance"] = min(o["clearance"], c)
            o["handler_clearance"] = min(o["handler_clearance"], clearance(o, (hx, hy)))
            near_any |= c < 5.0
            behind = (o["xy"][0] - x) * math.cos(yaw) + (o["xy"][1] - y) * math.sin(yaw) < -1.2
            if o["announced"] and not o["cleared"] and behind:
                o["cleared"] = True

        if new_event:
            kind, event, speech, ctx = new_event
            context = ctx
            events.append({"step": step, "kind": kind, "event": event, "speech": speech, "dog": [round(x, 2), round(y, 2)]})
            print(f"[{step:4d}] {kind:8s} {event} -> {speech}")

        # Camera follows the route heading (not the dog's weave), low-passed so it never jumps.
        k = int(np.argmin([math.dist(q, (x, y)) for q in route]))
        a, b = route[max(k - 3, 0)], route[min(k + 6, len(route) - 1)]
        ryaw = math.atan2(b[1] - a[1], b[0] - a[0])
        turn = (ryaw - cam_yaw + math.pi) % (2 * math.pi) - math.pi
        cam_yaw += (0.07 + 0.08 * cam_lift) * turn  # catch up faster in turns so the dog stays unoccluded
        cam_lift += 0.15 * (min(abs(turn) / 0.5, 1.0) - cam_lift)
        cam_focus = cam_focus + 0.25 * (np.array([x, y]) - cam_focus)
        chase, project = chase_view(world, cam_focus, cam_yaw, cam_lift)
        d = ImageDraw.Draw(chase)
        f15 = font(15)
        for o in obstacles:
            if not o["cleared"] and math.dist(o["xy"], (x, y)) < 18:
                pt = project((o["xy"][0], o["xy"][1], 2.3))
                if pt:
                    col = (200, 200, 200) if "illustrative" in o["tag"] and "FDOT" not in o["tag"] else (255, 170, 60)
                    tag(d, pt, o["tag"], col, f15)
        for wz in road_cones:
            if math.dist(wz["xy"], (x, y)) < 22 and (wz["xy"][1] - y) * math.sin(cam_yaw) > -2:
                pt = project((wz["xy"][0], wz["xy"][1], 1.3))
                if pt:
                    tag(d, pt, "FDOT work zone: real point", (255, 140, 40), f15)
        if math.dist((car_x, car_y), (x, y)) < 30:
            pt = project((car_x, car_y, 2.2))
            if pt:
                tag(d, pt, f"Waymo: {cho['road'].replace('Northeast', 'NE')} curb (scored)", (90, 230, 140), f15)

        hazard = hazard_frames > 0
        hazard_frames -= 1
        walked = trail.cum[-1] - 1.0
        progress = f"Walked {walked:.0f} m of {route_cum[-1]:.0f} m  |  to the Waymo {math.dist((x, y), goal_xy):.0f} m"
        frame = compose(chase, cam, dets, draw_local_map(dog_map, patch, blocked, path, (x, y), trail),
                        route_map(world, route, trail, obstacles, road_cones, (car_x, car_y), (x, y)), obstacles,
                        action, speech, context, progress, hazard,
                        "Biscayne Blvd west sidewalk" if route_cum[k] < corner_s else "NE 21st St north sidewalk")
        if step == 0:
            title = card(frame, [
                ("Obstacle course to the Waymo", 44, (255, 255, 255)),
                ("A robot guide dog leads a blind rider down Biscayne Blvd, Edgewater, Miami", 22, (200, 225, 255)),
                (f"{len(obstacles)} obstacles on {route_cum[-1]:.0f} m of real sidewalk, then the door of a waiting Waymo",
                 22, (200, 225, 255)),
                ("", 10, (0, 0, 0)),
                ("Real data: OpenStreetMap (sidewalks, curbs, buildings, bus stop)", 18, (255, 205, 120)),
                ("FDOT work-zone points (Biscayne Blvd signal project), FDOT crashes, Miami-Dade 311 floods (SafeRoute)",
                 18, (255, 205, 120)),
                ("Simulated: only the robot dog (PyBullet). Barrier shapes and the e-scooter are illustrative.",
                 18, (170, 180, 190)),
            ])
            frames.append(LazyFrame(title))
            durations.append(3500)
        frames.append(LazyFrame(frame))
        durations.append(EVENT_MS if new_event else NEAR_MS if near_any else FPS_MS)
        if arrived or stuck > 40:
            break

    durations[-1] = 2500
    last = frames[-1].convert("RGB")
    for o in obstacles:
        o["cleared"] = o["cleared"] or o["announced"]
    min_dog = min(o["clearance"] for o in obstacles)
    min_handler = min(o["handler_clearance"] for o in obstacles)
    avoided = sum(o["announced"] and o["cleared"] for o in obstacles)
    walked = trail.cum[-1] - 1.0
    frames.append(LazyFrame(card(last, [
        ("Arrived at the Waymo" if arrived else "Did not arrive", 44, (90, 230, 140) if arrived else (255, 90, 60)),
        (f"{walked:.0f} m walked  |  {avoided} obstacles avoided  |  closest pass {min_dog:.2f} m (dog), "
         f"{min_handler:.2f} m (rider)  |  max step drop {max_drop * 100:.1f} cm", 20, (255, 255, 255)),
        ("", 6, (0, 0, 0)),
        ("Why this curb (pickup_choice scoring, real data):", 22, (200, 225, 255)),
        (f"Requested: Biscayne Blvd curb at NE 21st St  ->  risk {req['total']}  = {breakdown(req)}",
         19, (255, 170, 120)),
        (f"Chosen: {cho['road'].replace('Northeast', 'NE')} north curb  ->  risk {cho['total']}  = {breakdown(cho)}",
         19, (120, 240, 160)),
        (f"{pick['scored']} real curbs within {PIN_RADIUS:.0f} m of the pin scored; "
         f"flood multiplier x{pick['flood_multiplier']} (cached live conditions)", 17, (170, 180, 190)),
        ("", 6, (0, 0, 0)),
        ("Real: OSM street + bus stop, FDOT work-zone points, pickup scoring.  "
         "Illustrative: barricade shapes/extent, e-scooter.", 16, (170, 180, 190)),
    ])))
    durations.append(6000)

    paths = write_video(frames, durations, args.output / "escort_obstacles")
    last.save(args.output / "last_frame.png")
    report = {
        "mission": "Guide-dog obstacle course to a Waymo, Biscayne Blvd west sidewalk, Miami",
        "arrived": arrived, "steps": step + 1, "walked_m": round(walked, 1), "route_m": round(float(route_cum[-1]), 1),
        "max_single_step_drop_m": round(max_drop, 3), "video_s": round(sum(durations) / 1000, 1),
        "obstacles": [{"name": o["name"], "xy": [round(v, 1) for v in o["xy"]], "tag": o["tag"], "side": o["side"],
                       "announced": o["announced"], "dog_clearance_m": round(o["clearance"], 2),
                       "handler_clearance_m": round(o["handler_clearance"], 2)} for o in obstacles],
        "fdot_points": road_cones,
        "pickup": {"requested": {k: req[k] for k in ("road", "xy", "total", "work_zones", "crashes", "lanes")},
                   "chosen": {k: cho[k] for k in ("road", "xy", "total", "work_zones", "crashes", "floods", "walk_m")},
                   "curbs_scored": pick["scored"]},
        "events": events,
    }
    (args.output / "mission_report.json").write_text(json.dumps(report, indent=2, default=list))
    p.disconnect()
    print(f"arrived={arrived} walked={walked:.1f} m avoided={avoided} min_clear dog={min_dog:.2f} "
          f"handler={min_handler:.2f} max_drop={max_drop:.3f} video={report['video_s']} s -> {[str(q) for q in paths]}")
    assert max_drop < 0.05, "dog stepped off a curb"
    assert arrived, "dog did not reach the Waymo"
    assert avoided >= 3, "fewer than three obstacles were announced and passed"
    assert min_dog > MIN_CLEAR and min_handler > MIN_CLEAR, "dog or rider brushed an obstacle"


if __name__ == "__main__":
    main()
