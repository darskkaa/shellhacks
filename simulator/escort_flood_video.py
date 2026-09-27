"""
Flooded sidewalk, new path: the guide dog finds the planned sidewalk under water and reroutes.

Real (simulator/real_block.py, simulator/data/): the Edgewater block around NE 2nd Ave & NE 21st St
from OpenStreetMap, the Miami-Dade 311 "drain clogged" report at 2100 NE 2nd Ave, the Waymo at the
curb simulator/pickup_choice.py chose with SafeRoute's hazard weights, and the king-tide conditions
cached in pickup.json. Illustrative: the extent of the standing water (the report is a point; the pool
is drawn from the curb drain to the building wall so it closes the sidewalk) and the rider's start.

The dog is only given the phone's walking route over the OSM map (it goes through the flooded
stretch). Its camera segmentation marks water in its map (DogMap); when perceived water sits on the
route ahead it stops, looks across the sidewalk, and replans the whole walking route with the water
it saw marked impassable. It then leads the rider along the new route to the Waymo.

    .venv/bin/python -m simulator.escort_flood_video --offline
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simulator.gemini_waymo_sim as gws  # noqa: E402
from simulator.gemini_waymo_sim import (  # noqa: E402
    CAM_W, GLOBAL_RES, HANDLER_LAG_M, HANDLER_RIGHT_M, LOOKAHEAD_M, MAX_TURN, RES, STEP_M, DogMap, Trail, astar,
    cell, center, detections, dilate, dog_camera, draw_local_map,
)
from simulator.pickup_choice import PICKUP_JSON  # noqa: E402
from simulator.real_block import load  # noqa: E402
from simulator.video import write_video  # noqa: E402
from simulator.waymo_scene import (  # noqa: E402
    CURB_H, RealBlock, build_handle, build_person, build_robot_dog, build_scene, create_box, create_cylinder,
    nearest_curb,
)

# The only way round the flooded frontage is round the whole building, so the window reaches its west end.
WINDOW = (-275.0, 5.0, -70.0, 82.0)
# DogMap and cell() read the map extent from gemini_waymo_sim's module globals; point them at this window.
gws.X0, gws.Y0 = WINDOW[0], WINDOW[1]
gws.NX, gws.NY = int((WINDOW[2] - WINDOW[0]) / RES), int((WINDOW[3] - WINDOW[1]) / RES)

FLOOD_ADDRESS = "2100 NE 2ND AVE"
START = (-94.0, 66.0)          # illustrative: NE 2nd Ave west sidewalk, 20 m north of the 311 pin
POOL_HALF_LEN = 1.5            # straight part of the pool along the curb; rounded ends add a sidewalk width
WATER_CLEAR = 8                # cells (0.8 m): the replanned route keeps this far from seen water
ROUTE_HIT = 6                  # cells: seen water this close to the route ahead means the route is wet
LOCAL_WATER = 6                # cells: local planner's water inflation (the rider walks 0.25 m to the right)
SCAN_STEPS = 18
FAST_EVERY = 4                 # save every 4th step on long uneventful stretches
W, H = 1280, 720
MAIN_W, MAIN_H = 860, 560
BG = (14, 17, 21)


def font(size):
    return ImageFont.load_default(size=size)


def pretty(address):
    """'2100 NE 2ND AVE' -> '2100 NE 2nd Ave'."""
    return " ".join(w if w in ("NE", "NW", "SE", "SW", "N", "S") else w.lower() if w[0].isdigit() else w.capitalize()
                    for w in address.split())


def path_len(pts):
    return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))


def plan(world, start, goal, water=None):
    """gws.global_route with seen water (DogMap cells, inflated) marked impassable."""
    walk = world.walkable(GLOBAL_RES)
    if water is not None and water.any():
        k = int(round(GLOBAL_RES / RES))
        nx, ny = walk.shape
        wet = dilate(water, WATER_CLEAR)[: nx * k, : ny * k].reshape(nx, k, ny, k).any(axis=(1, 3))
        walk &= ~wet
    to_c = lambda xy: (int((xy[0] - world.x0) / GLOBAL_RES), int((xy[1] - world.y0) / GLOBAL_RES))  # noqa: E731
    s, g = to_c(start), to_c(goal)
    walk[s] = walk[g] = True
    path = astar(~walk, s, lambda c: c == g, lambda c: math.hypot(c[0] - g[0], c[1] - g[1]))
    assert path, "no walking route to the Waymo"
    return [(world.x0 + (i + 0.5) * GLOBAL_RES, world.y0 + (j + 0.5) * GLOBAL_RES) for i, j in path]


def corners(route, cum):
    """Arc positions (m) and turn sides of the route's real corners (heading change > 50 deg over 6 m)."""
    out = []
    for s in np.arange(6.0, cum[-1] - 6.0, 1.0):
        a, b, c = (route[min(int(np.searchsorted(cum, t)), len(route) - 1)] for t in (s - 6, s, s + 6))
        turn = (math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi) \
            % (2 * math.pi) - math.pi
        if abs(turn) > 0.9 and (not out or s - out[-1][0] > 15):
            out.append((s, "left" if turn > 0 else "right"))
    return out


def route_to_car(world, start, base, fwd, water=None):
    """Route to the stop point beside the door; the dog stops HANDLER_LAG_M past it in its direction of travel
    so the rider, behind it, ends up at the handle."""
    r = plan(world, start, base, water)
    back = r[max(len(r) - 15, 0)]
    d = 1 if fwd[0] * (base[0] - back[0]) + fwd[1] * (base[1] - back[1]) > 0 else -1
    goal = (base[0] + d * fwd[0] * HANDLER_LAG_M, base[1] + d * fwd[1] * HANDLER_LAG_M)
    return plan(world, start, goal, water), goal


class Pool:
    """Illustrative extent of the standing water at a 311 report: curb drain to building wall, rounded ends."""

    def __init__(self, world, flood):
        (cx, cy), n = nearest_curb(world, flood["xy"])
        width = 0.2
        while world.ground_height(cx + n[0] * width, cy + n[1] * width) < CURB_H + 0.3 and width < 12:
            width += 0.1
        self.n, self.t = n, (-n[1], n[0])
        self.curb = (cx, cy)
        self.width = width
        self.r = width / 2 - 0.05
        self.mid = (cx + n[0] * width / 2, cy + n[1] * width / 2)
        self.ends = [(self.mid[0] + s * POOL_HALF_LEN * self.t[0], self.mid[1] + s * POOL_HALF_LEN * self.t[1])
                     for s in (-1, 1)]

    def build(self):
        blue = [0.22, 0.42, 0.78, 1]
        # The box runs 0.3 m into the wall (hidden) so no dry strip is left along it.
        a0, a1 = 0.05, self.width + 0.3
        ac = (a0 + a1) / 2
        pos = [self.curb[0] + self.n[0] * ac, self.curb[1] + self.n[1] * ac, CURB_H + 0.005]
        ori = p.getQuaternionFromEuler([0, 0, math.atan2(self.n[1], self.n[0])])
        ids = [create_box(pos, [(a1 - a0) / 2, POOL_HALF_LEN, 0.004], blue, orientation=ori)]
        ids += [create_cylinder([ex, ey, CURB_H + 0.005], self.r, 0.008, blue) for ex, ey in self.ends]
        return ids

    def dist(self, q):
        """Distance from q to the water (0 inside)."""
        dx, dy = q[0] - self.mid[0], q[1] - self.mid[1]
        a, b = dx * self.n[0] + dy * self.n[1], dx * self.t[0] + dy * self.t[1]
        box = math.hypot(max(abs(a) - self.width / 2, 0), max(abs(b) - POOL_HALF_LEN, 0))
        return min([box] + [max(math.dist(q, e) - self.r, 0) for e in self.ends])


def camera_blend(world, c, yaw):
    """1: behind-left oblique view (the rider walks on the dog's right, so from the left neither hides the
    other); 0: steep view from just behind, used when a building stands where the oblique camera would be."""
    ex, ey, _ = camera_eye(c, yaw, 1.0)
    clear = all(world.ground_height(c[0] + (ex - c[0]) * t, c[1] + (ey - c[1]) * t) < 5.0
                for t in np.linspace(0.1, 1.0, 12))
    return 1.0 if clear else 0.0


def camera_eye(c, yaw, b):
    fx, fy = math.cos(yaw), math.sin(yaw)
    off, back, up = 2.8 * b, 2.0 + 3.5 * b, 10.0 - 3.5 * b
    return c[0] - back * fx - off * fy, c[1] - back * fy + off * fx, c[2] + up


def chase_camera(eye_c, yaw, b):
    """Third-person view of the team; yaw and blend are smoothed by the caller."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y, z = eye_c
    eye = list(camera_eye(eye_c, yaw, b))
    view = p.computeViewMatrix(eye, [x + 3.0 * c, y + 3.0 * s, z], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(62, MAIN_W / MAIN_H, 0.1, 90)
    _, _, rgba, _, _ = p.getCameraImage(MAIN_W, MAIN_H, view, proj, renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(MAIN_H, MAIN_W, 4)).convert("RGB")


MAP_BOX = (-268.0, 10.0, -78.0, 78.0)  # route-map crop (m)


def draw_route_map(world, dog_map, old_route, route, trail, dog, car, pool):
    img = world.texture.copy()
    px = np.asarray(img).copy()
    ii, jj = np.nonzero(dog_map.water)
    u = ((gws.X0 + (ii + 0.5) * RES - world.x0) / 0.2).astype(int)
    v = ((world.y1 - (gws.Y0 + (jj + 0.5) * RES)) / 0.2).astype(int)
    ok = (u >= 0) & (u < px.shape[1]) & (v >= 0) & (v < px.shape[0])
    px[v[ok], u[ok]] = (40, 110, 255)
    img = Image.fromarray(px)
    d = ImageDraw.Draw(img)
    if old_route is not route:
        pts = [world.px(q) for q in old_route]
        for k in range(0, len(pts) - 1, 6):
            d.line(pts[k:k + 4], fill=(110, 110, 118), width=3)
    d.line([world.px(q) for q in route], fill=(30, 210, 90), width=4)
    d.line([world.px(q) for q in trail.pts[1:]], fill=(40, 110, 255), width=2)
    x, y = world.px(pool.mid)
    d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(20, 70, 220), width=2)
    for q, col, r in ((car, (240, 240, 240), 6), (dog, (255, 200, 30), 6)):
        x, y = world.px(q)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(0, 0, 0))
    x0, y0 = world.px((MAP_BOX[0], MAP_BOX[3]))
    x1, y1 = world.px((MAP_BOX[2], MAP_BOX[1]))
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    return crop.resize((420, int(420 * crop.height / crop.width)), Image.LANCZOS)


def compose(main_img, cam, dets, local_map, route_map, action, speech, context, hazard, fast, legend):
    out = Image.new("RGB", (W, H), BG)
    out.paste(main_img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, MAIN_W, 30], fill=(10, 12, 15))
    d.text((12, 6), "NE 2nd Ave & NE 21st St, Miami  |  OpenStreetMap + Miami-Dade 311 + SafeRoute",
           fill=(200, 225, 255), font=font(17))
    if fast:
        label = f">> FAST-FORWARD {FAST_EVERY}x"
        tw = int(d.textlength(label, font=font(16)))
        d.rectangle([12, 40, 12 + tw + 20, 68], fill=(10, 12, 15))
        d.text((22, 45), label, fill=(255, 200, 60), font=font(16))
    # Dog's own map, bottom-right of the main view.
    lx, ly = MAIN_W - local_map.width - 10, MAIN_H - local_map.height - 10
    d.rectangle([lx - 4, ly - 22, lx + local_map.width + 3, ly + local_map.height + 3], fill=(10, 12, 15))
    d.text((lx, ly - 19), "DOG'S MAP (seen only)", fill=(200, 210, 220), font=font(14))
    out.paste(local_map, (lx, ly))

    cam_h = 315
    k = 420 / CAM_W
    out.paste(cam.resize((420, cam_h)), (MAIN_W, 0))
    for label, (x0, y0, x1, y1), dist, _ in dets:
        good = label in ("Waymo", "Waymo door handle")
        col = (60, 230, 120) if good else (90, 160, 255) if label.startswith("standing") else (255, 90, 70)
        d.rectangle([MAIN_W + x0 * k, y0 * k, MAIN_W + x1 * k, y1 * k], outline=col, width=2)
        name = "standing water" if label.startswith("standing") else label
        d.text((MAIN_W + x0 * k + 3, max(y0 * k - 16, 30)), f"{name} {dist:.1f} m", fill=col, font=font(15))
    d.rectangle([MAIN_W, 0, W, 26], fill=(10, 12, 15))
    d.text((MAIN_W + 10, 5), "DOG CAMERA: segmentation + depth", fill=(255, 255, 255), font=font(15))

    my = cam_h
    d.rectangle([MAIN_W, my, W, MAIN_H], fill=(10, 12, 15))
    d.text((MAIN_W + 10, my + 5), "ROUTE (OSM walking grid)", fill=(220, 230, 240), font=font(15))
    out.paste(route_map, (MAIN_W, my + 26))
    ly2 = my + 26 + route_map.height + 6
    items = [((110, 110, 118), "planned"), ((30, 210, 90), "route now"), ((40, 110, 255), "walked / water seen")]
    x = MAIN_W + 10
    for col, name in items if legend else items[1:]:
        d.rectangle([x, ly2 + 5, x + 16, ly2 + 11], fill=col)
        d.text((x + 21, ly2), name, fill=(200, 210, 220), font=font(14))
        x += 35 + int(d.textlength(name, font=font(14)))

    d.rectangle([0, MAIN_H, W, H], fill=(46, 18, 14) if hazard else (22, 28, 36))
    col = (255, 100, 70) if hazard else (255, 190, 60) if action.startswith(("TURN", "STOP", "LOOK")) else (60, 230, 120)
    d.text((20, MAIN_H + 12), f"DOG: {action}", fill=col, font=font(24))
    if speech:
        d.text((20, MAIN_H + 50), f"“{speech}”", fill=(255, 255, 255), font=font(26))
    d.text((20, MAIN_H + 98), context[0], fill=(170, 195, 220), font=font(18))
    if len(context) > 1:
        d.text((20, MAIN_H + 126), context[1], fill=(170, 195, 220), font=font(18))
    return out


def card(lines):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 150
    for text, size, col in lines:
        d.text((90, y), text, fill=col, font=font(size))
        y += int(size * 1.6)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "simulator/artifacts/escort_flood")
    ap.add_argument("--offline", action="store_true", help="accepted for parity; this video never calls Gemini")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--frames-dir", type=Path, help="also save every 40th frame here as PNG")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pickup = json.loads(PICKUP_JSON.read_text())
    block = load()
    flood = next(f for f in block["flooding"] if f["address"] == FLOOD_ADDRESS)
    tide = (pickup.get("conditions") or {}).get("highTide") or {}

    p.connect(p.DIRECT)
    world = RealBlock(block, WINDOW)
    labels, water_ids, (hx_door, hy_door), (_, _, car_yaw), n = build_scene(world, pickup)
    pool = Pool(world, flood)
    # build_scene's generic 1.6 m drain disc at this report is replaced by the pool drawn to the wall.
    disc = min(water_ids, key=lambda b: math.dist(p.getBasePositionAndOrientation(b)[0][:2], pool.curb))
    p.removeBody(disc)
    water_ids.remove(disc)
    labels.pop(disc)
    # The other reports keep build_scene's 1.6 m drain puddles; the rider must avoid those too.
    drains = [p.getBasePositionAndOrientation(b)[0][:2] for b in water_ids]
    for body in pool.build():
        labels[body] = "standing water (311 flood report)"
        water_ids.append(body)
    water_dist = lambda q: min([pool.dist(q)] + [max(math.dist(q, c) - 1.6, 0) for c in drains])  # noqa: E731

    fwd = (math.cos(car_yaw), math.sin(car_yaw))
    base = (hx_door + n[0] * 1.6, hy_door + n[1] * 1.6)
    car_xy = (hx_door, hy_door)
    route, goal_xy = route_to_car(world, START, base, fwd)
    initial_route = route
    planned_m = path_len(route)
    assert min(pool.dist(q) for q in route) == 0.0, "the phone's route should run through the flooded stretch"
    route_cum = np.concatenate([[0.0], np.cumsum([math.dist(a, b) for a, b in zip(route, route[1:])])])
    print(f"start {START}  goal {goal_xy[0]:.1f},{goal_xy[1]:.1f}  planned {planned_m:.0f} m")

    dog_map = DogMap()
    x, y = START
    yaw = math.atan2(route[12][1] - y, route[12][0] - x)
    trail = Trail(START, yaw)
    cam_yaw, cam_c, cam_b = yaw, np.array([x, y, CURB_H]), 1.0
    actors, frames, durations, events = [], [], [], []
    announced = set()
    turns, new_total = [], 0.0
    speech = "Harness on. Heading to your Waymo on NE 21st Street."
    context = [f"Phone route: OpenStreetMap sidewalks, south on NE 2nd Ave then west on NE 21st St ({planned_m:.0f} m)",
               f"Waymo at the curb SafeRoute scoring chose (risk {pickup['chosen']['total']} vs "
               f"{pickup['requested']['total']} at the requested pin)"]
    phase, scan_i, scan_yaw, extra_m = "walk", 0, 0.0, 0.0
    arrived, max_drop, last_z, stuck, hazard_frames, quiet = False, 0.0, world.ground_height(x, y), 0, 0, 0
    min_handler_water, min_dog_water, road_steps = math.inf, math.inf, 0
    step = 0

    for step in range(args.max_steps):
        cam, depth, seg, eye, target = dog_camera(world, x, y, yaw)
        # The dog knows its own body, harness and rider; they are not obstacles in its map.
        depth = np.where(np.isin(seg, actors), 1.0, depth)
        dog_map.integrate(eye, target, depth, seg, water_ids, (x, y), yaw)
        patch = dog_map.patch((x, y))
        i0, i1, j0, j1 = patch
        obstacle, step_cells, water_cells = dog_map.hazards(patch)
        blocked = dilate(obstacle, 4) | dilate(step_cells, 3) | dilate(water_cells, LOCAL_WATER)
        # The rider is not something to walk through: a U-turn goes round them.
        s_h = trail.cum[-1] - HANDLER_LAG_M
        hx, hy = trail.at(s_h)
        hyaw = trail.heading(s_h)
        hx, hy = hx + HANDLER_RIGHT_M * math.sin(hyaw), hy - HANDLER_RIGHT_M * math.cos(hyaw)
        hi, hj = cell(hx, hy)
        ci, cj = np.ogrid[i0 - hi:i1 - hi, j0 - hj:j1 - hj]
        blocked |= ci * ci + cj * cj <= 9
        si, sj = cell(x, y)
        # Clear the dog's own footprint so it can always step away (the rider stands within 0.6 m).
        di, dj = np.ogrid[i0 - si:i1 - si, j0 - sj:j1 - sj]
        blocked &= ~((di * di + dj * dj <= 4) & ~dilate(obstacle | step_cells | water_cells, 1))

        k = int(np.argmin([math.dist(q, (x, y)) for q in route]))
        dets = detections(seg, depth, labels)
        new_event, action = None, "FORWARD"

        # Is water the dog has seen lying on the route ahead?
        wet = dilate(water_cells, ROUTE_HIT)
        ahead = [q for q, s in zip(route[k:], route_cum[k:]) if s - route_cum[k] < 12.0]
        route_wet = any(0 <= a - i0 < i1 - i0 and 0 <= b - j0 < j1 - j0 and wet[a - i0, b - j0]
                        for a, b in (cell(*q) for q in ahead))

        if phase == "walk" and route_wet:
            phase, scan_i, scan_yaw = "scan", 0, yaw
            hazard_frames = SCAN_STEPS + 20
            new_event = ("WATER", "Standing water ahead, across the route. Checking for a way around.")
        if phase == "scan":
            # Stop and look across the whole sidewalk so the map holds the full width of the water.
            scan_i += 1
            yaw = scan_yaw + 0.7 * math.sin(2 * math.pi * scan_i / SCAN_STEPS)
            action = "STOP, LOOKING ACROSS THE SIDEWALK"
            if scan_i >= SCAN_STEPS:
                yaw = scan_yaw
                old_left = float(route_cum[-1] - route_cum[k])
                new_route, goal_xy = route_to_car(world, (x, y), base, fwd, dog_map.water)
                extra_m = path_len(new_route) - old_left
                route = new_route
                route_cum = np.concatenate([[0.0], np.cumsum([math.dist(a, b) for a, b in zip(route, route[1:])])])
                k = 0
                turns = corners(route, route_cum)
                new_total = float(route_cum[-1])
                phase = "walk"
                hazard_frames = 30
                if extra_m > 5:
                    new_event = ("REROUTE", f"The sidewalk ahead is flooded. Taking another way; "
                                            f"about {int(round(extra_m / 10) * 10)} metres longer.")
                    context = [f"Replanned with the water the dog saw marked impassable: {path_len(route):.0f} m "
                               f"from here vs {old_left:.0f} m through the flood",
                               "Standing water at a real 311 flood-report location (extent illustrative)"]
                else:
                    new_event = ("AROUND", "Water ahead. Going around it.")
                print(f"[{step}] replan: {old_left:.0f} m -> {path_len(route):.0f} m (+{extra_m:.0f} m)")
            elif scan_i == SCAN_STEPS // 2:
                context = [f"Standing water at a real 311 flood-report location (extent illustrative): {pretty(flood['address'])}",
                           f"311 ticket {flood['ticket']}, {flood['issue'].lower()}; king tide today "
                           f"{tide.get('time', '')[11:]} ({tide.get('feet', 0):.1f} ft, SafeRoute conditions)"]

        wp = route[min(int(np.searchsorted(route_cum, route_cum[k] + LOOKAHEAD_M)), len(route) - 1)]
        if math.dist((x, y), goal_xy) < LOOKAHEAD_M:
            wp = goal_xy
        wi, wj = cell(*wp)
        wi, wj = wi - i0, wj - j0
        path = astar(blocked, (si - i0, sj - j0), lambda c: math.hypot(c[0] - wi, c[1] - wj) <= 3,
                     lambda c: math.hypot(c[0] - wi, c[1] - wj))

        left_m = float(route_cum[-1] - route_cum[k])
        if new_total and not arrived:
            context[0] = (f"New route {new_total:.0f} m avoids the water the dog saw (+{extra_m:.0f} m)   |   "
                          f"{left_m:.0f} m to the Waymo")
        if new_event is None and turns and route_cum[k] > turns[0][0] - 2.5:
            s_turn, side = turns.pop(0)
            if route_cum[k] < s_turn + 3:
                new_event = ("TURN", f"Turning {side}. {left_m:.0f} metres to your Waymo.")
        if new_event is None:
            for label, _, dist, cx in dets:
                if label in announced or dist > 3.0 or label.startswith("standing"):
                    continue
                if label in ("Waymo", "Waymo door handle"):
                    if "Waymo" in announced:
                        continue
                    label = "Waymo"
                    new_event = ("WAYMO", "I can see your Waymo at the curb just ahead.")
                elif 0.25 < cx < 0.75:
                    side = "left" if cx > 0.5 else "right"
                    new_event = ("OBSTACLE", f"{label.capitalize()} ahead. Going around it on the {side}.")
                else:
                    continue
                announced.add(label)
                break

        if phase == "scan":
            pass
        elif math.dist((x, y), goal_xy) < 0.25:
            arrived, action = True, "ARRIVED"
        elif path is None or len(path) < 2:
            action = "STOP"
            stuck += 1
            yaw += MAX_TURN
        else:
            stuck = 0
            ahead_cell = path[min(5, len(path) - 1)]
            tgt = center(i0 + ahead_cell[0], j0 + ahead_cell[1])
            want = math.atan2(tgt[1] - y, tgt[0] - x)
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += max(-MAX_TURN, min(MAX_TURN, err))
            action = "FORWARD" if abs(err) < 0.15 else ("TURN LEFT" if err > 0 else "TURN RIGHT")
            if abs(err) < 0.9:
                next_cell = path[min(2, len(path) - 1)]
                near = center(i0 + next_cell[0], j0 + next_cell[1])
                dd = math.dist(near, (x, y)) or 1.0
                nx_, ny_ = x + STEP_M * (near[0] - x) / dd, y + STEP_M * (near[1] - y) / dd
                here, there = dog_map.height[cell(x, y)], dog_map.height[cell(nx_, ny_)]
                if np.isnan(here) or np.isnan(there) or abs(there - here) <= 0.05:
                    x, y = nx_, ny_
                    trail.add((x, y))
        if phase == "walk" and extra_m > 5 and action.startswith("FORWARD"):
            action = "FORWARD ON THE NEW ROUTE"

        if arrived:
            s_h = trail.cum[-1] - HANDLER_LAG_M
            ahx, ahy = trail.at(s_h)
            side = "right" if math.sin(math.atan2(hy_door - ahy, hx_door - ahx) - trail.heading(s_h)) < 0 else "left"
            new_event = ("ARRIVED", f"We're at your Waymo. The door is on your {side}, handle at waist height.")
            context = [f"Arrived: walked {trail.cum[-1] - 1.0:.0f} m ({trail.cum[-1] - 1.0 - planned_m:.0f} m more than "
                       "the flooded route) "
                       "without touching the water", context[1]]

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
        min_handler_water = min(min_handler_water, water_dist((hx, hy)))
        min_dog_water = min(min_dog_water, water_dist((x, y)))
        road_steps += world.is_road(x, y) or world.is_road(hx, hy)
        hand = [hx + 0.4 * math.cos(hyaw) - 0.28 * math.sin(hyaw), hy + 0.4 * math.sin(hyaw) + 0.28 * math.cos(hyaw),
                hz + 0.85]
        back = [x - 0.15 * math.cos(yaw), y - 0.15 * math.sin(yaw), dz + 0.34]
        actors[:] = build_robot_dog(x, y, dz, yaw) + build_person(hx, hy, hz, hyaw) + build_handle(back, hand)
        if step % 8 == 0:
            create_box([x, y, dz + 0.01], [0.05, 0.05, 0.003], [0.25, 0.55, 1.0, 1])

        if new_event:
            kind, speech = new_event
            events.append({"step": step, "kind": kind, "speech": speech, "dog": [round(x, 2), round(y, 2)]})
            print(f"[{step:4d}] {kind:8s} {speech}")
            quiet = 0
        else:
            quiet += 1

        # Camera: follow the midpoint of dog and rider, turning slowly (no whip-pan on the U-turn or the scan).
        mid = np.array([(x + hx) / 2, (y + hy) / 2, dz])
        cam_c += 0.25 * (mid - cam_c)
        head = trail.heading(trail.cum[-1]) if phase == "walk" else cam_yaw
        cam_yaw += 0.08 * ((head - cam_yaw + math.pi) % (2 * math.pi) - math.pi)
        cam_b += 0.08 * (camera_blend(world, cam_c, cam_yaw) - cam_b)

        near_pool = pool.dist((x, y)) < 12
        near_goal = math.dist((x, y), goal_xy) < 10
        fast = not (near_pool or near_goal or phase == "scan" or hazard_frames > 0 or quiet < 25)
        hazard = hazard_frames > 0
        hazard_frames -= 1
        if new_event or arrived or step % (FAST_EVERY if fast else 1) == 0:
            frames.append(compose(chase_camera(cam_c, cam_yaw, cam_b), cam, dets,
                                  draw_local_map(dog_map, patch, blocked, path, (x, y), trail),
                                  draw_route_map(world, dog_map, initial_route, route, trail, (x, y), car_xy, pool),
                                  action, speech if quiet < 80 else "", context, hazard, fast, route is not initial_route))
            durations.append((2600 if new_event[0] in ("WATER", "REROUTE") else 1400) if new_event else 50)
            if args.frames_dir and len(frames) % 40 == 0:
                frames[-1].save(args.frames_dir / f"f{len(frames):04d}.png")
        if arrived or stuck > 60:
            break

    durations[-1] = 3000
    walked = trail.cum[-1] - 1.0
    # The walk differs from the phone's first plan around the flood: the rider's path never comes near it.
    trail_pts = trail.pts[1:]
    divergence = max(min(math.dist(q, r) for r in initial_route[::3]) for q in trail_pts[::5])
    cho = pickup["chosen"]
    title = card([
        ("Flooded sidewalk, new path", 54, (255, 255, 255)),
        ("Robot guide dog escort to a Waymo  |  Edgewater, Miami: NE 2nd Ave & NE 21st St", 26, (200, 225, 255)),
        ("", 16, BG),
        ("Real data: OpenStreetMap streets, sidewalks and buildings; Miami-Dade 311 flood report", 22, (190, 200, 210)),
        (f"  ({pretty(flood['address'])}, ticket {flood['ticket']}); SafeRoute Miami pickup scoring and today's", 22,
         (190, 200, 210)),
        (f"  king tide ({tide.get('feet', 0):.1f} ft at {tide.get('time', '')[11:]}, NWS Coastal Flood Statement).", 22,
         (190, 200, 210)),
        ("Simulated in PyBullet: the robot dog (Hiwonder MechDog) and the rider.", 22, (190, 200, 210)),
        ("Illustrative: the extent of the standing water, and the rider's start point.", 22, (255, 200, 120)),
    ])
    end = card([
        ("Arrived at the Waymo, dry", 50, (60, 230, 120)),
        (f"Planned route {planned_m:.0f} m ran through the water at the 311 report; the dog saw it,", 24,
         (230, 235, 240)),
        (f"replanned with it marked impassable and walked {walked:.0f} m (+{walked - planned_m:.0f} m) round the building.", 24,
         (230, 235, 240)),
        (f"Rider's closest approach to water {min_handler_water:.2f} m  |  max single-step drop {max_drop:.3f} m"
         "  |  roadway steps 0", 22, (190, 200, 210)),
        (f"Waymo: the NE 21st St curb pickup_choice picked (risk {cho['total']} vs {pickup['requested']['total']} "
         f"at the requested pin, {cho['floods']} flood reports", 22, (190, 200, 210)),
        ("within 60 m); the new route reaches it from the west, along the sidewalk, so it stayed the pickup.", 22,
         (190, 200, 210)),
        ("Water extent and rider start illustrative; everything else from OSM / 311 / SafeRoute data.", 20,
         (255, 200, 120)),
    ])
    frames = [title] + frames + [end]
    durations = [3500] + durations + [5000]
    paths = write_video(frames, durations, args.output / "escort_flood")
    frames[len(frames) // 2].save(args.output / "mid_frame.png")
    report = {"mission": "Flooded sidewalk, new path (NE 2nd Ave & NE 21st St, Miami)", "arrived": arrived,
              "steps": step + 1, "planned_m": round(planned_m, 1), "walked_m": round(walked, 1),
              "extra_m": round(extra_m, 1), "route_divergence_m": round(divergence, 1),
              "min_rider_water_m": round(min_handler_water, 2), "min_dog_water_m": round(min_dog_water, 2),
              "max_single_step_drop_m": round(max_drop, 3), "roadway_steps": int(road_steps),
              "flood_report": {k: flood[k] for k in ("address", "issue", "ticket")},
              "video_s": round(sum(durations) / 1000, 1), "outputs": [str(q) for q in paths], "events": events}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    p.disconnect()
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=1))
    assert max_drop < 0.05, "dog stepped off a curb"
    assert road_steps == 0, "dog or rider stepped into the roadway"
    assert min_handler_water > 0.2, "rider stepped into the water"
    assert arrived, "dog did not reach the Waymo"
    assert extra_m > 5 and divergence > 10, "the final walk did not leave the flooded route"


if __name__ == "__main__":
    main()
