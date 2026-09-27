"""
Waymo SafePoint 3D: autonomous guide-dog escort to a waiting Waymo, on a real Miami block.

World (all real, see waymo_scene.py): the Edgewater block around NE 2nd Ave and NE 21st St, Miami,
from OpenStreetMap, with Miami-Dade 311 flooding reports as standing water at the curb drains.
The rider asked to be picked up at 2100 NE 2nd Ave; simulator/pickup_choice.py scored every real
curb with SafeRoute Miami's hazard weights and live flood conditions and moved the Waymo to the
NE 21st St curb. This sim walks the rider there.

Only the dog is simulated. It is given what a phone would give it: a walking route over the OSM
map to the Waymo's door. Everything else it must see for itself. Every step it
  1. renders its own camera (RGB, depth, segmentation),
  2. turns the depth image into a height map; tall cells are obstacles, sharp height steps
     (curbs) are drop-offs, and ground it should see but can't is treated as a possible drop,
  3. marks standing water from the segmentation,
  4. replans a local A* path through what it has seen toward the next route waypoint.
Speech comes from what the camera sees; with a Gemini key (and quota) event frames are narrated by
Gemini 3.8 Flash instead.

    python -m simulator.pickup_choice      # once, needs the SafeRoute server for live conditions
    python -m simulator.gemini_waymo_sim   # add --offline to skip Gemini
"""

import argparse
import base64
import heapq
import io
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.pickup_choice import PICKUP_JSON  # noqa: E402
from simulator.real_block import load, ssl_context  # noqa: E402
from simulator.waymo_scene import (  # noqa: E402
    CURB_H, RealBlock, build_handle, build_person, build_robot_dog, build_scene, create_box,
)

WINDOW = (-195.0, 0.0, -75.0, 62.0)  # covers the rider's route plus margin
STEP_M = 0.15           # dog stride per step
MAX_TURN = 0.35         # rad per step
SAVE_EVERY = 3          # steps per GIF frame (event steps are always saved)
STEP_MS, EVENT_MS = 90, 1500
HANDLER_LAG_M, HANDLER_RIGHT_M = 0.6, 0.25

# Dog's map (0.1 m cells over the whole window; hazards are evaluated in a local patch)
RES = 0.1
X0, Y0 = WINDOW[0], WINDOW[1]
NX, NY = int((WINDOW[2] - WINDOW[0]) / RES), int((WINDOW[3] - WINDOW[1]) / RES)
LOCAL = 90              # local patch half-size in cells (9 m)
OBSTACLE_H = 0.45       # taller than this above the sidewalk is not walkable
STEP_H = 0.06           # a height jump bigger than this is a drop-off
OBSTACLE_INFLATE, STEP_INFLATE, WATER_INFLATE = 4, 3, 3
SENSE_RANGE = 5.0
LOOKAHEAD_M = 5.0
GLOBAL_RES = 0.4

CAM_W, CAM_H, CAM_FOV, NEAR, FAR = 480, 360, 70.0, 0.05, 40.0


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip("'\" ")
    return None


class GeminiNarrator:
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    PROMPT = ("You are the voice of a robot guide dog leading a blind passenger along a Miami sidewalk to their "
              "Waymo. This is the dog's camera. The dog's onboard perception reports: {event}. "
              'Reply in JSON: {{"speech": "one short, calm sentence for the passenger"}}')

    def __init__(self, api_key):
        self.api_key = api_key
        self.disabled = False

    def speak(self, image, event):
        if self.disabled:
            return None
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=80)
        payload = {
            "contents": [{"parts": [
                {"text": self.PROMPT.format(event=event)},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}},
            ]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        # Key in a header, not the URL, so it never lands in logs or tracebacks.
        req = urllib.request.Request(self.ENDPOINT, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=15, context=ssl_context()) as resp:
                res = json.loads(resp.read())
            return json.loads(res["candidates"][0]["content"]["parts"][0]["text"])["speech"]
        except Exception as e:  # noqa: BLE001 - narration is optional; onboard speech is used instead
            print(f"Gemini unavailable ({e}); using onboard speech from here on")
            self.disabled = True
            return None


def cell(x, y):
    # Clamped: a stride target or look-ahead probe near the window edge must not index outside the map.
    return min(max(int((x - X0) / RES), 0), NX - 1), min(max(int((y - Y0) / RES), 0), NY - 1)


def center(i, j):
    return X0 + (i + 0.5) * RES, Y0 + (j + 0.5) * RES


def _shift_or(dst, src, dx, dy):
    """dst[i + dx, j + dy] |= src[i, j] for arrays of equal shape."""
    nx, ny = src.shape
    dst[max(dx, 0):nx + min(dx, 0), max(dy, 0):ny + min(dy, 0)] |= \
        src[max(-dx, 0):nx - max(dx, 0), max(-dy, 0):ny - max(dy, 0)]


def dilate(mask, r):
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if 0 < dx * dx + dy * dy <= r * r:
                _shift_or(out, mask, dx, dy)
    return out


class DogMap:
    """What the dog has seen: highest point, height spread and water per 0.1 m cell."""

    def __init__(self):
        self.height = np.full((NX, NY), np.nan, np.float32)
        self.spread = np.zeros((NX, NY), np.float32)   # max - min in the cell: a curb face shows up here
        self.hole = np.zeros((NX, NY), bool)           # should be visible ground, got no return
        self.water = np.zeros((NX, NY), bool)
        self.cx, self.cy = np.meshgrid(X0 + (np.arange(NX) + 0.5) * RES, Y0 + (np.arange(NY) + 0.5) * RES,
                                       indexing="ij")

    def integrate(self, eye, target, depth, seg, water_ids, dog, yaw):
        f = np.subtract(target, eye)
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        vs, us = np.mgrid[0:CAM_H:3, 0:CAM_W:3]
        d = depth[vs, us]
        z_lin = FAR * NEAR / (FAR - (FAR - NEAR) * d)
        tan_v = math.tan(math.radians(CAM_FOV) / 2)
        ndc_x = 2 * (us + 0.5) / CAM_W - 1
        ndc_y = 1 - 2 * (vs + 0.5) / CAM_H
        dirs = f + (ndc_x * tan_v * CAM_W / CAM_H)[..., None] * r + (ndc_y * tan_v)[..., None] * u
        pts = np.asarray(eye) + z_lin[..., None] * dirs
        keep = (z_lin < SENSE_RANGE) & (d < 0.999)
        wet = np.isin(seg[vs, us], list(water_ids))[keep]
        pts = pts[keep]
        # floor, not truncation toward zero: points just outside the window must not land in cell 0.
        ix = np.floor((pts[:, 0] - X0) / RES).astype(int)
        iy = np.floor((pts[:, 1] - Y0) / RES).astype(int)
        ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
        ix, iy, z, wet = ix[ok], iy[ok], pts[ok, 2], wet[ok]
        i0, i1, j0, j1 = self.patch(dog)
        hi = np.full((i1 - i0, j1 - j0), -np.inf, np.float32)
        lo = np.full((i1 - i0, j1 - j0), np.inf, np.float32)
        inp = (ix >= i0) & (ix < i1) & (iy >= j0) & (iy < j1)
        np.maximum.at(hi, (ix[inp] - i0, iy[inp] - j0), z[inp])
        np.minimum.at(lo, (ix[inp] - i0, iy[inp] - j0), z[inp])
        seen = np.isfinite(hi)
        self.height[i0:i1, j0:j1][seen] = hi[seen]
        self.spread[i0:i1, j0:j1][seen] = hi[seen] - lo[seen]
        self.water[ix[wet], iy[wet]] = True

        h = self.height[i0:i1, j0:j1]
        cx, cy = self.cx[i0:i1, j0:j1], self.cy[i0:i1, j0:j1]
        dist = np.hypot(cx - dog[0], cy - dog[1])
        bearing = np.abs((np.arctan2(cy - dog[1], cx - dog[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        ci, cj = np.nonzero((dist > 0.9) & (dist < 1.8) & (bearing < 0.5) & np.isnan(h))
        # Unseen because something tall is in the way is occlusion, not a drop-off.
        occluded = np.zeros(len(ci), bool)
        for t in (0.3, 0.5, 0.7, 0.85):
            si = np.clip(((dog[0] + (cx[ci, cj] - dog[0]) * t - X0) / RES).astype(int) - i0, 0, i1 - i0 - 1)
            sj = np.clip(((dog[1] + (cy[ci, cj] - dog[1]) * t - Y0) / RES).astype(int) - j0, 0, j1 - j0 - 1)
            occluded |= np.nan_to_num(h[si, sj], nan=0.0) > CURB_H + OBSTACLE_H
        self.hole[i0 + ci[~occluded], j0 + cj[~occluded]] = True
        self.hole[i0:i1, j0:j1] &= np.isnan(h)

    @staticmethod
    def patch(dog):
        i, j = cell(*dog)
        return max(i - LOCAL, 0), min(i + LOCAL, NX), max(j - LOCAL, 0), min(j + LOCAL, NY)

    def hazards(self, patch):
        """(obstacle, drop-off, water) masks inside the patch."""
        i0, i1, j0, j1 = patch
        height = self.height[i0:i1, j0:j1]
        # Sidewalk tops sit at CURB_H, so obstacles are anything well above that.
        obstacle = np.nan_to_num(height, nan=0.0) > CURB_H + OBSTACLE_H
        h = np.where(obstacle, np.nan, height)
        step = (~obstacle & (self.spread[i0:i1, j0:j1] > STEP_H)) | self.hole[i0:i1, j0:j1]
        # Compare across up to 0.3 m so a drop is caught with unseen cells in between; only the
        # high side is marked, since that is where you would step off.
        nx, ny = h.shape
        for dx in range(0, 4):
            for dy in range(-3, 4):
                if (dx, dy) <= (0, 0) or dx * dx + dy * dy > 9:
                    continue
                a = (slice(0, nx - dx), slice(max(-dy, 0), ny - max(dy, 0)))
                b = (slice(dx, nx), slice(max(dy, 0), ny + min(dy, 0)))
                with np.errstate(invalid="ignore"):
                    step[a] |= (h[a] - h[b]) > STEP_H
                    step[b] |= (h[b] - h[a]) > STEP_H
        return obstacle, step & ~obstacle, self.water[i0:i1, j0:j1]

    def blocked(self, patch):
        obstacle, step, water = self.hazards(patch)
        return dilate(obstacle, OBSTACLE_INFLATE) | dilate(step, STEP_INFLATE) | dilate(water, WATER_INFLATE)


def astar(blocked, start, goal_test, heuristic):
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    nx, ny = blocked.shape
    g = {start: 0.0}
    parent = {start: None}
    heap = [(heuristic(start), start)]
    while heap:
        _, cur = heapq.heappop(heap)
        if goal_test(cur):
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for dx, dy, cost in moves:
            n = (cur[0] + dx, cur[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or blocked[n]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue
            ng = g[cur] + cost
            if ng < g.get(n, math.inf):
                g[n] = ng
                parent[n] = cur
                heapq.heappush(heap, (ng + heuristic(n), n))
    return None


def global_route(world, start, goal):
    """Walking route over the OSM map (sidewalks, alleys, plazas; no roadway, no buildings), like a phone gives."""
    walkable = world.walkable(GLOBAL_RES)
    to_c = lambda xy: (int((xy[0] - world.x0) / GLOBAL_RES), int((xy[1] - world.y0) / GLOBAL_RES))  # noqa: E731
    s, g = to_c(start), to_c(goal)
    walkable[s] = walkable[g] = True
    path = astar(~walkable, s, lambda c: c == g, lambda c: math.hypot(c[0] - g[0], c[1] - g[1]))
    assert path, "no sidewalk route from the rider to the Waymo in the OSM map"
    return [(world.x0 + (i + 0.5) * GLOBAL_RES, world.y0 + (j + 0.5) * GLOBAL_RES) for i, j in path]


class Trail:
    """Arc-length parameterised dog trail, so the handler walks where the dog walked."""

    def __init__(self, start, yaw):
        self.pts = [(start[0] - math.cos(yaw), start[1] - math.sin(yaw)), tuple(start)]
        self.cum = [0.0, 1.0]

    def add(self, pt):
        d = math.dist(pt, self.pts[-1])
        if d > 1e-6:
            self.pts.append(tuple(pt))
            self.cum.append(self.cum[-1] + d)

    def at(self, s):
        i = max(0, min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.pts) - 2))
        a, b = self.pts[i], self.pts[i + 1]
        t = (s - self.cum[i]) / (self.cum[i + 1] - self.cum[i])
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t

    def heading(self, s):
        a, b = self.at(s - 0.3), self.at(s + 0.05)
        return math.atan2(b[1] - a[1], b[0] - a[0])


def dog_camera(world, x, y, yaw):
    gz = world.ground_height(x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    eye = [x + 0.3 * c, y + 0.3 * s, gz + 0.35]
    target = [eye[0] + 3.0 * c, eye[1] + 3.0 * s, gz + 0.05]
    _, _, rgba, depth, seg = p.getCameraImage(
        CAM_W, CAM_H, p.computeViewMatrix(eye, target, [0, 0, 1]),
        p.computeProjectionMatrixFOV(CAM_FOV, CAM_W / CAM_H, NEAR, FAR), renderer=p.ER_TINY_RENDERER)
    rgb = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(CAM_H, CAM_W, 4)).convert("RGB")
    return rgb, np.asarray(depth).reshape(CAM_H, CAM_W), np.asarray(seg).reshape(CAM_H, CAM_W), eye, target


def chase_camera(world, x, y, yaw):
    w, h = 640, 480
    c, s = math.cos(yaw), math.sin(yaw)
    gz = world.ground_height(x, y)
    # High and close behind, so the camera stays above nearby building walls.
    eye = [x - 4.0 * c, y - 4.0 * s, gz + 8.0]
    _, _, rgba, _, _ = p.getCameraImage(w, h, p.computeViewMatrix(eye, [x + 2.0 * c, y + 2.0 * s, gz], [0, 0, 1]),
                                        p.computeProjectionMatrixFOV(65, w / h, 0.1, 80), renderer=p.ER_TINY_RENDERER)
    return Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)).convert("RGB")


def detections(seg, depth, labels):
    """Labelled objects in view: (label, bbox, distance m, horizontal centre 0..1)."""
    out = {}
    z_lin = FAR * NEAR / (FAR - (FAR - NEAR) * depth)
    for body in np.unique(seg):
        label = labels.get(int(body))
        if label is None:
            continue
        vs, us = np.nonzero(seg == body)
        if len(vs) < 80:
            continue
        dist = float(np.median(z_lin[vs, us]))
        box = (int(us.min()), int(vs.min()), int(us.max()), int(vs.max()))
        if label not in out or dist < out[label][2]:
            out[label] = (label, box, dist, float(us.mean()) / CAM_W)
    return list(out.values())


def draw_local_map(dog_map, patch, blocked, path, dog, trail):
    i0, i1, j0, j1 = patch
    h = dog_map.height[i0:i1, j0:j1]
    img = np.full(h.shape + (3,), 40, np.uint8)
    known = ~np.isnan(h)
    shade = np.clip(110 + np.nan_to_num(h) * 500, 0, 255).astype(np.uint8)
    img[known] = np.stack([shade[known]] * 3, axis=1)
    obstacle, step, water = dog_map.hazards(patch)
    img[blocked & known] = (150, 90, 60)
    img[step] = (230, 60, 40)
    img[obstacle] = (200, 30, 30)
    img[water] = (60, 110, 230)
    for i, j in path or []:
        img[i, j] = (60, 230, 120)
    for pt in trail.pts[-200::2]:
        i, j = cell(*pt)
        if i0 <= i < i1 and j0 <= j < j1:
            img[i - i0, j - j0] = (60, 140, 255)
    i, j = cell(*dog)
    img[max(i - i0 - 1, 0):i - i0 + 2, max(j - j0 - 1, 0):j - j0 + 2] = (255, 255, 255)
    return Image.fromarray(np.transpose(img, (1, 0, 2))[::-1]).resize((180, 180), Image.NEAREST)


def draw_route_map(world, route, dog, trail, handle):
    img = world.texture.copy()
    d = ImageDraw.Draw(img)
    d.line([world.px(q) for q in route], fill=(40, 200, 90), width=3)
    d.line([world.px(q) for q in trail.pts[1:]], fill=(40, 120, 255), width=3)
    for q, col in ((dog, (255, 255, 255)), (handle, (0, 220, 220))):
        x, y = world.px(q)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(0, 0, 0))
    return img.resize((img.width * 130 // img.height, 130))


def compose(chase, cam, dets, local_map, route_map, action, speech, source, hazard, stats):
    w, h = 640, 480
    out = Image.new("RGB", (w * 2, h + 100), (16, 20, 24))
    out.paste(chase, (0, 0))
    out.paste(cam.resize((w, h)), (w, 0))
    d = ImageDraw.Draw(out)
    k = w / CAM_W
    for label, (x0, y0, x1, y1), dist, _ in dets:
        good = label in ("Waymo", "Waymo door handle")
        col = (60, 230, 120) if good else (90, 150, 255) if label.startswith("standing") else (255, 80, 60)
        d.rectangle([w + x0 * k, y0 * k, w + x1 * k, y1 * k], outline=col, width=2)
        d.text((w + x0 * k + 3, max(y0 * k - 12, 30)), f"{label} {dist:.1f} m", fill=col)
    for img, (x, y), title in ((local_map, (w - 188, h - 196), "DOG'S MAP (seen only)"),
                               (route_map, (8, h - 146), "ROUTE (OSM)")):
        d.rectangle([x - 3, y - 15, x + img.width + 2, y + img.height + 2], fill=(10, 12, 15))
        d.text((x, y - 13), title, fill=(200, 210, 220))
        out.paste(img, (x, y))
    d.rectangle([0, 0, w, 28], fill=(10, 12, 15))
    d.text((12, 8), "NE 2nd Ave & NE 21st St, Miami (OpenStreetMap + SafeRoute data)", fill=(200, 225, 255))
    d.rectangle([w, 0, w * 2, 28], fill=(10, 12, 15))
    d.text((w + 12, 8), "ROBOT DOG CAMERA: segmentation + depth", fill=(255, 255, 255))
    d.rectangle([0, h, w * 2, h + 100], fill=(40, 16, 14) if hazard else (22, 28, 36))
    col = (255, 90, 60) if hazard else (255, 190, 60) if action.startswith("TURN") else (60, 230, 120)
    d.text((16, h + 10), f"ONBOARD PLANNER: [{action}]", fill=col)
    d.text((16, h + 30), stats, fill=(180, 200, 220))
    d.text((16, h + 56), f"Speech ({source}): \"{speech}\"", fill=(255, 255, 255))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "simulator/artifacts/waymo_escort")
    parser.add_argument("--offline", action="store_true", help="no Gemini narration; onboard speech only")
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pickup = json.loads(PICKUP_JSON.read_text())
    key = None if args.offline else get_api_key()
    narrator = GeminiNarrator(key) if key else None

    p.connect(p.DIRECT)
    world = RealBlock(load(), WINDOW)
    labels, water_ids, (hx_door, hy_door), (_, _, car_yaw), n = build_scene(world, pickup)
    # Stop the dog on the sidewalk 1.6 m in from the car's side, far enough past the door that the
    # handler, walking behind it, stands beside the handle.
    fwd = (math.cos(car_yaw), math.sin(car_yaw))
    start = tuple(pickup["rider_start"])
    route_dir = 1 if (fwd[0] * (hx_door - start[0]) + fwd[1] * (hy_door - start[1])) > 0 else -1
    goal_xy = (hx_door + n[0] * 1.6 + route_dir * fwd[0] * HANDLER_LAG_M,
               hy_door + n[1] * 1.6 + route_dir * fwd[1] * HANDLER_LAG_M)
    route = global_route(world, start, goal_xy)
    route_cum = np.concatenate([[0.0], np.cumsum([math.dist(a, b) for a, b in zip(route, route[1:])])])
    print(f"rider start {start[0]:.1f},{start[1]:.1f}  goal {goal_xy[0]:.1f},{goal_xy[1]:.1f}  "
          f"OSM route {route_cum[-1]:.0f} m")

    dog_map = DogMap()
    x, y = start
    yaw = math.atan2(route[min(8, len(route) - 1)][1] - y, route[min(8, len(route) - 1)][0] - x)
    trail = Trail(start, yaw)
    actors, frames, durations, events = [], [], [], []
    announced = set()
    speech, source, hazard_frames = "Harness on. Your Waymo moved to a safer curb on NE 21st Street. Let's go.", \
        "onboard", 0
    arrived, max_drop, last_z, stuck, step = False, 0.0, world.ground_height(x, y), 0, 0
    req, cho = pickup["requested"], pickup["chosen"]
    flood_note = (f"Pickup moved off NE 2nd Ave: {req['floods']} flood reports + {req['crashes']} crash at that curb "
                  f"(risk {req['total']} vs {cho['total']}, flood x{pickup['flood_multiplier']} live)")

    for step in range(args.max_steps):
        cam, depth, seg, eye, target = dog_camera(world, x, y, yaw)
        dog_map.integrate(eye, target, depth, seg, water_ids, (x, y), yaw)
        patch = dog_map.patch((x, y))
        i0, _, j0, _ = patch
        blocked = dog_map.blocked(patch)
        si, sj = cell(x, y)
        blocked[si - i0, sj - j0] = False

        # Next waypoint: LOOKAHEAD_M along the OSM route past the closest route point.
        k = int(np.argmin([math.dist(q, (x, y)) for q in route]))
        wp = route[min(int(np.searchsorted(route_cum, route_cum[k] + LOOKAHEAD_M)), len(route) - 1)]
        if math.dist((x, y), goal_xy) < LOOKAHEAD_M:
            wp = goal_xy
        wi, wj = cell(*wp)
        wi, wj = wi - i0, wj - j0
        path = astar(blocked, (si - i0, sj - j0), lambda c: math.hypot(c[0] - wi, c[1] - wj) <= 3,
                     lambda c: math.hypot(c[0] - wi, c[1] - wj))
        dets = detections(seg, depth, labels)

        new_event = None
        _, step_cells, _ = dog_map.hazards(patch)
        step_cells &= ~dog_map.hole[patch[0]:patch[1], patch[2]:patch[3]]
        ahead = [cell(x + d * math.cos(yaw + a), y + d * math.sin(yaw + a))
                 for d in np.arange(0.5, 1.6, 0.1) for a in (-0.4, 0, 0.4)]
        curb_ahead = any(0 <= i - i0 < step_cells.shape[0] and 0 <= j - j0 < step_cells.shape[1]
                         and step_cells[i - i0, j - j0] for i, j in ahead)
        if curb_ahead and "curb" not in announced:
            announced.add("curb")
            new_event = ("CURB", "curb drop-off beside the path; keeping to the sidewalk",
                         "Curb edge close by. I'm keeping us on the sidewalk.")
            hazard_frames = 12
        for label, _, dist, cx in dets:
            if label in announced or dist > 3.0:
                continue
            if label.startswith("standing water"):
                new_event = ("WATER", "standing water from a clogged storm drain (311 flood report) ahead; going around",
                             "Standing water ahead from a clogged drain. Going around it.")
                hazard_frames = 12
            elif label in ("Waymo", "Waymo door handle"):
                if "Waymo" in announced:
                    continue
                label = "Waymo"
                new_event = ("WAYMO", "Waymo in view at the curb", "I can see your Waymo at the curb just ahead.")
            elif 0.2 < cx < 0.8:
                side = "left" if cx > 0.5 else "right"
                new_event = ("OBSTACLE", f"{label} ahead, {dist:.1f} m; steering {side}",
                             f"{label.capitalize()} ahead. Going around it on the {side}.")
            else:
                continue
            announced.add(label)
            break

        if math.dist((x, y), goal_xy) < 0.25:
            arrived, action = True, "ARRIVED"
        elif path is None or len(path) < 2:
            action = "STOP"
            stuck += 1
            if "blocked" not in announced:
                announced.add("blocked")
                new_event = ("BLOCKED", "no safe path yet; looking around", "Hold on, I'm finding a safe way.")
            yaw += MAX_TURN  # look around to fill in the map
        else:
            stuck = 0
            ahead_cell = path[min(5, len(path) - 1)]
            tgt = center(i0 + ahead_cell[0], j0 + ahead_cell[1])
            want = math.atan2(tgt[1] - y, tgt[0] - x)
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += max(-MAX_TURN, min(MAX_TURN, err))
            action = "FORWARD" if abs(err) < 0.15 else ("TURN_LEFT" if err > 0 else "TURN_RIGHT")
            if abs(err) < 0.9:
                # Stride toward the next planned cell (already cleared), not blindly along the heading.
                next_cell = path[min(2, len(path) - 1)]
                near = center(i0 + next_cell[0], j0 + next_cell[1])
                d = math.dist(near, (x, y)) or 1.0
                nx_, ny_ = x + STEP_M * (near[0] - x) / d, y + STEP_M * (near[1] - y) / d
                here, there = dog_map.height[cell(x, y)], dog_map.height[cell(nx_, ny_)]
                if np.isnan(here) or np.isnan(there) or abs(there - here) <= 0.05:
                    x, y = nx_, ny_
                    trail.add((x, y))

        if arrived:
            hx, hy = trail.at(trail.cum[-1] - HANDLER_LAG_M)
            hyaw = trail.heading(trail.cum[-1] - HANDLER_LAG_M)
            side = "right" if math.sin(math.atan2(hy_door - hy, hx_door - hx) - hyaw) < 0 else "left"
            new_event = ("ARRIVED", f"arrived; Waymo door handle on the passenger's {side}",
                         f"We're at your Waymo. The door is on your {side}, handle at waist height, "
                         "curb edge just past it.")

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
            create_box([x, y, dz + 0.01], [0.05, 0.05, 0.003], [0.25, 0.55, 1.0, 1])  # dog's trail

        if new_event:
            kind, event, onboard = new_event
            spoken = narrator.speak(cam, event) if narrator else None
            speech, source = (spoken, "Gemini") if spoken else (onboard, "onboard")
            events.append({"step": step, "kind": kind, "event": event, "speech": speech, "source": source,
                           "dog": [round(x, 2), round(y, 2)]})
            print(f"[{step:4d}] {kind:8s} {event}  ->  {speech}")

        hazard = hazard_frames > 0
        hazard_frames -= 1
        if new_event or step % SAVE_EVERY == 0 or arrived:
            stats = f"{flood_note}  |  to Waymo {math.dist((x, y), goal_xy):.0f} m"
            frames.append(compose(chase_camera(world, x, y, yaw), cam, dets,
                                  draw_local_map(dog_map, patch, blocked, path, (x, y), trail),
                                  draw_route_map(world, route, (x, y), trail, (hx_door, hy_door)),
                                  action, speech, source, hazard, stats))
            durations.append(EVENT_MS if new_event else STEP_MS)
        if arrived or stuck > 40:
            break

    durations[-1] = 3500
    gif_path = args.output / "gemini_waymo_escort.gif"
    frames[0].save(args.output / "first_frame.png")
    frames[-1].save(args.output / "last_frame.png")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    report = {"mission": "Autonomous guide-dog escort to Waymo, NE 21st St, Miami", "arrived": arrived,
              "steps": step + 1, "walked_m": round(trail.cum[-1] - 1.0, 1), "osm_route_m": round(float(route_cum[-1]), 1),
              "max_single_step_drop_m": round(max_drop, 3), "pickup": cho["road"], "events": events}
    (args.output / "mission_report.json").write_text(json.dumps(report, indent=2))
    p.disconnect()
    print(f"arrived={arrived} frames={len(frames)} walked={report['walked_m']} m max_step_drop={max_drop:.3f} m "
          f"-> {gif_path}")
    # A single-stride drop bigger than a sidewalk undulation means the dog walked off a curb.
    assert max_drop < 0.05, "dog stepped off a curb"
    assert arrived, "dog did not reach the Waymo"


if __name__ == "__main__":
    main()
