"""
Waymo SafePoint 3D: the real Edgewater block (Miami) as a PyBullet world for the guide-dog escort.

Everything placed here comes from simulator/real_block.py (OpenStreetMap + SafeRoute Miami data):
- streets at their OSM lane count (3.3 m per lane) sunk 0.15 m below the sidewalks, so every
  curb is a real drop-off; alleys and driveways (OSM service ways) are flush, as curb cuts are
- buildings extruded from their OSM footprints and heights
- trees, poles, bus stops and benches at their OSM positions
- standing water at the curb drain in front of each Miami-Dade 311 flooding report
- the Waymo parked at the curb chosen by simulator/pickup_choice.py, door facing the curb
The ground is one heightfield textured with a top-down render of the same raster.
"""

import heapq
import math
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

CURB_H = 0.15
LANE_W = 3.3
RES = 0.2  # heightfield / texture resolution (m)

SIDEWALK_RGB, ROAD_RGB, SERVICE_RGB = (196, 196, 200), (70, 72, 78), (150, 150, 155)
BUILDING_RGB, CROSSWALK_RGB, LINE_RGB = (186, 170, 150), (240, 240, 240), (230, 200, 60)


class RealBlock:
    """Rasterised block inside a window (x0, y0, x1, y1) in local metres."""

    def __init__(self, block, window):
        self.block = block
        self.x0, self.y0, self.x1, self.y1 = window
        self.nx = int(round((self.x1 - self.x0) / RES))
        self.ny = int(round((self.y1 - self.y0) / RES))
        self.height, self.texture = self._rasterise()

    def px(self, pt):
        """Local metres to raster pixel (column = x, row = y from the top)."""
        return (pt[0] - self.x0) / RES, (self.y1 - pt[1]) / RES

    def _rasterise(self):
        size = (self.nx, self.ny)
        tex = Image.new("RGB", size, SIDEWALK_RGB)
        road = Image.new("L", size, 0)
        heights = Image.new("F", size, CURB_H)
        dt, dr, dh = ImageDraw.Draw(tex), ImageDraw.Draw(road), ImageDraw.Draw(heights)
        for r in self.block["roads"]:
            pts = [self.px(q) for q in r["pts"]]
            service = r["highway"] == "service"
            w = max(1, int(r["lanes"] * LANE_W / RES))
            dt.line(pts, fill=SERVICE_RGB if service else ROAD_RGB, width=w, joint="curve")
            if not service:
                dr.line(pts, fill=255, width=w, joint="curve")
                for q in pts:
                    dr.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=255)
                    dt.ellipse([q[0] - w / 2, q[1] - w / 2, q[0] + w / 2, q[1] + w / 2], fill=ROAD_RGB)
        for r in self.block["roads"]:
            if r["highway"] != "service" and r["lanes"] >= 2:
                dt.line([self.px(q) for q in r["pts"]], fill=LINE_RGB, width=1)
        for c in self.block["crossings"]:
            dt.line([self.px(q) for q in c["pts"]], fill=CROSSWALK_RGB, width=int(3.0 / RES))
        heights.paste(0.0, mask=road)
        for b in self.block["buildings"]:
            if len(b["pts"]) > 2:
                poly = [self.px(q) for q in b["pts"]]
                dt.polygon(poly, fill=BUILDING_RGB, outline=(120, 110, 100))
                dh.polygon(poly, fill=CURB_H + b["height"])
        # Rows of the raster run north to south; flip so index [i, j] is (x, y) increasing.
        h = np.asarray(heights, dtype=np.float32)[::-1].T.copy()
        return h, tex

    def ground_height(self, x, y):
        i = int(np.clip((x - self.x0) / RES, 0, self.nx - 1))
        j = int(np.clip((y - self.y0) / RES, 0, self.ny - 1))
        return float(self.height[i, j])

    def is_road(self, x, y):
        return self.ground_height(x, y) < CURB_H / 2

    def build(self):
        """Create the heightfield body; returns its id."""
        nx, ny = self.nx, self.ny
        data = self.height.flatten(order="F").tolist()  # index i + j * rows, rows along x
        shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, meshScale=[RES, RES, 1], heightfieldData=data,
                                       numHeightfieldRows=nx, numHeightfieldColumns=ny)
        zmid = (float(self.height.max()) + float(self.height.min())) / 2
        body = p.createMultiBody(0, shape, basePosition=[(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, zmid])
        # The heightfield texture runs x right-to-left and y bottom-to-top relative to the raster.
        with tempfile.TemporaryDirectory() as tmp:
            tex_path = Path(tmp) / "ground.png"
            self.texture.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM).save(tex_path)
            p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(str(tex_path)), rgbaColor=[1, 1, 1, 1])
        return body

    def walkable(self, res):
        """Grid (cell size res) of sidewalk-level ground, kept one cell clear of curbs and walls."""
        k = int(round(res / RES))
        assert abs(k * RES - res) < 1e-9, f"walk grid {res} m must be a multiple of the {RES} m raster"
        nx, ny = self.nx // k, self.ny // k
        coarse = self.height[: nx * k, : ny * k].reshape(nx, k, ny, k)
        ok = (coarse.max(axis=(1, 3)) <= CURB_H + 0.01) & (coarse.min(axis=(1, 3)) >= CURB_H - 0.01)
        near_edge = np.zeros_like(ok)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near_edge[max(dx, 0):nx + min(dx, 0), max(dy, 0):ny + min(dy, 0)] |= \
                ~ok[max(-dx, 0):nx - max(dx, 0), max(-dy, 0):ny - max(dy, 0)]
        return ok & ~near_edge

    def walk_distances(self, start, res):
        """Walking distance (m) from start to every cell of walkable(res); inf where unreachable
        without stepping into the roadway (there is no crossing data at mid-block curbs)."""
        ok = self.walkable(res)
        dist = np.full(ok.shape, np.inf)
        s = (int((start[0] - self.x0) / res), int((start[1] - self.y0) / res))
        dist[s] = 0.0
        heap = [(0.0, s)]
        moves = [(1, 0, res), (-1, 0, res), (0, 1, res), (0, -1, res)] + \
            [(dx, dy, res * math.sqrt(2)) for dx in (1, -1) for dy in (1, -1)]
        while heap:
            d, (i, j) = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for dx, dy, c in moves:
                n = (i + dx, j + dy)
                if 0 <= n[0] < ok.shape[0] and 0 <= n[1] < ok.shape[1] and ok[n] and d + c < dist[n]:
                    dist[n] = d + c
                    heapq.heappush(heap, (d + c, n))
        return dist

    def nearest_sidewalk(self, xy):
        """Closest point on an OSM sidewalk way."""
        best = None
        for way in self.block["sidewalks"]:
            for a, b in zip(way["pts"], way["pts"][1:]):
                ab = (b[0] - a[0], b[1] - a[1])
                t = max(0.0, min(1.0, ((xy[0] - a[0]) * ab[0] + (xy[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2 or 1)))
                q = (a[0] + ab[0] * t, a[1] + ab[1] * t)
                if self.inside(q) and (best is None or math.dist(q, xy) < math.dist(best, xy)):
                    best = q
        return best

    def inside(self, xy, margin=1.0):
        return self.x0 + margin < xy[0] < self.x1 - margin and self.y0 + margin < xy[1] < self.y1 - margin


def create_box(position, half_extents, rgba_color, orientation=None):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba_color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=position,
                             baseOrientation=orientation or [0, 0, 0, 1])


def create_cylinder(position, radius, length, rgba_color):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba_color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=position)


def create_sphere(position, radius, rgba_color):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=position)


def nearest_curb(world, xy, search=25.0):
    """Nearest sidewalk-side curb point to xy and the unit normal pointing from road to sidewalk."""
    best = None
    for a in np.arange(0, 2 * math.pi, math.pi / 36):
        for d in np.arange(0.2, search, RES):
            q = (xy[0] + d * math.cos(a), xy[1] + d * math.sin(a))
            if world.is_road(*q) != world.is_road(*xy):
                if best is None or d < best[0]:
                    best = (d, q, a)
                break
    assert best, f"no curb within {search} m of {xy}"
    d, q, a = best
    n = (-math.cos(a), -math.sin(a)) if not world.is_road(*xy) else (math.cos(a), math.sin(a))
    return q, n


def build_scene(world, pickup):
    """Ground, furniture, standing water and the parked Waymo. Returns labels, water ids, door handle, car pose."""
    labels = {}
    world.build()
    for f in world.block["furniture"]:
        x, y = f["xy"]
        if not world.inside((x, y)):
            continue
        z = world.ground_height(x, y)
        kind = f["kind"]
        if kind == "tree":
            ids = [create_cylinder([x, y, z + 1.5], 0.18, 3.0, [0.4, 0.28, 0.18, 1]),
                   create_sphere([x, y, z + 3.6], 1.6, [0.2, 0.5, 0.2, 1])]
        elif kind in ("pole", "utility_pole", "street_lamp"):
            ids = [create_cylinder([x, y, z + 4.0], 0.12, 8.0, [0.45, 0.45, 0.45, 1])]
        elif kind == "bus_stop":
            kind = "bus stop sign"
            ids = [create_cylinder([x, y, z + 1.25], 0.05, 2.5, [0.5, 0.5, 0.52, 1]),
                   create_box([x, y, z + 2.4], [0.04, 0.3, 0.2], [0.1, 0.35, 0.75, 1])]
        elif kind == "bench":
            ids = [create_box([x, y, z + 0.45], [0.8, 0.25, 0.04], [0.5, 0.33, 0.18, 1])]
        else:
            ids = [create_cylinder([x, y, z + 0.4], 0.15, 0.8, [0.6, 0.6, 0.6, 1])]
        labels.update({i: kind.replace("_", " ") for i in ids})

    # 311 "drain clogged" tickets are filed at a street address; the water pools at the curb drain in front.
    water = []
    for f in world.block["flooding"]:
        if not world.inside(f["xy"], margin=4):
            continue
        (cx, cy), n = nearest_curb(world, f["xy"])
        body = create_cylinder([cx, cy, CURB_H + 0.004], 1.6, 0.008, [0.25, 0.45, 0.75, 1])
        labels[body] = "standing water (311 flood report)"
        water.append(body)

    # Waymo at the chosen curb, facing so the curb is on its right (passenger) side.
    (kx, ky), n = pickup["chosen"]["xy"], pickup["chosen"]["side_normal"]  # n points from the road to the curb
    yaw = math.atan2(n[0], -n[1])
    # The curb point sits 0.3 m onto the sidewalk; park 0.4 m off the curb (car half-width 0.95 m).
    cx, cy = kx - n[0] * 1.65, ky - n[1] * 1.65
    first = p.getNumBodies()
    handle = build_waymo_car(cx, cy, 0.32, yaw)
    labels.update({p.getBodyUniqueId(i): "Waymo" for i in range(first, p.getNumBodies())})
    labels[p.getBodyUniqueId(p.getNumBodies() - 1)] = "Waymo door handle"
    return labels, water, handle, (cx, cy, yaw), n


def _build_parts(parts, x, y, z, yaw):
    """Place boxes given in the actor's local frame (x forward, y left) at a world pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    ori = p.getQuaternionFromEuler([0, 0, yaw])
    return [create_box([x + lx * c - ly * s, y + lx * s + ly * c, z + lz], half, rgba, orientation=ori)
            for (lx, ly, lz), half, rgba in parts]


def build_waymo_car(x, y, z, yaw):
    """Waymo (Jaguar I-PACE) with rooftop LiDAR; the door handle, created last, is on the right side."""
    white, glass, black = [0.96, 0.96, 0.98, 1], [0.1, 0.12, 0.15, 1], [0.12, 0.12, 0.13, 1]
    parts = [((0, 0, 0.25), [2.35, 0.95, 0.35], white),
             ((-0.1, 0, 0.72), [1.3, 0.85, 0.2], glass),
             ((-0.1, 0, 0.94), [1.2, 0.8, 0.04], white),
             ((0.05, 0, 1.06), [0.3, 0.3, 0.08], [0.2, 0.2, 0.22, 1]),
             ((0.05, 0, 1.2), [0.2, 0.2, 0.08], [0.0, 0.8, 0.78, 1])]
    parts += [((wx, wy, 0.0), [0.36, 0.13, 0.36], black) for wx in (1.45, -1.45) for wy in (0.9, -0.9)]
    parts += [((-0.3, -0.96, 0.42), [0.15, 0.02, 0.05], [0.2, 0.7, 1.0, 1])]
    _build_parts(parts, x, y, z, yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + (-0.3) * c - (-0.96) * s, y + (-0.3) * s + (-0.96) * c


def build_person(x, y, z, yaw):
    """Visually impaired passenger facing yaw, left arm forward to hold the harness."""
    pants, jacket = [0.15, 0.18, 0.25, 1], [0.8, 0.25, 0.2, 1]
    return _build_parts([
        ((0, -0.12, 0.4), [0.1, 0.1, 0.4], pants),
        ((0, 0.12, 0.4), [0.1, 0.1, 0.4], pants),
        ((0, 0, 1.1), [0.16, 0.24, 0.35], jacket),
        ((0, 0, 1.6), [0.12, 0.12, 0.14], [0.85, 0.7, 0.6, 1]),
        ((0.05, 0, 1.63), [0.09, 0.13, 0.03], [0.05, 0.05, 0.05, 1]),  # dark glasses
        ((0.2, 0.28, 0.85), [0.2, 0.05, 0.05], jacket),
    ], x, y, z, yaw)


def build_robot_dog(x, y, z, yaw):
    """Hiwonder MechDog body with high-vis guide harness, facing yaw."""
    parts = [
        ((0, 0, 0.22), [0.22, 0.14, 0.09], [0.12, 0.45, 0.85, 1]),
        ((0.22, 0, 0.26), [0.07, 0.10, 0.07], [0.1, 0.1, 0.1, 1]),
        ((0.29, -0.03, 0.27), [0.01, 0.02, 0.02], [0.2, 0.8, 1.0, 1]),
        ((0.29, 0.03, 0.27), [0.01, 0.02, 0.02], [0.2, 0.8, 1.0, 1]),
        ((0, 0, 0.32), [0.2, 0.15, 0.02], [0.95, 0.75, 0.1, 1]),
    ]
    parts += [((lx, ly, 0.11), [0.04, 0.04, 0.11], [0.15, 0.15, 0.18, 1])
              for lx, ly in [(0.16, 0.15), (0.16, -0.15), (-0.16, 0.15), (-0.16, -0.15)]]
    return _build_parts(parts, x, y, z, yaw)


def build_handle(a, b):
    """Rigid harness handle: a thin bar from point a (dog's back) to point b (handler's hand)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    horiz = math.hypot(dx, dy)
    ori = p.getQuaternionFromEuler([0, -math.atan2(dz, horiz), math.atan2(dy, dx)])
    mid = [(a[i] + b[i]) / 2 for i in range(3)]
    return [create_box(mid, [math.hypot(horiz, dz) / 2, 0.015, 0.015], [0.95, 0.75, 0.1, 1], orientation=ori)]
