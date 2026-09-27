"""
Final approach to the pickup curb, simulated in Waymax (github.com/waymo-research/waymax, Waymax License
Agreement for Non-Commercial Use).

No Waymo Open Motion Dataset: the scenario is built from our own data.
- Roadgraph: road edges and lane centres of every OSM drive way within ROADGRAPH_R of the stop, offset from
  the OSM centreline by the way's lane count x LANE_W. Edge points that fall inside another road's carriageway
  (intersections) are dropped so the junction itself is not "offroad".
- SDC: the Waymo (Jaguar I-PACE footprint). Its reference trajectory is ours: the last PULLIN_M of the router's
  path in the rightmost lane, easing over the last EASE_M to a stop CURB_GAP_M off the curb, with a speed profile
  capped by the speed limit, A_LAT_MAX in turns and A_LONG_MAX braking.
- No other agents: there is no real traffic data for this street, so none are invented.
Waymax's expert actor inverts that reference into bicycle-model controls (acceleration, curvature, clipped to
the model's limits) at every 0.1 s step, and InvertibleBicycleModel integrates them; the car's pose comes from
that rollout, not from the reference. Waymax's own metrics are computed at every step.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from waymax import agents, config, datatypes, dynamics, env

# Starts just past the last turn: OSM has no kerb-return geometry, so the square inside corner of a turn would
# read as offroad in Waymax even for a turn a real car makes.
PULLIN_M = 80.0
EASE_M = 25.0
TURN_SMOOTH_M = 6.0     # corner cutting at this spacing gives roughly a 6 m turning radius
TURN_R_M = 6.0          # I-PACE-class turning radius, sets the speed the car leaves a corner at
CURB_GAP_M = 0.4
CAR_L, CAR_W, CAR_H = 4.7, 1.9, 1.6
LANE_W = 3.3
ROADGRAPH_R = 160.0
A_LAT_MAX, A_LONG_MAX = 2.0, 1.5
DT = 0.1
DWELL_S = 2.0
ROAD_EDGE = int(datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
LANE_SURFACE_STREET = int(datatypes.MapElementIds.LANE_SURFACE_STREET)
METRICS = ("log_divergence", "overlap", "offroad", "kinematic_infeasibility", "sdc_wrongway", "sdc_progression",
           "sdc_off_route")


def _resample(pts, step):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, cum[-1], step).tolist() + [cum[-1]]
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1), np.array(s)


def _right_normals(p):
    d = np.gradient(p, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.stack([d[:, 1], -d[:, 0]], 1)


def _dist_to_polyline(q, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((q[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(q[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)


def _smooth(p, iters=4):
    """Chaikin corner cutting; keeps both end points so the stop pose is exact."""
    for _ in range(iters):
        q = np.empty((2 * len(p) - 2, 2))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def reference(center_pts, lanes, speed_mps):
    """Reference path + speed for the last PULLIN_M. center_pts/lanes/speed are per centreline vertex."""
    pts = np.asarray(center_pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    start = max(0.0, cum[-1] - PULLIN_M)
    c, s = _resample(pts, 0.5)
    keep = s >= start
    c, s = c[keep], s[keep] - start
    idx = np.clip(np.searchsorted(cum, s + start, side="right") - 1, 0, len(lanes) - 1)
    lane_n = np.asarray(lanes, float)[idx]
    vmax = np.asarray(speed_mps, float)[idx]
    # Rightmost lane centre, then ease to the curb: car side CURB_GAP_M from the curb face.
    offset = (lane_n - 1) * LANE_W / 2
    stop_off = lane_n[-1] * LANE_W / 2 - CURB_GAP_M - CAR_W / 2
    u = np.clip((s - (s[-1] - EASE_M)) / EASE_M, 0, 1)
    offset = offset + (stop_off - offset[-1]) * u * u * (3 - 2 * u)
    q = c + offset[:, None] * _right_normals(c)
    # Offsetting inside a turn folds the path into a loop; drop points nearer another leg of the centreline.
    q = q[_dist_to_polyline(q, c) >= offset - 0.2]
    coarse, _ = _resample(q, TURN_SMOOTH_M)
    path, s = _resample(_smooth(coarse), 0.5)
    d = np.gradient(path, axis=0)
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kappa = np.abs(np.gradient(yaw) / np.maximum(np.gradient(s), 1e-6))
    vmax = np.interp(s, np.linspace(0, s[-1], len(vmax)), vmax)
    v = np.minimum(vmax, np.sqrt(A_LAT_MAX / np.maximum(kappa, 1e-6)))
    # Entering right after a turn (heading change > 45 deg within 20 m), the car is still at cornering speed.
    s_pre = np.linspace(max(0.0, start - 20), start, 11)
    step = np.diff(np.stack([np.interp(s_pre, cum, pts[:, 0]), np.interp(s_pre, cum, pts[:, 1])], 1), axis=0)
    step = step[np.linalg.norm(step, axis=1) > 1e-6]
    if len(step) and np.ptp(np.unwrap(np.arctan2(step[:, 1], step[:, 0]))) > math.pi / 4:
        v[0] = min(v[0], math.sqrt(A_LAT_MAX * TURN_R_M))
    v[-1] = 0.0
    ds = np.diff(s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * A_LONG_MAX * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * A_LONG_MAX * ds[i - 1]))
    # Time-parametrise, then sample every DT and hold the stop for DWELL_S.
    t = np.concatenate([[0], np.cumsum(2 * ds / np.maximum(v[1:] + v[:-1], 1e-3))])
    tt = np.arange(0, t[-1] + DWELL_S, DT)
    ref = {k: np.interp(tt, t, a) for k, a in (("x", path[:, 0]), ("y", path[:, 1]), ("yaw", yaw), ("v", v))}
    ref["path"] = path
    return ref


def roadgraph(segments, xy, stop_xy):
    """Road-edge and lane-centre points of the drive ways near the stop, in Waymax's conventions."""
    near = [sg for sg in segments
            if min(math.dist(xy[sg["a"]], stop_xy), math.dist(xy[sg["b"]], stop_xy)) < ROADGRAPH_R]
    a = np.array([xy[sg["a"]] for sg in near])
    b = np.array([xy[sg["b"]] for sg in near])
    half = np.array([sg["lanes"] * LANE_W / 2 for sg in near])
    pts, dirs, types, ids = [], [], [], []
    next_id = 0
    for k, sg in enumerate(near):
        p, _ = _resample([a[k], b[k]], 0.5)
        if len(p) < 2:
            continue
        n = _right_normals(p)
        fwd = (b[k] - a[k]) / max(np.linalg.norm(b[k] - a[k]), 1e-6)
        # Right edge runs a->b and left edge b->a, so the road is always on the edge's left (Waymax's sign).
        for side, direction in ((1, fwd), (-1, -fwd)):
            e = p + side * half[k] * n
            e = e if side == 1 else e[::-1]
            ab = b - a
            t = np.clip(((e[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
            dist = np.linalg.norm(e[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1)
            dist[:, k] = np.inf
            inside = (dist < half[None] - 0.3).any(1)
            # Each unbroken run of edge points is its own polyline id: Waymax pairs a point with its predecessor.
            cuts = np.flatnonzero(np.diff(np.concatenate([[1], inside.astype(int), [1]])))
            for lo, hi in zip(cuts[::2], cuts[1::2]):
                if hi - lo > 1:
                    pts += list(e[lo:hi])
                    dirs += [direction] * (hi - lo)
                    types += [ROAD_EDGE] * (hi - lo)
                    ids += [next_id] * (hi - lo)
                    next_id += 1
        for i in range(sg["lanes"]):
            off = (i + 0.5) * LANE_W - half[k]
            pts += list(p + off * n)
            dirs += [fwd] * len(p)
            types += [LANE_SURFACE_STREET] * len(p)
            ids += [next_id] * len(p)
            next_id += 1
    return np.array(pts), np.array(dirs), np.array(types), np.array(ids)


def simulate(ref, rg):
    """Roll the SDC out in Waymax; returns the simulated trajectory and per-metric summaries."""
    t_n = len(ref["x"])

    def arr(v, dtype=jnp.float32):
        return jnp.asarray(np.asarray(v)[None], dtype=dtype)

    log = datatypes.Trajectory(
        x=arr(ref["x"]), y=arr(ref["y"]), z=arr(np.zeros(t_n)),
        vel_x=arr(ref["v"] * np.cos(ref["yaw"])), vel_y=arr(ref["v"] * np.sin(ref["yaw"])),
        yaw=arr(ref["yaw"]), valid=arr(np.ones(t_n, bool), jnp.bool_),
        timestamp_micros=arr(np.arange(t_n) * int(DT * 1e6), jnp.int32),
        length=arr(np.full(t_n, CAR_L)), width=arr(np.full(t_n, CAR_W)), height=arr(np.full(t_n, CAR_H)))
    one = jnp.ones(1, jnp.bool_)
    meta = datatypes.ObjectMetadata(ids=jnp.zeros(1, jnp.int32), object_types=jnp.ones(1, jnp.int32), is_sdc=one,
                                    is_modeled=one, is_valid=one, objects_of_interest=one, is_controlled=one)
    pts, dirs, types, ids = rg
    n = len(pts)
    road = datatypes.RoadgraphPoints(
        x=jnp.asarray(pts[:, 0], jnp.float32), y=jnp.asarray(pts[:, 1], jnp.float32), z=jnp.zeros(n, jnp.float32),
        dir_x=jnp.asarray(dirs[:, 0], jnp.float32), dir_y=jnp.asarray(dirs[:, 1], jnp.float32),
        dir_z=jnp.zeros(n, jnp.float32), types=jnp.asarray(types, jnp.int32), ids=jnp.asarray(ids, jnp.int32),
        valid=jnp.ones(n, jnp.bool_))
    lights = datatypes.TrafficLights(
        x=jnp.zeros((1, t_n), jnp.float32), y=jnp.zeros((1, t_n), jnp.float32), z=jnp.zeros((1, t_n), jnp.float32),
        state=jnp.zeros((1, t_n), jnp.int32), lane_ids=jnp.zeros((1, t_n), jnp.int32),
        valid=jnp.zeros((1, t_n), jnp.bool_))
    path = ref["path"]
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    m = len(path)
    paths = datatypes.Paths(
        x=jnp.asarray(path[None, :, 0], jnp.float32), y=jnp.asarray(path[None, :, 1], jnp.float32),
        z=jnp.zeros((1, m), jnp.float32), ids=jnp.zeros((1, m), jnp.int32), valid=jnp.ones((1, m), jnp.bool_),
        arc_length=jnp.asarray(arc[None], jnp.float32), on_route=jnp.ones((1, 1), jnp.bool_))
    state = datatypes.SimulatorState(sim_trajectory=log, log_trajectory=log, log_traffic_light=lights,
                                     object_metadata=meta, timestep=jnp.array(0), sdc_paths=paths,
                                     roadgraph_points=road)

    model = dynamics.InvertibleBicycleModel()
    sim = env.BaseEnvironment(dynamics_model=model, config=config.EnvironmentConfig(
        max_num_objects=1, init_steps=1, compute_reward=False,
        metrics=config.MetricsConfig(metrics_to_run=METRICS)))
    actor = agents.create_expert_actor(dynamics_model=model)
    step, select, metrics = jax.jit(sim.step), jax.jit(actor.select_action), jax.jit(sim.metrics)
    state = sim.reset(state)
    per_step = {k: [] for k in METRICS}
    while not bool(state.is_done):
        state = step(state, agents.merge_actions([select({}, state, None, None)]))
        for k, r in metrics(state).items():
            value, valid = np.ravel(r.value)[0], np.ravel(r.valid)[0]  # SDC-only metrics are scalars
            per_step[k].append(float(value) if valid else math.nan)
    traj = state.sim_trajectory
    out = {k: np.asarray(getattr(traj, k)[0]) for k in ("x", "y", "yaw", "vel_x", "vel_y")}
    out["speed"] = np.hypot(out["vel_x"], out["vel_y"])
    out["offroad"] = np.array([0.0] + per_step["offroad"])
    summary = {k: {"max": round(float(np.nanmax(v)), 3), "mean": round(float(np.nanmean(v)), 3)}
               for k, v in per_step.items()}
    summary["sdc_progression"]["final"] = round(per_step["sdc_progression"][-1], 3)
    return out, summary
