"""16:9 video of the real Biscayne Blvd & NE 19th St crossing (real_crosswalk_demo.py) with traffic.

Runs the unchanged demo (same checks, report, gif) and, through its frame hook, renders a chase camera every
0.1 s of simulated time. Adds the dog-camera inset with the map-guided signal ROI and the model's WALK / DON'T WALK
confidence, the time-left countdown, a caption bar, a title card and an end card. Writes crossing.webm + .mp4.

    .venv/bin/python -m simulator.crossing_video < /dev/null  # needs the meshes from ASSETS.md
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__:
    from . import real_crosswalk_demo as demo
    from .video import write_video
else:
    import real_crosswalk_demo as demo
    from video import write_video

W, H = 1280, 720
MAIN_H = 600  # main camera; the caption bar fills the rest
FOV = 66
TAIL_S = 2.0  # seconds of video kept after the dog reaches the far ramp
FRAME_MS = 100  # one rendered frame per 0.1 s of simulated time: real-time playback
CYAN, GREEN, RED, AMBER, WHITE, GREY = (0, 220, 255), (60, 225, 110), (255, 80, 50), (255, 190, 40), (240, 240, 240), (170, 170, 175)
PHASE_RGB = {"green": GREEN, "amber": AMBER, "red": RED}


def font(size: int):
    return ImageFont.load_default(size=size)


def unit(a, b):
    d = (b[0] - a[0], b[1] - a[1])
    n = math.hypot(*d)
    return d[0] / n, d[1] / n


def chase(site, s: float):
    """Behind the start kerb on the dog's left (the handler walks on its right), sliding along at 85% of its progress."""
    start, d = site.pts[0], unit(site.pts[0], site.pts[-1])
    r = (d[1], -d[0])
    a = 0.85 * s
    eye = [start[0] + d[0] * (a - 7) - r[0] * 4, start[1] + d[1] * (a - 7) - r[1] * 4, 4.5]
    target = [start[0] + d[0] * (a + 9) + r[0] * 1.5, start[1] + d[1] * (a + 9) + r[1] * 1.5, 0.0]
    return eye, target


def render_main(eye, target) -> Image.Image:
    """Chase view; TinyRenderer leaves the background white, so paint a sky where nothing was hit."""
    import pybullet as p
    view = p.computeViewMatrix(eye, target, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(FOV, W / MAIN_H, 0.05, 200)
    _, _, rgba, _, seg = p.getCameraImage(W, MAIN_H, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgba, np.uint8).reshape(MAIN_H, W, 4)[..., :3]
    sky = np.linspace([95, 150, 210], [200, 222, 240], MAIN_H).astype(np.uint8)[:, None, :]
    return Image.fromarray(np.where((np.asarray(seg).reshape(MAIN_H, W) < 0)[..., None], sky, rgb))


def project(point, eye, target):
    """Pixel of a world point in the main view, or None when it is behind the camera."""
    import pybullet as p
    view = np.array(p.computeViewMatrix(eye, target, [0, 0, 1])).reshape(4, 4).T
    proj = np.array(p.computeProjectionMatrixFOV(FOV, W / MAIN_H, 0.05, 200)).reshape(4, 4).T
    clip = proj @ view @ np.array([*point, 1.0])
    if clip[3] <= 0:
        return None
    return (clip[0] / clip[3] + 1) / 2 * W, (1 - clip[1] / clip[3]) / 2 * MAIN_H


def tag(d, xy, text, color):
    if xy is None:
        return
    x, y = xy
    f = font(16)
    w = d.textlength(text, font=f)
    d.line([x, y, x, y - 34], fill=color, width=2)
    d.rounded_rectangle([x - w / 2 - 6, y - 58, x + w / 2 + 6, y - 34], radius=5, fill=(0, 0, 0))
    d.text((x - w / 2, y - 55), text, fill=color, font=f)


def dog_cam_panel(rec) -> Image.Image:
    """Dog camera (downscaled) with model boxes and the cyan ROI, plus the ROI crop and the two confidences."""
    cam, roi, fv = rec["cam"], rec["roi"], rec["fv"]
    pw, ph = 400, 225
    k = pw / demo.CAM_W
    panel = Image.new("RGB", (pw + 8, ph + 34 + 164), (10, 10, 14))
    panel.paste(cam.resize((pw, ph)), (4, 34))
    d = ImageDraw.Draw(panel)
    d.text((8, 7), "DOG CAMERA (Brio 105, 1080p) + YOLO, 5 fps", fill=WHITE, font=font(16))
    for det in rec["detections"]:
        color = {"ped_signal_walk": GREEN, "ped_signal_stop": RED}.get(det["class"], GREY)
        b = det["box"]
        d.rectangle([4 + b[0] * k, 34 + b[1] * k, 4 + b[2] * k, 34 + b[3] * k], outline=color, width=2)
    if roi is not None:
        d.rectangle([4 + roi[0] * k, 34 + roi[1] * k, 4 + roi[2] * k, 34 + roi[3] * k], outline=CYAN, width=2)
        crop = cam.crop(roi).resize((150, 150))
        cd = ImageDraw.Draw(crop)
        rk = 150 / (roi[2] - roi[0])
        for det in rec["detections"]:
            if det.get("roi"):
                b = det["box"]
                color = GREEN if det["class"] == "ped_signal_walk" else RED
                cd.rectangle([(b[0] - roi[0]) * rk, (b[1] - roi[1]) * rk, (b[2] - roi[0]) * rk, (b[3] - roi[1]) * rk],
                             outline=color, width=2)
        panel.paste(crop, (4, ph + 42))
        d.rectangle([3, ph + 41, 155, ph + 193], outline=CYAN, width=2)
    else:
        d.text((12, ph + 100), "signal head\nout of view", fill=GREY, font=font(16))
    x0, y0 = 166, ph + 40
    d.text((x0, y0), "map-guided signal ROI crop", fill=CYAN, font=font(15))
    for row, (label, conf, color) in enumerate((("DON'T WALK", fv["stop_conf"], RED), ("WALK", fv["walk_conf"], GREEN))):
        y = y0 + 22 + row * 38
        d.text((x0, y), f"{label} {conf:.2f}", fill=color, font=font(17))
        d.rectangle([x0, y + 21, x0 + 230, y + 30], outline=GREY)
        d.rectangle([x0, y + 21, x0 + 230 * conf, y + 30], fill=color)
    vote = {"walk": ("vote: WALK", GREEN), "dont_walk": ("vote: DON'T WALK", RED)}.get(fv["vote"], ("vote: unknown", GREY))
    d.text((x0, y0 + 100), vote[0], fill=vote[1], font=font(18))
    streak = rec["ctl"]["streak"] if rec["ctl"]["state"] == "waiting" else demo.CONFIRM_FRAMES
    d.text((x0, y0 + 126), f"WALK confirm {streak}/{demo.CONFIRM_FRAMES} frames", fill=WHITE, font=font(16))
    return panel


def hud(d, rec):
    """Top-left: the simulated signal state (ground truth; the dog never reads it)."""
    x = W - 420
    d.rounded_rectangle([x - 12, 12, W - 12, 118], radius=8, fill=(0, 0, 0))
    d.text((x, 18), f"t = {rec['t']:5.1f} s   SIMULATED SIGNAL (not seen by dog)", fill=GREY, font=font(14))
    color = PHASE_RGB[rec["phase"]]
    d.ellipse([x, 44, x + 20, 64], fill=color)
    d.text((x + 30, 43), f"Biscayne cars: {rec['phase'].upper()}", fill=color, font=font(19))
    ped = {"walk": ("WALK", GREEN), "clearance": ("DON'T WALK - clearance", AMBER), "dont_walk": ("DON'T WALK", RED)}[rec["ped"]]
    d.text((x, 72), f"Ped signal: {ped[0]}", fill=ped[1], font=font(19))
    d.text((x, 97), "Robot dog, handler, cars simulated (PyBullet)", fill=GREY, font=font(13))


def countdown(d, rec):
    """Bottom-left: true WALK + clearance left vs the time the pair still needs at 0.9 m/s."""
    site = rec["site"]
    total = rec["timing"]["walk_s"] + rec["timing"]["ped_clearance_s"]
    left = rec["true_left_s"]
    need = max(0.0, site.length - rec["s"]) / demo.ESCORT_SPEED
    x, y = 12, MAIN_H - 112
    d.rounded_rectangle([x, y, x + 470, y + 100], radius=8, fill=(0, 0, 0))
    scale = 250 / total
    for row, (label, value, color) in enumerate((("WALK+clear left", left, CYAN), ("needed @0.9m/s", need, WHITE))):
        yy = y + 12 + row * 34
        d.text((x + 10, yy), label, fill=color, font=font(15))
        d.rectangle([x + 128, yy + 2, x + 128 + total * scale, yy + 18], outline=(80, 80, 85))
        d.rectangle([x + 128, yy + 2, x + 128 + min(value, total) * scale, yy + 18], fill=color)
        shown = f"{value:4.1f} s" if value or row else "--"
        d.text((x + 390, yy - 1), shown, fill=color, font=font(17))
    if rec["ctl"]["state"] == "waiting":
        note, color = "standard timing, not measured", GREY
    else:
        spare = left - need
        note, color = (f"margin {spare:+.1f} s", GREEN if spare > 0 else RED) if rec["s"] < site.length else ("off the road", GREEN)
    d.text((x + 10, y + 78), note, fill=color, font=font(15))


def caption(rec, state_text, speech, context):
    bar = Image.new("RGB", (W, H - MAIN_H), (14, 14, 20))
    d = ImageDraw.Draw(bar)
    d.text((20, 10), state_text[0], fill=state_text[1], font=font(26))
    d.text((20, 48), f"Dog says: “{speech}”" if speech else "", fill=(255, 235, 170), font=font(21))
    d.text((20, 88), context, fill=(140, 200, 240), font=font(16))
    return bar


def stage(rec, first_cross_t, arrive_t):
    """Caption state line, the latest guidance phrase, and a line of real-data context for this moment."""
    site, st, s = rec["site"], rec["ctl"]["state"], rec["s"]
    facts = f"{site.length:.1f} m, {site.road['lanes']} lanes, {site.road.get('maxspeed', '')}"
    if st == "waiting":
        if rec["ctl"]["streak"]:
            return ((f"WALK seen on {rec['ctl']['streak']}/{demo.CONFIRM_FRAMES} frames - confirming before stepping off", GREEN),
                    "Wait.", "Dog needs 3 consecutive WALK frames AND enough WALK + clearance time left to cover the crossing")
        moving = rec["phase"] != "red"
        return ((f"Holding at the kerb ramp: DON'T WALK{' - traffic moving through' if moving else ', cars stopping'}", AMBER),
                "Wait. Traffic.", f"REAL (OpenStreetMap): crossing {facts}; lowered kerbs + tactile paving both ends")
    if st == "crossing":
        lane = min(site.road["lanes"], int(s / (site.length / site.road["lanes"])) + 1)
        speech = "Walk sign. Forward." if s < site.length / 2 else "Halfway."
        if s >= site.length - demo.RAMP_LEN:
            speech = "Kerb ramp ahead."
        ctx = ("Signal timing: MUTCD standard, not measured - WALK 7 s + clearance 16.9 s (18.07 m / 1.07 m/s)"
               if rec["t"] - first_cross_t < 10 else
               "REAL (FDOT): 3 crashes within 40 m (2018-19); FDOT traffic-signal work zone 17.6 m away")
        on_road = s < site.length
        where = f" lane {lane} of {site.road['lanes']}" if on_road else " - up the far kerb ramp"
        return ((f"CROSSING{where} - cars held on red", GREEN), speech, ctx)
    return (("ARRIVED on the far kerb ramp - dog stops, handler beside it", GREEN), "Stop. Kerb.",
            "REAL: far kerb lowered with tactile paving (OSM kerb node)")


def frame(rec, panel, first_cross_t, arrive_t, banner=None) -> Image.Image:
    out = Image.new("RGB", (W, H))
    main = rec["main"].copy()
    d = ImageDraw.Draw(main)
    pose, eye, target = rec["pose"], rec["eye"], rec["target"]
    tag(d, project([pose["x"], pose["y"], pose["z"] + 0.35], eye, target), "guide dog", AMBER)
    tag(d, project([pose["hx"], pose["hy"], 1.9], eye, target), "blind handler", WHITE)
    placed = []
    for xy in rec["held"]:  # one tag per queue: neighbouring lanes project a few pixels apart
        q = project([xy[0], xy[1], 1.4], eye, target)
        if q and all(abs(q[0] - o[0]) > 160 for o in placed):
            placed.append(q)
            tag(d, q, "cars stopped at the line", RED)
    hud(d, rec)
    countdown(d, rec)
    if banner:
        f = font(28)
        w = d.textlength(banner, font=f)
        d.rounded_rectangle([W / 2 - w / 2 - 20, 250, W / 2 + w / 2 + 20, 300], radius=10, fill=(0, 90, 40))
        d.text((W / 2 - w / 2, 258), banner, fill=WHITE, font=f)
    out.paste(main, (0, 0))
    out.paste(panel, (W - panel.width - 8, MAIN_H - panel.height - 8))
    state_text, speech, ctx = stage(rec, first_cross_t, arrive_t)
    out.paste(caption(rec, state_text, speech, ctx), (0, MAIN_H))
    return out


def card(lines, footer) -> Image.Image:
    im = Image.new("RGB", (W, H), (12, 14, 20))
    d = ImageDraw.Draw(im)
    y = 90
    for text, size, color in lines:
        if text:
            d.text((90, y), text, fill=color, font=font(size))
        y += size + 22
    d.text((90, H - 70), footer, fill=GREY, font=font(16))
    return im


def title_card(facts) -> Image.Image:
    t = facts["timing_standard_not_measured"]
    zone = facts["work_zones_within_30m"][0]["distance_m"] if facts["work_zones_within_30m"] else None
    return card([
        ("Crossing Biscayne Blvd with traffic", 48, WHITE),
        ("Robot guide dog + blind handler at US 1 & NE 19th St, Edgewater, Miami", 26, (200, 210, 225)),
        ("", 6, WHITE),
        ((f"REAL  crossing {facts['crossing_length_m']:.1f} m over {facts['lanes']} lanes "
         f"({facts['lanes_backward']} SB + {facts['lanes_forward']} NB), {facts['maxspeed']}, signalized, "
         "kerb ramps + tactile paving both ends  (OpenStreetMap)"), 20, (140, 220, 255)),
        (f"REAL  {facts['crashes_within_40m']} FDOT crashes within 40 m ({facts['pedestrian_crashes_within_40m']} pedestrian, "
         f"{'-'.join(str(y) for y in facts['crash_years'])})"
         + (f";  FDOT traffic-signal work zone {zone:.1f} m away" if zone is not None else ""), 20, (255, 205, 120)),
        ((f"STANDARD, NOT MEASURED  signal timing: WALK {t['walk_s']:.0f} s + clearance {t['ped_clearance_s']:.1f} s "
         f"(MUTCD), amber {t['amber_s']:.1f} s (ITE)"), 20, (230, 230, 230)),
        ("SIMULATED  robot dog, handler, cars, signal face (PyBullet). Dog moves only on the YOLO model's WALK votes.", 20, GREY),
    ], "Data: OpenStreetMap, SafeRoute Miami (FDOT crashes, FDOT work zones, Miami-Dade 311). Model: blind_escort_yolo (ONNX).")


def end_card(report) -> Image.Image:
    cs = report["crossing_start"]
    spare = report["ped_clearance_end_s"] - report["dog_left_road_s"]
    checks = report["checks"]
    return card([
        ("Outcome", 48, WHITE),
        ((f"Waited {cs['seconds']:.1f} s on DON'T WALK: {report['model_grading_while_waiting']['non_walk_votes_on_dont_walk']} "
         f"frames, 0 false WALK votes"), 24, (230, 230, 230)),
        ((f"WALK confirmed {report['detection_latency_s']:.1f} s after it lit (3 frames); started with "
         f"{cs['true_walk_plus_clearance_left_s']:.1f} s left vs {cs['time_needed_s']:.1f} s needed"), 24, (230, 230, 230)),
        ((f"Crossed {report['site']['crossing_length_m']:.1f} m in {report['escort_crossing_time_s']:.1f} s at "
         f"{report['escort_speed_mps']} m/s; off the road {spare:.1f} s before clearance ended"), 24, GREEN),
        (f"Closest car (centre) while in the road: {report['min_car_center_distance_in_road_m']:.1f} m, held at the stop bar",
         24, (230, 230, 230)),
        (f"Self-checks: {sum(checks.values())}/{len(checks)} passed", 24, GREEN if report["passed"] else RED),
        ((f"MechDog prototype tops out at {report['mechdog_top_speed_mps']} m/s: it would need "
         f"{report['mechdog_crossing_time_s']:.0f} s, but WALK + clearance is {report['walk_plus_clearance_s']:.1f} s"), 24, AMBER),
    ], "Synthetic signal face, ROI margin tuned on this scene: shows the control loop at real geometry, not real-street safety.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-seconds", type=float, default=None,
                        help="stop the video this many simulated seconds in (quick framing cuts); default: arrival + 2 s")
    args, rest = parser.parse_known_args()
    records = []
    arrived = []

    def hook(ctx):
        if args.video_seconds is not None and ctx["t"] > args.video_seconds + 1e-6:
            return
        if arrived and ctx["t"] > arrived[0] + TAIL_S + 1e-6:
            return
        eye, target = chase(ctx["site"], ctx["s"])
        rec = {**ctx, "eye": eye, "target": target, "main": render_main(eye, target),
               "held": [ctx["cars"].xy(i) for i, (_, _, stop) in enumerate(ctx["cars"].lanes) if abs(ctx["cars"].s[i] - stop) < 1e-6]}
        if "cam" in ctx:
            if ctx["ctl"]["state"] == "arrived" and not arrived:
                arrived.append(ctx["t"])
        else:
            rec.update({k: records[-1][k] for k in ("cam", "roi", "detections", "fv", "ctl")})
        records.append(rec)

    out_dir = Path(next((rest[i + 1] for i, a in enumerate(rest) if a == "--output"), demo.ROOT / "simulator/artifacts/real_crosswalk"))
    try:
        demo.main(rest, frame_hook=hook)
        failed = None
    except SystemExit as e:  # failed checks (or a short --seconds cut): still render what ran, then exit non-zero
        failed = e
    report = json.loads((out_dir / "report.json").read_text())
    if not report.get("completed"):
        raise SystemExit(1)
    first_cross = next((r["t"] for r in records if r["ctl"]["state"] != "waiting"), math.inf)
    arrive_t = arrived[0] if arrived else math.inf
    frames, durations = [title_card(report["site"])], [5000]
    panel, last_cam = None, None
    for rec in records:
        if rec["cam"] is not last_cam or panel is None:
            panel, last_cam = dog_cam_panel(rec), rec["cam"]
        new_state = "cam" in rec and (rec["ctl"]["streak"] and rec["ctl"]["state"] == "waiting" or rec["t"] in (first_cross, arrive_t))
        banner = None
        if "cam" in rec and rec["t"] == first_cross:
            cs = report["crossing_start"]
            banner = (f"WALK confirmed 3/3  -  {cs['dog_estimate_left_s']:.1f} s left >= {cs['time_needed_s']:.1f} s needed  ->  GO")
        elif "cam" in rec and rec["t"] == arrive_t:
            banner = "Far kerb ramp reached"
        frames.append(frame(rec, panel, first_cross, arrive_t, banner))
        durations.append(1500 if banner else 700 if new_state else FRAME_MS)
    if report["crossing_start"] and report["dog_left_road_s"] and report["ped_clearance_end_s"]:
        frames.append(end_card(report))
        durations.append(7000)
    paths = write_video(frames, durations, out_dir / "crossing")
    print(f"video: {', '.join(map(str, paths))}  ({sum(durations) / 1000:.1f} s)", flush=True)
    if failed is not None:
        raise failed


if __name__ == "__main__":
    sys.exit(main())
