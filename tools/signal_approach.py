"""Standby -> target seen -> walk toward it -> stop on arrival -> celebrate -> standby. Fully automatic.

States:
  wait       Standby. While armed (not paused), the detector for the chosen target watches the camera: Gemini for
             the walk signal, YOLO26 for any other object (a phone, a person). After an arrival the same target
             must leave the view for REARM_ABSENT_S before it can start another approach.
  walk       Approach. Steers toward the target (angle from its horizontal position), at full stride while far and
             slowing to a short stride once the sonar reads under NEAR_CM. The sonar alone decides arrival:
             ARRIVE_READINGS readings in a row under ARRIVE_CM. The camera steers; it never stops the dog. A YOLO26
             target unseen for LOST_S ends the approach ("lost") only if the sonar shows nothing within NEAR_CM:
             a close target often drops out of the camera's view, and then the dog keeps going straight on sonar. A walk-signal crossing keeps going when the signal changes mid-street (a
             signal turning to DON'T WALK mid-crossing means hurry, not stop) until the sonar says it arrived or
             MAX_WALK_S passes.
  celebrate  Stop, three rising beeps, bow, stand on the hind legs, back to normal posture. Then standby.
  hold       Follow targets only (a person, a dog, a cat: things that move on their own). Arriving next to one
             doesn't celebrate: the dog stops, keeps turning in place to face it, and walks after it again as soon as
             the sonar shows it has moved away. It drops back to standby if the target is gone for HOLD_LOST_S.
The operator can still force a start (W), steer (A / D held) and stop (Space) from the panel; none are needed.

Detection: the panel's target picks the detector.
  "walk_signal" -> Gemini. Two workers, one per free-tier quota (15 req/min per model; budgeted at 14), staggered
          so requests interleave. Each request carries the full frame plus a 2x zoom of the centre (a 15-23 cm
          signal symbol across a 20 m street is only a few pixels wide), from vision_stream's /raw.jpg without
          boxes drawn over it; Gemini also reports the signal's horizontal position for steering. To save the
          ~500 requests/day free quota, Gemini runs at full rate only while YOLO26 sees a traffic light (a signal
          head) or during a crossing, and otherwise once every IDLE_GEMINI_S.
  any YOLO26 (COCO) class, e.g. "cell phone" -> YOLO26. The camera stream (tools/vision_stream.py) runs
          models/yolo26n.pt at 960 px; this module reads its /detections rather than running a second copy, so the
          boxes on the stream are exactly what triggers and steers. Gemini stays idle.
          Tracking locks onto ONE instance: it starts on the largest (nearest) match, then each new camera frame
          keeps the match closest to where the track was, so a second person walking through the frame doesn't
          steal the dog; the heading is smoothed so single-frame jitter in the box doesn't swing the steering.
          Only new camera frames count (the loop ticks faster than YOLO26 runs), and detections from a frame older
          than DET_STALE_S are ignored, so a frozen camera can't keep the dog walking toward a ghost.
"""
import json
import threading
import time
import urllib.request

from gemini_dog_agent import GeminiClient, get_api_keys
from real_world_escort import STOP_CLASSES, det_conf, det_label, seen_labels

RAW_URL = "http://127.0.0.1:8001/raw.jpg"
DETECTIONS_URL = "http://127.0.0.1:8001/detections"
CLASSES_URL = "http://127.0.0.1:8001/classes"
# Hazard switch (panel, off by default): the escort model's classes plus YOLO26's moving-traffic classes.
HAZARD_CLASSES = STOP_CLASSES | {"car", "truck", "bus", "motorcycle", "bicycle"}
# One worker per free-tier quota, with its requests/min budget one under Google's limit. Aliases share their
# target's quota (gemini-flash-lite-latest counts as gemini-3.5-flash-lite, gemini-3.1-flash-lite-preview as
# gemini-3.1-flash-lite), so running an alias too only triggers 429s. gemini-3.8-flash is left out: its free tier
# allows 20 requests per day. gemini-3-flash-preview spends its output budget thinking and returns no JSON.
# 2026-09-27: 3.5-flash-lite was the only model answering reliably (~1 s, correct on real signal photos);
# 3.1-flash-lite and the rest timed out or returned 503 "high demand".
# 3.1-flash-lite has its own free quota (500/day) and was also correct on the walk / raised-hand photos (~1 s), so
# it keeps checking when 3.5's quota or a key's credit runs out. A worker whose key is out of quota or credit just
# stays benched; the others keep polling.
GO_MODELS = {"gemini-3.5-flash-lite": 15, "gemini-3.1-flash-lite": 15}  # free tier: 15 requests/min each
GO_MIN_CONF = 0.5
# The operator's rule (2026-09-27): cross on the walking person AND on the flashing hand / countdown that follows it
# (a real cycle is ~5 s of WALK then ~20 s of flashing hand); hold only on a steady (solid) hand. One still frame can't
# tell a flashing hand from a solid one, so every check sends a burst of BURST_FRAMES zooms BURST_GAP_S apart.
GO_PHASES = ("walk", "flashing_hand")
MIN_COUNTDOWN_S = 3     # don't set off on the last few seconds of a countdown
BURST_FRAMES = 3
BURST_GAP_S = 0.4       # signal heads flash at ~1 Hz, so 3 frames over 0.8 s catch it both lit and dark
VERDICT_FRESH_S = 5.0   # a verdict about an older frame than this is ignored (burst ~0.8 s + reply ~1-2 s)
IDLE_GEMINI_S = 0.0     # no idle throttle: while armed, poll at the full free-tier rate
SIGNAL_HEAD_S = 3.0     # a traffic light seen this recently counts as "a signal head is in view"
FULL_WIDTH = 768        # full view sent to Gemini
ZOOM_FRACTION = 0.5     # centre crop (this fraction of width and height), enlarged 2x
BLACK_FRAME_MEAN = 8    # mean pixel value below this = camera delivering black frames (stuck stream, lens covered)
YOLO_HITS = 2           # consecutive camera frames with the target needed to start an approach
DET_STALE_S = 1.0       # detections from a frame older than this (camera frozen or reopening) count as nothing
FOLLOW_CLASSES = {"person", "dog", "cat"}  # moving targets: follow and hold next to them instead of celebrating
TRACK_ALPHA = 0.5       # weight of each new frame in the smoothed heading (1 = raw, jittery box centre)
TRACK_GATE = 0.22       # a match more than this far (fraction of frame width) from the track is someone else...
REACQUIRE_S = 0.6       # the heading is smoothed only while the track is continuous; after a gap it snaps
FORGET_S = 2.0          # unseen this long, the track is forgotten and the nearest (largest) match takes over
# Driving. Hiwonder's API: move(stride_mm, turn_deg) and set_gait_params(air_ms, ground_ms, lift_mm); its init sets
# 150/200/25 and its examples walk forward with an 80 mm stride. A foot on the ground carries the body one stride
# per ground phase, so speed = stride / ground time. The old "sprint" (80/120/18 at stride 100) asked each foot to
# swing 100 mm in 80 ms (~1.25 m/s); the servos can't track that, so steps came out short and scuffing.
# Stock Hiwonder walk (what MechDog() sets at power-on). 150/350/20 was tried on smooth concrete (2026-09-27) and
# walked no better; a quicker gait (120/160) veered off line and was no faster.
GAIT = (150, 200, 25)   # air ms, ground ms, lift mm
# Hiwonder's phone app drives full-speed forward with move(120, 0) on this gait; the bridge allows 120.
FAST_STRIDE = 120       # indoors on the real dog: 132 -> 44 cm in 3.6 s (~0.25 m/s); stride 80 crawled at ~6 cm/s
# The sonar decides arrival, and slowing down near the target only made the last metre drag, so one speed throughout.
SLOW_STRIDE = FAST_STRIDE
NEAR_CM = 100           # sonar under this switches to SLOW, and keeps an approach alive when the camera loses it
# Real dog, 2026-09-27: with 45 cm x 3 readings it stopped at 17-19 cm (readings 30, 19, 17): the bridge's sonar
# filter lags and the dog walks ~0.25 m/s, so it overshot and the bridge's own 20 cm obstacle guard fired first.
ARRIVE_CM = 60          # arrival: ARRIVE_READINGS readings in a row under this (sonar glitches on single readings)
ARRIVE_READINGS = 2
TURN_GAIN = 50          # degrees of steering per unit of horizontal offset from centre; positive turns left
MAX_TURN = 25
TURN_STEP = 5           # steering is sent in 5-degree steps so box jitter doesn't change the command every tick
PIVOT_OFFSET = 0.3      # target this far off centre: shorten the stride and turn toward it before it leaves the frame
PIVOT_STRIDE = 50
LOST_S = 3.0            # a YOLO26 object unseen this long ends the approach
FOLLOW_LOST_S = 2.0     # a follow target (person) unseen this long ends the approach: never walk on blind
HOLD_LOST_S = 4.0       # holding next to a follow target: back to standby once it's been gone this long
RESUME_CM = 90          # holding: walk after the target again once every recent sonar reading is past this
HOLD_FACE_OFFSET = 0.15  # holding: turn in place to face the target when it's this far off centre
HOLD_TURN = 15
MAX_WALK_S = 45.0       # object approaches
CROSS_MAX_S = 180.0     # walk-signal crossing safety cap (~45 m at 0.25 m/s); the operator's Stop ends it
# The camera sits on the laptop, not on the dog, so where a target appears in the picture says nothing about the dog's
# heading: steering toward it curved the dog off line (off the table, in the tabletop demo). Every approach, walk signal
# or dropdown object, goes dead straight; the operator's A / D still nudge it. Set True if the camera rides on the dog.
CAMERA_ON_DOG = False
SONAR_PAUSE = False     # 2026-09-27: off on the operator's call; the sonar's false short echoes stop-started the walk
IMU_PAUSE = False       # same for the IMU's "fall" reports (walking shake reads as >50 deg tilt)
BLOCKED_CM = 25
# Stall: walking, something within NEAR_CM, and the sonar not getting closer means the dog is pressed against it.
# It walked into a chair and kept pushing for 8 s, because the sonar saw the chair back at ~40 cm while the legs hit
# the frame (2026-09-27). Progress = median distance over the older half of the last STALL_S minus the newer half;
# medians ride out the sonar's few-cm jitter, which made a "readings all within N cm" test miss the stall.
STALL_S = 2.0
STALL_CM = 3            # less progress than this between the two halves (1 s apart) = stalled
CLEAR_CM = 50           # device demo: only set off with nothing this close in front (after the handshake the person
                        # must step aside first, or the dog walks into them and "arrives" there)         # crossing: pause while the sonar reads this close (the bridge's own guard stops at 20 cm)
REARM_ABSENT_S = 1.5    # after an arrival, the target must be gone this long before it can trigger again
EST_SPEED_MPS = 0.25    # real dog on the floor, stride 120: 78-88 cm in 3.1-3.6 s (2026-09-27)
STEER_DEG = 20          # operator A / D override
STEER_FRESH_S = 0.7     # the panel re-sends a held key every 250 ms; a key that stops arriving counts as released
CELEBRATE = [           # (command, seconds to wait after it)
    ({"t": "stop"}, 0.3),
    ({"t": "buzzer", "freq": 1319, "ms": 120}, 0.18),
    ({"t": "buzzer", "freq": 1568, "ms": 120}, 0.18),
    ({"t": "buzzer", "freq": 2093, "ms": 260}, 0.5),
    ({"t": "action", "id": 7}, 3.0),    # bow (scrape_a_bow)
    ({"t": "action", "id": 5}, 3.5),    # stand on the hind legs (stand_two_legs)
    ({"t": "action", "id": 15}, 2.0),   # back to normal posture
]
# The main demo's finish (2026-09-27): a short victory jingle on the buzzer, then the paw goes up. playTone doesn't
# block, so each note's wait covers its length plus a small gap. {"t": "say"} entries are voice lines, not dog commands.
CELEBRATE_PAW = [
    ({"t": "stop"}, 0.3),
    ({"t": "buzzer", "freq": 1047, "ms": 90}, 0.11),    # C6
    ({"t": "buzzer", "freq": 1319, "ms": 90}, 0.11),    # E6
    ({"t": "buzzer", "freq": 1568, "ms": 90}, 0.11),    # G6
    ({"t": "buzzer", "freq": 2093, "ms": 160}, 0.22),   # C7
    ({"t": "buzzer", "freq": 1568, "ms": 90}, 0.12),    # G6
    ({"t": "buzzer", "freq": 2093, "ms": 320}, 0.5),    # C7, held
    ({"t": "say", "text": "Paw five!"}, 0.0),
    ({"t": "action", "id": 6}, 3.5),    # raises a front paw (handshake)
    ({"t": "action", "id": 15}, 2.0),   # back to normal posture
]
CELEBRATE_HANDSHAKE = [  # device demo: walk up to the person holding the GO signal and shake their hand, like a guide
    ({"t": "stop"}, 0.3),
    ({"t": "buzzer", "freq": 1568, "ms": 120}, 0.2),
    ({"t": "buzzer", "freq": 2093, "ms": 220}, 0.6),
    ({"t": "action", "id": 6}, 4.0),    # handshake: offers a front paw
    ({"t": "action", "id": 15}, 2.0),   # back to normal posture
]
CELEBRATE_BOW = [       # the device demo on a table: beeps and a bow, no hind-leg stand
    ({"t": "stop"}, 0.3),
    ({"t": "buzzer", "freq": 1319, "ms": 120}, 0.18),
    ({"t": "buzzer", "freq": 1568, "ms": 120}, 0.18),
    ({"t": "buzzer", "freq": 2093, "ms": 260}, 0.5),
    ({"t": "action", "id": 7}, 3.0),    # bow (scrape_a_bow)
    ({"t": "action", "id": 15}, 2.0),   # back to normal posture
]

PROMPT = """You are the eyes of a guide dog robot at a street crossing, possibly at night.
Image 1 is the full camera view. Images 2, 3 and 4 are 2x zooms of the centre of the view, taken about 0.4 s apart
(Image 2 is the same moment as Image 1). Use all of them: a FLASHING signal is lit in some of Images 2-4 and dark in
others.
Classify the pedestrian crossing signal:
- "walk": ANYTHING that tells a pedestrian to GO: a WALKING PERSON icon (a figure mid-stride), a green pedestrian
  light or green "go" man, the word WALK or GO, a crosswalk "go" sign or graphic. It counts wherever it appears: a real
  signal head, or a picture on an iPhone, tablet or laptop screen, or a printout held up to the camera. The camera's
  colours are unreliable: a walking person that looks white, cream, yellow, amber or orange is still "walk", because
  the SHAPE decides, not the colour.
- "flashing_hand": a raised HAND that is flashing (lit in some zoom images, dark in others), OR a hand shown next to a
  countdown number (the countdown only runs while the hand flashes).
- "solid_hand": a raised HAND (any colour), or the words DON'T WALK, lit steadily in every zoom image with no
  countdown number.
- "none": no pedestrian signal visible, or an unlit signal.
The signal may be small and far away (5 to 30 metres, across a street); at night it shows as a bright glowing shape.
A signal shown on a phone, laptop, screen or printed photo counts exactly like a real one; it may be small in the
view (a phone a metre or two away), so look carefully, especially in the zoomed images.
countdown: the countdown number shown next to the hand, or null if there is none.
x: the signal's horizontal centre in Image 1 (0 = left edge, 1 = right edge), or null.
Respond with JSON only: {"phase": "walk"|"flashing_hand"|"solid_hand"|"none", "countdown": integer|null,
"confidence": 0.0-1.0, "x": number|null, "seen": "a few words on what you see and where"}"""


def fetch_frame(url=RAW_URL):
    """(BGR image at camera resolution, capture time) from tools/vision_stream.py; (None, t) when the stream is
    down or the frame is black."""
    import cv2
    import numpy as np
    t = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            img = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None, t
    if img is None:
        return None, t
    if img.mean() < BLACK_FRAME_MEAN:
        print("[Camera] black frame; restart tools/vision_stream.py if this persists")
        return None, t
    return img, t


def fetch_detections(url=DETECTIONS_URL):
    """(detections, capture time of their frame on the time.monotonic() clock) from tools/vision_stream.py;
    ([], None) when the stream is down or its newest frame is older than DET_STALE_S."""
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            data = json.loads(resp.read())
    except (OSError, ValueError):
        return [], None
    age = _num(data.get("age"))
    if age is None or age > DET_STALE_S:
        return [], None
    return data.get("dets") or [], time.monotonic() - age


def gemini_burst(img, more):
    """[full view of the first frame, then a 2x centre zoom of every frame] as JPEG bytes (see gemini_images)."""
    first = gemini_images(img)
    return first + [gemini_images(f)[1] for f in more] if more else first


def verdict_go(v):
    """GO for a walk signal or a flashing hand with enough countdown left; never for a solid hand or nothing."""
    if not v or (_num(v.get("confidence")) or 0) < GO_MIN_CONF:
        return False
    phase = v.get("phase")
    if phase == "flashing_hand":
        n = _num(v.get("countdown"))
        return n is None or n >= MIN_COUNTDOWN_S
    return phase == "walk"


def gemini_images(img):
    """[full view downscaled to FULL_WIDTH, centre crop enlarged 2x] as JPEG bytes. Measured with a real WALK photo
    pasted into live frames: both inputs read it to ~24 px tall (~9 m at 720p, ~13 m at 1080p); the enlarged crop
    also caught a 14 px signal at night that a native-size crop and a 4-quadrant split missed."""
    import cv2
    h, w = img.shape[:2]
    full = cv2.resize(img, (FULL_WIDTH, h * FULL_WIDTH // w)) if w > FULL_WIDTH else img
    cw, ch = int(w * ZOOM_FRACTION), int(h * ZOOM_FRACTION)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    zoom = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (min(w, 1280), min(w, 1280) * ch // cw),
                      interpolation=cv2.INTER_CUBIC)
    return [cv2.imencode(".jpg", full, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(),
            cv2.imencode(".jpg", zoom, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()]


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class SignalApproach:
    def __init__(self, agent, rpm_per_model=None, models=GO_MODELS, panel=None, frame_source=fetch_frame,
                 classes_url=CLASSES_URL, det_source=fetch_detections):
        self.agent = agent
        self.dog = agent.dog
        self.panel = panel
        self.frame_source = frame_source
        self.det_source = det_source
        self.classes_url = classes_url
        self.lock = threading.Lock()
        self.verdict = None
        self.verdict_frame_t = 0.0
        self.dets = []                  # latest camera-stream detections, refreshed every tick
        self.yolo_live = False          # the camera stream answered with its class list
        self.target_hits = 0            # consecutive camera frames with the YOLO26 target in view
        self.target_seen_at = 0.0
        self.target_pos = None          # (smoothed cx, area) of the target, or (cx, None) from Gemini
        self.track = None               # the locked YOLO26 detection (drawn as the target on the page)
        self.det_frame_t = None         # capture time of the frame self.dets came from; None = no fresh frame
        self.last_frame_t = None        # last frame the tracker consumed
        self.signal_head_at = 0.0       # last time YOLO26 saw a traffic light
        self.need_absence = False       # after an arrival: wait for the target to leave before re-triggering
        self.was_paused = True
        self.pose_sent = None           # (pitch, height) last applied to the dog
        self.mission_from = None        # device demo: the object target met before heading for the crossing signal
        self.said_wait = False          # "the signal says wait" is spoken once per Arm
        self.armed_at = None            # when Arm was last pressed, for the go_timer_s demo timer; None once used
        self.absent_since = time.monotonic()
        self.state = "wait"
        self.state_at = time.monotonic()
        self.go_by = None
        self.phase = None
        self.last_log = 0.0
        self.last_said = None
        self.moving = False
        self.last_stop = 0.0
        # One worker per (key, model): each key from a separate Google project has its own quota. The workers are
        # staggered evenly so their checks interleave (2 keys at 15/min each = a check every 2 s).
        keys = get_api_keys() or [agent.api_key]
        slots = [(k, m, rpm_per_model or rpm) for k in keys for m, rpm in models.items()]
        total = sum(r for *_, r in slots) or 1
        self.threads = [threading.Thread(target=self._gemini_worker, args=(m, i * 60.0 / total, rpm, k), daemon=True)
                        for i, (k, m, rpm) in enumerate(slots)]
        self.gemini_keys = len(keys)
        self.gemini_errors = {}  # (key, model) -> last failure reason, None when that worker's last call succeeded
        self.gemini_workers = len(self.threads)
        self.threads.append(threading.Thread(target=self._classes_loader, daemon=True))

    def settings(self):
        if self.panel:
            return self.panel.get_settings()
        from approach_panel import DEFAULT_SETTINGS
        return dict(DEFAULT_SETTINGS)

    # -- detectors --
    def _gemini_wanted(self):
        """Gemini runs for the walk signal while armed: at full rate during a crossing or while a signal head is in
        view, otherwise at most once per IDLE_GEMINI_S."""
        s = self.settings()
        if s["paused"] or s["target"] != "walk_signal" or self.state not in ("wait", "walk"):
            return False, False
        full_rate = self.state == "walk" or time.monotonic() - self.signal_head_at < SIGNAL_HEAD_S
        return True, full_rate

    def _gemini_worker(self, model, stagger_s, rpm, key=None):
        client = GeminiClient(key or self.agent.api_key, min_interval=60.0 / rpm, rpm_per_model=rpm, models=(model,))
        time.sleep(stagger_s)
        last_call = 0.0
        while True:
            wanted, full_rate = self._gemini_wanted()
            if not wanted or (not full_rate and time.monotonic() - last_call < IDLE_GEMINI_S):
                time.sleep(0.2)
                continue
            img, frame_t = self.frame_source()
            if img is None:
                time.sleep(0.5)
                continue
            more = []
            for _ in range(BURST_FRAMES - 1):  # the flash: same view a moment later, lit or dark
                time.sleep(BURST_GAP_S)
                f, _t = self.frame_source()
                if f is not None:
                    more.append(f)
            last_call = time.monotonic()
            jpegs = gemini_burst(img, more)
            verdict = client.generate_json(PROMPT, jpeg=jpegs)
            self.gemini_errors[(key, model)] = client.last_error
            if verdict is not None:
                verdict["go"] = verdict_go(verdict)
                verdict["signal"] = ("walk" if verdict.get("phase") == "walk" else
                                     "dont_walk" if verdict.get("phase") in ("flashing_hand", "solid_hand") else "none")
                if self.panel and hasattr(self.panel, "set_shot"):
                    self.panel.set_shot(jpegs[0], {"go": verdict["go"], "phase": verdict.get("phase"),
                                                   "seen": verdict.get("seen"), "model": model})
                print(f"👁️  {model}: {verdict.get('phase')} countdown={verdict.get('countdown')} -> "
                      f"{'GO' if verdict['go'] else 'hold'} conf={verdict.get('confidence')} x={verdict.get('x')} | "
                      f"{verdict.get('seen')}")
                with self.lock:
                    if frame_t > self.verdict_frame_t:
                        self.verdict, self.verdict_frame_t = verdict, frame_t
            ready_at = max(client.next_call, client.benched_until.get(model, 0.0))
            time.sleep(max(0.05, ready_at - time.monotonic()))

    def _classes_loader(self):
        """Fill the panel's target list from the camera stream's model."""
        while not self.yolo_live:
            try:
                with urllib.request.urlopen(self.classes_url, timeout=1.0) as resp:
                    names = json.loads(resp.read())
                if self.panel:
                    self.panel.set_classes(names)
                self.yolo_live = True
                print(f"[YOLO26] reading camera-stream detections ({len(names)} classes)")
            except (OSError, ValueError):
                time.sleep(2.0)

    def latest_verdict(self):
        with self.lock:
            if self.verdict and time.monotonic() - self.verdict_frame_t < VERDICT_FRESH_S:
                return dict(self.verdict)
        return None

    def observe(self, s, verdict, now):
        """Update target tracking from this tick's detections (YOLO26 targets) or verdict (walk signal)."""
        if any(det_label(d) == "traffic light" for d in self.dets):
            self.signal_head_at = now
        if s["target"] == "walk_signal":
            seen = bool(verdict) and verdict.get("signal") in ("walk", "dont_walk")
            if seen and _num(verdict.get("x")) is not None:
                self.target_pos, self.target_seen_at = (min(1.0, max(0.0, verdict["x"])), None), now
            go = bool(verdict) and verdict.get("go") is True  # set by verdict_go() when the reply arrived
            seen = go  # re-arming only needs GO gone for REARM_ABSENT_S
        else:
            seen = self.track_yolo(s, now)
            go = self.target_hits >= YOLO_HITS
        # Absence only counts while standing by: the target leaving the frame during the bow doesn't re-arm.
        if seen or self.state != "wait":
            self.absent_since = None
        elif self.absent_since is None:
            self.absent_since = now
        if self.need_absence and self.absent_since is not None and now - self.absent_since >= REARM_ABSENT_S:
            self.need_absence = False
        return go

    def track_yolo(self, s, now):
        """Advance the single-target track by one camera frame. Returns whether the target is in view."""
        if self.det_frame_t is None:  # camera down or frozen
            self.target_hits = 0
            return False
        if self.det_frame_t == self.last_frame_t:  # same frame as last tick: nothing new
            return now - self.target_seen_at < 0.5
        self.last_frame_t = self.det_frame_t
        matches = [d for d in self.dets if det_label(d) == s["target"] and det_conf(d) >= s["min_conf"]]
        pick = None
        unseen = now - self.target_seen_at
        locked = self.target_pos is not None and unseen < REACQUIRE_S
        if matches and self.target_pos is not None and unseen < FORGET_S:
            # Look for the same one near where it was last seen; the window widens the longer it's been missing.
            cx0, a0 = self.target_pos
            gate = min(0.5, TRACK_GATE + 0.3 * unseen)
            near = min(matches, key=lambda d: abs(float(d.get("cx", 0.5)) - cx0))
            a = _num(near.get("area"))
            similar = not a0 or not a or 1 / 3 <= a / a0 <= 3
            if abs(float(near.get("cx", 0.5)) - cx0) <= gate and similar:
                pick = near
        elif matches:
            pick = max(matches, key=lambda d: _num(d.get("area")) or 0)  # nearest = biggest in frame
        if pick is None:
            self.target_hits = 0
            return False
        cx = float(pick.get("cx", 0.5))
        if locked:
            cx = TRACK_ALPHA * cx + (1 - TRACK_ALPHA) * self.target_pos[0]
        self.target_pos, self.target_seen_at, self.track = (cx, _num(pick.get("area"))), now, pick
        self.target_hits += 1
        return True

    @staticmethod
    def follow_mode(s):
        return s["target"] in FOLLOW_CLASSES and not s.get("finish_cm")

    # -- operator overrides (none needed) --
    def handle_commands(self):
        for cmd in self.panel.pop_commands() if self.panel else []:
            if cmd == "go" and self.state == "wait":
                self.start_walk("operator")
            elif cmd == "end" and self.state == "walk" and self.settings()["target"] == "walk_signal":
                self.arrive("operator Stop")  # Stop is the finish line of a crossing: stop, then celebrate
            elif cmd == "end" and self.state in ("walk", "hold"):
                # Stopped by hand: like an arrival, the target has to leave the view before it can start another.
                self.need_absence = True
                self.halt()
                self.set_state("wait", "Stopping.")

    def start_walk(self, source):
        print(f"🟢 GO from {source}")
        self.go_by = source
        # Whatever started this walk, the Arm timer is used up: without this, a manual GO while disarmed counted as
        # pressing Arm, and the timer started a second walk 6.72 s later, right after the celebration (2026-09-27).
        self.armed_at, self.was_paused = None, False
        if self.panel and self.settings()["paused"]:
            self.panel.update_settings({"paused": False})  # a manual GO while paused means "go now"
        st = self.settings()
        target, device = st["target"], bool(st.get("finish_cm"))
        detected = source.startswith(("Gemini", "YOLO26"))
        speech = ("Crossing now, stay with me." if not detected
                  else "Walk signal detected! Crossing now, stay with me." if target == "walk_signal"
                  else "I see you. I'm coming." if device and target == "person"
                  else f"I see the {target}. Coming to you." if device
                  else f"I see a {target}. Following." if target in FOLLOW_CLASSES
                  else f"I see the {target}. Heading to it.")
        self.set_state("walk", speech)

    def arrive(self, reason):
        walked = time.monotonic() - self.state_at
        print(f"🏁 arrived ({reason}) after {walked:.1f} s (~{walked * EST_SPEED_MPS:.1f} m est.)")
        if self.follow_mode(self.settings()):
            self.halt()
            self.set_state("hold", "I'm right here.")
            return
        self.need_absence = True
        s = self.settings()
        style = self.celebration_style(s)
        speech = ("I'm here with you. Take my paw, I'll get you across safely." if style == "handshake"
                  else "We made it across!" if s["target"] == "walk_signal" else "We're here!")
        self.set_state("celebrate", speech)
        threading.Thread(target=self._celebrate, args=(style, s), daemon=True).start()

    @staticmethod
    def celebration_style(s):
        """Device demo: meeting the person (a dropdown object) is a handshake; reaching the crossing signal is the jingle
        and a raised paw. Street: the operator's choice (the jingle and paw by default)."""
        if s.get("finish_cm"):
            return "paw" if s["target"] == "walk_signal" else "handshake"
        return s.get("celebration") or "paw"

    def _celebrate(self, style=None, s=None):
        s = s or self.settings()
        style = style or self.celebration_style(s)
        self.dog.set_rgb(0, 200, 255)
        for cmd, pause in {"bow": CELEBRATE_BOW, "handshake": CELEBRATE_HANDSHAKE,
                           "paw": CELEBRATE_PAW}.get(style, CELEBRATE):
            if cmd["t"] == "say":
                self.say(cmd["text"])
            else:
                self.dog.send_cmd(cmd)
            time.sleep(pause)
        self.moving = False
        self.pose_sent = None  # the "normal posture" action undid the pitch / height tuning: re-apply it
        if s.get("finish_cm") and self.panel:
            if style == "handshake":
                # Guardian mission, part 2: having met the person, head for the crossing signal (Gemini). It sets off
                # once the sign shows GO and the person has stepped out of the way (nothing within finish_cm).
                self.mission_from = s["target"]
                self.panel.update_settings({"target": "walk_signal"})
                self.need_absence = False
                self.set_state("wait", "Now let's find the crossing signal.")
                return
            if s["target"] == "walk_signal":
                # Across: the take is over. Disarm, and put the dropdown back on the person / object for the next one.
                patch = {"paused": True}
                if self.mission_from:
                    patch["target"], self.mission_from = self.mission_from, None
                self.panel.update_settings(patch)
        self.set_state("wait")

    # -- control loop --
    def say(self, text):
        if self.panel:
            self.panel.say(text)  # the panel drops repeats itself
        elif self.last_said != text:
            print(f"🗣️  {text}")
        self.last_said = text

    def set_state(self, state, speech=None):
        if state != self.state:
            print(f"➡️  [{state.upper()}]")
            self.state, self.state_at = state, time.monotonic()
            with self.lock:
                self.verdict = None  # a new state never acts on the previous one's evidence
            if state != "hold":
                self.target_hits = 0
            if speech:
                self.say(speech)

    def sonar_cm(self):
        fresh = time.monotonic() - self.dog.dist_at < 1.0
        return self.dog.dist_cm if fresh else None

    def sonar_close(self, cm=ARRIVE_CM):
        """True when the last ARRIVE_READINGS sonar readings, all recent, are every one under cm."""
        recent = list(self.dog.dist_history)[-ARRIVE_READINGS:]
        now = time.monotonic()
        return len(recent) == ARRIVE_READINGS and all(
            d is not None and d <= cm and now - t < 1.0 for t, d in recent)

    def path_clear(self, cm):
        """The last two sonar readings are recent and both farther than cm (or no echo at all)."""
        recent = list(self.dog.dist_history)[-2:]
        now = time.monotonic()
        return len(recent) == 2 and all(now - t < 1.0 and (d is None or d > cm) for t, d in recent)

    def step(self):
        now = time.monotonic()
        self.handle_commands()
        s = self.settings()
        self.dets, self.det_frame_t = self.det_source()
        verdict = self.latest_verdict()
        go = self.observe(s, verdict, now)
        dist = self.sonar_cm()
        hazards = seen_labels(self.dets) & HAZARD_CLASSES if s["hazard_stop"] else set()
        label, stride, angle, self.phase = None, None, 0, None

        if self.was_paused and not s["paused"]:  # Arm pressed: a fresh start, and the demo timer runs from here
            self.armed_at, self.need_absence, self.said_wait = now, False, False
            if s["target"] == "walk_signal":
                self.say("I'm watching the crossing signal.")
        self.was_paused = s["paused"]
        timer = s.get("go_timer_s") or 0
        timer_due = (timer > 0 and self.armed_at is not None and s["target"] == "walk_signal"
                     and now - self.armed_at >= timer)

        pose = (s.get("pitch", 0), s.get("height", 80), s.get("shift", 0))
        if pose != self.pose_sent and self.dog.link.ready and self.state != "celebrate":
            ok = self.dog.send_cmd({"t": "posture", "x": 0, "y": pose[0], "shift": pose[2]}) and                 self.dog.send_cmd({"t": "height", "mm": pose[1]})
            self.pose_sent = pose if ok else None
            print(f"🦴 posture pitch {pose[0]}, height {pose[1]} mm, shift {pose[2]} mm")
        sign = -1 if s.get("reverse") else 1
        drive = self.panel.drive(STEER_FRESH_S) if self.panel and hasattr(self.panel, "drive") else None
        nudge_only = bool(drive) and self.state == "walk" and drive[0] == 0  # A / D while it walks: steer, don't stop
        if drive and (drive[0] or drive[1]) and self.state != "celebrate" and not nudge_only:
            # Manual W / S driving (or A / D while standing) overrides the autonomy (and ends a crossing).
            if self.state != "wait":
                self.set_state("wait")
            self.dog.set_gait(*GAIT)
            angle = drive[1] if drive[1] or not drive[0] else int(s.get("trim") or 0)  # W / S alone: trimmed straight
            self.dog.send_cmd({"t": "move", "stride": sign * drive[0], "angle": angle})
            self.moving = True
            self._publish("Manual drive", dist, drive[0], drive[1], verdict)
            return

        if self.state == "celebrate":
            label = "Celebrating"  # the celebration thread owns the dog; sending stop here would cancel the bow
        elif s["paused"]:
            self.halt()
            if self.state != "wait":
                self.set_state("wait")
            label = "Paused"
        elif hazards and self.state in ("walk", "hold"):
            self.halt()
            self.say("Stop. Traffic ahead.")
            label = "Hazard: " + ", ".join(sorted(hazards))
            self._log(now, f"⚠️  {label}: holding")
        else:
            if self.state == "wait":
                if go and self.follow_mode(s) and self.sonar_close():  # already next to them: just stay with them
                    self.set_state("hold", "I'm right here.")
                elif go and not self.need_absence and (
                        self.path_clear(CLEAR_CM) if s.get("finish_cm")  # device demo: nobody right in front
                        else not self.sonar_close() if s["target"] != "walk_signal" else True):
                    self.armed_at = None
                    self.start_walk("Gemini (walk signal)" if s["target"] == "walk_signal"
                                    else f"YOLO26 ({s['target']})")
                elif timer_due:
                    self.armed_at = None  # once per Arm
                    self.start_walk(f"timer ({timer:g} s after Arm)")
                else:
                    self.halt()
                    label = "Waiting for it to leave view" if self.need_absence else "Standby"
                    if go and not self.need_absence:
                        # Say why a detected GO isn't moving the dog (it looked like "Arm does nothing", 2026-09-27).
                        link = self.dog.link
                        if not (link.ready and link.connected):
                            label = "GO seen · robot not connected"
                        elif s.get("finish_cm") and not self.path_clear(CLEAR_CM):
                            label = f"GO seen · something within {CLEAR_CM} cm in front"
                    if (s["target"] == "walk_signal" and not self.said_wait and verdict
                            and verdict.get("phase") == "solid_hand"):
                        self.said_wait = True  # once per Arm
                        self.say("The signal says wait. I'll tell you when it's safe.")
                    self._log(now, f"🔴 standby ({(verdict or {}).get('seen', 'no fresh verdict')})")
            if self.state == "hold":
                label, stride, angle = self.hold(now)
            if self.state == "walk":
                label, stride, angle = self.drive(s, dist, now)
        self._publish(label, dist, stride, angle, verdict)

    def drive(self, s, dist, now):
        """One approach tick: arrival / lost checks, then steer toward the target at the right speed."""
        fresh = now - self.target_seen_at < 1.0
        sonar_sees = dist is not None and dist < NEAR_CM
        problem = getattr(self.dog, "problem", None)
        if problem and "batt" in problem and now - self.dog.problem_at < 2.0:
            self.halt()
            self.set_state("wait", "I can't walk: my battery is low.")
            return "Dog stopped: " + problem, None, 0
        if IMU_PAUSE and problem in ("fall", "move refused: fallen"):
            # The bridge reports "fall" past 50 degrees of tilt and "upright" once level again (which replaces
            # dog.problem). On the real dog it fired for a moment mid-walk (2026-09-27), so a fall only pauses the
            # crossing; it carries on after "upright".
            self.halt()
            self.phase = "tipped"
            return "Tipped · waiting until upright", None, 0
        finish = s.get("finish_cm") or 0
        if finish and self.sonar_close(finish):
            # Device demo: the target (a phone showing GO, the object picked in the dropdown, the person holding it) is
            # at the end of the table; stop this far from it and greet.
            self.arrive("sonar " + ", ".join(f"{d:.0f}" for _, d in list(self.dog.dist_history)[-ARRIVE_READINGS:]))
            return "Arrived", None, 0
        if finish and now - self.state_at > STALL_S + 0.5:
            recent = [(t, d) for t, d in self.dog.dist_history if now - t <= STALL_S and d is not None and d < 400]
            older = sorted(d for t, d in recent if now - t > STALL_S / 2)
            newer = sorted(d for t, d in recent if now - t <= STALL_S / 2)
            if len(older) >= 4 and len(newer) >= 4 and max(older + newer) < NEAR_CM:
                progress = older[len(older) // 2] - newer[len(newer) // 2]
                if progress < STALL_CM:
                    self.arrive(f"stalled at ~{newer[len(newer) // 2]:.0f} cm: {progress:.1f} cm closer in 1 s")
                    return "Arrived", None, 0
        if s["target"] == "walk_signal":
            # Street mode (finish_cm 0): a crossing ends only on Stop (or the time cap). The sonar ending it made the
            # dog "arrive" the moment GO fired whenever anything stood within 60 cm (2026-09-27).
            recent = list(self.dog.dist_history)[-2:]
            if SONAR_PAUSE and len(recent) == 2 and all(
                    d is not None and d <= BLOCKED_CM and now - t < 1.0 for t, d in recent):
                self.halt()
                self.phase = "blocked"
                self._log(now, f"✋ something {recent[-1][1]:.0f} cm in front: waiting for it to clear")
                return "Something in front · waiting", None, 0
        elif not finish and self.sonar_close():
            self.arrive("sonar " + ", ".join(f"{d:.0f}" for _, d in list(self.dog.dist_history)[-ARRIVE_READINGS:]))
            return "Arrived", None, 0
        unseen = now - self.target_seen_at
        # A close object often drops out of the low camera's view, so the sonar keeps an object approach alive. A
        # person moves: walking on without seeing them would only walk into whatever the sonar happens to see.
        lost = unseen > FOLLOW_LOST_S if self.follow_mode(s) else unseen > LOST_S and not sonar_sees
        if s["target"] != "walk_signal" and lost:
            self.halt()
            self.set_state("wait", f"I lost the {s['target']}.")
            return "Lost the target", None, 0
        cap = (s.get("cross_max_s") or CROSS_MAX_S) if finish or s["target"] == "walk_signal" else MAX_WALK_S
        if now - self.state_at > cap:
            if finish:
                # Device demo: the backstop timer still ends with the finish (jingle, paw); a crossing that just
                # stopped, "Stopping here.", looked like a failure (2026-09-27).
                self.arrive("time limit")
                return "Arrived", None, 0
            self.halt()
            self.set_state("wait", "Stopping here.")
            return "Time limit", None, 0

        steer = self.panel.steer(STEER_FRESH_S) if self.panel else 0
        nudge = self.panel.drive(STEER_FRESH_S) if self.panel and hasattr(self.panel, "drive") else None
        if nudge and nudge[0] == 0 and nudge[1]:
            angle, self.phase = max(-STEER_DEG, min(STEER_DEG, nudge[1])), "operator nudge"
        elif steer:
            angle, self.phase = STEER_DEG * steer, "operator steering"
        elif fresh and self.target_pos and CAMERA_ON_DOG:
            angle = self.heading(self.target_pos[0], MAX_TURN)
            self.phase = "tracking"
        else:
            # "Straight" includes the operator's trim for this dog's drift (the page's Straight trim buttons).
            angle, self.phase = int(s.get("trim") or 0), "on sonar" if sonar_sees else "dead reckoning"
        near = sonar_sees
        stride = SLOW_STRIDE if near else FAST_STRIDE
        if self.phase == "tracking" and abs(self.target_pos[0] - 0.5) >= PIVOT_OFFSET:
            stride, self.phase = PIVOT_STRIDE, "turning to it"  # at full stride it would walk out of view
        self.dog.set_gait(*GAIT)  # sent only when it differs from the last gait the bridge acknowledged
        self.dog.send_cmd({"t": "move", "stride": -stride if s.get("reverse") else stride, "angle": angle})
        self.moving = True
        self.dog.set_rgb(0, 255, 0)
        label = ("Closing in" if near else "Heading to target") + (f" · {self.phase}" if self.phase else "")
        self._log(now, f"🟢 {label}: stride {stride} mm (~{EST_SPEED_MPS:.2f} m/s measured), angle {angle}, "
                       f"sonar {'-' if dist is None else f'{dist:.0f} cm'}")
        return label, stride, angle

    @staticmethod
    def heading(cx, limit):
        """Steering angle toward a target at horizontal position cx (0 left .. 1 right); positive turns left."""
        a = max(-limit, min(limit, (0.5 - cx) * TURN_GAIN))
        return int(TURN_STEP * round(a / TURN_STEP))

    def hold(self, now):
        """Next to a follow target: face it, and walk after it again once it moves away."""
        unseen = now - self.target_seen_at
        if unseen > HOLD_LOST_S:
            self.halt()
            self.set_state("wait", "I lost you.")
            return "Lost the target", None, 0
        recent = list(self.dog.dist_history)[-ARRIVE_READINGS:]
        moved_away = len(recent) == ARRIVE_READINGS and all(
            d is not None and d > RESUME_CM and now - t < 1.0 for t, d in recent)
        if moved_away and unseen < 0.5:
            self.set_state("walk", "Following.")
            return None, None, 0  # drive() takes over this same tick
        if unseen < 0.5 and self.target_pos and abs(self.target_pos[0] - 0.5) > HOLD_FACE_OFFSET:
            angle = self.heading(self.target_pos[0], HOLD_TURN) or (
                TURN_STEP if self.target_pos[0] < 0.5 else -TURN_STEP)
            self.dog.send_cmd({"t": "move", "stride": 0, "angle": angle})
            self.moving = True
            self.phase = "facing it"
            return "With you · turning to face", 0, angle
        self.halt()
        return "With you", None, 0

    def _publish(self, label, dist, stride, angle, verdict):
        if not self.panel:
            return
        s = self.settings()
        yolo = ", ".join(f"{det_label(d)} {det_conf(d):.2f}" for d in
                         sorted(self.dets, key=det_conf, reverse=True)[:4]) or "nothing"
        if not self.yolo_live:
            yolo = "camera stream not running"
        wanted, full_rate = self._gemini_wanted()
        errors = list(self.gemini_errors.values())
        down = errors and all(errors) and s["target"] == "walk_signal" and not s["paused"]
        gemini = (f"DOWN: {errors[0]} on every key; add GEMINI_API_KEY_2 from another project" if down and not verdict
                  else f"{'GO' if verdict.get('go') else 'no go'} ({verdict.get('seen', '')})" if verdict
                  else "off (disarmed)" if s["paused"] else
                  f"idle (target is {s['target']}: YOLO26's job)" if s["target"] != "walk_signal" else
                  "watching" + ("" if full_rate or not IDLE_GEMINI_S else f" · light duty, 1 check / "
                                f"{IDLE_GEMINI_S:.0f} s until a signal head is in view"))
        state = "paused" if label == "Paused" else "hazard" if label and label.startswith("Hazard") else self.state
        walking_s = time.monotonic() - self.state_at if self.state == "walk" else None
        link = self.dog.link
        self.panel.set_status(state=state, label=label, sonar_cm=dist, stride=stride, angle=angle, gemini=gemini,
                              yolo=yolo, go_by=self.go_by if self.state == "walk" else None, phase=self.phase,
                              target_x=self.target_pos[0] if self.target_pos else None,
                              track=({"label": det_label(self.track), "conf": det_conf(self.track),
                                      "nbox": self.track.get("nbox")}
                                     if self.track and s["target"] != "walk_signal"
                                     and time.monotonic() - self.target_seen_at < 1.0 else None),
                              camera="live" if self.det_frame_t is not None else "down",
                              link="connected" if link.ready and link.connected else "connecting", target=link.target,
                              batt_v=getattr(self.dog, "batt_v", None),
                              timer_left=(round(max(0.0, s["go_timer_s"] - (time.monotonic() - self.armed_at)), 1)
                                          if s.get("go_timer_s") and self.armed_at is not None
                                          and self.state == "wait" and not s["paused"] else None),
                              dog_mode=getattr(self.dog, "mode", None),
                              imu=getattr(self.dog, "imu_ang", None),
                              dog_problem=(self.dog.problem if getattr(self.dog, "problem", None)
                                           and time.monotonic() - self.dog.problem_at < 10 else None),
                              walking_s=walking_s,
                              est_distance_m=None if walking_s is None else round(walking_s * EST_SPEED_MPS, 1))

    def halt(self):
        # A stopped dog stays stopped, so repeat "stop" once a second rather than every tick: over Bluetooth each
        # command takes ~130 ms per 20-byte chunk. Any move since the last stop forces an immediate one.
        now = time.monotonic()
        if self.moving or now - self.last_stop >= 1.0:
            self.dog.send_cmd({"t": "stop"})
            self.last_stop, self.moving = now, False
        self.dog.set_rgb(255, 0, 0)

    def _log(self, now, line):
        if now - self.last_log >= 1.0:
            self.last_log = now
            print(line)

    def run(self, tick_s=0.12):
        print(f"Go-signal mode: {self.gemini_workers} Gemini workers (walk signal) + YOLO26 from the camera stream "
              f"(other targets). Automatic: standby -> approach -> arrive -> celebrate.")
        self.halt()
        for t in self.threads:
            t.start()
        try:
            while True:
                self.step()
                time.sleep(tick_s)
        finally:
            self.halt()
