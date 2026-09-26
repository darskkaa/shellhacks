# MechDog WiFi command bridge — MicroPython main.py for the Hiwonder MechDog ESP32 controller.
#
# Implements PROTOCOL.md (newline-delimited JSON over TCP, and over the USB-serial REPL port) on top of Hiwonder's stock motion library so the
# Arduino UNO Q can drive the dog without touching servo wiring or calibration. Back up the factory main.py
# before uploading this file (see README.md in this folder).

import gc
import json
import select
import socket
import sys
import time

import config

try:
    import network
except ImportError:
    network = None

try:
    import Hiwonder
    import Hiwonder_IIC
    from HW_MechDog import MechDog
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def num(v, default=0):
    """int() that never raises: strings, None, floats and junk all become a usable number."""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


EAGAIN = 11


class SonarFilter:
    """Median of the last SONAR_MEDIAN_N valid reads; a None/out-of-range read only becomes "unknown" after
    SONAR_DROPOUT_POLLS in a row (Hiwonder's IOT lesson uses the same dis_count > 4 rule)."""

    def __init__(self):
        self.hist = []
        self.last = None
        self.dropouts = 0

    def update(self, d):
        if d is None:
            self.dropouts += 1
            if self.dropouts >= config.SONAR_DROPOUT_POLLS:
                self.last = None
                self.hist = []
            return self.last
        self.dropouts = 0
        self.hist.append(d)
        if len(self.hist) > config.SONAR_MEDIAN_N:
            self.hist.pop(0)
        self.last = sorted(self.hist)[len(self.hist) // 2]
        return self.last


# ---------------------------------------------------------------------------------------------------------
# Hardware abstraction. Hiwonder ships the motion library inside the firmware; only move()/set_servo()/
# sensors are documented for Python, so every optional feature is probed at boot and reported in `hello`.
# ---------------------------------------------------------------------------------------------------------
class DogHAL:
    ACTION_NAMES = [
        "left_foot_kick", "right_foot_kick", "stand_four_legs", "sit_dowm", "go_prone", "stand_two_legs",
        "handshake", "scrape_a_bow", "nodding_motion", "boxing", "stretch_oneself", "pee", "press_up",
        "rotation_pitch", "rotation_roll", "normal_attitude",
    ]
    HEIGHT_NOMINAL_MM = 80   # stock default pose; transform's z is an offset from it
    HEIGHT_Z_MIN = -25       # clamps taken from the stock main.py
    HEIGHT_Z_MAX = 15
    TILT_MAX_DEG = 17

    def __init__(self):
        self.caps = {"move": False, "action": None, "gait": None, "height": None, "posture": None,
                     "imu_raw": None, "imu_angle": False, "sonar": False, "rgb": False, "buzzer": False,
                     "battery": None, "sonar_unit": config.SONAR_UNIT}
        self.dog = None
        self.sonar = None
        self.imu = None
        self.buzzer = None
        self._last_ang = None
        self._last_ang_t = None
        self._pose_z = 0
        self._roll = 0
        self._pitch = 0
        if not HW_AVAILABLE:
            return
        self.dog = MechDog()
        self.caps["move"] = hasattr(self.dog, "move")
        self.caps["action"] = self._probe(self.dog, ("action_run", "run_action", "action", "do_action"))
        self.caps["gait"] = self._probe(self.dog, ("set_gait_params", "set_gait", "gait_params"))
        # Height and posture are both `transform([x,y,z], [roll,pitch,yaw], ms)`; there is no set_height or
        # set_posture. Confirmed from dir(MechDog()) on the device, 2026-09-08.
        has_tf = hasattr(self.dog, "transform")
        self.caps["height"] = "transform" if has_tf else None
        self.caps["posture"] = "transform" if has_tf else None
        # Battery is a module function returning millivolts, not a MechDog method.
        self.caps["battery"] = self._probe(Hiwonder, ("Battery_power", "battery_power", "get_battery"))
        try:
            iic1 = Hiwonder_IIC.IIC(1)
            self.sonar = Hiwonder_IIC.I2CSonar(iic1)
            self.caps["sonar"] = True
            self.caps["rgb"] = hasattr(self.sonar, "setRGB")
        except Exception:
            self.sonar = None
        try:
            self.imu = Hiwonder_IIC.MPU()
            self.caps["imu_angle"] = hasattr(self.imu, "read_angle")
            self.caps["imu_raw"] = self._probe(self.imu, ("read_raw", "read_accel", "get_accel", "read_data"))
        except Exception:
            self.imu = None
        try:
            self.buzzer = Hiwonder.Buzzer()
            self.caps["buzzer"] = hasattr(self.buzzer, "playTone")
        except Exception:
            self.buzzer = None

    @staticmethod
    def _probe(obj, names):
        for n in names:
            if hasattr(obj, n):
                return n
        return None

    def _call(self, cap, *args):
        name = self.caps.get(cap)
        if not name:
            return False
        try:
            getattr(self.dog, name)(*args)
            return True
        except Exception:
            return False

    # -- motion --
    def move(self, stride, angle):
        if self.dog:
            self.dog.move(int(stride), int(angle))

    def stop(self):
        self.move(0, 0)

    def action(self, idx):
        idx = int(idx)
        name = self.ACTION_NAMES[idx] if 0 <= idx < len(self.ACTION_NAMES) else None
        # Try by name first (matches the Arduino API), then by 1-based index.
        return self._call("action", name) or self._call("action", idx + 1)

    def gait(self, lift_ms, contact_ms, lift_mm):
        return self._call("gait", int(lift_ms), int(contact_ms), int(lift_mm))

    def _transform(self, ms=100):
        if not self.caps.get("height") and not self.caps.get("posture"):
            return False
        try:
            self.dog.transform([0, 0, self._pose_z], [self._roll, self._pitch, 0], ms)
            return True
        except Exception:
            return False

    def height(self, mm):
        self._pose_z = clamp(int(mm) - self.HEIGHT_NOMINAL_MM, self.HEIGHT_Z_MIN, self.HEIGHT_Z_MAX)
        return self._transform()

    def posture(self, x, y):
        self._roll = clamp(int(x), -self.TILT_MAX_DEG, self.TILT_MAX_DEG)
        self._pitch = clamp(int(y), -self.TILT_MAX_DEG, self.TILT_MAX_DEG)
        return self._transform()

    # -- sensors --
    def dist_cm(self):
        if not self.sonar:
            return None
        try:
            d = self.sonar.getDistance()
        except Exception:
            return None
        if d is None or d < 0:
            return None
        return round(d / 10.0, 1) if config.SONAR_UNIT == "mm" else round(float(d), 1)

    def imu_read(self):
        if not self.imu:
            return None, None
        ang = None
        if self.caps["imu_angle"]:
            try:
                ang = [float(a) for a in self.imu.read_angle()]
            except Exception:
                ang = None
        raw = None
        if self.caps["imu_raw"]:
            try:
                r = getattr(self.imu, self.caps["imu_raw"])()
                if r and len(r) >= 6:
                    raw = {"ax": r[0], "ay": r[1], "az": r[2], "gx": r[3], "gy": r[4], "gz": r[5]}
            except Exception:
                raw = None
        if raw is None and ang is not None:
            # Derive angular rates from consecutive angle samples so the brain still gets a vibration signal.
            now = time.ticks_ms()
            gx = gy = gz = 0.0
            if self._last_ang is not None:
                dt = time.ticks_diff(now, self._last_ang_t) / 1000.0
                if dt > 0:
                    gx = (ang[0] - self._last_ang[0]) / dt
                    gy = (ang[1] - self._last_ang[1]) / dt if len(ang) > 1 else 0.0
                    gz = (ang[2] - self._last_ang[2]) / dt if len(ang) > 2 else 0.0
            self._last_ang, self._last_ang_t = ang, now
            raw = {"ax": 0.0, "ay": 0.0, "az": 1.0, "gx": round(gx, 2), "gy": round(gy, 2), "gz": round(gz, 2)}
        return raw, ang

    def battery_v(self):
        name = self.caps.get("battery")
        if not name:
            return None
        try:
            v = getattr(Hiwonder, name)()
            return round(v / 1000.0, 2) if v > 100 else round(float(v), 2)
        except Exception:
            return None

    def rgb(self, r, g, b):
        if self.sonar and self.caps["rgb"]:
            try:
                self.sonar.setRGB(0, int(r), int(g), int(b))
                self.sonar.setRGB(1, int(r), int(g), int(b))
            except Exception:
                pass

    def beep(self, freq, ms):
        if self.buzzer:
            try:
                self.buzzer.playTone(int(freq), int(ms), False)
            except Exception:
                pass


# ---------------------------------------------------------------------------------------------------------
# WiFi
# ---------------------------------------------------------------------------------------------------------
def wifi_up():
    if network is None:
        raise RuntimeError("MicroPython 'network' module missing: this firmware build has no WiFi support")
    if config.WIFI_MODE == "sta":
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        if not wlan.isconnected():
            wlan.connect(config.STA_SSID, config.STA_PASSWORD)
            t0 = time.ticks_ms()
            while not wlan.isconnected():
                if time.ticks_diff(time.ticks_ms(), t0) > config.STA_CONNECT_TIMEOUT_S * 1000:
                    print("STA connect timeout, falling back to AP")
                    return wifi_ap()
                time.sleep_ms(200)
        print("STA connected:", wlan.ifconfig())
        return wlan
    return wifi_ap()


def wifi_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    try:
        ap.config(essid=config.AP_SSID, password=config.AP_PASSWORD, authmode=3)
    except Exception:
        ap.config(essid=config.AP_SSID, password=config.AP_PASSWORD)
    while not ap.active():
        time.sleep_ms(100)
    print("AP up:", ap.ifconfig())
    return ap


# ---------------------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------------------
class UsbLink:
    """The USB-serial REPL port (CH340, 115200) as a permanent bridge client: JSON lines in on stdin, out on
    stdout. print() output shares the port, so hosts must skip non-JSON lines. A UART cannot tell when the
    cable is pulled, so the watchdog, not link_lost, is what stops the dog on a USB drop."""

    def __init__(self):
        self._in = sys.stdin.buffer
        self._out = sys.stdout.buffer
        self._p = select.poll()
        self._p.register(sys.stdin, select.POLLIN)

    def recv(self, n):
        # stdin.read(n) blocks until n bytes arrive, so take only what is already buffered.
        out = b""
        while len(out) < n and self._p.poll(0):
            out += self._in.read(1)
        return out

    def send(self, data):
        self._out.write(data)

    def close(self):
        pass


class Bridge:
    def __init__(self, hal):
        self.hal = hal
        self.mode = "idle"
        self.stride = 0
        self.angle = 0
        self.tel_hz = config.TELEMETRY_HZ_DEFAULT
        self.last_msg = time.ticks_ms()
        self.last_tel = time.ticks_ms()
        self.last_poll = time.ticks_ms()
        self.action_until = 0
        self.t0 = time.ticks_ms()
        self.dist_cm = None
        self.batt_v = None
        self.imu_raw = None
        self.imu_ang = None
        self.low_batt_sent = False
        self.fallen = False
        self.sonar_filter = SonarFilter()
        self.clients = []
        self.bufs = {}
        self.running = True
        self.poller = select.poll()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("0.0.0.0", config.PORT))
        self.srv.listen(2)
        self.srv.setblocking(False)
        self.poller.register(self.srv, select.POLLIN)
        self.usb = UsbLink()
        self.clients.append(self.usb)
        self.bufs[id(self.usb)] = b""
        self.poller.register(sys.stdin, select.POLLIN)
        try:
            self.port = self.srv.getsockname()[1]
        except Exception:
            self.port = config.PORT

    def uptime(self):
        return time.ticks_diff(time.ticks_ms(), self.t0)

    # -- io --
    def send(self, c, msg):
        data = (json.dumps(msg) + "\n").encode()
        for attempt in range(3):
            try:
                c.send(data)
                return
            except OSError as e:
                if e.args and e.args[0] == EAGAIN and attempt < 2:
                    time.sleep_ms(2)  # TX buffer momentarily full under telemetry load: retry, don't drop the client
                    continue
                break
            except Exception:
                break
        self.drop(c)

    def broadcast(self, msg):
        for c in list(self.clients):
            self.send(c, msg)

    def drop(self, c):
        if c is self.usb:
            return  # never unplug the UART client; the watchdog covers a dead cable
        if c in self.clients:
            self.clients.remove(c)
            self.bufs.pop(id(c), None)
            try:
                self.poller.unregister(c)
            except Exception:
                pass
            try:
                c.close()
            except Exception:
                pass
        if not any(x is not self.usb for x in self.clients):
            self.safe_stop("link_lost")

    def hello(self, c):
        self.send(c, {"t": "hello", "proto": 1, "fw": config.FW_VERSION, "name": "MechDog", "caps": self.hal.caps})

    def accept(self):
        try:
            c, addr = self.srv.accept()
        except OSError:
            return
        c.setblocking(False)
        self.clients.append(c)
        self.bufs[id(c)] = b""
        self.poller.register(c, select.POLLIN)
        self.hello(c)
        print("client", addr)

    def read(self, c):
        try:
            data = c.recv(512)
        except OSError:
            return
        if not data:
            self.drop(c)
            return
        buf = self.bufs.get(id(c), b"") + data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(c, {"t": "err", "msg": "bad json"})
                continue
            if not isinstance(msg, dict) or "t" not in msg:
                self.send(c, {"t": "err", "msg": "bad message"})
                continue
            try:
                self.handle(c, msg)
            except Exception as e:  # a malformed field must never take the bridge (and its watchdog) down
                self.send(c, {"t": "err", "msg": "handler error: %s" % e, "for": msg.get("t")})
        self.bufs[id(c)] = buf

    # -- commands --
    def handle(self, c, msg):
        self.last_msg = time.ticks_ms()
        t = msg.get("t")
        mid = msg.get("id")
        clamped = False

        def ack(ok=True, **extra):
            a = {"t": "ack", "for": t, "ok": ok}
            if mid is not None:
                a["id"] = mid
            if clamped:
                a["clamped"] = True
            a.update(extra)
            self.send(c, a)

        def cl(key, lo, hi, default=0):
            nonlocal clamped
            v = num(msg.get(key, default), default)
            cv = int(clamp(v, lo, hi))
            clamped = clamped or (cv != v)
            return cv

        if t == "hello":
            self.hello(c)  # USB hosts attach without a connect event, so they ask for caps
        elif t == "ping":
            p = {"t": "pong", "ts": self.uptime()}
            if mid is not None:
                p["id"] = mid
            self.send(c, p)
        elif t == "hb":
            if self.mode == "safe" and not self.fallen:
                self.mode = "idle"
        elif t == "move":
            if self.batt_v is not None and self.batt_v < config.LOW_BATTERY_V:
                ack(False, msg="low_battery")
                return
            if self.fallen:
                ack(False, msg="fallen")
                return
            self.stride = cl("stride", -100, 100)
            self.angle = cl("angle", -30, 30)
            self.hal.move(self.stride, self.angle)
            self.mode = "walk" if (self.stride or self.angle) else "idle"
            ack()
        elif t == "stop" or t == "reset":
            self.stride = self.angle = 0
            self.hal.stop()
            self.mode = "idle"
            if t == "reset":
                self.hal.height(80)
                self.hal.gait(200, 300, 20)
            ack()
        elif t == "action":
            if self.batt_v is not None and self.batt_v < config.LOW_BATTERY_V:
                ack(False, msg="low_battery")
                return
            self.stride = self.angle = 0
            self.hal.stop()
            ok = self.hal.action(cl("id", 0, 15))
            self.mode = "action"
            self.action_until = time.ticks_add(time.ticks_ms(), 2500)
            ack(ok, msg=None if ok else "action_unsupported")
        elif t == "height":
            ack(self.hal.height(cl("mm", 50, 120, 80)))
        elif t == "gait":
            ok = self.hal.gait(cl("lift_ms", 50, 1000, 200), cl("contact_ms", 50, 1000, 300), cl("lift_mm", 5, 40, 20))
            ack(ok)
        elif t == "posture":
            ack(self.hal.posture(cl("x", -20, 20), cl("y", -20, 20)))
        elif t == "rgb":
            self.hal.rgb(cl("r", 0, 255), cl("g", 0, 255), cl("b", 0, 255))
            ack()
        elif t == "buzzer":
            self.hal.beep(cl("freq", 100, 8000, 2000), cl("ms", 10, 5000, 100))
            ack()
        elif t == "sub":
            self.tel_hz = cl("hz", 0, 50)
            ack(hz=self.tel_hz)
        else:
            self.send(c, {"t": "err", "msg": "unknown type %s" % t})

    # -- periodic --
    def safe_stop(self, reason):
        if self.mode == "walk":
            self.stride = self.angle = 0
            self.hal.stop()
            self.mode = "safe"
            self.broadcast({"t": "event", "name": "watchdog_stop", "reason": reason, "ts": self.uptime()})

    def tick(self):
        now = time.ticks_ms()
        if self.mode == "walk" and time.ticks_diff(now, self.last_msg) > config.WATCHDOG_MS:
            self.safe_stop("watchdog")
        if self.mode == "action" and time.ticks_diff(now, self.action_until) > 0:
            self.mode = "idle"
        if time.ticks_diff(now, self.last_poll) >= config.SENSOR_POLL_MS:
            self.last_poll = now
            self.poll_sensors()  # local safety runs whether or not anyone subscribed to telemetry
        if self.tel_hz > 0 and time.ticks_diff(now, self.last_tel) >= 1000 // self.tel_hz:
            self.last_tel = now
            self.telemetry()

    def poll_sensors(self):
        self.imu_raw, self.imu_ang = self.hal.imu_read()
        self.dist_cm = self.sonar_filter.update(self.hal.dist_cm())
        self.batt_v = self.hal.battery_v()
        self.check_fall()
        if config.OBSTACLE_GUARD and self.mode == "walk" and self.stride > 0 and self.dist_cm is not None \
                and self.dist_cm < config.OBSTACLE_STOP_CM:
            self.stride = self.angle = 0
            self.hal.stop()
            self.mode = "idle"
            self.broadcast({"t": "event", "name": "obstacle_stop", "ts": self.uptime()})
        if self.batt_v is not None and self.batt_v < config.LOW_BATTERY_V and not self.low_batt_sent:
            self.low_batt_sent = True
            self.safe_stop("low_battery")
            self.broadcast({"t": "event", "name": "low_battery", "batt_v": self.batt_v, "ts": self.uptime()})

    def check_fall(self):
        """Hiwonder's IOT lesson treats |angle| > 50° as an impact; we stop the servos and tell the brain."""
        ang = self.imu_ang
        if not ang or len(ang) < 2:
            return
        roll, pitch = abs(ang[0]), abs(ang[1])
        if not self.fallen and max(roll, pitch) > config.FALL_DEG:
            self.fallen = True
            self.stride = self.angle = 0
            self.hal.stop()
            self.mode = "safe"
            self.broadcast({"t": "event", "name": "fall", "roll": ang[0], "pitch": ang[1], "ts": self.uptime()})
        elif self.fallen and max(roll, pitch) < config.FALL_CLEAR_DEG:
            self.fallen = False
            self.broadcast({"t": "event", "name": "upright", "ts": self.uptime()})

    def telemetry(self):
        msg = {"t": "tel", "ts": self.uptime(), "mode": self.mode, "imu": self.imu_raw, "dist_cm": self.dist_cm,
               "batt_v": self.batt_v, "stride": self.stride, "angle": self.angle}
        if self.imu_ang is not None:
            msg["ang"] = self.imu_ang
        self.broadcast(msg)

    def run(self):
        print("bridge listening on", config.PORT, "and USB serial")
        self.hello(self.usb)
        while self.running:
            for sock, ev in self.poller.poll(config.LOOP_MS):
                if sock is self.srv:
                    self.accept()
                elif sock is sys.stdin:
                    self.read(self.usb)
                elif ev & select.POLLIN:
                    self.read(sock)
                else:
                    self.drop(sock)
            self.tick()
            if self.uptime() % 5000 < config.LOOP_MS:
                gc.collect()


def main():
    hal = DogHAL()
    print("HAL caps:", hal.caps)
    wifi_up()
    hal.rgb(0, 40, 0)
    hal.beep(1500, 80)
    Bridge(hal).run()


if __name__ == "__main__":
    main()
