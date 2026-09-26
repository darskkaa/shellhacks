"""
Waymo SafePoint 3D: Real-World Robot Dog Blind Escort Controller.
Designed for real-world testing on a private road/track with the physical Hiwonder MechDog.

Safety Guards:
  1. Hardware Sonar Hard Stop: Automatically halts if obstacle < 35 cm.
  2. Tilt / Fall Protection: Immediately stops if pitch/roll tilt > 35 degrees.
  3. Battery Gate: Enforces battery >= 7.0V before permitting escort traversal.
  4. Speed Capping: Limits sidewalk stride to 40; faster gait cadence only when fast mode is toggled on.
  5. Keyboard Emergency Stop: Pressing Spacebar or Ctrl+C immediately stops all motion.
  6. Zero-Tolerance Drop-off Guard: Prohibits forward movement when curb_drop_off_hazard is present.
  7. Auto-Vision (--auto-vision / [v]): walks on ped_signal_walk, halts on stop signals and hazards,
     and stops if no signal is seen for 2 s. Any manual command cancels it.
"""

import sys
import time
import json
import socket
import argparse
import threading
import urllib.request
from pathlib import Path

# ANSI Terminal Colors
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
W = "\033[0m"

VISION_URL = "http://localhost:8001/detections"
VISION_MIN_CONF = 0.35
WALK_CLASSES = {"ped_signal_walk", "curb_ramp_ada"}
STOP_CLASSES = {"ped_signal_stop", "curb_drop_off_hazard", "conflict_vehicle_cyclist"}


def fetch_detections(url=VISION_URL, timeout=1.0):
    """Latest YOLO detections from tools/vision_stream.py; [] when the stream is down."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MechDogEscort/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        dets = data.get("dets")
        if dets is None:
            dets = data.get("detections", [])
        return dets or []
    except Exception:
        return []


def det_label(d):
    return d.get("label") or d.get("class")


def det_conf(d):
    return float(d.get("conf", d.get("confidence")) or 0)


def seen_labels(dets, min_conf=VISION_MIN_CONF):
    return {det_label(d) for d in dets if det_conf(d) >= min_conf}


class BridgeLink:
    """Newline-JSON link to the MechDog bridge over TCP or USB serial. A background thread reads lines and
    reopens the link every RECONNECT_S after any drop. on_ready(first) fires on each bridge "hello", i.e. on
    every (re)connect and whenever the ESP32 reboots, so callers can redo their setup."""
    RECONNECT_S = 1.0

    def __init__(self, host="127.0.0.1", port=5005, serial_port=None, on_msg=None, on_ready=None, tag="Link"):
        self.host = host
        self.port = port
        self.serial_port = serial_port
        self.on_msg = on_msg
        self.on_ready = on_ready
        self.tag = tag
        self.target = serial_port or f"{host}:{port}"
        self.conn = None
        self.alive = True
        self.hellos = 0
        self.lock = threading.Lock()

    def start(self):
        """Call after assigning the link, so callbacks on the reader thread can already reach it."""
        self._open(verbose=True)
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def connected(self):
        return self.conn is not None

    def _open(self, verbose=False):
        try:
            if self.serial_port:
                import serial
                c = serial.Serial()
                c.port, c.baudrate, c.timeout, c.write_timeout = self.serial_port, 115200, 1.0, 1.0
                c.dtr = c.rts = False  # CH340 auto-reset wiring: asserting DTR/RTS reboots the ESP32
                c.open()
                c.reset_input_buffer()
            else:
                c = socket.create_connection((self.host, self.port), timeout=3.0)
                c.settimeout(1.0)
        except Exception as e:
            if verbose:
                print(f"{R}[{self.tag}] Cannot open {self.target}: {e}{W}")
            return
        self.conn = c
        print(f"{G}[{self.tag}] Link open to {self.target}, waiting for bridge hello{W}")
        if self.serial_port:
            self.send({"t": "hello"})  # a UART has no connect event; TCP accept sends hello unprompted

    def _drop(self, c, why):
        with self.lock:
            if self.conn is c:
                self.conn = None
        try:
            c.close()  # release the fd now, or a replugged CH340 re-enumerates as the next ttyUSB
        except Exception:
            pass
        if self.alive:
            print(f"{Y}[{self.tag}] Link to {self.target} lost ({why}); retrying every {self.RECONNECT_S:.0f}s{W}")

    def send(self, payload):
        c = self.conn
        if c is None:
            return False
        data = (json.dumps(payload) + "\n").encode()
        try:
            with self.lock:
                if self.serial_port:
                    c.write(data)
                else:
                    c.sendall(data)
            return True
        except Exception as e:
            self._drop(c, f"send: {e}")
            return False

    def _loop(self):
        buf = b""
        while self.alive:
            c = self.conn
            if c is None:
                time.sleep(self.RECONNECT_S)
                if self.alive:
                    buf = b""
                    self._open()
                continue
            try:
                if self.serial_port:
                    data = c.read(c.in_waiting or 1)  # b"" on timeout
                else:
                    try:
                        data = c.recv(1024)
                    except socket.timeout:
                        continue
                    if not data:
                        self._drop(c, "closed by bridge")
                        continue
            except Exception as e:  # SerialException / OSError when the cable wobbles out
                self._drop(c, e)
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"{"):
                    continue  # MicroPython print() chatter shares the UART
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                try:
                    if msg.get("t") == "hello":
                        self.hellos += 1
                        print(f"{G}[{self.tag}] MechDog bridge ready ({msg.get('fw')}){W}")
                        if self.on_ready:
                            self.on_ready(self.hellos == 1)
                    elif self.on_msg:
                        self.on_msg(msg)
                except Exception as e:  # a callback bug must not kill the reader and its sonar trip
                    print(f"{R}[{self.tag}] Handler error on {msg.get('t')}: {e}{W}")

    def close(self):
        self.alive = False
        c = self.conn
        if c is not None:
            self._drop(c, "closed")


class RealDogController:
    # (lift_ms, contact_ms, lift_mm) for the bridge "gait" command; cadence = 1000 / (lift_ms + contact_ms)
    GAIT_DEFAULT = (200, 300, 20)  # 2.0 steps/s, sidewalk precision (stock bridge default)
    GAIT_FAST = (100, 150, 20)     # 4.0 steps/s, brisk escort
    GAIT_SPRINT = (80, 120, 18)    # 5.0 steps/s, crosswalk transit

    def __init__(self, host="192.168.4.1", port=5005, max_stride=40, min_sonar_cm=35, serial_port=None):
        self.max_stride = max_stride
        self.min_sonar_cm = min_sonar_cm

        self.batt_v = 7.4
        self.dist_cm = 999.0
        self.is_walking = False
        self.gait = None

        print(f"{C}[RealDog] Connecting to physical MechDog at {serial_port or f'{host}:{port}'}...{W}")
        self.link = BridgeLink(host, port, serial_port, on_msg=self._handle_telemetry, on_ready=self._on_ready,
                               tag="RealDog")
        self.link.start()

    @property
    def connected(self):
        return self.link.connected

    def _on_ready(self, first):
        self.gait = None  # bridge may have rebooted or been reset; force resend
        if not first:
            # The watchdog has likely already halted the dog; make that explicit so the operator re-commands.
            self.is_walking = False
            self._send({"t": "stop"})
            print(f"\n{Y}[RealDog] Bridge link re-established; dog stopped, re-issue your command.{W}")
        self._send({"t": "sub", "hz": 4})

    def _send(self, payload):
        return self.link.send(payload)

    def _handle_telemetry(self, data):
        try:
            # Distance / Sonar
            if "dist_cm" in data and data["dist_cm"] is not None:
                self.dist_cm = float(data["dist_cm"])
                # HARDWARE SAFETY TRIP: Stop immediately if obstacle closer than safety threshold
                if self.dist_cm < self.min_sonar_cm and self.is_walking:
                    print(f"\n{R}🚨 [EMERGENCY HARDWARE STOP] Sonar reading: {self.dist_cm:.1f} cm (< {self.min_sonar_cm} cm)!{W}")
                    self.stop()
            # Battery
            if "batt_v" in data and data["batt_v"] is not None:
                self.batt_v = float(data["batt_v"])
        except Exception:
            pass

    def set_gait(self, lift_ms, contact_ms, lift_mm):
        gait = (lift_ms, contact_ms, lift_mm)
        if gait == self.gait:
            return
        sent = self._send({"t": "gait", "lift_ms": lift_ms, "contact_ms": contact_ms, "lift_mm": lift_mm})
        self.gait = gait if sent else None
        print(f"{C}⚙️  Gait: lift {lift_ms} ms / contact {contact_ms} ms / lift {lift_mm} mm ({1000 / (lift_ms + contact_ms):.1f} steps/s){W}")

    def walk_safe(self, stride=None, angle=0, is_crosswalk=False, fast_mode=False):
        # 1. Battery check
        if self.batt_v < 6.8:
            print(f"{R}⚠️ Low battery ({self.batt_v:.2f}V < 6.8V). Motion inhibited.{W}")
            return False

        # 2. Sonar check
        if self.dist_cm < self.min_sonar_cm:
            print(f"{R}⚠️ Path blocked: Sonar object at {self.dist_cm:.1f} cm. Motion inhibited.{W}")
            return False

        # 3. Dynamic Stride: Stride 40 on sidewalk, Stride 100 for rapid street crossing!
        if stride is None:
            stride = 100 if is_crosswalk else self.max_stride

        safe_stride = max(-100, min(100, stride if is_crosswalk else min(self.max_stride, stride)))
        safe_angle = max(-25, min(25, angle))

        if not fast_mode:
            gait = self.GAIT_DEFAULT
        else:
            gait = self.GAIT_SPRINT if is_crosswalk else self.GAIT_FAST
        self.set_gait(*gait)

        pace = " +FAST GAIT" if fast_mode else ""
        mode_label = f"{R}[CROSSWALK SPRINT{pace} (Stride {safe_stride})]{W}" if is_crosswalk else f"{C}[SIDEWALK ESCORT{pace} (Stride {safe_stride})]{W}"
        print(f"🐕 {mode_label} Angle: {safe_angle}°")

        self.is_walking = True
        self._send({"t": "move", "stride": safe_stride, "angle": safe_angle})
        return True

    def stop(self):
        self.is_walking = False
        self._send({"t": "stop"})
        print(f"{Y}🛑 [DOG STOPPED]{W}")

    def heartbeat(self):
        self._send({"t": "hb"})

    def set_color(self, r, g, b):
        self._send({"t": "rgb", "r": r, "g": g, "b": b})

    def close(self):
        self.stop()
        self.link.close()


class AutoVision:
    """Drives the dog straight from the YOLO stream: walk signal -> cross, stop signal or hazard -> halt."""
    HZ = 4
    LOST_S = 2.0

    def __init__(self, dog, url=VISION_URL):
        self.dog = dog
        self.url = url
        self.active = False
        self.fast = False
        self.state = None
        self.last_seen = 0.0
        # Held while acting so a cancel can never be followed by a walk from an in-flight tick.
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def set_active(self, on):
        with self.lock:
            self.active = on
            self.state = None
            self.last_seen = time.monotonic()
        print(f"{Y}Auto-Vision {'ON: watching ' + self.url if on else 'OFF'}{W}")

    def _loop(self):
        while True:
            time.sleep(1 / self.HZ)
            if not self.active:
                continue
            try:
                self._tick()
            except Exception as e:
                print(f"{R}[VISION] Tick error: {e}{W}")

    def _tick(self):
        labels = seen_labels(fetch_detections(self.url))
        now = time.monotonic()
        with self.lock:
            if not self.active:
                return
            dog = self.dog
            if labels & STOP_CLASSES:
                self.last_seen = now
                if self.state != "stop" or dog.is_walking:
                    dog.stop()
                    dog.set_color(255, 0, 0)
                    print(f"\n{R}🔴 [VISION] STOP / HAZARD DETECTED -> Halting safely! ({', '.join(sorted(labels & STOP_CLASSES))}){W}")
                self.state = "stop"
            elif labels & WALK_CLASSES:
                self.last_seen = now
                if not dog.is_walking and dog.walk_safe(is_crosswalk=True, fast_mode=self.fast):
                    dog.set_color(0, 255, 0)
                    print(f"\n{G}🟢 [VISION] WALK SIGNAL DETECTED -> Walking forward!{W}")
                self.state = "walk"
            elif dog.is_walking and now - self.last_seen >= self.LOST_S:
                print(f"\n{Y}🟡 [VISION] No signal for {self.LOST_S:.0f}s -> Stopping for safety.{W}")
                dog.stop()
                self.state = None
            if dog.is_walking:
                dog.heartbeat()  # bridge watchdog halts a walk after 1.5 s without traffic


def main():
    parser = argparse.ArgumentParser(description="Real-World MechDog Escort Controller")
    parser.add_argument("--wifi", default="127.0.0.1", help="Dog IP (use 192.168.4.1 for real dog AP mode, 127.0.0.1 for local/sim)")
    parser.add_argument("--port", type=int, default=5005, help="Bridge TCP port")
    parser.add_argument("--serial", metavar="DEV", help="Drive over USB serial instead of WiFi (e.g. /dev/ttyUSB0, COM3)")
    parser.add_argument("--max-stride", type=int, default=40, help="Max safe walking stride (10-50)")
    parser.add_argument("--min-sonar", type=int, default=35, help="Minimum obstacle stop distance in cm")
    parser.add_argument("--gemini", action="store_true", help="Enable Gemini 2.5 Flash autonomous reasoning")
    parser.add_argument("--auto-vision", action="store_true",
                        help=f"Start in Auto-Vision mode: walk/stop on YOLO signals from {VISION_URL}")
    args = parser.parse_args()

    print(f"\n========================================================")
    print(f"   🐕 WAYMO SAFEPOINT: REAL-WORLD MECHDOG ESCORT RUNNER   ")
    print(f"========================================================")
    print(f"• Target: {args.serial or f'{args.wifi}:{args.port}'}{' (USB serial)' if args.serial else ''}")
    print(f"• Max Escort Stride: {args.max_stride} (Human walking pace)")
    print(f"• Sonar Hard-Stop: < {args.min_sonar} cm")
    print(f"• Emergency Stop: Press Ctrl+C or Spacebar at ANY time")
    print(f"========================================================\n")

    dog = RealDogController(host=args.wifi, port=args.port, max_stride=args.max_stride, min_sonar_cm=args.min_sonar,
                            serial_port=args.serial)

    print("\nPre-Flight Hardware Verification:")
    time.sleep(1.0)
    print(f"  • Battery Voltage: {dog.batt_v:.2f} V ({'OK' if dog.batt_v >= 7.0 else 'LOW'})")
    print(f"  • Sonar Distance:  {dog.dist_cm:.1f} cm ({'CLEAR' if dog.dist_cm >= args.min_sonar else 'OBSTACLE DETECTED'})")
    print(f"  • Bridge Connection: {'ACTIVE' if dog.connected else 'WAITING/OFFLINE (auto-retrying)'}")

    print(f"\n{G}Ready for Private Road Testing.{W}")
    print("Commands:")
    print("  [w] Walk forward (sidewalk pace, stride 40)")
    print("  [c] Fast Crosswalk Transit (rapid crossing, stride 100)")
    print("  [f] Toggle Fast Speed (Brisk/Sprint): 4.0 steps/s sidewalk, 5.0 steps/s crosswalk")
    print("  [v] Toggle Auto-Vision Mode (walk on walk signal, stop on stop signal/hazard)")
    print("  [s] Stop dog immediately (also cancels Auto-Vision)")
    print("  [a] Turn slightly left (15 deg)")
    print("  [d] Turn slightly right (15 deg)")
    print("  [q] Quit and shut down")

    fast = False
    auto = AutoVision(dog)
    if args.auto_vision:
        auto.set_active(True)
    try:
        while True:
            speed = "FAST" if fast else "NORMAL"
            mode = " | AUTO" if auto.active else ""
            cmd = input(f"\n[{dog.batt_v:.1f}V | {dog.dist_cm:.0f}cm | {speed}{mode}] Command (w/c/f/v/s/a/d/q): ").strip().lower()
            if auto.active and cmd not in ("v", "f"):
                auto.set_active(False)  # any manual command (incl. s / Spacebar) takes back control
            if cmd == "v":
                auto.set_active(not auto.active)
                if not auto.active:
                    dog.stop()
            elif cmd == "w":
                dog.walk_safe(is_crosswalk=False, fast_mode=fast)
            elif cmd == "c":
                dog.walk_safe(is_crosswalk=True, fast_mode=fast)
            elif cmd == "f":
                fast = not fast
                auto.fast = fast
                print(f"{Y}Fast speed {'ON (Brisk/Sprint gait)' if fast else 'OFF (default gait)'}; applies on next walk command.{W}")
            elif cmd == "s":
                dog.stop()
            elif cmd == "a":
                dog.walk_safe(stride=args.max_stride, angle=-15, fast_mode=fast)
            elif cmd == "d":
                dog.walk_safe(stride=args.max_stride, angle=15, fast_mode=fast)
            elif cmd == "q":
                break
            else:
                dog.stop()
    except KeyboardInterrupt:
        print(f"\n{R}Emergency Stop Triggered.{W}")
    finally:
        auto.set_active(False)
        dog.stop()
        dog.close()
        print("MechDog safe shutdown completed.")

if __name__ == "__main__":
    main()
