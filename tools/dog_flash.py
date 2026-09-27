"""Copy files to and from the MechDog's flash over USB serial.

mpremote does not work on this dog: it enters the raw REPL with a soft reset, and a soft reset panics the
Hiwonder firmware (IntegerDivideByZero). This tool instead hard-resets the ESP32 through the CH340's RTS line,
interrupts main.py right after the IMU init (before WiFi/Bluetooth start), and uses the raw REPL directly.
The dog is hard-reset again at the end so the new main.py runs.

  python tools/dog_flash.py --port COM4 get main.py config.py --to robot_dump/device_backup
  python tools/dog_flash.py --port COM4 put robot_dump/main.py robot_dump/config.py
  python tools/dog_flash.py --ble put robot_dump/main.py          # no cable: through the running bridge

--ble uploads through the bridge's fopen/fwrite/fclose commands over Bluetooth (needs a main.py that has them,
flashed once over USB). Each piece is acknowledged before the next is sent; ~150 bytes/s, so main.py takes about
3 minutes. The file is written as NAME.tmp and renamed only when complete, so a dropped link is harmless.
"""
import argparse
import base64
import binascii
import os
import queue
import time

import serial

CHUNK = 256  # bytes of file data per raw-REPL exec


class RawRepl:
    def __init__(self, port):
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 0.1
        self.s.dtr = self.s.rts = False  # CH340 auto-reset wiring: only pulse RTS deliberately
        self.s.open()

    def _read_until(self, marker, timeout):
        buf, t0 = b"", time.time()
        while time.time() - t0 < timeout:
            buf += self.s.read(self.s.in_waiting or 1)
            if marker in buf:
                return buf
        raise TimeoutError(f"waiting for {marker!r}; got {buf[-200:]!r}")

    def enter(self):
        """Get to the raw REPL. First try interrupting the running bridge (Ctrl-C) with no reset: resetting makes
        the servos jump to their start pose, and that current spike can knock the CH340 off USB mid-flash. Fall
        back to a hard reset that stops main.py before it brings up the radio."""
        self.s.reset_input_buffer()
        self.s.write(b"\r\x03\x03")
        try:
            self._read_until(b">>> ", 3)
            self.s.write(b"\x01")  # raw REPL
            self._read_until(b"raw REPL; CTRL-B to exit\r\n>", 5)
            return
        except TimeoutError:
            pass
        self.s.reset_input_buffer()
        self.s.rts = True
        time.sleep(0.15)
        self.s.rts = False
        self._read_until(b"MPU6050 init", 15)
        self.s.write(b"\x03\x03")
        time.sleep(0.2)
        self.s.write(b"\x03")
        self._read_until(b">>> ", 5)
        self.s.write(b"\x01")  # raw REPL
        self._read_until(b"raw REPL; CTRL-B to exit\r\n>", 5)

    def exec(self, code, timeout=10):
        self.s.write(code.encode() + b"\x04")
        self._read_until(b"OK", timeout)
        out = self._read_until(b"\x04>", timeout)
        stdout, _, rest = out.partition(b"\x04")
        err = rest.rstrip(b"\x04>")
        if err.strip():
            raise RuntimeError(err.decode(errors="replace"))
        return stdout

    def get(self, name):
        out = self.exec(f"import ubinascii\nwith open({name!r},'rb') as f: print(ubinascii.hexlify(f.read()).decode())")
        return binascii.unhexlify(out.strip())

    def put(self, name, data):
        tmp = name + ".tmp"
        self.exec(f"f=open({tmp!r},'wb')")
        for i in range(0, len(data), CHUNK):
            self.exec(f"f.write(bytes.fromhex({data[i:i + CHUNK].hex()!r}))")
        self.exec("f.close()")
        # Only replace the live file once the whole upload landed, so an interrupted flash can't brick boot.
        self.exec(f"import os\ntry: os.remove({name!r})\nexcept OSError: pass\nos.rename({tmp!r},{name!r})")
        size = int(self.exec(f"import os\nprint(os.stat({name!r})[6])").strip())
        if size != len(data):
            raise RuntimeError(f"{name}: wrote {size} bytes, expected {len(data)}")

    def reset(self):
        self.s.write(b"\x02")  # leave raw REPL
        self.s.rts = True
        time.sleep(0.15)
        self.s.rts = False
        self.s.close()


BLE_CHUNK = 384  # raw bytes per fwrite: 512 base64 chars, well under the bridge's line buffer


def ble_put(paths, name, reset):
    from real_world_escort import BridgeLink
    acks, ready = queue.Queue(), queue.Queue()
    link = BridgeLink(serial_port=f"ble:{name}", tag="BLE",
                      on_msg=lambda m: acks.put(m) if m.get("t") == "ack" else None,
                      on_ready=lambda first: ready.put(True))
    link.start()
    try:
        ready.get(timeout=40)
    except queue.Empty:
        link.close()
        raise SystemExit(f"no bridge hello from {name!r} over Bluetooth")
    next_id = [1000]

    def call(msg, timeout=20):
        next_id[0] += 1
        msg["id"] = next_id[0]
        link.send(msg)
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                a = acks.get(timeout=timeout)
            except queue.Empty:
                break
            if a.get("id") == msg["id"]:
                if not a.get("ok"):
                    raise RuntimeError(f"{msg['t']} refused: {a.get('msg')}")
                return a
        raise TimeoutError(f"no ack for {msg['t']}")

    try:
        for path in paths:
            with open(path, "rb") as f:
                data = f.read()
            target = os.path.basename(path)
            t0 = time.time()
            call({"t": "fopen", "name": target})
            for i in range(0, len(data), BLE_CHUNK):
                call({"t": "fwrite", "d": base64.b64encode(data[i:i + BLE_CHUNK]).decode()})
                done = min(len(data), i + BLE_CHUNK)
                print(f"  {target}: {done}/{len(data)} bytes ({done / max(1e-6, time.time() - t0):.0f} B/s)",
                      flush=True)
            call({"t": "fclose", "size": len(data)})
            print(f"put {path} ({len(data)} bytes) over Bluetooth in {time.time() - t0:.0f} s")
        if reset:
            call({"t": "reset"})
            print("dog rebooting into the new code")
    finally:
        link.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--ble", action="store_true",
                    help="upload over Bluetooth through the running bridge instead of USB (put only)")
    ap.add_argument("--ble-name", default="MechDog", help="--ble: the dog's Bluetooth name (config.BLE_NAME)")
    ap.add_argument("--no-reset", action="store_true", help="--ble: don't reboot the dog after uploading")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("names", nargs="+")
    g.add_argument("--to", default=".")
    p = sub.add_parser("put")
    p.add_argument("paths", nargs="+")
    args = ap.parse_args()

    if args.ble:
        if args.cmd != "put":
            raise SystemExit("--ble supports put only")
        ble_put(args.paths, args.ble_name, reset=not args.no_reset)
        return
    repl = RawRepl(args.port)
    try:
        repl.enter()
        if args.cmd == "get":
            os.makedirs(args.to, exist_ok=True)
            for name in args.names:
                data = repl.get(name)
                with open(os.path.join(args.to, name), "wb") as f:
                    f.write(data)
                print(f"got {name} ({len(data)} bytes) -> {args.to}")
        else:
            for path in args.paths:
                with open(path, "rb") as f:
                    data = f.read()
                repl.put(os.path.basename(path), data)
                print(f"put {path} ({len(data)} bytes)")
    finally:
        repl.reset()
        print("dog reset")


if __name__ == "__main__":
    main()
