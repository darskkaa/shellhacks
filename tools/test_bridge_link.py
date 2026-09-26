"""Drive RealDogController over a fake USB-serial bridge (a pty) and MechDogBridgeClient over TCP, including
cable drops. Linux/macOS only (pty). Run: python tools/test_bridge_link.py"""

import json
import os
import pty
import select
import socket
import sys
import threading
import time
import tty
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemini_dog_agent import MechDogBridgeClient  # noqa: E402
from real_world_escort import RealDogController  # noqa: E402


def fake_serial_bridge(fd, log, stop):
    os.write(fd, b"MicroPython print() chatter\r\n")
    buf = b""
    while not stop.is_set():
        if not select.select([fd], [], [], 0.1)[0]:
            continue
        buf += os.read(fd, 256)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            msg = json.loads(line)
            log.append(msg)
            if msg["t"] == "hello":
                os.write(fd, b'{"t":"hello","fw":"fake"}\n')
            elif msg["t"] == "sub":
                os.write(fd, b'{"t":"tel","dist_cm":80.0,"batt_v":7.6}\n')


class BridgeLinkTest(unittest.TestCase):
    def test_serial(self):
        master, slave = pty.openpty()
        tty.setraw(slave)
        log, stop = [], threading.Event()
        threading.Thread(target=fake_serial_bridge, args=(master, log, stop), daemon=True).start()
        dog = RealDogController(serial_port=os.ttyname(slave))
        time.sleep(0.5)
        self.assertTrue(dog.connected)
        self.assertEqual((dog.batt_v, dog.dist_cm), (7.6, 80.0))

        self.assertTrue(dog.walk_safe(is_crosswalk=True, fast_mode=True))
        time.sleep(0.2)
        self.assertEqual([m["t"] for m in log], ["hello", "sub", "gait", "move"])
        self.assertEqual((log[-2]["lift_ms"], log[-1]["stride"]), (80, 100))

        os.write(master, b'{"t":"tel","dist_cm":20.0}\n')
        time.sleep(0.3)
        self.assertEqual(log[-1]["t"], "stop")
        self.assertFalse(dog.is_walking)

        dog.is_walking = True
        os.write(master, b'{"t":"hello","fw":"fake"}\n')  # ESP32 rebooted mid-walk
        time.sleep(0.3)
        self.assertEqual([m["t"] for m in log[-2:]], ["stop", "sub"])
        self.assertIsNone(dog.gait)
        self.assertFalse(dog.is_walking)

        stop.set()
        time.sleep(0.3)
        os.close(master)  # cable yanked
        os.close(slave)
        time.sleep(1.5)
        self.assertFalse(dog.connected)
        dog.link.close()

    def test_tcp_reconnect(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)

        def accept():
            c, _ = srv.accept()
            c.sendall(b'{"t":"hello","fw":"tcp"}\n')
            return c, c.makefile("rb")

        cli = MechDogBridgeClient(port=srv.getsockname()[1])
        c, f = accept()
        time.sleep(0.2)
        cli.set_speed_mode("sprint")
        cli.move(80, 0)
        self.assertEqual([json.loads(f.readline())["t"] for _ in range(2)], ["gait", "move"])

        f.close()
        c.close()
        time.sleep(0.3)
        self.assertFalse(cli.link.connected)

        c, f = accept()
        time.sleep(0.4)
        self.assertTrue(cli.link.connected)
        self.assertIsNone(cli.gait)
        cli.set_speed_mode("sprint")
        self.assertEqual(json.loads(f.readline())["t"], "gait")
        cli.close()
        f.close()
        c.close()
        srv.close()


if __name__ == "__main__":
    unittest.main()
