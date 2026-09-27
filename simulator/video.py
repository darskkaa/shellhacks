"""Write sim frames to video: WebM (VP9, plays in every browser incl. Brave on Linux) and MP4 (H.264, for sharing).

    from simulator.video import write_video
    write_video(frames, durations_ms, out_dir / "demo")   # -> demo.webm, demo.mp4

Uses the static ffmpeg bundled with the imageio-ffmpeg wheel, so no system ffmpeg is needed.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

FPS = 20


def write_video(frames, durations_ms, base, fps=FPS):
    """frames: PIL images of equal size; durations_ms: how long each is shown. Returns the written paths."""
    assert frames and len(frames) == len(durations_ms), "need one duration per frame"
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    w, h = w - w % 2, h - h % 2  # yuv420p needs even dimensions
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    codecs = {
        ".webm": ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"],
        ".mp4": ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart"],
    }
    written = []
    for ext, codec in codecs.items():
        out = base.with_suffix(ext)
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-", *codec, "-pix_fmt", "yuv420p", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        carry = 0.0
        for frame, ms in zip(frames, durations_ms):
            # Hold each frame for its duration, carrying the rounding remainder so total time stays exact.
            carry += ms * fps / 1000
            n, carry = int(carry), carry - int(carry)
            data = frame.convert("RGB").crop((0, 0, w, h)).tobytes()
            for _ in range(n):
                proc.stdin.write(data)
        proc.stdin.close()
        if proc.wait(timeout=600) != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")
        written.append(out)
    return written


if __name__ == "__main__":
    import tempfile

    demo = [Image.new("RGB", (64, 48), (i * 40, 0, 0)) for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        paths = write_video(demo, [500, 250, 250, 1000], Path(d) / "check")
        assert all(p.stat().st_size > 0 for p in paths)
        print("ok", [p.name for p in paths])
