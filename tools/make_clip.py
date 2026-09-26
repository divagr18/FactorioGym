"""Cut a raw demo recording into a clip ready to post on X.

    uv run python tools/make_clip.py runtime/videos/demo-raw.mp4 runtime/videos/demo-x.mp4

X takes H.264 in an MP4, 4:2:0, up to 2:20 and 512 MB. The raw recording from
`tools/demo_play.py --record` is four to five minutes, most of it the agent
standing by a site waiting for plates, and the tool writes a
`<recording>.timeline.json` beside it saying when it was only waiting. Those
stretches are sped up hard (the drills and furnaces keep moving, just faster);
everything the agent does -- walking, placing, fuelling, taking plates -- keeps
its real speed unless the clip still would not fit, and then it is sped up
evenly too. The final wide shot is kept at real speed, trimmed to `--hold`.

Without a timeline the whole recording is sped up evenly to fit.

Output: 1920x1080 (or `--height 720`), 30 fps, H.264 High, yuv420p, CRF 20 with
a bitrate cap, `+faststart` so it plays while it downloads. No audio track.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

X_MAX_SECONDS = 140.0
X_MAX_BYTES = 512 * 1024 * 1024


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


def segments_from_timeline(timeline: dict, duration: float, min_idle: float, hold: float):
    """[(start, end, kind)] covering the recording; kind is 'act', 'idle' or 'hold'."""
    phases = dict((name, t) for name, t in timeline.get("phases", []))
    hold_start = phases.get("hold")
    end = min(duration, timeline.get("end", duration))
    idle = sorted(
        (a, min(b, end)) for a, b in timeline.get("idle", []) if b - a >= min_idle and a < end
    )
    segments, cursor = [], 0.0
    for a, b in idle:
        if a > cursor:
            segments.append((cursor, a, "act"))
        segments.append((max(a, cursor), b, "idle"))
        cursor = b
    last = hold_start if hold_start is not None and hold_start > cursor else end
    if last > cursor:
        segments.append((cursor, last, "act"))
    if hold_start is not None and hold_start >= cursor:
        segments.append((hold_start, min(end, hold_start + hold), "hold"))
    return [(a, b, k) for a, b, k in segments if b - a > 0.05]


def choose_speeds(segments, target: float, max_idle_speed: float, max_act_speed: float):
    """Idle stretches first, up to `max_idle_speed`; then everything else evenly."""
    length = {k: sum(b - a for a, b, kk in segments if kk == k) for k in ("act", "idle", "hold")}
    act_speed, idle_speed = 1.0, 1.0
    fixed = length["act"] + length["hold"]
    if fixed + length["idle"] > target:
        room = target - fixed
        idle_speed = length["idle"] / room if room > 0 else max_idle_speed
        idle_speed = min(max(idle_speed, 1.0), max_idle_speed)
    if length["act"] + length["hold"] + length["idle"] / idle_speed > target:
        room = target - length["hold"] - length["idle"] / idle_speed
        act_speed = (
            min(max(length["act"] / room, 1.0), max_act_speed) if room > 0 else max_act_speed
        )
    total = length["act"] / act_speed + length["idle"] / idle_speed + length["hold"]
    return {"act": act_speed, "idle": idle_speed, "hold": 1.0}, total


def build_filter(segments, speeds, height: int, fps: int) -> str:
    width = height * 16 // 9
    parts, labels = [], []
    for i, (a, b, kind) in enumerate(segments):
        speed = speeds[kind]
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{speed:.4f}[s{i}]")
        labels.append(f"[s{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0[cat]")
    parts.append(f"[cat]fps={fps},scale={width}:{height}:flags=lanczos,format=yuv420p[out]")
    return ";".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--timeline", type=Path, help="default: <raw>.timeline.json")
    parser.add_argument("--target", type=float, default=138.0, help="clip length cap, s")
    parser.add_argument("--height", type=int, default=1080, choices=(720, 1080))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold", type=float, default=8.0, help="s of the final wide shot")
    parser.add_argument("--min-idle", type=float, default=2.0, help="shorter waits stay 1x")
    parser.add_argument("--max-idle-speed", type=float, default=8.0)
    parser.add_argument("--max-act-speed", type=float, default=2.0)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args()

    target = min(args.target, X_MAX_SECONDS)
    duration = probe_duration(args.raw)
    timeline_path = args.timeline or args.raw.with_suffix(".timeline.json")
    if timeline_path.exists():
        timeline = json.loads(timeline_path.read_text("utf-8"))
        segments = segments_from_timeline(timeline, duration, args.min_idle, args.hold)
        speeds, total = choose_speeds(segments, target, args.max_idle_speed, args.max_act_speed)
    else:
        print(f"no timeline at {timeline_path}; speeding the whole recording up evenly")
        speed = max(1.0, duration / target)
        segments, speeds, total = [(0.0, duration, "act")], {"act": speed}, duration / speed
    print(
        f"raw {duration:.1f}s -> clip {total:.1f}s in {len(segments)} segments; "
        f"speeds {', '.join(f'{k} x{v:.2f}' for k, v in speeds.items())}"
    )
    if total > target + 0.5:
        print(f"warning: still {total:.1f}s; raise --max-idle-speed or --max-act-speed")
    if args.dry_run:
        for a, b, k in segments:
            print(f"  {a:7.2f} - {b:7.2f}  {k}")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    maxrate = "12M" if args.height == 1080 else "6M"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-stats",
        "-i", str(args.raw),
        "-filter_complex", build_filter(segments, speeds, args.height, args.fps),
        "-map", "[out]", "-an",
        "-t", f"{target:.2f}",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
        "-maxrate", maxrate, "-bufsize", "24M",
        "-profile:v", "high", "-level:v", "4.2", "-pix_fmt", "yuv420p",
        "-r", str(args.fps), "-g", str(args.fps * 2),
        "-movflags", "+faststart",
        str(args.out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    size = args.out.stat().st_size
    length = probe_duration(args.out)
    print(f"wrote {args.out}: {length:.1f}s, {size / 1e6:.1f} MB")
    if size > X_MAX_BYTES or length > X_MAX_SECONDS:
        print("warning: over X's limits (140 s, 512 MB)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
