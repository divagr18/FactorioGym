"""Cut a demo recording into a short showreel: the agent at work, a grid that
keeps splitting, then a title card.

    uv run --with opencv-python-headless python tools/make_showreel.py \
        runtime/videos/demo-raw.mp4 runtime/videos/showreel.mp4 --music
    ... --stills 3,12,20     # save those frames as PNG instead of a video

Structure (1920x1080, 60 fps, about 24 s):
  1. the agent at work, full screen (shown at 2x);
  2. a 3x2 grid of six moments, labelled with the results (top) and the
     features (bottom);
  3. a 6x4 grid (24 moments), then 12x8 (96);
  4. the title card: plain type on black.

Every change is a jump cut: no transitions. Every frame before the title gets
bloom, grain, a little chromatic aberration and a vignette. Footage is sped up
(2x on the opening shot, 3-6x in the grids); the labels say so. With --music, a
synthesised soundtrack is mixed in (nothing sampled).
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 60
ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "runtime" / "fonts"
SYS_FONTS = Path("C:/Windows/Fonts")
BG = (9, 11, 14)
INK = (244, 239, 230)
DIM = (244, 239, 230, 170)
AMBER = (255, 150, 40)
GRADE = "eq=contrast=1.12:saturation=1.18:brightness=0.015:gamma=1.04"
GAPS = (14, 8, 4)  # between cells, and round the edge, at 3x2, 6x4 and 12x8
CELL_ASPECT = ((W - 7 * GAPS[1]) / 6) / ((H - 5 * GAPS[1]) / 4)
# The part of the raw frame the grids show: centred on the character, clear of
# the action panel and the minimap, at the cells' aspect.
_CH = 810
_CW = int(round(_CH * CELL_ASPECT))
CROP = (_CW, _CH, 960 - _CW // 2, 560 - _CH // 2)

# ---------------------------------------------------------------- timing (s)
SPLIT1 = 5.0  # cut: full screen -> 3x2
SPLIT2 = 10.5  # cut: 6x4
SPLIT3 = 15.0  # cut: 12x8
STAMP = 18.6  # cut: the title card
END = STAMP + 5.0
BPM = 104
BEAT = 60 / BPM

HERO = (6.0, 2.0)  # raw start, speed: the opening shot (coal site)
SIX = [  # raw start, speed, label: results on top, features below. Each clip stays inside
    # one stretch of demo-raw.timeline.json for its whole run (about 16 s of raw at 3x).
    (None, None, "PPO", "trained in the sim · 90.6% held-out in the real game"),  # coal build
    (
        123.0,
        3.0,
        "PROGRAM SEARCH",
        "evolved builder programs · 99.3% in the real game",
    ),  # iron running
    (68.0, 3.0, "GRPO · QWEN3.5-9B", "held-out 1.6% to 84% · 85% in the real game"),  # copper build
    (36.0, 3.0, "REAL FACTORIO 2.0", "agents play the actual game through a mod"),  # iron build
    (
        151.2,
        3.0,
        "TICK-EXACT SIMULATOR",
        "matches the engine tick for tick · ~3,700× faster",
    ),  # copper running
    (
        271.0,
        3.0,
        "MACHINE-OUTPUT VERIFIER",
        "scores what the factory actually produces",
    ),  # overview
]

# ------------------------------------------------------------------- helpers


def clamp01(t):
    return min(max(t, 0.0), 1.0)


def ease_io(t):
    t = clamp01(t)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t, p=3):
    return 1 - (1 - clamp01(t)) ** p


def ease_in(t, p=3):
    return clamp01(t) ** p


def ease_expo(t):
    t = clamp01(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def font(path: Path, size, axes=None):
    f = ImageFont.truetype(str(path), size)
    if axes is not None:
        f.set_variation_by_axes(list(axes))
    return f


def rounded_mask(w, h, r, scale=3):
    m = Image.new("L", (w * scale, h * scale), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * scale - 1, h * scale - 1), r * scale, fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0


def paste(canvas, img, x, y, alpha=None, opacity=1.0):
    """Blend an HxWx3 image onto the float canvas at (x, y)."""
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    sub = img[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32)
    a = opacity if alpha is None else alpha[y0 - y : y1 - y, x0 - x : x1 - x, None] * opacity
    region = canvas[y0:y1, x0:x1]
    region += (sub - region) * a


def paste_rgba(canvas, rgba: Image.Image, x, y, opacity=1.0):
    arr = np.asarray(rgba, dtype=np.float32)
    paste(canvas, arr[..., :3], int(round(x)), int(round(y)), arr[..., 3] / 255.0, opacity)


def fill_rect(canvas, x, y, w, h, colour, alpha=1.0):
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), W), min(int(round(y + h)), H)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    region += (np.array(colour[:3], np.float32) - region) * alpha


def cover_crop(img: np.ndarray, aspect: float, zoom: float = 1.0) -> np.ndarray:
    """Centre-crop `img` to `aspect`, then by `zoom` more."""
    h, w = img.shape[:2]
    if w / h > aspect:
        cw, ch = h * aspect, h
    else:
        cw, ch = w, w / aspect
    cw, ch = cw / zoom, ch / zoom
    x, y = (w - cw) / 2, (h - ch) / 2
    return img[int(y) : int(y + ch), int(x) : int(x + cw)]


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(
        np.ascontiguousarray(img), (max(w, 1), max(h, 1)), interpolation=cv2.INTER_AREA
    )


def grid(cols, rows, gap):
    cw = (W - (cols + 1) * gap) // cols
    ch = (H - (rows + 1) * gap) // rows
    ox = (W - (cols * cw + (cols - 1) * gap)) // 2
    oy = (H - (rows * ch + (rows - 1) * gap)) // 2
    cells = [(ox + c * (cw + gap), oy + r * (ch + gap)) for r in range(rows) for c in range(cols)]
    return cells, cw, ch


# ------------------------------------------------------------------- sources


class Reader:
    """Frames of one stretch of the raw recording, graded, at one size."""

    def __init__(self, raw, start, speed, w, h, crop=None, seconds=40.0):
        vf = []
        if crop:
            vf.append("crop={}:{}:{}:{}".format(*crop))
        vf += [
            GRADE,
            f"setpts=(PTS-STARTPTS)/{speed}",
            f"fps={FPS}",
            f"scale={w}:{h}:flags=lanczos",
        ]
        cmd = [
            "ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{seconds * speed}", "-i", str(raw),
            "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]  # fmt: skip
        self.w, self.h = w, h
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last = np.zeros((h, w, 3), np.uint8)

    def next(self):
        buf = self.proc.stdout.read(self.w * self.h * 3)
        if len(buf) == self.w * self.h * 3:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
        return self.last

    def close(self):
        self.proc.kill()


def build_cache(raw, w, h, fps, path: Path):
    """The whole recording, cropped and graded, at `fps` and one small size."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(raw), "-vf",
        "crop={}:{}:{}:{},{},fps={},scale={}:{}:flags=lanczos".format(*CROP, GRADE, fps, w, h),
        "-f", "rawvideo", "-pix_fmt", "rgb24", str(path), "-y",
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    frames = path.stat().st_size // (w * h * 3)
    return np.memmap(path, np.uint8, "r", shape=(frames, h, w, 3))


class CacheCell:
    def __init__(self, cache, fps, start, speed, t0):
        self.cache, self.fps, self.start, self.speed, self.t0 = cache, fps, start, speed, t0

    def at(self, t):
        s = self.start + max(0.0, t - self.t0) * self.speed
        i = min(int(s * self.fps), len(self.cache) - 1)
        return np.asarray(self.cache[i])


# --------------------------------------------------------------- typography


def text_image(text, fnt, fill, tracking=0.0, pad=4):
    """RGBA image of a line of text, with optional letter spacing (px)."""
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + tracking * max(len(text) - 1, 0)
    asc, desc = fnt.getmetrics()
    img = Image.new("RGBA", (int(math.ceil(total)) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if tracking == 0:
        d.text((pad, pad), text, font=fnt, fill=fill)
    else:
        x = pad
        for c, w in zip(text, widths, strict=True):
            d.text((x, pad), c, font=fnt, fill=fill)
            x += w + tracking
    return img


SCRAMBLE = "#%&@$*+=/<>?!0123456789"


def scrambled(text, t, seed=0, per_char=0.028, settle=0.12):
    """`text` decoding from noise, `t` seconds after it started."""
    out = []
    for i, c in enumerate(text):
        if c == " " or t > settle + i * per_char:
            out.append(c)
        elif t > i * per_char * 0.5:
            out.append(SCRAMBLE[hash((seed, i, int(t * 30))) % len(SCRAMBLE)])
        else:
            out.append(" ")
    return "".join(out)


class TextCache:
    """Rendered lines of text, re-used until the string changes."""

    def __init__(self):
        self.items = {}

    def get(self, text, fnt, fill, tracking=0.0):
        key = (text, id(fnt), fill, tracking)
        img = self.items.get(key)
        if img is None:
            img = self.items[key] = text_image(text, fnt, fill, tracking)
        return img


def cell_label(index: int, title: str, sub: str, speed: str):
    label = font(SYS_FONTS / "bahnschrift.ttf", 25, (600, 100))
    label_sub = font(SYS_FONTS / "bahnschrift.ttf", 21, (350, 100))
    mono_small = font(SYS_FONTS / "CascadiaMono.ttf", 17)
    title_img = text_image(title, label, INK + (255,), tracking=2.5)
    sub_img = text_image(sub, label_sub, (220, 214, 204, 235))
    num_img = text_image(f"{index:02d}", mono_small, AMBER + (255,))
    w = 22 + num_img.width + 10 + max(title_img.width, sub_img.width) + 18
    h = 70
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 8, fill=(8, 10, 13, 196))
    d.rectangle((0, 12, 3, h - 12), fill=AMBER + (255,))
    img.alpha_composite(num_img, (16, 10))
    img.alpha_composite(title_img, (16 + num_img.width + 8, 4))
    img.alpha_composite(sub_img, (16 + num_img.width + 8, 36))
    chip = text_image(speed, mono_small, INK + (230,))
    badge = Image.new("RGBA", (chip.width + 14, chip.height + 4), (0, 0, 0, 0))
    ImageDraw.Draw(badge).rounded_rectangle(
        (0, 0, badge.width - 1, badge.height - 1), 6, fill=(8, 10, 13, 180),
        outline=AMBER + (150,), width=1,
    )  # fmt: skip
    badge.alpha_composite(chip, (7, 2))
    return img, badge


# ---------------------------------------------------------------- title card


class TitleCard:
    """FactorioGym, with "Gym" in amber; under it what it is, and where. Plain
    type, no effects: it is on, whole, from the frame the scan line opens."""

    def __init__(self):
        big = font(FONTS / "Archivo.ttf", 150, (700, 100))
        a, b = "Factorio", "Gym"
        asc, desc = big.getmetrics()
        wa = big.getlength(a)
        img = Image.new("RGBA", (int(wa + big.getlength(b)) + 24, asc + desc + 24), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((12, 12), a, font=big, fill=INK + (255,))
        d.text((12 + wa, 12), b, font=big, fill=AMBER + (255,))
        self.word = img.crop(img.getbbox())
        sub = font(FONTS / "Archivo.ttf", 48, (500, 100))
        mono = font(FONTS / "JetBrainsMono.ttf", 30, (400,))
        self.sub = text_image("An RL environment + verifier for Factorio", sub, INK + (255,))
        self.url = text_image("github.com/divagr18/FactorioGym", mono, INK + (190,))
        gap1, gap2 = 34, 40
        total = self.word.height + gap1 + self.sub.height + gap2 + self.url.height
        self.word_y = (H - total) / 2
        self.sub_y = self.word_y + self.word.height + gap1
        self.url_y = self.sub_y + self.sub.height + gap2

    def draw(self, canvas, u):
        paste_rgba(canvas, self.word, (W - self.word.width) / 2, self.word_y)
        paste_rgba(canvas, self.sub, (W - self.sub.width) / 2, self.sub_y)
        paste_rgba(canvas, self.url, (W - self.url.width) / 2, self.url_y)


# ------------------------------------------------------------------- finish


class Finish:
    """Chromatic aberration, bloom, flash, vignette, grain, soft shoulder."""

    def __init__(self, seed=11):
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = (xs + 0.5) / W - 0.5, (ys + 0.5) / H - 0.5
        r2 = cx * cx + cy * cy
        a = 0.9 * 0.008
        ox, oy = cx * a * r2 * W, cy * a * r2 * H
        self.map_r = (xs - ox, ys - oy)
        self.map_b = (xs + ox, ys + oy)
        s = np.clip((r2 * 1.6 - 0.18) / (0.75 - 0.18), 0, 1)
        s = s * s * (3 - 2 * s)
        self.vig = (1 - 0.55 * s)[..., None].astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __call__(self, canvas, flash=0.0, bloom=0.6, grain=0.032):
        img = np.clip(canvas / 255.0, 0, 1).astype(np.float32)
        r = cv2.remap(img[..., 0], *self.map_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b = cv2.remap(img[..., 2], *self.map_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        col = np.dstack([r, img[..., 1], b])
        lum = col.max(2)
        k = np.clip((lum - 0.72) / 0.28, 0, 1) ** 2
        small = cv2.resize(col * k[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bl = (cv2.GaussianBlur(small, (0, 0), 3) * 0.5 + cv2.GaussianBlur(small, (0, 0), 9) * 0.35
              + cv2.GaussianBlur(small, (0, 0), 24) * 0.25)  # fmt: skip
        col += cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * bloom
        col += flash
        col *= self.vig
        col += ((self.rng.random((H, W), dtype=np.float32) - 0.5) * grain)[..., None]
        col = col / (1.0 + 0.06 * col)
        return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------- main


def render(raw: Path, out: Path, work: Path, stills: list[float]) -> None:
    rng = np.random.default_rng(7)
    g1, w1, h1 = grid(3, 2, GAPS[0])
    g2, w2, h2 = grid(6, 4, GAPS[1])
    g3, w3, h3 = grid(12, 8, GAPS[2])
    m1, m2, m3 = rounded_mask(w1, h1, 12), rounded_mask(w2, h2, 7), rounded_mask(w3, h3, 3)
    print(f"cells {w1}x{h1}, {w2}x{h2}, {w3}x{h3}; crop {CROP}; building caches", flush=True)
    cache2 = build_cache(raw, w2, h2, 10, work / "c2.rgb")
    cache3 = build_cache(raw, w3, h3, 10, work / "c3.rgb")
    duration = len(cache2) / 10

    hero = Reader(raw, HERO[0], HERO[1], W, H, seconds=SPLIT2 + 1)
    six = [None] + [
        Reader(raw, s - SPLIT1 * sp, sp, w1, h1, CROP, seconds=SPLIT2 + 1) for s, sp, *_ in SIX[1:]
    ]
    labels = [cell_label(i + 1, t, s, f"{int(HERO[1] if sp is None else sp)}×")
              for i, (_, sp, t, s) in enumerate(SIX)]  # fmt: skip
    starts2 = np.linspace(8, duration - 24, len(g2))
    rng.shuffle(starts2)
    cells2 = [CacheCell(cache2, 10, s, 3.0, SPLIT2) for s in starts2]
    starts3 = np.linspace(4, duration - 30, len(g3))
    rng.shuffle(starts3)
    cells3 = [CacheCell(cache3, 10, s, (3.0, 6.0)[i % 2], SPLIT3) for i, s in enumerate(starts3)]
    card, finish = TitleCard(), Finish()
    bgc = np.array(BG, np.float32)

    enc = None
    if not stills:
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
    want = sorted({min(int(round(s * FPS)), int(END * FPS) - 1) for s in stills})
    frames = (want[-1] + 1) if want else int(END * FPS)
    for n in range(frames):
        t = n / FPS
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = bgc
        if t < SPLIT2:
            hero_frame = hero.next()
            six_frames = [None] + [r.next() for r in six[1:]]

        if t < SPLIT1:  # the opening shot, full frame, with a slow push
            zoom = 1.0 + 0.035 * t / SPLIT1
            paste(canvas, resize(cover_crop(hero_frame, W / H, zoom), W, H), 0, 0)
        elif t < SPLIT2:  # 3x2: the opening shot keeps playing in the first cell
            for i, (cx, cy) in enumerate(g1):
                if i == 0:
                    im = resize(cover_crop(hero_frame, w1 / h1, H / CROP[1]), w1, h1)
                else:
                    im = six_frames[i]
                paste(canvas, im, cx, cy, m1)
                lab, badge = labels[i]
                paste_rgba(canvas, lab, cx + 18, cy + h1 - lab.height - 18)
                paste_rgba(canvas, badge, cx + w1 - badge.width - 16, cy + 16)
        elif t < SPLIT3:  # 6x4
            for i, (cx, cy) in enumerate(g2):
                paste(canvas, cells2[i].at(t), cx, cy, m2)
        elif t < STAMP:  # 12x8
            for i, (cx, cy) in enumerate(g3):
                paste(canvas, cells3[i].at(t), cx, cy, m3)

        if t < STAMP:
            frame = finish(canvas)
        else:  # the title card gets no film finish
            card.draw(canvas, t - STAMP)
            frame = (np.clip(canvas, 0, 255) + 0.5).astype(np.uint8)
        if enc is not None:
            enc.stdin.write(frame.tobytes())
        elif n in want:
            path = out.with_name(f"{out.stem}_{t:05.2f}.png")
            Image.fromarray(frame).save(path)
            print(f"  wrote {path}", flush=True)
        if n % 120 == 0:
            print(f"  frame {n}/{frames}", flush=True)
    for r in [hero] + six[1:]:
        r.close()
    if enc is not None:
        enc.stdin.close()
        enc.wait()


def music(path: Path, seconds: float) -> None:
    """A synthesised bed: a slow pad, a pulse from the first split, risers into
    each split and a low hit when the title card cuts in."""
    sr = 48000
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    notes = [55.0, 82.41, 110.0, 130.81, 164.81]  # A1 E2 A2 C3 E3
    env = np.clip(t / 2.5, 0, 1) * np.clip((seconds - t) / 1.2, 0, 1)
    for i, f in enumerate(notes):
        for det in (-0.35, 0.35):
            ph = 2 * np.pi * (f + det) * t
            tone = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
            out += tone * (0.05 if i == 0 else 0.028) * env
    out *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.11 * t)
    rng = np.random.default_rng(3)
    k = 0
    while SPLIT1 + k * BEAT < STAMP - 0.2:
        b0 = SPLIT1 + k * BEAT
        i0 = int(b0 * sr)
        ln = int(0.35 * sr)
        tt = np.arange(ln) / sr
        kick = np.sin(2 * np.pi * (48 + 60 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9) * 0.22
        out[i0 : i0 + ln] += kick[: max(0, min(ln, n - i0))]
        h0 = int((b0 + BEAT / 2) * sr)
        hl = int(0.05 * sr)
        hat = np.diff(rng.standard_normal(hl) * np.exp(-np.arange(hl) / sr * 90) * 0.03, prepend=0)
        out[h0 : h0 + hl] += hat[: max(0, min(hl, n - h0))]
        k += 1
    for at in (SPLIT1, SPLIT2, SPLIT3, STAMP - 0.35):
        ln = int(1.4 * sr)
        i1 = int((at + 0.35) * sr)
        i0 = max(0, i1 - ln)
        noise = rng.standard_normal(i1 - i0)
        ramp = np.linspace(0, 1, i1 - i0) ** 2.5
        lp = np.convolve(noise, np.ones(40) / 40, mode="same")
        out[i0:i1] += (lp * (1 - ramp) + np.diff(noise, prepend=0) * 0.3 * ramp) * ramp * 0.06
    hit = int(STAMP * sr)
    ln = int(3.5 * sr)
    tt = np.arange(ln) / sr
    boom = np.sin(2 * np.pi * (38 + 40 * np.exp(-tt * 6)) * tt) * np.exp(-tt * 1.6) * 0.45
    out[hit : hit + ln] += boom[: max(0, min(ln, n - hit))]
    sh = np.zeros(n)
    for f in (880.0, 1318.5, 1760.0):
        sh += np.sin(2 * np.pi * f * t) * 0.012
    sh *= np.clip((t - STAMP - 0.3) / 1.5, 0, 1) * np.clip((seconds - t) / 1.2, 0, 1)
    out += sh
    wet = np.zeros(n)
    for d, g in ((0.061, 0.35), (0.113, 0.25), (0.187, 0.18), (0.29, 0.12)):
        s = int(d * sr)
        wet[s:] += out[:-s] * g
    out = out + wet
    out /= max(1e-9, np.abs(out).max()) / 0.7
    pcm = (np.stack([out, np.roll(out, int(0.0007 * sr))], 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--music", action="store_true", help="mix in the soundtrack")
    parser.add_argument(
        "--stills", default="", help="comma list of times (s) to save as PNG instead"
    )
    args = parser.parse_args()
    stills = [float(s) for s in args.stills.split(",") if s.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        silent = work / "silent.mp4" if args.music and not stills else args.out
        render(args.raw, silent, work, stills)
        if stills:
            return 0
        if args.music:
            wav = work / "music.wav"
            music(wav, END)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav), "-map", "0:v",
                 "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                 "-movflags", "+faststart", str(args.out)],
                check=True,
            )  # fmt: skip
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
