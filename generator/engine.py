"""Мини-движок: рендерит процедурную сцену кадр за кадром в 4K и кодирует в HEVC 10-bit.

Каждая сцена — класс с методом frame(p) -> np.ndarray (3, H, W) float32 в диапазоне ~[0, 1],
где p = 2π * f / N — фаза петли. Всё, что зависит от времени, использует целые кратные p,
поэтому кадр N совпадает с кадром 0 и петля бесшовная.
"""
import math
import subprocess
import sys
import time

import cv2
import numpy as np

W, H = 3840, 2160
ASPECT = W / H

# Сетки координат: x в [0, ASPECT], y в [0, 1] сверху вниз. Формы (1, W) и (H, 1) — для broadcast.
XX = ((np.arange(W, dtype=np.float32) + 0.5) / H)[None, :]
YY = ((np.arange(H, dtype=np.float32) + 0.5) / H)[:, None]

TAU = 2 * math.pi

_rng_dither = np.random.default_rng(7)
DITHER = ((_rng_dither.random((H, W), dtype=np.float32) - 0.5) / 1023.0).astype(np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fbm(rng, h, w, octaves=6, base=4, persistence=0.55):
    """Фрактальный шум: сумма увеличенных случайных сеток разного масштаба."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * (2 ** o) * h / max(h, w)) + 2)
        gw = max(2, int(base * (2 ** o) * w / max(h, w)) + 2)
        g = rng.random((gh, gw), dtype=np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    out /= total
    out -= out.min()
    out /= out.max()
    return out


def splat(img, xs, ys, amps, colors, sigma, r):
    """Рисует звёзды гауссовыми точками. img: (3, H, W); colors: (N, 3)."""
    ix = np.floor(xs).astype(np.int64)
    iy = np.floor(ys).astype(np.int64)
    flat = img.reshape(3, -1)
    inv = -1.0 / (2 * sigma * sigma)
    for dy in range(-r, r + 1):
        py = iy + dy
        for dx in range(-r, r + 1):
            px = ix + dx
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if not ok.any():
                continue
            d2 = (px[ok] + 0.5 - xs[ok]) ** 2 + (py[ok] + 0.5 - ys[ok]) ** 2
            w = amps[ok] * np.exp(d2 * inv[ok] if np.ndim(inv) else d2 * inv)
            idx = py[ok] * W + px[ok]
            for c in range(3):
                np.add.at(flat[c], idx, (w * colors[ok, c]).astype(np.float32))


def bloom(img, threshold=0.6, strength=0.5, sigma=24):
    """Мягкое свечение ярких областей (считается в 1/4 разрешения — быстро)."""
    small = cv2.resize(np.ascontiguousarray(img.transpose(1, 2, 0)), (W // 4, H // 4),
                       interpolation=cv2.INTER_AREA)
    bright = np.maximum(small - threshold, 0)
    glow = cv2.GaussianBlur(bright, (0, 0), sigma / 4)
    glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + strength * glow.transpose(2, 0, 1)


def finish(img):
    """Мягкое ограничение пересветов + дизеринг против полос в градиентах."""
    img = np.maximum(img, 0)
    hi = img > 0.85
    img[hi] = 0.85 + 0.15 * np.tanh((img[hi] - 0.85) / 0.15)
    img += DITHER
    return np.clip(img, 0, 1)


def render(scene, out_path, seconds, fps=30, preview=None):
    n = int(round(seconds * fps))
    if preview is not None:
        # preview — список фаз в долях петли, сохраняем PNG
        for i, frac in enumerate(preview):
            img = finish(scene.frame(TAU * frac))
            bgr = (img[::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
            cv2.imwrite(f"{out_path}_preview{i}.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "gbrp16le", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
        "-c:v", "libx265", "-preset", "fast", "-crf", "19", "-tag:v", "hvc1",
        "-x265-params", "log-level=error:aq-mode=3",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for f in range(n):
        img = finish(scene.frame(TAU * f / n))
        img16 = (img * 65535 + 0.5).astype(np.uint16)
        proc.stdin.write(img16[1].tobytes())  # gbrp: G, B, R
        proc.stdin.write(img16[2].tobytes())
        proc.stdin.write(img16[0].tobytes())
        if f % 30 == 0:
            el = time.time() - t0
            print(f"{out_path}: {f}/{n}  {el:.0f}s", file=sys.stderr, flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"{out_path}: done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
