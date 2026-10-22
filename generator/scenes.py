import math

import cv2
import numpy as np

from engine import (ASPECT, H, TAU, W, XX, YY, bloom, fbm, smoothstep, splat)

XX_, YY_ = XX, YY


def col(*c):
    return np.array(c, np.float32)[:, None, None]


def ridge_line(rng, base, amp, n_peaks, jag=0.35, smooth=9):
    """Горный хребет: y(x) для каждой колонки пикселей."""
    xs = np.linspace(0, W, n_peaks)
    ys = base - amp * (0.5 + 0.5 * rng.random(n_peaks))
    ys[1::2] += amp * jag * rng.random(len(ys[1::2]))
    line = np.interp(np.arange(W), xs, ys)
    # мелкая «зазубренность» скал
    detail = fbm(rng, 1, W, octaves=6, base=40)[0] - 0.5
    line = line + detail * amp * 0.25
    k = np.ones(smooth) / smooth
    line = np.convolve(np.pad(line, smooth, mode="edge"), k, mode="same")[smooth:-smooth]
    return line.astype(np.float32)[None, :]


def mask_below(line):
    """Антиалиасная маска: 1 ниже линии line (в долях высоты)."""
    return np.clip((YY - line) * H + 0.5, 0, 1)


def make_stars(rng, n, y_max, bright_frac=0.04):
    xs = rng.random(n) * W
    ys = rng.random(n) ** 1.3 * y_max * H
    amps = rng.random(n) ** 3 * 0.9 + 0.08
    big = rng.random(n) < bright_frac
    amps[big] = 0.9 + rng.random(big.sum()) * 0.8
    tint = rng.random(n)
    colors = np.stack([0.85 + 0.15 * tint, 0.9 * np.ones(n), 1.0 - 0.2 * tint], 1)
    sig = np.where(big, 1.6, 0.9)
    m = rng.integers(1, 4, n)
    ph = rng.random(n) * math.tau
    return xs, ys, amps, colors, sig, big, m, ph


def draw_stars(img, stars, p, twinkle=0.35):
    xs, ys, amps, colors, sig, big, m, ph = stars
    a = amps * (1 - twinkle + twinkle * (0.5 + 0.5 * np.sin(m * p + ph)))
    small = ~big
    splat(img, xs[small], ys[small], a[small], colors[small], sig[small], 2)
    splat(img, xs[big], ys[big], a[big], colors[big], sig[big], 4)


# ---------------------------------------------------------------- 1. Северное сияние
class Aurora:
    def __init__(self):
        rng = np.random.default_rng(11)
        self.stars = make_stars(rng, 2600, 0.8)
        self.lake = 0.80
        self.far = ridge_line(rng, 0.795, 0.19, 18, smooth=15)
        self.near = ridge_line(rng, 0.80, 0.07, 36, smooth=5)
        self.far_m = mask_below(self.far)
        self.near_m = mask_below(self.near)
        self.layers = []
        for base, h, s in [(0.34, 0.26, 1.0), (0.42, 0.18, 0.7), (0.27, 0.30, 0.55)]:
            self.layers.append(dict(base=base, h=h, s=s, ph=rng.random(8) * math.tau))
        # отражение: координаты для cv2.remap (только область озера)
        y0 = int(self.lake * H)
        self.y0 = y0
        ys = np.arange(y0, H, dtype=np.float32)[:, None]
        self.ref_y = 2 * y0 - ys - 1
        self.ref_x = np.repeat(np.arange(W, dtype=np.float32)[None, :], H - y0, 0)
        self.ref_rows = ys - y0
        sky_t = np.clip(YY / 0.8, 0, 1)
        self.sky = (col(0.004, 0.007, 0.022) * (1 - sky_t) + col(0.012, 0.03, 0.055) * sky_t)

    def frame(self, p):
        img = np.broadcast_to(self.sky, (3, H, W)).copy()
        draw_stars(img, self.stars, p, 0.4)

        X2, Y2 = XX[:, ::2], YY[::2]
        aur = np.zeros((3, H // 2, W // 2), np.float32)
        for L in self.layers:
            ph = L["ph"]
            x = X2[0]
            yc = (L["base"] + 0.06 * np.sin(TAU * 0.33 * x + p + ph[0])
                  + 0.03 * np.sin(TAU * 0.85 * x - 2 * p + ph[1])
                  + 0.014 * np.sin(TAU * 2.1 * x + 3 * p + ph[2]))[None, :]
            rays = (0.5 * np.sin(TAU * 6.3 * x + 2 * p + ph[3])
                    + 0.3 * np.sin(TAU * 15.7 * x - 3 * p + ph[4])
                    + 0.2 * np.sin(TAU * 33.1 * x + 5 * p + ph[5]))
            rays = (0.68 + 0.32 * rays)[None, :]
            env = (0.06 + 0.94 * (0.5 + 0.5 * np.sin(TAU * 0.45 * x + p + ph[6])) ** 2.2)[None, :]
            d = yc - Y2                        # >0 — выше нижней кромки
            dp = np.maximum(d, 0)
            dn = np.minimum(d, 0)
            prof = np.exp(-dp / L["h"] + dn / 0.012) * (1 + 0.9 * np.exp(-dp / 0.01))
            inten = prof * (rays * env * L["s"] * 0.95)
            t = np.clip(d / 0.22, 0, 1)
            aur[0] += inten * (0.10 + 0.55 * t)
            aur[1] += inten * (1.00 - 0.75 * t)
            aur[2] += inten * (0.45 + 0.35 * t)
        aur = cv2.GaussianBlur(aur.transpose(1, 2, 0), (0, 0), 1.5)
        img += cv2.resize(aur, (W, H), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

        # свечение неба у горизонта от сияния
        glow = np.exp(-np.abs(self.lake - YY) / 0.16) * 0.13
        img[0] += glow * 0.15
        img[1] += glow
        img[2] += glow * 0.75

        # дальние горы: тёмно-синие, с зелёным отсветом на гребне
        depth = np.clip((YY - 0.6) / 0.2, 0, 1)
        far_col = [0.006 - 0.003 * depth, 0.011 - 0.005 * depth, 0.016 - 0.007 * depth]
        for c in range(3):
            img[c] = img[c] * (1 - self.far_m) + far_col[c] * self.far_m

        # озеро с отражением и рябью
        ry = self.ref_rows
        wob_x = (2.5 * np.sin(ry * 0.21 + 3 * p) + 1.5 * np.sin(ry * 0.53 - 5 * p)) * (0.3 + ry / 300)
        wob_y = 1.2 * np.sin(XX * 900 + ry * 0.4 + 4 * p)
        mx = (self.ref_x + wob_x).astype(np.float32)
        my = np.clip(self.ref_y + wob_y, 0, H - 1).astype(np.float32)
        src = np.ascontiguousarray(img[:, : self.y0 + 1].transpose(1, 2, 0))
        refl = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=2, sigmaY=5)
        fade = (0.7 - 0.3 * (ry / (H - self.y0)))[..., None]
        img[:, self.y0:] = (refl * fade * np.array([0.8, 0.95, 1.1], np.float32)).transpose(2, 0, 1)
        return bloom(img, 0.35, 0.35, 30)


# ---------------------------------------------------------------- 2. Звёзды над степью
class Steppe:
    def __init__(self):
        rng = np.random.default_rng(23)
        self.horizon = 0.84
        self.cx, self.cy = 0.42 * W, -1.1 * H        # центр вращения неба — за кадром
        self.total = math.radians(2.2)                # поворот за одну петлю
        n = 7000
        r = rng.random(n)
        # звёзды на кольцах вокруг центра вращения, покрывающих кадр
        ang0 = rng.uniform(-0.9, 0.9, n) + math.pi / 2
        rad = np.sqrt(rng.uniform((1.0 * H) ** 2, (3.3 * H) ** 2, n))
        # часть звёзд гуще в полосе Млечного Пути
        self.r, self.a0, self.tau = rad, ang0, rng.random(n)
        amps = r ** 3 * 0.9 + 0.08
        big = rng.random(n) < 0.03
        amps[big] = 1.0 + rng.random(big.sum()) * 0.9
        tint = rng.random(n)
        self.colors = np.stack([0.8 + 0.2 * tint, 0.88 * np.ones(n), 1.0 - 0.25 * tint], 1)
        self.amps, self.big = amps, big
        self.sig = np.where(big, 1.5, 0.85)

        # Млечный Путь — статичная текстура на холсте с запасом, поворачивается вместе с небом
        pad = 700
        self.pad = pad
        ch, cw = H + 2 * pad, W + 2 * pad
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        ang = math.radians(-32)
        px, py = 0.58 * cw, 0.5 * ch
        dist = ((xx - px) * math.sin(ang) - (yy - py) * math.cos(ang))
        band = np.exp(-(dist / (0.16 * H)) ** 2)
        core = np.exp(-(dist / (0.06 * H)) ** 2)
        n1 = fbm(rng, ch, cw, 7, 5, 0.6)
        n2 = fbm(rng, ch, cw, 7, 8, 0.65)
        dust = smoothstep(0.45, 0.75, n2) * core * 0.85
        dens = band * (0.35 + 0.9 * n1 ** 2) * (1 - dust)
        along = ((xx - px) * math.cos(ang) + (yy - py) * math.sin(ang)) / cw
        warm = np.clip(0.5 - along * 1.2, 0, 1)
        mw = np.stack([dens * (0.55 + 0.45 * warm), dens * (0.55 + 0.2 * warm), dens * (0.75 - 0.25 * warm)], 2)
        self.mw = (mw * 0.42).astype(np.float32)

        sky_t = np.clip(YY / self.horizon, 0, 1)
        sky = col(0.003, 0.005, 0.016) * (1 - sky_t ** 2) + col(0.02, 0.035, 0.06) * sky_t ** 2
        hg = np.exp(-np.clip(self.horizon - YY, 0, None) / 0.05)
        warm_side = np.exp(-((XX - 0.25 * ASPECT) / 0.6) ** 2)
        sky = sky + hg * (col(0.05, 0.08, 0.11) + col(0.10, 0.05, 0.015) * warm_side)
        self.sky = sky.astype(np.float32)

        # степь: пологие холмы + юрта
        x = np.arange(W, dtype=np.float32)
        hills = (self.horizon - 0.012 * np.sin(x / W * 5.1 + 0.7) - 0.008 * np.sin(x / W * 13.3 + 2.1)
                 - 0.004 * (fbm(rng, 1, W, 6, 30)[0] - 0.5))
        self.hills = hills[None, :].astype(np.float32)
        ground = mask_below(self.hills)
        # юрта
        yx = 0.69 * W
        gy = float(hills[int(yx)]) * H
        bw, bh, rh = 0.068 * W, 0.050 * H, 0.040 * H
        X = np.arange(W, dtype=np.float32)[None, :]
        Y = np.arange(H, dtype=np.float32)[:, None]
        body = (np.abs(X - yx) < bw / 2) & (Y > gy - bh) & (Y < gy + 4)
        dome = ((X - yx) / (bw / 2 * 1.04)) ** 2 + ((Y - (gy - bh)) / rh) ** 2 < 1
        dome &= Y <= gy - bh
        top = ((X - yx) / (bw * 0.09)) ** 2 + ((Y - (gy - bh - rh)) / (rh * 0.18)) ** 2 < 1
        yurt = (body | dome | top).astype(np.float32)
        yurt = cv2.GaussianBlur(yurt, (0, 0), 0.7)
        self.ground_m = np.maximum(ground, yurt)
        gt = np.clip((YY - self.horizon) / (1 - self.horizon), 0, 1)
        self.ground_col = col(0.012, 0.014, 0.018) * (1 - gt) + col(0.004, 0.004, 0.006) * gt
        # светящаяся дверь и тёплый отсвет
        dw, dh = 0.009 * W, 0.034 * H
        door = (np.abs(X - yx) < dw / 2) & (Y > gy - dh) & (Y < gy)
        door = cv2.GaussianBlur(door.astype(np.float32), (0, 0), 1.2)
        r2 = ((X - yx) ** 2 + ((Y - gy + dh / 2) * 1.6) ** 2)
        halo = np.exp(-r2 / (2 * (0.02 * H) ** 2)) * 0.18 + np.exp(-r2 / (2 * (0.07 * H) ** 2)) * 0.05
        self.door = door
        self.halo = halo.astype(np.float32)

    def frame(self, p):
        s = (p / TAU) % 1.0
        img = np.broadcast_to(self.sky, (3, H, W)).copy()

        # Млечный Путь: поворот с бесшовной сшивкой петли (кроссфейд в последних 30%)
        def rot(angle):
            cx, cy = self.cx + self.pad, self.cy + self.pad
            M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(angle), 1.0)
            M[:, 2] -= self.pad
            return cv2.warpAffine(self.mw, M, (W, H), flags=cv2.INTER_LINEAR)
        wB = float(smoothstep(0.7, 1.0, s))
        mw = rot(self.total * s) * (1 - wB)
        if wB > 0:
            mw += rot(self.total * (s - 1)) * wB
        img += mw.transpose(2, 0, 1)

        # звёзды: у каждой своя фаза; «перескок» в начало дуги происходит, пока звезда погашена
        u = (s + self.tau) % 1.0
        ang = self.a0 + self.total * (u - 0.5)
        xs = self.cx + self.r * np.cos(ang)
        ys = self.cy + self.r * np.sin(ang)
        env = np.minimum(1, np.minimum(u, 1 - u) / 0.12)
        a = self.amps * env
        vis = ys < self.horizon * H
        sm, bg = vis & ~self.big, vis & self.big
        splat(img, xs[sm], ys[sm], a[sm], self.colors[sm], self.sig[sm], 2)
        splat(img, xs[bg], ys[bg], a[bg], self.colors[bg], self.sig[bg], 4)

        img = img * (1 - self.ground_m) + self.ground_col * self.ground_m
        flicker = 0.92 + 0.05 * math.sin(3 * p) + 0.03 * math.sin(7 * p + 1)
        img += col(1.0, 0.62, 0.28) * (self.door * 0.95 * flicker + self.halo * flicker)
        return bloom(img, 0.5, 0.4, 16)


# ---------------------------------------------------------------- 3. Градиенты
class Gradients:
    def __init__(self):
        rng = np.random.default_rng(5)
        palette = [(0.04, 0.05, 0.22), (0.10, 0.30, 0.95), (0.42, 0.16, 0.88),
                   (0.92, 0.24, 0.58), (1.00, 0.52, 0.28), (0.05, 0.62, 0.78), (0.06, 0.08, 0.35)]
        self.blobs = []
        for c in palette:
            self.blobs.append(dict(
                c=np.array(c, np.float32), cx=rng.uniform(0.1, ASPECT - 0.1), cy=rng.uniform(0.1, 0.9),
                ax=rng.uniform(0.15, 0.4), ay=rng.uniform(0.1, 0.3), m=int(rng.integers(1, 3)),
                n=int(rng.integers(1, 3)), ph=rng.random(2) * math.tau, r=rng.uniform(0.32, 0.55)))
        self.vig = (1 - 0.35 * (((XX - ASPECT / 2) / ASPECT) ** 2 * 2.2 + ((YY - 0.5)) ** 2 * 1.6)).astype(np.float32)

    def frame(self, p):
        XX, YY = XX_[:, ::2], YY_[::2]
        qx = XX + 0.07 * np.sin(TAU * 0.9 * YY + p) + 0.035 * np.sin(TAU * (1.9 * YY + 0.4 * XX) - 2 * p)
        qy = YY + 0.06 * np.sin(TAU * 0.7 * XX + 2 * p + 1.0) + 0.03 * np.sin(TAU * (1.6 * XX - 0.8 * YY) + p)
        acc = np.zeros((3, H // 2, W // 2), np.float32)
        wsum = np.full((H // 2, W // 2), 1e-4, np.float32)
        for b in self.blobs:
            bx = b["cx"] + b["ax"] * math.sin(b["m"] * p + b["ph"][0])
            by = b["cy"] + b["ay"] * math.cos(b["n"] * p + b["ph"][1])
            w = np.exp(-((qx - bx) ** 2 + (qy - by) ** 2) / (b["r"] ** 2))
            w = w * w
            wsum += w
            acc += w[None] * b["c"][:, None, None]
        img = acc / wsum[None]
        # шёлковые «складки» света
        for k, (a0, amp, fr, m, sh) in enumerate([(0.42, 0.14, 0.55, 1, 0.16), (0.66, 0.10, 0.8, 2, 0.10)]):
            cy = a0 + amp * np.sin(TAU * fr * qx + m * p + k * 2.1) + 0.04 * np.sin(TAU * 1.7 * qx - p)
            d = qy - cy
            sheen = np.exp(-(d / 0.035) ** 2) * sh + np.exp(-(np.maximum(d, 0) / 0.12)) * (d > 0) * sh * 0.25
            img += sheen[None] * np.array([1.0, 0.95, 1.0], np.float32)[:, None, None]
        img *= self.vig[None, ::2, ::2]
        img = cv2.resize(np.ascontiguousarray(img.transpose(1, 2, 0)), (W, H), interpolation=cv2.INTER_CUBIC)
        return img.transpose(2, 0, 1).copy()


# ---------------------------------------------------------------- 4. Ретровейв
class Retrowave:
    def __init__(self):
        rng = np.random.default_rng(3)
        self.hz = 0.60
        self.stars = make_stars(rng, 900, 0.5, 0.02)
        t = np.clip(YY / self.hz, 0, 1)
        sky = (col(0.03, 0.0, 0.10) * (1 - t) ** 2 + col(0.30, 0.04, 0.42) * 2 * t * (1 - t)
               + col(1.0, 0.32, 0.45) * t ** 2)
        self.sky = sky.astype(np.float32)
        self.sun_c = (ASPECT / 2, self.hz - 0.13)
        self.sun_r = 0.25
        # горы по бокам, долина в центре
        x = np.arange(W) / W
        base = self.hz - 0.004
        valley = np.clip(np.abs(x - 0.5) / 0.5, 0, 1) ** 1.3
        pts = 34
        xs = np.linspace(0, 1, pts)
        ys = rng.random(pts)
        jag = np.interp(x, xs, ys)
        self.ridge = (base - valley * (0.05 + 0.17 * jag))[None, :].astype(np.float32)
        self.mtn = mask_below(self.ridge) * (YY < self.hz)
        self.edge = np.exp(-np.abs(YY - self.ridge) * H / 2.5) * (YY < self.hz + 0.002)
        # пол
        dy = np.maximum(YY - self.hz, 1e-4)
        self.z = 0.06 / dy                                 # глубина
        self.dzdy = 0.06 / dy ** 2 / H                      # на пиксель
        self.xw = (XX - ASPECT / 2) / dy                    # мировой x
        self.dxw = 1 / dy / H
        self.floor_m = (YY > self.hz).astype(np.float32)

    def frame(self, p):
        s = (p / TAU) % 1.0
        img = np.broadcast_to(self.sky, (3, H, W)).copy()
        draw_stars(img, self.stars, p, 0.5)

        # солнце с полосами
        cx, cy = self.sun_c
        r = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
        disc = np.clip((self.sun_r - r) * H + 0.5, 0, 1)
        rel = (YY - cy) / self.sun_r                        # -1 верх … +1 низ
        period = 0.11
        ph = (rel / period - s * 2) % 1.0
        gap = np.clip(0.08 + 0.5 * rel, 0, 0.85)
        stripe = np.where(rel > 0.0, np.clip((ph - gap) * 60, 0, 1) * np.clip((1 - ph) * 60, 0, 1), 1.0)
        sunk = disc * stripe
        tt = np.clip((rel + 1) / 2, 0, 1)
        sun_col = [1.0, 0.95 - 0.75 * tt, 0.35 + 0.25 * tt]
        glow = np.exp(-np.maximum(r - self.sun_r, 0) / 0.09) * 0.35
        for c, g in enumerate((1.0, 0.35, 0.55)):
            img[c] = img[c] * (1 - sunk) + sun_col[c] * sunk + glow * g * (YY < self.hz)

        # горы + неоновый контур
        m = self.mtn
        for c, v in enumerate((0.06, 0.01, 0.12)):
            img[c] = img[c] * (1 - m) + v * m
        img += self.edge[None] * col(0.2, 0.9, 1.0) * 0.9

        # неоновая сетка пола, бегущая к зрителю
        fz = self.z * 6.0 + s * 2
        gz = np.abs(fz - np.round(fz)) / (self.dzdy * 6.0)          # расстояние до линии в пикселях
        fx = self.xw * 2.5
        gx = np.abs(fx - np.round(fx)) / (self.dxw * 2.5)
        line = np.exp(-(gz / 1.3) ** 2) + np.exp(-(gx / 1.3) ** 2)
        haze = np.exp(-self.z * 0.18)
        fade = smoothstep(0.0, 0.03, YY - self.hz)
        line = np.minimum(line, 1.3) * haze * fade
        floor = col(0.02, 0.0, 0.05) + col(0.35, 0.05, 0.3) * np.exp(-(YY - self.hz) / 0.05)
        refl = np.exp(-((XX - ASPECT / 2) / 0.35) ** 2) * np.exp(-(YY - self.hz) / 0.12) * 0.25
        floor = floor + refl * col(1.0, 0.3, 0.5)
        floor = floor + line[None] * col(1.0, 0.15, 0.85)
        fm = self.floor_m
        img = img * (1 - fm) + floor * fm
        # линия горизонта
        img += np.exp(-np.abs(YY - self.hz) * H / 2) * col(1.0, 0.4, 0.8) * 0.8
        return bloom(img, 0.45, 0.65, 28)


# ---------------------------------------------------------------- 5. Туманность
class Nebula:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.h2, self.w2 = H // 2, W // 2
        pad = 160
        self.pad = pad
        h, w = self.h2 + 2 * pad, self.w2 + 2 * pad
        self.n1 = fbm(rng, h, w, 8, 3, 0.58)
        self.n2 = fbm(rng, h, w, 8, 5, 0.6)
        self.n3 = fbm(rng, h, w, 8, 9, 0.62)
        yy, xx = np.mgrid[0:self.h2, 0:self.w2].astype(np.float32)
        self.gx, self.gy = xx + pad, yy + pad
        self.ux, self.uy = xx / self.h2, yy / self.h2
        cx, cy = 0.55 * self.w2 / self.h2, 0.48
        self.core = np.exp(-(((self.ux - cx) / 0.55) ** 2 + ((self.uy - cy) / 0.38) ** 2)).astype(np.float32)
        self.stars = make_stars(rng, 5000, 1.0, 0.025)
        # дифракционные лучи у ярких звёзд
        xs, ys, amps, colors, sig, big, m, ph = self.stars
        bright = np.argsort(amps)[-14:]
        self.spikes = [(xs[i], ys[i], amps[i] * 0.9, m[i], ph[i]) for i in bright]

    def warp(self, n, p, k, a):
        mx = self.gx + a * (np.sin(TAU * 0.9 * self.uy + k * p + 1.3 * k) + 0.5 * np.sin(TAU * 2.1 * self.ux - 2 * p))
        my = self.gy + a * (np.cos(TAU * 0.7 * self.ux + p + 0.7 * k) + 0.5 * np.cos(TAU * 1.8 * self.uy + 2 * p))
        return cv2.remap(n, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)

    def frame(self, p):
        a = self.warp(self.n1, p, 1, 70)
        b = self.warp(self.n2, p, 2, 45)
        c = self.warp(self.n3, p, 1, 28)
        gas = smoothstep(0.35, 0.9, a * 0.6 + b * 0.4) * (0.35 + 0.9 * self.core)
        hot = smoothstep(0.55, 0.95, b * 0.7 + c * 0.3) * self.core
        dust = smoothstep(0.5, 0.8, c) * 0.75
        r = gas * 0.55 + hot * 1.0
        g = gas * 0.12 + hot * 0.42
        bl = gas * 0.75 + hot * 0.25
        blue = smoothstep(0.4, 0.9, a) * (1 - self.core) * 0.35
        r, g, bl = r + blue * 0.1, g + blue * 0.35, bl + blue * 0.9
        neb = np.stack([r, g, bl], 2) * (1 - dust)[..., None] * 0.95
        neb = cv2.resize(neb.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        img = neb.transpose(2, 0, 1) + col(0.004, 0.003, 0.012)
        draw_stars(img, self.stars, p, 0.3)
        for (x, y, amp, m, ph) in self.spikes:
            amp = amp * (0.8 + 0.2 * math.sin(m * p + ph))
            ix, iy = int(x), int(y)
            L = int(60 + 140 * amp)
            x0, x1 = max(ix - L, 0), min(ix + L, W)
            y0, y1 = max(iy - L, 0), min(iy + L, H)
            dx = np.abs(np.arange(x0, x1) - x)
            dy = np.abs(np.arange(y0, y1) - y)
            img[:, iy, x0:x1] += (amp * 0.5 * np.exp(-dx / (L * 0.25)))[None] * col(0.8, 0.9, 1.0)[:, :, 0]
            img[:, y0:y1, ix] += (amp * 0.5 * np.exp(-dy / (L * 0.25)))[None] * col(0.8, 0.9, 1.0)[:, :, 0]
        return bloom(img, 0.4, 0.5, 20)


SCENES = {
    "aurora": (Aurora, 15),
    "steppe": (Steppe, 20),
    "gradients": (Gradients, 20),
    "retrowave": (Retrowave, 10),
    "nebula": (Nebula, 20),
}


# ---------------------------------------------------------------- 6. Ущелье с путником
class Canyon:
    def __init__(self):
        rng = np.random.default_rng(77)
        self.cx = ASPECT / 2
        cx = self.cx
        Y1 = np.arange(H, dtype=np.float32) / H
        # фон: светлая щель неба и туман
        t = np.clip(YY / 0.82, 0, 1)
        bg = col(0.62, 0.63, 0.64) * (1 - t) ** 1.5 + col(0.30, 0.31, 0.32) * (1 - (1 - t) ** 1.5)
        bg = np.broadcast_to(bg, (3, H, W)).copy()
        # пол ущелья
        floor_y = 0.80
        gt = np.clip((YY - floor_y) / (1 - floor_y), 0, 1)
        rocks = fbm(rng, H, W, 7, 12, 0.6)
        ground = (col(0.22, 0.22, 0.23) * (1 - gt) ** 1.2 + col(0.04, 0.04, 0.045) * gt) * (0.75 + 0.5 * rocks)
        gm = smoothstep(floor_y, floor_y + 0.01, YY)
        img = bg * (1 - gm) + ground * gm

        # стены: слои от дальних к ближним
        K = 7
        for k in range(K):
            d = k / (K - 1)                                    # 0 — дальний, 1 — ближний
            e = 0.028 + 0.62 * d ** 1.7                       # отступ кромки от центра
            fog = 0.9 * (1 - d) ** 1.8 + 0.01
            base = floor_y + 0.012 + 0.25 * d ** 2 if d < 0.5 else 1.2   # ближние стены уходят за кадр
            for side in (-1, 1):
                # «неестественная» кромка: крупные ступени + мелкая зазубренность
                n_steps = int(6 + 10 * rng.random())
                knots = np.sort(rng.random(n_steps))
                offs = rng.random(n_steps + 1) * (0.02 + 0.05 * d)
                idx = np.searchsorted(knots, Y1)
                jag = offs[idx] + (fbm(rng, H, 1, 6, 20)[:, 0] - 0.5) * (0.006 + 0.02 * d)
                jag = cv2.GaussianBlur(jag.astype(np.float32)[:, None], (1, 0), sigmaX=0, sigmaY=2)[:, 0]
                edge = cx + side * (e + jag)                   # (H,)
                dist = (XX - edge[:, None]) * side             # >0 — внутри стены
                m = np.clip(dist * H + 0.5, 0, 1) * (1 - smoothstep(base, base + 0.004, YY))
                # текстура скалы: вертикальные трещины + горизонтальные пласты
                tex_v = cv2.resize(rng.random((14, 160), dtype=np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
                tex_f = fbm(rng, H, W, 7, 10, 0.62)
                strata = smoothstep(0.85, 1.0, np.sin(YY * (38 + 25 * rng.random()) + tex_f * 9))
                rock = 0.02 + 0.11 * tex_f ** 1.6 + 0.04 * tex_v - 0.015 * strata
                rock = rock * (0.45 + 0.55 * (1 - YY) ** 1.5)    # свет сверху
                rim = np.exp(-np.maximum(dist, 0) / (0.004 + 0.01 * d)) * 0.10 * (1 - YY)
                v = rock + rim
                wall = np.stack([v * 0.98, v, v * 1.03]) * (1 - fog) + col(0.34, 0.35, 0.36) * fog * (1 - 0.4 * YY)
                img = img * (1 - m) + wall * m
        self.static = img.astype(np.float32)

        # туман (половинное разрешение)
        self.h2, self.w2 = H // 2, W // 2
        pad = 120
        self.pad = pad
        self.f1 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 4, 0.55)
        self.f2 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 7, 0.6)
        yy, xx = np.mgrid[0:self.h2, 0:self.w2].astype(np.float32)
        self.gx, self.gy = xx + pad, yy + pad
        self.uy, self.ux = yy / self.h2, xx / self.h2
        self.fog_shape = (np.exp(-((self.uy - 0.78) / 0.14) ** 2) * 0.9 + 0.25).astype(np.float32)
        self.shaft = (np.exp(-((XX - cx) / 0.05) ** 2) * (1 - smoothstep(0.1, 0.85, YY)) * 0.10).astype(np.float32)

        # пыль в воздухе
        n = 900
        self.dx0 = rng.random(n) * W
        self.dy0 = rng.random(n) * H
        self.dax = rng.uniform(10, 40, n)
        self.day = rng.uniform(15, 60, n)
        self.dm = rng.integers(1, 3, n)
        self.dph = rng.random((n, 2)) * math.tau
        self.damp = rng.random(n) ** 2 * 0.10 + 0.01
        self.dsig = rng.uniform(0.8, 2.2, n)

        # путник
        self.feet = (cx * H, 0.835 * H)
        self.ph_h = 0.058 * H

    def person(self, img, p):
        h = self.ph_h
        fx, fy = self.feet
        S = 4                                                  # суперсэмплинг
        pw, ph = int(h * 0.6), int(h * 1.15)
        x0, y0 = int(fx - pw / 2), int(fy - ph)
        ys, xs = np.mgrid[0:ph * S, 0:pw * S].astype(np.float32) / S
        px = xs + x0 - fx
        py = ys + y0 - fy                                      # 0 — у ступней, отрицательно вверх
        g = 13 * p                                             # 13 шагов за петлю
        sway = 0.012 * h * math.sin(g)
        bob = 0.012 * h * abs(math.cos(g))

        def capsule(ax, ay, bx, by, r):
            vx, vy = bx - ax, by - ay
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-6), 0, 1)
            return np.hypot(px - ax - t * vx, py - ay - t * vy) < r

        hip_y = -0.47 * h - bob
        lift_l = 0.07 * h * max(0.0, math.sin(g))
        lift_r = 0.07 * h * max(0.0, -math.sin(g))
        sh = mask = capsule(-0.06 * h + sway, hip_y, -0.06 * h, -lift_l, 0.055 * h)
        mask = mask | capsule(0.06 * h + sway, hip_y, 0.06 * h, -lift_r, 0.055 * h)
        mask = mask | capsule(sway, hip_y, sway, -0.80 * h - bob, 0.12 * h)       # туловище
        mask = mask | capsule(-0.13 * h + sway, -0.78 * h - bob, -0.15 * h + sway, -0.45 * h - bob - 0.02 * h * math.sin(g), 0.04 * h)
        mask = mask | capsule(0.13 * h + sway, -0.78 * h - bob, 0.15 * h + sway, -0.45 * h - bob + 0.02 * h * math.sin(g), 0.04 * h)
        mask = mask | (np.hypot(px - sway, py + 0.92 * h + bob) < 0.085 * h)       # голова
        cov = mask.reshape(ph, S, pw, S).mean(axis=(1, 3)).astype(np.float32)
        reg = img[:, y0:y0 + ph, x0:x0 + pw]
        reg *= (1 - cov * 0.92)[None]
        reg += (cov * 0.012)[None]

    def frame(self, p):
        img = self.static.copy()
        self.person(img, p)
        # дрейфующий туман
        ax = self.gx + 90 * np.sin(TAU * 0.6 * self.uy + p) + 40 * np.sin(TAU * 1.3 * self.ux - 2 * p)
        ay = self.gy + 25 * np.cos(TAU * 0.8 * self.ux + p)
        a = cv2.remap(self.f1, ax.astype(np.float32), ay.astype(np.float32), cv2.INTER_LINEAR)
        bx = self.gx - 60 * np.sin(TAU * 0.4 * self.uy + 2 * p + 1) + 30 * np.sin(TAU * 0.9 * self.ux + p)
        by = self.gy + 20 * np.sin(TAU * 1.1 * self.ux - p)
        b = cv2.remap(self.f2, bx.astype(np.float32), by.astype(np.float32), cv2.INTER_LINEAR)
        fog = smoothstep(0.3, 0.85, 0.6 * a + 0.4 * b) * self.fog_shape * 0.30
        fog = cv2.resize(fog, (W, H), interpolation=cv2.INTER_CUBIC)
        img = img * (1 - fog[None] * 0.6) + col(0.40, 0.41, 0.42) * fog[None] * 0.6
        img += self.shaft[None] * (0.85 + 0.15 * math.sin(p)) * col(1.0, 1.0, 1.02)
        # пыль
        xs = self.dx0 + self.dax * np.sin(self.dm * p + self.dph[:, 0])
        ys = self.dy0 + self.day * np.sin(self.dm * p + self.dph[:, 1])
        a = self.damp * (0.6 + 0.4 * np.sin(2 * self.dm * p + self.dph[:, 0]))
        cols = np.ones((len(xs), 3), np.float32) * np.array([0.9, 0.9, 0.95], np.float32)
        splat(img, xs, ys, a, cols, self.dsig, 4)
        return bloom(img, 0.55, 0.3, 30)


SCENES["canyon"] = (Canyon, 15)


# ---------------------------------------------------------------- 6b. Ущелье — реалистичнее
def ridged(rng, h, w, octaves, base):
    n = fbm(rng, h, w, octaves, base, 0.55)
    return 1 - np.abs(2 * n - 1)


class Canyon2(Canyon):
    def __init__(self):
        rng = np.random.default_rng(77)
        self.cx = ASPECT / 2
        cx = self.cx
        Y1 = np.arange(H, dtype=np.float32) / H
        floor_y = 0.80
        self.floor_y = floor_y

        # небо в щели: холодный верх, лёгкое тепло ниже
        t = np.clip(YY / floor_y, 0, 1)
        sky = col(0.70, 0.72, 0.74) * (1 - t) ** 1.4 + col(0.33, 0.33, 0.33) * (1 - (1 - t) ** 1.4)
        img = np.broadcast_to(sky, (3, H, W)).copy()

        # дно: камни в перспективе
        dy = np.maximum(YY - floor_y, 1e-3)
        pebble_src = fbm(rng, 512, 2048, 7, 10, 0.62)
        u = ((XX - cx) / dy * 0.10 + 0.5) % 1.0
        v = np.clip((0.06 / dy) % 1.0, 0, 1) * np.ones_like(XX)
        mx = (u * 2047).astype(np.float32) * np.ones_like(YY)
        my = (v * 511).astype(np.float32)
        peb = cv2.remap(pebble_src, mx, my, cv2.INTER_LINEAR)
        gt = np.clip((YY - floor_y) / (1 - floor_y), 0, 1)
        g_light = 0.22 * (1 - gt) ** 1.5 + 0.03
        side = np.abs(XX - cx)
        rubble = smoothstep(0.08, 0.35, side + 0.04 * (peb - 0.5)) * gt
        ground_v = g_light * (1 + (0.6 * peb - 0.3) * smoothstep(0.0, 0.5, gt)) * (1 - 0.6 * rubble)
        ground = np.stack([ground_v * 0.99, ground_v, ground_v * 1.02])
        gm = smoothstep(floor_y, floor_y + 0.006, YY)
        img = img * (1 - gm) + ground * gm

        # стены с рельефом и освещением
        K = 7
        for k in range(K):
            d = k / (K - 1)
            e = 0.028 + 0.62 * d ** 1.7
            fog = 0.9 * (1 - d) ** 1.8 + 0.01
            base = floor_y + 0.012 + 0.25 * d ** 2 if d < 0.5 else 1.2
            for side_s in (-1, 1):
                n_steps = int(6 + 10 * rng.random())
                knots = np.sort(rng.random(n_steps))
                offs = rng.random(n_steps + 1) * (0.02 + 0.05 * d)
                idx = np.searchsorted(knots, Y1)
                jag = offs[idx] + (fbm(rng, H, 1, 6, 20)[:, 0] - 0.5) * (0.006 + 0.02 * d)
                jag = cv2.GaussianBlur(jag.astype(np.float32)[:, None], (1, 0), sigmaX=0, sigmaY=2)[:, 0]
                edge = cx + side_s * (e + jag)
                dist = (XX - edge[:, None]) * side_s
                bnoise = (fbm(rng, 1, W, 6, 24)[0] - 0.5)[None, :] * 0.03
                m = np.clip(dist * H + 0.5, 0, 1) * (1 - smoothstep(base + bnoise, base + bnoise + 0.02, YY))
                if m.max() == 0:
                    continue
                # рельеф: крупные массы, трещины, вертикальные борозды эрозии, пласты
                scale = 0.6 + 0.8 * d
                big = fbm(rng, H, W, 6, int(5 * scale) + 3, 0.55)
                crack = ridged(rng, H, W, 7, int(14 * scale) + 6)
                groove = cv2.resize(rng.random((10, int(90 * scale) + 40), dtype=np.float32), (W, H),
                                    interpolation=cv2.INTER_CUBIC)
                strata = np.sin(YY * (30 + 30 * rng.random()) / scale + big * 7)
                hmap = 1.0 * big + 0.35 * crack ** 3 + 0.25 * groove + 0.08 * strata
                hmap = cv2.GaussianBlur(hmap, (0, 0), 1.0)
                gx = cv2.Sobel(hmap, cv2.CV_32F, 1, 0, ksize=3) * (H / 260)
                gy = cv2.Sobel(hmap, cv2.CV_32F, 0, 1, ksize=3) * (H / 260)
                L = np.array([-side_s * 0.35, -0.75, 0.55], np.float32)
                L /= np.linalg.norm(L)
                nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1)
                diff = np.maximum((-gx * L[0] - gy * L[1] + L[2]) * nz, 0)
                albedo = 0.10 + 0.06 * big - 0.04 * (crack ** 4)
                topfall = (1 - YY) ** 1.3                        # света меньше на дне
                ao = 0.55 + 0.45 * smoothstep(0.0, 0.06, dist)  # кромка чуть светлее (свет из щели)
                v = albedo * (0.25 + 1.1 * diff * (0.35 + 0.65 * topfall))
                v = v * (0.6 + 0.4 * topfall)
                rim = np.exp(-np.maximum(dist, 0) / (0.003 + 0.008 * d)) * 0.08 * topfall
                # светлые полки на уступах
                step_y = knots[None, :]
                ledge = np.zeros_like(YY)
                for ky in knots:
                    ledge = np.maximum(ledge, np.exp(-((YY - ky) * H / 2.2) ** 2))
                ledge = ledge * np.exp(-np.maximum(dist, 0) / 0.05) * 0.10 * topfall
                v = v * (2 - ao) * 0.5 + rim + ledge
                wall = np.stack([v * 0.97, v, v * 1.03]) * (1 - fog) + col(0.35, 0.355, 0.36) * fog * (1 - 0.35 * YY)
                img = img * (1 - m) + wall * m
                del big, crack, groove, hmap, gx, gy, diff
        self.static = img.astype(np.float32)

        # туман
        self.h2, self.w2 = H // 2, W // 2
        pad = 120
        self.pad = pad
        self.f1 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 4, 0.55)
        self.f2 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 7, 0.6)
        yy, xx = np.mgrid[0:self.h2, 0:self.w2].astype(np.float32)
        self.gx, self.gy = xx + pad, yy + pad
        self.uy, self.ux = yy / self.h2, xx / self.h2
        self.fog_shape = (np.exp(-((self.uy - 0.80) / 0.12) ** 2) * 1.0 + 0.22).astype(np.float32)
        # лучи света из щели (косые)
        beams = np.zeros((H, W), np.float32)
        for off, ang, wdt, st in [(-0.012, 0.05, 0.010, 0.10), (0.006, -0.03, 0.007, 0.08),
                                  (0.018, -0.08, 0.012, 0.07), (-0.02, 0.10, 0.006, 0.06)]:
            xc = cx + off + ang * YY
            beams += np.exp(-((XX - xc) / (wdt + 0.03 * YY)) ** 2) * st
        beams *= (1 - smoothstep(0.05, 0.85, YY))
        self.beams = beams
        self.shaft = (np.exp(-((XX - cx) / 0.05) ** 2) * (1 - smoothstep(0.1, 0.85, YY)) * 0.06).astype(np.float32)
        self.vig = (1 - 0.45 * (((XX - cx) / ASPECT) ** 2 * 2.0 + (YY - 0.5) ** 2 * 1.3)).astype(np.float32)
        g = np.random.default_rng(5)
        self.grain = [((g.random((H // 2, W // 2), dtype=np.float32) - 0.5) * 0.016) for _ in range(5)]

        n = 1100
        self.dx0 = rng.random(n) * W
        self.dy0 = rng.random(n) * H
        self.dax = rng.uniform(10, 40, n)
        self.day = rng.uniform(15, 60, n)
        self.dm = rng.integers(1, 3, n)
        self.dph = rng.random((n, 2)) * math.tau
        self.damp = rng.random(n) ** 2 * 0.08 + 0.008
        self.dsig = rng.uniform(0.8, 2.0, n)

        self.feet = (cx * H, 0.835 * H)
        self.ph_h = 0.056 * H

    def person(self, img, p):
        fx, fy = self.feet
        h = self.ph_h
        # тень под ногами
        sw, sh_ = int(h * 0.9), int(h * 0.14)
        ys, xs = np.mgrid[-sh_:sh_, -sw:sw].astype(np.float32)
        sh = np.exp(-((xs / (h * 0.32)) ** 2 + (ys / (h * 0.045)) ** 2)) * 0.45
        img[:, int(fy) - sh_:int(fy) + sh_, int(fx) - sw:int(fx) + sw] *= (1 - sh)[None]
        # силуэт: стройнее, с длинным плащом
        S = 4
        pw, ph = int(h * 0.6), int(h * 1.15)
        x0, y0 = int(fx - pw / 2), int(fy - ph)
        ys, xs = np.mgrid[0:ph * S, 0:pw * S].astype(np.float32) / S
        px = xs + x0 - fx
        py = ys + y0 - fy
        g = 13 * p
        sway = 0.012 * h * math.sin(g)
        bob = 0.012 * h * abs(math.cos(g))

        def capsule(ax, ay, bx, by, r):
            vx, vy = bx - ax, by - ay
            t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-6), 0, 1)
            return np.hypot(px - ax - t * vx, py - ay - t * vy) < r

        hip_y = -0.47 * h - bob
        lift_l = 0.07 * h * max(0.0, math.sin(g))
        lift_r = 0.07 * h * max(0.0, -math.sin(g))
        mask = capsule(-0.05 * h + sway, hip_y, -0.05 * h, -lift_l, 0.042 * h)
        mask |= capsule(0.05 * h + sway, hip_y, 0.05 * h, -lift_r, 0.042 * h)
        # плащ — трапеция от плеч до колен
        top_y, bot_y = -0.76 * h - bob, -0.30 * h
        tt = np.clip((py - top_y) / (bot_y - top_y), 0, 1)
        half = 0.11 * h + 0.05 * h * tt + 0.01 * h * math.sin(g) * tt
        coat = (py > top_y) & (py < bot_y) & (np.abs(px - sway) < half)
        mask |= coat
        mask |= capsule(-0.09 * h + sway, -0.78 * h - bob, 0.09 * h + sway, -0.78 * h - bob, 0.035 * h)  # плечи
        mask |= capsule(-0.12 * h + sway, -0.77 * h - bob, -0.14 * h + sway, -0.46 * h - bob - 0.02 * h * math.sin(g), 0.032 * h)
        mask |= capsule(0.12 * h + sway, -0.77 * h - bob, 0.14 * h + sway, -0.46 * h - bob + 0.02 * h * math.sin(g), 0.032 * h)
        mask |= capsule(sway, -0.86 * h - bob, sway, -0.80 * h - bob, 0.04 * h)          # шея
        mask |= np.hypot((px - sway) / 0.9, py + 0.93 * h + bob) < 0.075 * h               # голова
        cov = mask.reshape(ph, S, pw, S).mean(axis=(1, 3)).astype(np.float32)
        cov = cv2.GaussianBlur(cov, (0, 0), 0.6)
        reg = img[:, y0:y0 + ph, x0:x0 + pw]
        person_col = col(0.035, 0.036, 0.04)                    # не чисто чёрный — дымка расстояния
        reg[:] = reg * (1 - cov[None]) + person_col * cov[None]

    def frame(self, p):
        img = super().frame(p)
        img += self.beams[None] * (0.8 + 0.2 * math.sin(p + 0.5)) * col(1.0, 0.99, 0.96)
        img *= self.vig[None]
        f = int(round(p / TAU * 450)) % 5
        img += cv2.resize(self.grain[f], (W, H), interpolation=cv2.INTER_NEAREST)[None]
        return img


SCENES["canyon"] = (Canyon2, 15)


# ---------------------------------------------------------------- 6c. Ущелье — идём вместе с путником
def periodic_noise(rng, h, w, beta=2.0):
    """Шум 1/f^beta, бесшовно повторяющийся по обеим осям (через FFT)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    spec = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / f ** (beta / 2)
    spec[0, 0] = 0
    n = np.fft.irfft2(spec, s=(h, w)).astype(np.float32)
    n -= n.min()
    return n / n.max()


class CanyonWalk:
    PPU = 64          # пикселей текстуры на единицу мира
    CW, CH = 12, 60   # размер «карты» скалы в единицах мира
    K, DZ = 8, 4.0    # уникальных карт на сторону, шаг по глубине
    NC = 32           # карт на сторону в коридоре
    ZMIN = 1.85
    CAM = 1.75        # высота камеры
    PERSON_Z = 32.0

    def __init__(self):
        rng = np.random.default_rng(91)
        self.F = float(H)
        self.cxp = W / 2
        self.yh = 0.78 * H
        self.span = self.NC * self.DZ
        self.loop = self.K * self.DZ

        # небо в щели
        t = np.clip(YY / 0.78, 0, 1)
        self.sky = (col(0.70, 0.72, 0.74) * (1 - t) ** 1.4 + col(0.34, 0.345, 0.35) * (1 - (1 - t) ** 1.4)).astype(np.float32)
        self.fogc = col(0.35, 0.355, 0.36)

        # карты скал (для каждой стороны K штук) + мип-уровни
        self.cards = {}
        self.e = {}
        for side in (-1, 1):
            for k in range(self.K):
                self.cards[side, k] = self.make_card(rng)
                self.e[side, k] = 1.8 + 1.4 * rng.random()

        d1 = periodic_noise(rng, 512, 512, 1.6)
        d2 = 1 - np.abs(2 * periodic_noise(rng, 512, 512, 2.0) - 1)
        self.detail = (0.6 * d1 + 0.4 * d2 ** 3).astype(np.float32)
        self.detail = (self.detail - self.detail.mean()) / (self.detail.std() * 4) + 0.5
        # пол: периодическая текстура
        self.gtile = periodic_noise(rng, 512, 512, 2.2) * 0.6 + periodic_noise(rng, 512, 512, 1.2) * 0.4
        ys = np.arange(H, dtype=np.float32)
        fl = ys > self.yh + 1
        self.fy0 = int(np.argmax(fl))
        yy = ys[self.fy0:, None]
        self.gz = (self.CAM * self.F / (yy - self.yh)).astype(np.float32)               # глубина
        xw = (np.arange(W, dtype=np.float32)[None, :] - self.cxp) * self.gz / self.F       # мировой x
        self.gcol = (((xw + 8) % 16) / 16 * 511).astype(np.float32)
        self.gfog = (1 - np.exp(-self.gz / 38)).astype(np.float32)
        side = np.clip(np.abs(xw) / 2.6, 0, 1)
        self.gdark = (0.55 + 0.45 * (1 - side ** 2)).astype(np.float32)                 # у стен темнее

        # объёмный туман (экранный), лучи, виньетка, зерно
        self.h2, self.w2 = H // 2, W // 2
        pad = 120
        self.f1 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 4, 0.55)
        self.f2 = fbm(rng, self.h2 + 2 * pad, self.w2 + 2 * pad, 7, 7, 0.6)
        yy2, xx2 = np.mgrid[0:self.h2, 0:self.w2].astype(np.float32)
        self.fgx, self.fgy = xx2 + pad, yy2 + pad
        self.fuy, self.fux = yy2 / self.h2, xx2 / self.h2
        self.fog_shape = (np.exp(-((self.fuy - 0.79) / 0.11) ** 2) + 0.2).astype(np.float32)
        cx = ASPECT / 2
        beams = np.zeros((H, W), np.float32)
        for off, ang, wdt, st in [(-0.012, 0.05, 0.010, 0.09), (0.006, -0.03, 0.007, 0.07),
                                  (0.018, -0.08, 0.012, 0.06), (-0.02, 0.10, 0.006, 0.05)]:
            beams += np.exp(-((XX - (cx + off + ang * YY)) / (wdt + 0.03 * YY)) ** 2) * st
        self.beams = beams * (1 - smoothstep(0.05, 0.8, YY))
        self.vig = (1 - 0.45 * (((XX - cx) / ASPECT) ** 2 * 2.0 + (YY - 0.5) ** 2 * 1.3)).astype(np.float32)
        g = np.random.default_rng(5)
        self.grain = [((g.random((H // 2, W // 2), dtype=np.float32) - 0.5) * 0.016) for _ in range(5)]

        # пыль в 3D
        n = 1600
        self.px = rng.uniform(-3.5, 3.5, n)
        self.ph_ = rng.uniform(0, 9, n)
        self.pz0 = rng.uniform(0, self.loop, n)
        self.pamp = rng.random(n) ** 2 * 0.5 + 0.05
        self.pw = rng.random((n, 2)) * math.tau

        # путник
        self.feet = (self.cxp, self.yh + self.CAM * self.F / self.PERSON_Z)
        self.ph_h = 1.8 * self.F / self.PERSON_Z

    # ---- текстура одной «карты» скалы: RGBA, u — наружу от кромки, v — сверху вниз
    def make_card(self, rng):
        tw, th = self.CW * self.PPU, self.CH * self.PPU
        big = fbm(rng, th, tw, 6, 6, 0.55)
        crack = 1 - np.abs(2 * fbm(rng, th, tw, 7, 18, 0.55) - 1)
        groove = cv2.resize(rng.random((12, 60), dtype=np.float32), (tw, th), interpolation=cv2.INTER_CUBIC)
        vv = np.arange(th, dtype=np.float32)[:, None] / self.PPU          # от верха, ед.
        uu = np.arange(tw, dtype=np.float32)[None, :] / self.PPU          # от кромки, ед.
        strata = np.sin(vv * (1.4 + rng.random()) + big * 7)
        hmap = big + 0.35 * crack ** 3 + 0.25 * groove + 0.08 * strata
        hmap = cv2.GaussianBlur(hmap, (0, 0), 1.0)
        gx = cv2.Sobel(hmap, cv2.CV_32F, 1, 0, ksize=3) * (self.PPU / 9)
        gy = cv2.Sobel(hmap, cv2.CV_32F, 0, 1, ksize=3) * (self.PPU / 9)
        L = np.array([-0.35, -0.75, 0.55], np.float32)
        L /= np.linalg.norm(L)
        nz = 1 / np.sqrt(gx * gx + gy * gy + 1)
        diff = np.maximum((-gx * L[0] - gy * L[1] + L[2]) * nz, 0)
        hgt = (self.CH - vv) / self.CH                                    # 1 — верх, 0 — земля
        topfall = hgt ** 1.2
        albedo = 0.10 + 0.06 * big - 0.04 * crack ** 4
        v = albedo * (0.22 + 1.1 * diff * (0.3 + 0.7 * topfall)) * (0.55 + 0.45 * topfall)
        # кромка ступенями + мелкая зазубренность
        n_steps = int(5 + 8 * rng.random())
        knots = np.sort(rng.random(n_steps)) * self.CH
        offs = rng.random(n_steps + 1) * 1.2
        jag = offs[np.searchsorted(knots, vv[:, 0])] + (fbm(rng, th, 1, 6, 30)[:, 0] - 0.5) * 0.35
        jag = cv2.GaussianBlur(jag.astype(np.float32)[:, None], (1, 0), sigmaX=0, sigmaY=3)
        dist = uu - jag
        a = np.clip(dist * self.PPU + 0.5, 0, 1)
        rim = np.exp(-np.maximum(dist, 0) / 0.12) * 0.07 * topfall
        ledge = np.zeros_like(vv)
        for ky in knots:
            ledge = np.maximum(ledge, np.exp(-((vv - ky) * self.PPU / 2.0) ** 2))
        ledge = ledge * np.exp(-np.maximum(dist, 0) / 2.5) * 0.08 * topfall
        v = v + rim + ledge
        # низ карты растворяется в осыпи
        bot = 1 - smoothstep(self.CH - 0.6, self.CH, vv + (fbm(rng, 1, tw, 5, 10)[0] - 0.5)[None, :] * 0.8)
        a = a * bot
        rgba = np.stack([v * 0.97, v, v * 1.03, np.broadcast_to(a, v.shape)], 2).astype(np.float32)
        mips = [rgba]
        for _ in range(4):
            mips.append(cv2.pyrDown(mips[-1]))
        return mips

    def draw_card(self, img, mips, side, e, z):
        s = self.F / (z * self.PPU)                         # пикселей экрана на пиксель текстуры
        lvl = 0
        while lvl < len(mips) - 1 and s < 0.5:
            s *= 2
            lvl += 1
        tex = mips[lvl]
        th, tw = tex.shape[:2]
        x0 = self.cxp + side * e * self.F / z
        y0 = self.yh + (self.CAM - self.CH) * self.F / z
        xa, xb = (x0, x0 + side * s * tw)
        xmin, xmax = max(int(min(xa, xb)), 0), min(int(math.ceil(max(xa, xb))), W)
        ymin, ymax = max(int(y0), 0), min(int(math.ceil(y0 + s * th)), H)
        if xmin >= xmax or ymin >= ymax:
            return
        M = np.float32([[side * s, 0, x0 - xmin], [0, s, y0 - ymin]])
        out = cv2.warpAffine(tex, M, (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if lvl == 0 and s > 1.3:
            # мелкая детализация для близких скал (иначе они размыты при увеличении)
            Md = np.float32([[side * s / 4, 0, x0 - xmin], [0, s / 4, y0 - ymin]])
            det = cv2.warpAffine(self.detail, Md, (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)
            k = min(1.0, (s - 1.3) / 2.0)
            out[..., :3] *= (1 + k * (det - 0.5) * 0.9)[..., None]
        fog = 1 - math.exp(-z / 40)
        fade = float(smoothstep(self.ZMIN + self.span, self.ZMIN + self.span - 20, z))
        a = out[..., 3] * fade
        reg = img[:, ymin:ymax, xmin:xmax]
        fc = self.fogc[:, :, 0]
        for c in range(3):
            colr = out[..., c] * (1 - fog) + fc[c] * fog
            reg[c] = reg[c] * (1 - a) + colr * a

    def frame(self, p):
        s = (p / TAU) % 1.0
        travel = s * self.loop
        img = np.broadcast_to(self.sky, (3, H, W)).copy()

        # пол
        row = np.ascontiguousarray(np.broadcast_to((((self.gz + travel) % 32) / 32 * 511).astype(np.float32), self.gcol.shape))
        tex = cv2.remap(self.gtile, self.gcol, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        detail = 0.6 + 0.8 * tex
        gv = 0.16 * self.gdark * detail * (1 - self.gfog) + 0.0
        floor = np.stack([gv * 0.99, gv, gv * 1.02])
        fg = self.gfog[None]
        floor = floor * (1 - fg) + self.fogc * fg * (0.9 - 0.25 * (self.gz < 6))
        img[:, self.fy0:] = floor

        # скалы: от дальних к ближним
        cards = []
        for side in (-1, 1):
            for k in range(self.NC):
                z = self.ZMIN + ((k * self.DZ - travel) % self.span)
                cards.append((z, side, k % self.K))
        cards.sort(key=lambda c: -c[0])
        person_drawn = False
        for z, side, k in cards:
            if not person_drawn and z < self.PERSON_Z:
                Canyon2.person(self, img, p * 21 / 13)
                person_drawn = True
            self.draw_card(img, self.cards[side, k], side, self.e[side, k], z)

        # дрейфующий туман
        ax = self.fgx + 90 * np.sin(TAU * 0.6 * self.fuy + p) + 40 * np.sin(TAU * 1.3 * self.fux - 2 * p)
        ay = self.fgy + 25 * np.cos(TAU * 0.8 * self.fux + p)
        fa = cv2.remap(self.f1, ax.astype(np.float32), ay.astype(np.float32), cv2.INTER_LINEAR)
        bx = self.fgx - 60 * np.sin(TAU * 0.4 * self.fuy + 2 * p + 1) + 30 * np.sin(TAU * 0.9 * self.fux + p)
        by = self.fgy + 20 * np.sin(TAU * 1.1 * self.fux - p)
        fb = cv2.remap(self.f2, bx.astype(np.float32), by.astype(np.float32), cv2.INTER_LINEAR)
        fog = smoothstep(0.3, 0.85, 0.6 * fa + 0.4 * fb) * self.fog_shape * 0.28
        fog = cv2.resize(fog, (W, H), interpolation=cv2.INTER_CUBIC)
        img = img * (1 - fog[None] * 0.6) + col(0.40, 0.41, 0.42) * fog[None] * 0.6
        img += self.beams[None] * (0.8 + 0.2 * math.sin(p + 0.5)) * col(1.0, 0.99, 0.96)

        # пыль летит навстречу
        z = 0.6 + ((self.pz0 - travel) % self.loop)
        xs = self.cxp + (self.px + 0.15 * np.sin(p + self.pw[:, 0])) * self.F / z
        ys = self.yh + (self.CAM - self.ph_ - 0.15 * np.sin(2 * p + self.pw[:, 1])) * self.F / z
        amp = self.pamp / z * smoothstep(0.6, 2.0, z) * (1 - z / self.loop) * 0.6
        sig = np.clip(0.04 * self.F / z / 6, 0.7, 2.3)
        cols = np.ones((len(xs), 3), np.float32) * np.array([0.9, 0.9, 0.95], np.float32)
        splat(img, xs, ys, amp, cols, sig, 4)

        img = bloom(img, 0.55, 0.3, 30)
        img *= self.vig[None]
        f = int(round(s * 600)) % 5
        img += cv2.resize(self.grain[f], (W, H), interpolation=cv2.INTER_NEAREST)[None]
        return img


SCENES["canyon"] = (CanyonWalk, 20)
