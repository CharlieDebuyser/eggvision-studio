"""Générateur d'images synthétiques d'une table à rouleaux vue de dessus.

Sert à développer et tester toute la chaîne sans matériel (``SimulatedSource``,
agent Pi en mode simulé, jeu de données de démonstration, tests automatisés).

Modèle de scène :
  * les œufs avancent en colonnes de ``n_lanes`` œufs, le long de l'axe +y de
    l'image (du haut vers le bas) ; les couloirs sont des bandes verticales ;
  * chaque œuf repose entre deux rouleaux, grand axe horizontal, et tourne sur
    lui-même autour de ce grand axe ;
  * les défauts sont peints sur la surface de l'œuf (coordonnées « position le
    long du grand axe » × « angle autour de l'axe ») : ils apparaissent et
    disparaissent donc avec la rotation, comme sur une vraie table.

Tout est déterministe : même configuration + même graine = mêmes images.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .constants import DEFECT_TYPES, SENSOR_HEIGHT, SENSOR_WIDTH, parse_lot_type
from .preprocess import extract_crop

_TEX_U = 72  # résolution de la texture le long du grand axe
_TEX_T = 192  # résolution de la texture autour de l'axe (360°)


@dataclass
class SceneConfig:
    """Paramètres d'une scène synthétique."""

    width: int = SENSOR_WIDTH
    height: int = SENSOR_HEIGHT
    n_lanes: int = 6
    mm_per_px: float = 0.27
    pitch_mm: float = 76.0  # pas entre deux colonnes
    column_period_s: float = 0.5  # une colonne toutes les 0,5 s
    fps: float = 30.0
    lot_type: str = "viable"  # viable | declasse | declasse:<défaut> | production
    defect_rate: float = 0.05  # proportion de non viables d'un lot « production »
    empty_lane_prob: float = 0.0  # probabilité qu'un emplacement soit vide
    rotations_per_pass: float = 1.25  # tours effectués pendant la traversée de l'image
    noise_sigma: float = 2.0
    motion_blur_px: int = 0
    n_columns: int | None = None  # nombre de colonnes du lot ; None = défilement sans fin
    seed: int = 0

    @classmethod
    def scaled(cls, scale: float, **kwargs: Any) -> SceneConfig:
        """Scène à résolution réduite (``scale`` = 0,5 → 728 × 544), même champ de vue."""
        w = int(round(SENSOR_WIDTH * scale / 8)) * 8
        h = int(round(SENSOR_HEIGHT * scale / 8)) * 8
        return cls(width=w, height=h, mm_per_px=0.27 * SENSOR_WIDTH / w, **kwargs)

    @property
    def speed_mm_s(self) -> float:
        return self.pitch_mm / self.column_period_s

    @property
    def speed_px_s(self) -> float:
        return self.speed_mm_s / self.mm_per_px

    @property
    def pitch_px(self) -> float:
        return self.pitch_mm / self.mm_per_px

    @property
    def lane_width_px(self) -> float:
        return self.width / self.n_lanes


@dataclass
class EggSpec:
    """Un œuf physique de la scène."""

    egg_id: int
    column: int
    lane: int
    viable: bool
    defect: str | None
    major_mm: float
    minor_mm: float
    color_bgr: tuple[float, float, float]
    rot0: float
    rot_speed: float  # rad/s
    dx_mm: float  # décalage latéral dans le couloir
    dy_mm: float  # décalage le long du défilement
    asym: float  # asymétrie « bout pointu »
    bulge: tuple[float, float, float]  # (amplitude, fréquence, phase) de déformation du contour
    seed: int
    _texture: np.ndarray | None = field(default=None, repr=False, compare=False)

    def texture(self) -> np.ndarray:
        """Texture de surface RGBA float32 ``(_TEX_U, _TEX_T, 4)`` (BGR + alpha), mise en cache."""
        if self._texture is None:
            self._texture = _make_texture(self)
        return self._texture


@dataclass
class EggTruth:
    """Vérité terrain d'un œuf dans une image."""

    egg_id: int
    column: int
    lane: int
    cx: float
    cy: float
    major_px: float
    minor_px: float
    rot: float
    viable: bool
    defect: str | None
    fully_visible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "egg_id": self.egg_id,
            "column": self.column,
            "lane": self.lane,
            "cx": round(self.cx, 2),
            "cy": round(self.cy, 2),
            "major_px": round(self.major_px, 2),
            "minor_px": round(self.minor_px, 2),
            "viable": self.viable,
            "defect": self.defect,
            "fully_visible": self.fully_visible,
        }


# ---------------------------------------------------------------------------
# Textures de défauts
# ---------------------------------------------------------------------------
def _blob(rng: np.random.Generator, cu: float, ct: float, su: float, st: float, rough: float = 0.5) -> np.ndarray:
    """Tache irrégulière (carte alpha 0..1) centrée en (cu, ct), périodique selon l'angle."""
    uu = np.arange(_TEX_U, dtype=np.float32)[:, None]
    tt = np.arange(_TEX_T, dtype=np.float32)[None, :]
    dt = np.abs(tt - ct)
    dt = np.minimum(dt, _TEX_T - dt)
    d2 = ((uu - cu) / su) ** 2 + (dt / st) ** 2
    base = np.exp(-d2)
    noise = rng.random((_TEX_U // 6 + 1, _TEX_T // 6 + 1)).astype(np.float32)
    noise = cv2.resize(noise, (_TEX_T, _TEX_U), interpolation=cv2.INTER_CUBIC)
    a = base * (1.0 - rough + 2.0 * rough * noise)
    return np.clip((a - 0.35) * 4.0, 0.0, 1.0)


def _polyline_mask(points: list[tuple[float, float]], thickness: int) -> np.ndarray:
    """Trace une ligne brisée sur une texture périodique en angle (alpha 0..1)."""
    canvas = np.zeros((_TEX_U, _TEX_T * 3), dtype=np.uint8)
    pts = np.array([[t + _TEX_T, u] for u, t in points], dtype=np.float32)
    cv2.polylines(canvas, [np.round(pts * 4).astype(np.int32)], False, 255, thickness, cv2.LINE_AA, shift=2)
    folded = np.maximum.reduce([canvas[:, :_TEX_T], canvas[:, _TEX_T : 2 * _TEX_T], canvas[:, 2 * _TEX_T :]])
    return folded.astype(np.float32) / 255.0


def _over(tex: np.ndarray, alpha: np.ndarray, color: tuple[float, float, float]) -> None:
    """Compose une couche (alpha, couleur BGR) sur la texture RGBA, en place."""
    a = alpha[..., None]
    tex[..., :3] = tex[..., :3] * (1.0 - a) + np.asarray(color, dtype=np.float32) * a
    tex[..., 3] = np.maximum(tex[..., 3], alpha)


def _make_texture(egg: EggSpec) -> np.ndarray:
    rng = np.random.default_rng(egg.seed + 7919)
    tex = np.zeros((_TEX_U, _TEX_T, 4), dtype=np.float32)

    # Variations bénignes présentes sur tous les œufs : fines mouchetures claires.
    n_speck = int(rng.integers(0, 25))
    for _ in range(n_speck):
        a = _blob(rng, rng.uniform(8, _TEX_U - 8), rng.uniform(0, _TEX_T), rng.uniform(0.6, 1.3), rng.uniform(0.6, 1.3), 0.2)
        light = tuple(min(255.0, c + rng.uniform(8, 22)) for c in egg.color_bgr)
        _over(tex, a * rng.uniform(0.25, 0.5), light)

    d = egg.defect
    if d == "souillure":
        for _ in range(int(rng.integers(1, 5))):
            a = _blob(rng, rng.uniform(12, _TEX_U - 12), rng.uniform(0, _TEX_T), rng.uniform(3, 9), rng.uniform(5, 16), 0.6)
            color = (rng.uniform(25, 60), rng.uniform(50, 85), rng.uniform(60, 105))  # brun / verdâtre
            _over(tex, a * rng.uniform(0.6, 0.95), color)
    elif d == "coulure":
        ct = rng.uniform(0, _TEX_T)
        cu = rng.uniform(18, _TEX_U - 18)
        a = _blob(rng, cu, ct, rng.uniform(3, 6), rng.uniform(18, 40), 0.35)
        for _ in range(int(rng.integers(1, 4))):  # gouttes
            a = np.maximum(a, _blob(rng, cu + rng.uniform(-10, 10), ct + rng.uniform(-30, 30), rng.uniform(2, 4), rng.uniform(3, 7), 0.3))
        _over(tex, a * rng.uniform(0.55, 0.85), (rng.uniform(20, 60), rng.uniform(170, 215), rng.uniform(225, 250)))  # jaune d'œuf
    elif d == "felure":
        u, t = rng.uniform(15, _TEX_U - 15), rng.uniform(0, _TEX_T)
        pts = [(u, t)]
        heading = rng.uniform(0, 2 * math.pi)
        for _ in range(int(rng.integers(5, 12))):
            heading += rng.normal(0, 0.7)
            step = rng.uniform(4, 10)
            u = float(np.clip(u + math.sin(heading) * step * 0.6, 6, _TEX_U - 6))
            t = t + math.cos(heading) * step
            pts.append((u, t))
        a = _polyline_mask(pts, 1)
        if rng.random() < 0.5:  # ramification
            k = int(rng.integers(1, len(pts) - 1))
            bu, bt = pts[k]
            branch = [(bu, bt)]
            for _ in range(int(rng.integers(2, 5))):
                bu = float(np.clip(bu + rng.uniform(-6, 6), 6, _TEX_U - 6))
                bt = bt + rng.uniform(-7, 7)
                branch.append((bu, bt))
            a = np.maximum(a, _polyline_mask(branch, 1))
        dark = tuple(c * rng.uniform(0.3, 0.5) for c in egg.color_bgr)
        _over(tex, np.clip(a * rng.uniform(0.75, 1.0), 0, 1), dark)
    elif d == "sang":
        ct, cu = rng.uniform(0, _TEX_T), rng.uniform(14, _TEX_U - 14)
        for _ in range(int(rng.integers(1, 5))):
            a = _blob(rng, cu + rng.uniform(-9, 9), ct + rng.uniform(-14, 14), rng.uniform(1.5, 5), rng.uniform(2, 9), 0.45)
            _over(tex, a * rng.uniform(0.7, 0.95), (rng.uniform(15, 45), rng.uniform(10, 40), rng.uniform(120, 190)))
    elif d == "plume":
        u, t = rng.uniform(18, _TEX_U - 18), rng.uniform(0, _TEX_T)
        heading = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(26, 48)
        curve = rng.normal(0, 0.035)
        rachis = [(u, t)]
        vane = np.zeros((_TEX_U, _TEX_T), dtype=np.float32)
        for i in range(int(length / 3)):
            heading += curve
            u = float(np.clip(u + math.sin(heading) * 3 * 0.6, 4, _TEX_U - 4))
            t = t + math.cos(heading) * 3
            rachis.append((u, t))
            w = 3.2 * math.sin(math.pi * (i + 1) / (length / 3 + 1)) + 0.6
            for side in (-1, 1):  # barbes
                bu = u + side * w * math.cos(heading) * 0.6 * 1.6
                bt = t - side * w * math.sin(heading) * 1.6
                vane = np.maximum(vane, _polyline_mask([(u, t), (bu, bt)], 1))
        vane = cv2.GaussianBlur(vane, (0, 0), 0.8)
        lum = float(np.mean(egg.color_bgr))
        feather = (235.0, 236.0, 238.0) if lum < 200 else (150.0, 155.0, 165.0)
        _over(tex, np.clip(vane * rng.uniform(0.7, 1.0), 0, 1), feather)
        shaft = tuple(c * 0.75 for c in feather)
        _over(tex, _polyline_mask(rachis, 1) * 0.9, shaft)
    elif d == "deformation":
        # Bourrelets / stries calcaires en plus du contour irrégulier.
        uu = np.arange(_TEX_U, dtype=np.float32)[:, None]
        tt = np.arange(_TEX_T, dtype=np.float32)[None, :]
        freq = rng.uniform(0.25, 0.55)
        ridges = np.sin(uu * freq + np.sin(tt * 2 * math.pi / _TEX_T * int(rng.integers(1, 4))) * 1.5 + rng.uniform(0, 6))
        band = np.exp(-((uu - rng.uniform(18, _TEX_U - 18)) / rng.uniform(8, 16)) ** 2)
        a = np.clip(ridges, 0, 1) * band * rng.uniform(0.25, 0.45) * np.ones_like(tt)
        _over(tex, a.astype(np.float32), tuple(c * 0.72 for c in egg.color_bgr))
        a2 = np.clip(-ridges, 0, 1) * band * 0.25 * np.ones_like(tt)
        _over(tex, a2.astype(np.float32), tuple(min(255.0, c * 1.12) for c in egg.color_bgr))
    return tex


# ---------------------------------------------------------------------------
# Rendu d'un œuf
# ---------------------------------------------------------------------------
def render_egg_patch(egg: EggSpec, rot: float, mm_per_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Rend un œuf seul. Retourne (patch BGR float32, alpha float32), centrés sur l'œuf."""
    a = egg.major_mm / mm_per_px / 2.0
    b = egg.minor_mm / mm_per_px / 2.0
    amp, freq, phase = egg.bulge
    half_w = int(math.ceil(a)) + 2
    half_h = int(math.ceil(b * (1.0 + abs(egg.asym) + abs(amp)))) + 2
    xs = np.arange(-half_w, half_w + 1, dtype=np.float32)[None, :]
    ys = np.arange(-half_h, half_h + 1, dtype=np.float32)[:, None]
    u = xs / a
    local_b = b * (1.0 + egg.asym * u) * (1.0 + amp * np.sin(freq * math.pi * u + phase))
    v = ys / np.maximum(local_b, 1e-3)
    r2 = u * u + v * v
    inside = r2 < 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    alpha = np.clip((1.0 - np.sqrt(r2)) * min(a, b) * 0.9, 0.0, 1.0).astype(np.float32)
    alpha = np.where(np.abs(u) < 1.0, alpha, 0.0).astype(np.float32)

    # Angle autour du grand axe.
    cross = np.sqrt(np.clip(1.0 - u * u, 1e-4, 1.0))
    phi = np.arcsin(np.clip(v / cross, -1.0, 1.0))
    theta = np.mod(phi + rot, 2.0 * math.pi)
    map_t = (theta / (2.0 * math.pi) * _TEX_T).astype(np.float32)
    map_u = np.broadcast_to((u + 1.0) * 0.5 * (_TEX_U - 1), map_t.shape).astype(np.float32)
    tex = cv2.remap(egg.texture(), map_t, map_u, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    base = np.empty(map_t.shape + (3,), dtype=np.float32)
    base[...] = np.asarray(egg.color_bgr, dtype=np.float32)
    ta = tex[..., 3:4]
    base = base * (1.0 - ta) + tex[..., :3] * ta

    shade = 0.50 + 0.50 * np.power(z, 0.65)
    spec = 0.20 * np.exp(-(((u + 0.18) / 0.22) ** 2 + ((v + 0.28) / 0.18) ** 2))
    patch = base * shade[..., None] + 255.0 * spec[..., None] * inside[..., None]
    return np.clip(patch, 0.0, 255.0), alpha


def _roller_background(cfg: SceneConfig) -> np.ndarray:
    """Fond de rouleaux, périodique le long de y (hauteur = image + un pas)."""
    pitch = cfg.pitch_px
    h = cfg.height + int(math.ceil(pitch)) + 2
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(cfg.width, dtype=np.float32)[None, :]
    # Les œufs sont à y ≡ 0 (mod pas) ; les rouleaux sont centrés à mi-pas.
    d = np.mod(y, pitch) - pitch / 2.0
    radius = pitch * 0.44
    cyl = np.sqrt(np.clip(1.0 - (d / radius) ** 2, 0.0, 1.0))
    lane_w = cfg.lane_width_px
    lane_pos = np.mod(x, lane_w) / lane_w - 0.5
    diabolo = 0.82 + 0.18 * np.cos(lane_pos * 2 * math.pi)  # creux au centre de chaque couloir
    lum = 0.12 + 0.88 * cyl * (0.55 + 0.45 * cyl) * diabolo
    color = np.array([96.0, 78.0, 66.0], dtype=np.float32)  # plastique bleu ardoise (BGR)
    bg = lum[..., None] * color
    rng = np.random.default_rng(cfg.seed + 101)
    grain = cv2.resize(
        rng.normal(0, 1, (h // 4 + 1, cfg.width // 4 + 1)).astype(np.float32), (cfg.width, h), interpolation=cv2.INTER_LINEAR
    )
    bg += grain[..., None] * 3.0
    return np.clip(bg, 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Scène
# ---------------------------------------------------------------------------
class SyntheticScene:
    """Scène déterministe à accès aléatoire : ``frame(i)`` rend l'image n° i."""

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.cfg = config or SceneConfig()
        self.lot_base, self.lot_defect = parse_lot_type(self.cfg.lot_type)
        self._bg = _roller_background(self.cfg)
        self._columns: OrderedDict[int, list[EggSpec | None]] = OrderedDict()
        rng = np.random.default_rng(self.cfg.seed + 5)
        self._noise = [
            rng.normal(0, self.cfg.noise_sigma, (self.cfg.height, self.cfg.width, 1)).astype(np.float32)
            for _ in range(4)
        ] if self.cfg.noise_sigma > 0 else []
        self._entry_margin = self.cfg.pitch_px * 0.5

    # -- géométrie --------------------------------------------------------
    def lane_center_x(self, lane: int) -> float:
        return (lane + 0.5) * self.cfg.lane_width_px

    def column_y(self, column: int, t: float) -> float:
        """Position y (px) du centre de la colonne à l'instant ``t`` (s)."""
        return self.cfg.speed_px_s * (t - column * self.cfg.column_period_s) - self._entry_margin

    def calibration(self) -> dict[str, Any]:
        """Calibration de scène correspondante (même schéma que ``config.yaml`` → ``scene``)."""
        w, h = self.cfg.width, self.cfg.height
        lw = self.cfg.lane_width_px
        return {
            "image_width": w,
            "image_height": h,
            "direction": "down",
            "mm_per_px": round(self.cfg.mm_per_px, 5),
            "lanes": [
                {"index": i, "polygon": [[round(i * lw, 1), 0], [round((i + 1) * lw, 1), 0], [round((i + 1) * lw, 1), h], [round(i * lw, 1), h]]}
                for i in range(self.cfg.n_lanes)
            ],
        }

    # -- œufs -------------------------------------------------------------
    def _make_egg(self, column: int, lane: int) -> EggSpec | None:
        cfg = self.cfg
        seed = (cfg.seed * 1_000_003 + column * 97 + lane * 13 + 1) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        if rng.random() < cfg.empty_lane_prob:
            return None
        if self.lot_base == "viable":
            defect = None
        elif self.lot_base == "declasse":
            defect = self.lot_defect or str(rng.choice(DEFECT_TYPES))
        else:
            defect = str(rng.choice(DEFECT_TYPES)) if rng.random() < cfg.defect_rate else None
        tone = rng.random()
        white = np.array([226.0, 236.0, 240.0])
        brown = np.array([118.0, 158.0, 204.0])
        color = white * (1 - tone) + brown * tone + rng.normal(0, 4, 3)
        traverse_s = cfg.height / cfg.speed_px_s
        rot_speed = 2 * math.pi * cfg.rotations_per_pass / traverse_s * rng.uniform(0.9, 1.15)
        if defect == "deformation":
            asym = float(rng.choice([-1, 1]) * rng.uniform(0.2, 0.34))
            bulge = (float(rng.uniform(0.06, 0.13)), float(rng.uniform(1.2, 2.6)), float(rng.uniform(0, 6.28)))
            major, minor = rng.uniform(50, 68), rng.uniform(38, 47)
        else:
            asym = float(rng.choice([-1, 1]) * rng.uniform(0.05, 0.13))
            bulge = (0.0, 1.0, 0.0)
            major, minor = rng.uniform(54, 62), rng.uniform(41.5, 46)
        return EggSpec(
            egg_id=column * cfg.n_lanes + lane,
            column=column,
            lane=lane,
            viable=defect is None,
            defect=defect,
            major_mm=float(major),
            minor_mm=float(minor),
            color_bgr=tuple(float(np.clip(c, 60, 250)) for c in color),  # type: ignore[arg-type]
            rot0=float(rng.uniform(0, 2 * math.pi)),
            rot_speed=float(rot_speed),
            dx_mm=float(rng.normal(0, 1.2)),
            dy_mm=float(rng.normal(0, 0.8)),
            asym=asym,
            bulge=bulge,
            seed=seed,
        )

    def column(self, column: int) -> list[EggSpec | None]:
        col = self._columns.get(column)
        if self.cfg.n_columns is not None and column >= self.cfg.n_columns:
            return [None] * self.cfg.n_lanes
        if col is None:
            col = [self._make_egg(column, lane) for lane in range(self.cfg.n_lanes)]
            self._columns[column] = col
            while len(self._columns) > 64:
                self._columns.popitem(last=False)
        else:
            self._columns.move_to_end(column)
        return col

    def visible_columns(self, t: float) -> range:
        cfg = self.cfg
        margin = cfg.pitch_px
        first = math.floor((t - (cfg.height + margin + self._entry_margin) / cfg.speed_px_s) / cfg.column_period_s)
        last = math.floor((t + (margin - self._entry_margin) / cfg.speed_px_s) / cfg.column_period_s) + 1
        return range(max(0, first), max(0, last) + 1)

    def truth_at(self, t: float) -> list[EggTruth]:
        cfg = self.cfg
        out: list[EggTruth] = []
        for c in self.visible_columns(t):
            y_col = self.column_y(c, t)
            for egg in self.column(c):
                if egg is None:
                    continue
                major = egg.major_mm / cfg.mm_per_px
                minor = egg.minor_mm / cfg.mm_per_px
                cx = self.lane_center_x(egg.lane) + egg.dx_mm / cfg.mm_per_px
                cy = y_col + egg.dy_mm / cfg.mm_per_px
                if cy < -minor / 2 or cy > cfg.height + minor / 2:
                    continue
                full = (cy - minor / 2 >= 0) and (cy + minor / 2 <= cfg.height - 1)
                out.append(
                    EggTruth(egg.egg_id, c, egg.lane, cx, cy, major, minor, egg.rot0 + egg.rot_speed * t, egg.viable, egg.defect, full)
                )
        return out

    # -- rendu ------------------------------------------------------------
    def frame_time(self, index: int) -> float:
        return index / self.cfg.fps

    def frame(self, index: int) -> tuple[np.ndarray, list[EggTruth]]:
        """Rend l'image n° ``index`` (BGR uint8) et sa vérité terrain."""
        return self.render_at(self.frame_time(index), noise_index=index)

    def render_at(self, t: float, noise_index: int = 0) -> tuple[np.ndarray, list[EggTruth]]:
        cfg = self.cfg
        # Les rouleaux défilent avec les œufs.
        offset = (-(cfg.speed_px_s * t - self._entry_margin)) % cfg.pitch_px
        y0 = min(int(round(offset)), self._bg.shape[0] - cfg.height)
        img = self._bg[y0 : y0 + cfg.height].copy()
        truths = self.truth_at(t)
        for tr in truths:
            egg = self.column(tr.column)[tr.lane]
            assert egg is not None
            patch, alpha = render_egg_patch(egg, tr.rot, cfg.mm_per_px)
            _composite(img, patch, alpha, tr.cx, tr.cy)
        if self._noise:
            img += self._noise[noise_index % len(self._noise)]
        out = np.clip(img, 0, 255).astype(np.uint8)
        if cfg.motion_blur_px > 1:
            out = cv2.blur(out, (1, int(cfg.motion_blur_px)))
        return out, truths

    def render_views(self, egg: EggSpec, n_views: int = 12, size: int = 224, margin: float = 1.15) -> list[np.ndarray]:
        """Rend directement ``n_views`` vues recadrées d'un œuf couvrant une rotation complète.

        Raccourci pour fabriquer vite un jeu d'entraînement sans rendre les images entières.
        """
        cfg = self.cfg
        major = egg.major_mm / cfg.mm_per_px
        minor = egg.minor_mm / cfg.mm_per_px
        side = int(math.ceil(major * margin)) + 8
        rng = np.random.default_rng(egg.seed + 31)
        views: list[np.ndarray] = []
        lane_x = int(self.lane_center_x(egg.lane) - side / 2)
        lane_x = int(np.clip(lane_x, 0, cfg.width - side))
        for k in range(n_views):
            rot = egg.rot0 + 2 * math.pi * k / n_views
            # L'œuf est à y ≡ 0 (mod pas) dans le fond : on le place au centre d'une tuile.
            ybg = int(round(cfg.pitch_px)) - side // 2
            ybg = int(np.clip(ybg, 0, self._bg.shape[0] - side))
            canvas = self._bg[ybg : ybg + side, lane_x : lane_x + side].copy()
            patch, alpha = render_egg_patch(egg, rot, cfg.mm_per_px)
            jx, jy = rng.normal(0, 1.0, 2)
            cx, cy = side / 2 + jx, (round(cfg.pitch_px) - ybg) + jy
            _composite(canvas, patch, alpha, cx, cy)
            if cfg.noise_sigma > 0:
                canvas += rng.normal(0, cfg.noise_sigma, canvas.shape[:2] + (1,)).astype(np.float32)
            canvas8 = np.clip(canvas, 0, 255).astype(np.uint8)
            views.append(extract_crop(canvas8, cx, cy, major, minor, out_size=size, margin=margin))
        return views

    def frames_for_columns(self, n_columns: int) -> int:
        """Nombre d'images nécessaires pour que ``n_columns`` colonnes traversent entièrement l'image."""
        cfg = self.cfg
        t_end = (n_columns - 1) * cfg.column_period_s + (cfg.height + cfg.pitch_px + self._entry_margin) / cfg.speed_px_s
        return int(math.ceil(t_end * cfg.fps)) + 1


def _composite(img: np.ndarray, patch: np.ndarray, alpha: np.ndarray, cx: float, cy: float) -> None:
    """Compose ``patch`` (centré en cx, cy) sur ``img`` float32, en place, avec découpe aux bords."""
    ph, pw = alpha.shape
    x0 = int(round(cx)) - pw // 2
    y0 = int(round(cy)) - ph // 2
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(img.shape[1], x0 + pw), min(img.shape[0], y0 + ph)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    px0, py0 = ix0 - x0, iy0 - y0
    a = alpha[py0 : py0 + (iy1 - iy0), px0 : px0 + (ix1 - ix0), None]
    region = img[iy0:iy1, ix0:ix1]
    region *= 1.0 - a
    region += patch[py0 : py0 + (iy1 - iy0), px0 : px0 + (ix1 - ix0)] * a
