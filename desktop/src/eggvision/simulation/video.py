"""Vidéo MP4 de l'animation 2D (vue de dessus), rendue côté serveur.

Mêmes éléments que l'animation de l'écran : table à rouleaux, zone caméra,
colonnes qui défilent, couleur de chaque œuf selon vérité × décision (gris tant
que la décision n'est pas disponible), retards entourés, trappes qui s'ouvrent,
bacs « viables » et « rebut » avec compteurs, horloge et légende.

Dessin : OpenCV (formes) + Pillow (texte accentué, police Inter embarquée).
Encodage : ffmpeg embarqué (``imageio-ffmpeg``) en H.264 / yuv420p, lisible partout ;
repli sur ``cv2.VideoWriter`` (MPEG-4 Part 2) si ffmpeg est indisponible.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from .engine import ACCEPTED_OK, EJECTED_OK, FALSE_REJECT, MISSED, MISSED_LATE, OUTCOME_LABELS

log = logging.getLogger(__name__)

FONT_PATH = Path(__file__).with_name("assets") / "inter-latin.woff2"

# Palette commune à l'écran et à la vidéo (hexadécimal RGB).
PALETTE = {
    "pending": "#C9C2B2",
    ACCEPTED_OK: "#3FB950",
    EJECTED_OK: "#4C9AFF",
    MISSED: "#F85149",
    MISSED_LATE: "#B370F0",
    FALSE_REJECT: "#E3A21A",
    "late": "#FFE14D",
    "camera": "#4C9AFF",
    "table": "#3A3F47",
    "roller": "#4A505A",
    "trap_closed": "#6E7681",
    "trap_open": "#0D1117",
    "background": "#1E2228",
    "text": "#E6EDF3",
    "muted": "#9AA4AE",
}


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    b, g, r = _bgr(hex_color)
    return (r, g, b)


class VideoOptions(BaseModel):
    """Options de rendu de la vidéo."""

    model_config = ConfigDict(extra="forbid")

    start_column: int = Field(0, ge=0, description="Première colonne")
    end_column: int | None = Field(None, ge=0, description="Dernière colonne (incluse) ; défaut : la dernière")
    speed: float | None = Field(None, gt=0.0, le=1000.0, description="Vitesse de lecture (×) ; défaut : automatique")
    max_duration_s: float = Field(60.0, ge=2.0, le=3600.0, description="Durée visée en vitesse automatique (s)")
    fps: int = Field(25, ge=5, le=60)
    width: int = Field(1280, ge=320, le=3840)
    height: int = Field(720, ge=240, le=2160)


@dataclass
class _Layout:
    w: int
    h: int
    x_min: float  # mm (monde)
    x_end: float  # fin de table (mm) : entrée du bac viables
    scale: float  # px par mm
    left: int  # px de x_min
    table_top: int
    lane_h: float
    egg_rx: int
    egg_ry: int

    def px(self, x_mm: float) -> int:
        return int(round(self.left + (x_mm - self.x_min) * self.scale))

    def lane_y(self, lane: int) -> int:
        return int(round(self.table_top + (lane + 0.5) * self.lane_h))


def _fonts(height: int) -> tuple[Any, Any, Any]:
    base = max(10, int(height / 40))
    try:
        mk = lambda s: ImageFont.truetype(str(FONT_PATH), s)  # noqa: E731
        return mk(int(base * 1.25)), mk(base), mk(int(base * 0.85))
    except Exception:  # police indisponible : police intégrée de Pillow (sans accents)
        log.warning("Police Inter indisponible, repli sur la police par défaut de Pillow")
        return ImageFont.load_default(int(base * 1.25)), ImageFont.load_default(base), ImageFont.load_default(int(base * 0.85))


class Renderer:
    """Dessine une image de l'animation à l'instant virtuel ``t``."""

    def __init__(self, timeline: dict[str, Any], width: int, height: int, c0: int, c1: int) -> None:
        self.g = timeline["geometry"]
        self.cols = timeline["columns"]
        self.c0, self.c1 = c0, c1
        self.w, self.h = width, height
        g = self.g
        self.pitch = float(g["pitch_mm"])
        self.v = float(g["speed_mm_s"])
        self.period = float(g["period_s"])
        self.n_lanes = int(g["n_lanes"])
        self.trap_x = float(g["trap_x_mm"])
        self.cam = float(g["camera_zone_mm"])
        self.font_big, self.font, self.font_small = _fonts(height)

        header_h = int(height * 0.09)
        footer_h = int(height * 0.24)
        bin_w = int(width * 0.11)
        margin = int(width * 0.02)
        x_min = -1.5 * self.pitch
        x_end = self.trap_x + 1.5 * self.pitch
        label_w = int(width * 0.025)  # numéros de couloir
        usable_w = width - 2 * margin - label_w - bin_w - margin
        scale = usable_w / (x_end - x_min)
        avail_h = height - header_h - footer_h
        lane_h = min(avail_h / self.n_lanes, self.pitch * scale * 1.1)
        table_h = lane_h * self.n_lanes
        table_top = header_h + int((avail_h - table_h) / 2)
        egg_len_mm = min(0.33 * self.pitch, 30.0)  # demi-grand axe affiché
        rx = max(2, int(egg_len_mm * scale))
        ry = max(2, min(int(rx * 0.75), int(lane_h * 0.34)))
        self.L = _Layout(width, height, x_min, x_end, scale, margin + label_w, table_top, lane_h, rx, ry)
        self.table_bottom = int(table_top + table_h)
        self.header_h, self.footer_h, self.bin_w, self.margin = header_h, footer_h, bin_w, margin

        # Cumuls par colonne (compteurs des bacs).
        sub = self.cols[c0 : c1 + 1]
        self.t_trap = [c["t_trap"] for c in sub]
        self.t_end = [c["t_enter"] + x_end / self.v for c in sub]  # arrivée en bout de table (bac viables)
        rej = np.array([sum(1 for e in c["lanes"] if e and e["ejected"]) for c in sub])
        rej_bad = np.array([sum(1 for e in c["lanes"] if e and e["outcome"] == FALSE_REJECT) for c in sub])
        ok = np.array([sum(1 for e in c["lanes"] if e and not e["ejected"]) for c in sub])
        ok_bad = np.array([sum(1 for e in c["lanes"] if e and e["outcome"] in (MISSED, MISSED_LATE)) for c in sub])
        self.cum = {k: np.concatenate([[0], np.cumsum(a)]) for k, a in (("rej", rej), ("rej_bad", rej_bad), ("ok", ok), ("ok_bad", ok_bad))}
        self.background = self._static_layer()

    # -- couche statique ----------------------------------------------------
    def _static_layer(self) -> np.ndarray:
        L = self.L
        img = np.full((self.h, self.w, 3), _bgr(PALETTE["background"]), np.uint8)
        x0, x1 = L.px(L.x_min), L.px(L.x_end)
        cv2.rectangle(img, (x0, L.table_top), (x1, self.table_bottom), _bgr(PALETTE["table"]), -1)
        for lane in range(1, self.n_lanes):
            y = int(L.table_top + lane * L.lane_h)
            cv2.line(img, (x0, y), (x1, y), _bgr(PALETTE["roller"]), 1, cv2.LINE_AA)
        # Zone caméra (bande translucide).
        overlay = img.copy()
        cv2.rectangle(overlay, (L.px(0), L.table_top - 6), (L.px(self.cam), self.table_bottom + 6), _bgr(PALETTE["camera"]), -1)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.rectangle(img, (L.px(0), L.table_top - 6), (L.px(self.cam), self.table_bottom + 6), _bgr(PALETTE["camera"]), 2, cv2.LINE_AA)
        # Bac viables (bout de table) et bac rebut (sous les trappes).
        bx0 = x1 + self.margin
        cv2.rectangle(img, (bx0, L.table_top), (bx0 + self.bin_w, self.table_bottom), _bgr("#22303A"), -1)
        cv2.rectangle(img, (bx0, L.table_top), (bx0 + self.bin_w, self.table_bottom), _bgr(PALETTE[ACCEPTED_OK]), 2, cv2.LINE_AA)
        rx0 = L.px(self.trap_x) - int(self.bin_w * 0.8)
        ry0 = self.table_bottom + int(self.h * 0.02)
        self.reject_box = (rx0, ry0, rx0 + int(self.bin_w * 1.6), ry0 + int(self.footer_h * 0.42))
        cv2.rectangle(img, self.reject_box[:2], self.reject_box[2:], _bgr("#3A2226"), -1)
        cv2.rectangle(img, self.reject_box[:2], self.reject_box[2:], _bgr(PALETTE[MISSED]), 2, cv2.LINE_AA)
        self.viable_box = (bx0, L.table_top, bx0 + self.bin_w, self.table_bottom)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.text((L.px(0) + 4, L.table_top - 8), "Caméra", font=self.font_small, fill=_rgb(PALETTE["camera"]), anchor="lb")
        d.text((L.px(self.trap_x), L.table_top - 8), "Trappes", font=self.font_small, fill=_rgb(PALETTE["text"]), anchor="mb")
        d.text((bx0 + self.bin_w / 2, L.table_top - 8), "Bac viables", font=self.font_small, fill=_rgb(PALETTE[ACCEPTED_OK]), anchor="mb")
        d.text((self.reject_box[0] - 8, (self.reject_box[1] + self.reject_box[3]) / 2), "Bac rebut", font=self.font_small, fill=_rgb(PALETTE[MISSED]), anchor="rm")
        for lane in range(self.n_lanes):
            d.text((x0 - 4, L.lane_y(lane)), str(lane + 1), font=self.font_small, fill=_rgb(PALETTE["muted"]), anchor="rm")
        # Légende.
        items = [
            ("pending", "Décision en attente"),
            (ACCEPTED_OK, OUTCOME_LABELS[ACCEPTED_OK]),
            (EJECTED_OK, OUTCOME_LABELS[EJECTED_OK]),
            (MISSED, OUTCOME_LABELS[MISSED]),
            (MISSED_LATE, OUTCOME_LABELS[MISSED_LATE]),
            (FALSE_REJECT, OUTCOME_LABELS[FALSE_REJECT]),
            ("late", "Décision en retard"),
        ]
        ly = self.h - int(self.footer_h * 0.25)
        lx = self.margin
        r = max(4, int(self.h / 90))
        for key, label in items:
            if key == "late":
                d.ellipse((lx, ly - r, lx + 2 * r, ly + r), outline=_rgb(PALETTE["late"]), width=max(2, r // 2))
            else:
                d.ellipse((lx, ly - r, lx + 2 * r, ly + r), fill=_rgb(PALETTE[key]))
            d.text((lx + 2 * r + 6, ly), label, font=self.font_small, fill=_rgb(PALETTE["text"]), anchor="lm")
            lx += 2 * r + 12 + int(d.textlength(label, font=self.font_small)) + 18
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    # -- image à l'instant t -------------------------------------------------
    def frame(self, t: float, speed: float) -> np.ndarray:
        L = self.L
        img = self.background.copy()
        # Rouleaux qui défilent.
        spacing = self.pitch / 2
        offset = (t * self.v) % spacing
        x = L.x_min + offset
        while x < L.x_end:
            px = L.px(x)
            cv2.line(img, (px, L.table_top + 1), (px, self.table_bottom - 1), _bgr(PALETTE["roller"]), 1)
            x += spacing

        # Colonnes visibles.
        t0 = self.cols[0]["t_enter"]
        k_lo = max(self.c0, int(math.floor((t - t0 - L.x_end / self.v) / self.period)) - 1)
        k_hi = min(self.c1, int(math.ceil((t - t0 - L.x_min / self.v) / self.period)) + 1)
        trap_open = [False] * self.n_lanes
        half_p = 0.5 * self.period
        for k in range(k_lo, k_hi + 1):
            c = self.cols[k]
            if c["decided"] and not c["late"] and c["t_deadline"] <= t <= c["t_trap"] + half_p:
                for lane, e in enumerate(c["lanes"]):
                    if e and e["ejected"]:
                        trap_open[lane] = True
        # Trappes.
        tx = L.px(self.trap_x)
        tw = max(4, int(self.pitch * 0.55 * L.scale / 2))
        for lane in range(self.n_lanes):
            y = L.lane_y(lane)
            th = int(L.lane_h * 0.42)
            color = PALETTE["trap_open"] if trap_open[lane] else PALETTE["trap_closed"]
            cv2.rectangle(img, (tx - tw, y - th), (tx + tw, y + th), _bgr(color), -1)
            if trap_open[lane]:
                cv2.rectangle(img, (tx - tw, y - th), (tx + tw, y + th), _bgr(PALETTE["late"]), 1, cv2.LINE_AA)
        # Œufs.
        fall = 0.25 * self.period
        for k in range(k_lo, k_hi + 1):
            c = self.cols[k]
            x_mm = (t - c["t_enter"]) * self.v
            if x_mm < L.x_min:
                continue
            for lane, e in enumerate(c["lanes"]):
                if e is None:
                    continue
                rx, ry = L.egg_rx, L.egg_ry
                xx = x_mm
                if e["ejected"] and t > c["t_trap"]:
                    dt = t - c["t_trap"]
                    if dt > fall:
                        continue
                    f = 1.0 - dt / fall
                    rx, ry = max(1, int(rx * f)), max(1, int(ry * f))
                    xx = self.trap_x
                elif x_mm > L.x_end:
                    continue
                decided = c["decided"] and t >= c["t_decision"] and e["decision"] is not None
                color = PALETTE[e["outcome"]] if decided else PALETTE["pending"]
                if e["decision"] is None and x_mm >= self.cam:
                    color = PALETTE[e["outcome"]]  # aucun avis possible (0 vue) : issue connue dès la sortie caméra
                center = (L.px(xx), L.lane_y(lane))
                cv2.ellipse(img, center, (rx, ry), 0, 0, 360, _bgr(color), -1, cv2.LINE_AA)
                if decided and e["late"]:
                    cv2.ellipse(img, center, (rx + 2, ry + 2), 0, 0, 360, _bgr(PALETTE["late"]), 2, cv2.LINE_AA)

        # Textes dynamiques (en-tête, compteurs).
        n_trap = bisect.bisect_right(self.t_trap, t)
        n_end = bisect.bisect_right(self.t_end, t)
        rej, rej_bad = int(self.cum["rej"][n_trap]), int(self.cum["rej_bad"][n_trap])
        ok, ok_bad = int(self.cum["ok"][n_end]), int(self.cum["ok_bad"][n_end])
        current = min(self.c1, max(self.c0, int(math.floor((t - t0) / self.period))))
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        hy = self.header_h // 2
        d.text((self.margin, hy), f"t = {t:.2f} s", font=self.font_big, fill=_rgb(PALETTE["text"]), anchor="lm")
        d.text((self.margin + int(self.w * 0.2), hy), f"Colonne {current + 1} / {len(self.cols)}", font=self.font, fill=_rgb(PALETTE["text"]), anchor="lm")
        d.text((self.w - self.margin, hy), f"Lecture ×{speed:.2f}", font=self.font, fill=_rgb(PALETTE["muted"]), anchor="rm")
        vb = self.viable_box
        d.text(((vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2 - 12), f"{ok}", font=self.font_big, fill=_rgb(PALETTE[ACCEPTED_OK]), anchor="mm")
        d.text(((vb[0] + vb[2]) / 2, (vb[1] + vb[3]) / 2 + 14), f"dont {ok_bad} défaut(s)", font=self.font_small, fill=_rgb(PALETTE["muted"]), anchor="mm")
        rb = self.reject_box
        d.text(((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2 - 9), f"{rej}", font=self.font_big, fill=_rgb(PALETTE[MISSED]), anchor="mm")
        d.text(((rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2 + 13), f"dont {rej_bad} à tort", font=self.font_small, fill=_rgb(PALETTE["muted"]), anchor="mm")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def plan(timeline: dict[str, Any], opts: VideoOptions) -> dict[str, Any]:
    """Plage de colonnes, instants de début/fin, vitesse effective et nombre d'images."""
    cols = timeline["columns"]
    if not cols:
        raise ValueError("Chronologie vide")
    g = timeline["geometry"]
    c0 = min(opts.start_column, len(cols) - 1)
    c1 = len(cols) - 1 if opts.end_column is None else min(opts.end_column, len(cols) - 1)
    if c1 < c0:
        raise ValueError("Plage de colonnes invalide (fin avant début)")
    v, pitch = float(g["speed_mm_s"]), float(g["pitch_mm"])
    t_start = cols[c0]["t_enter"] - 1.5 * pitch / v
    t_end = cols[c1]["t_enter"] + (float(g["trap_x_mm"]) + 1.5 * pitch) / v + 0.3 * float(g["period_s"])
    span = t_end - t_start
    speed = opts.speed if opts.speed is not None else max(1.0, span / opts.max_duration_s)
    duration = span / speed
    n_frames = max(1, int(math.ceil(duration * opts.fps)))
    return {"c0": c0, "c1": c1, "t_start": t_start, "t_end": t_end, "speed": speed, "duration_s": duration, "n_frames": n_frames}


def _even(x: int) -> int:
    return x - (x % 2)


def render_video(
    timeline: dict[str, Any],
    out_path: Path,
    opts: VideoOptions,
    *,
    check_cancelled: Callable[[], None] = lambda: None,
    progress: Callable[[float, str | None], None] = lambda *_: None,
) -> dict[str, Any]:
    """Rend la vidéo dans ``out_path`` (écriture dans un fichier temporaire puis renommage)."""
    p = plan(timeline, opts)
    w, h = _even(opts.width), _even(opts.height)
    renderer = Renderer(timeline, w, h, p["c0"], p["c1"])
    tmp = out_path.with_name(out_path.stem + ".partial.mp4")
    if tmp.exists():
        tmp.unlink()
    encoder = "libx264"
    writer: Any = None
    gen: Any = None
    try:
        try:
            import imageio_ffmpeg

            gen = imageio_ffmpeg.write_frames(
                str(tmp),
                (w, h),
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                fps=opts.fps,
                codec="libx264",
                quality=None,
                macro_block_size=2,
                output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"],
            )
            gen.send(None)
        except Exception as exc:  # ffmpeg indisponible : repli OpenCV
            log.warning("ffmpeg indisponible (%s) : repli sur cv2.VideoWriter", exc)
            gen = None
            encoder = "opencv-mp4v"
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), opts.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Aucun encodeur vidéo disponible") from exc
        n = p["n_frames"]
        for i in range(n):
            if i % 5 == 0:
                check_cancelled()
                progress(i / n, f"Image {i + 1}/{n}")
            t = p["t_start"] + i / opts.fps * p["speed"]
            frame = renderer.frame(t, p["speed"])
            if gen is not None:
                gen.send(np.ascontiguousarray(frame).tobytes())
            else:
                writer.write(frame)
        if gen is not None:
            gen.close()
            gen = None
        if writer is not None:
            writer.release()
            writer = None
        os.replace(tmp, out_path)
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        if writer is not None:
            writer.release()
        if tmp.exists():
            tmp.unlink()
    progress(1.0, "Vidéo terminée")
    return {
        "columns": [p["c0"], p["c1"]],
        "speed": round(p["speed"], 4),
        "fps": opts.fps,
        "width": w,
        "height": h,
        "n_frames": p["n_frames"],
        "duration_s": round(p["duration_s"], 3),
        "encoder": encoder,
        "size_bytes": out_path.stat().st_size,
    }


def probe(path: Path) -> dict[str, Any]:
    """Nombre d'images et durée d'un fichier vidéo (ffmpeg embarqué, sinon OpenCV)."""
    try:
        import imageio_ffmpeg

        n, secs = imageio_ffmpeg.count_frames_and_secs(str(path))
        return {"n_frames": int(n), "duration_s": float(secs), "tool": "ffmpeg"}
    except Exception:
        cap = cv2.VideoCapture(str(path))
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            return {"n_frames": n, "duration_s": n / fps if fps else 0.0, "tool": "opencv"}
        finally:
            cap.release()
