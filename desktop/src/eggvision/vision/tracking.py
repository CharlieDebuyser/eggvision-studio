"""Suivi des œufs d'image en image, sans codeur.

Principe :

* chaque détection est projetée sur l'axe de défilement (``s``) et l'axe latéral (``l``) ;
* la table avance d'un bloc : une **vitesse unique** (px/s le long de ``s``) est
  estimée en continu à partir des déplacements observés de toutes les pistes
  (médiane par image, puis médiane glissante des dernières images) ;
* la position de chaque piste est prédite à l'instant de la nouvelle image à
  partir de l'**horodatage capteur** (des images peuvent manquer) ;
* association gloutonne par coût croissant à l'intérieur d'une fenêtre
  (le long du défilement et latéralement) : un œuf ne change pas de couloir ;
* une piste non retrouvée pendant ``max_missed`` images est close ; elle l'est
  immédiatement si sa position prédite a quitté l'image par le bord de sortie ;
* une détection non associée fait naître une piste.

Le traqueur ne connaît ni la base de données ni les fichiers : il reçoit des
listes de détections et rend des :class:`TrackState` terminées.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .params import TrackingParams
from .scene import SceneGeometry

SPEED_BASELINE = 8


@dataclass
class Observation:
    frame_index: int
    ts: float  # secondes (horloge capteur, origine = début du lot)
    det: dict[str, Any]
    s: float
    l: float
    payload: Any = None  # données associées (ex. vue recadrée encodée)


@dataclass
class TrackState:
    track_no: int
    obs: list[Observation] = field(default_factory=list)
    missed: int = 0  # images consécutives sans détection
    total_missed: int = 0  # images sans détection pendant la vie de la piste

    @property
    def last(self) -> Observation:
        return self.obs[-1]

    def lane(self) -> tuple[int, float]:
        """Couloir majoritaire et part des détections qui l'appuient."""
        votes: dict[int, int] = {}
        for o in self.obs:
            votes[o.det.get("lane", -1)] = votes.get(o.det.get("lane", -1), 0) + 1
        lane, n = max(votes.items(), key=lambda kv: (kv[1], kv[0] >= 0))
        return lane, n / max(1, len(self.obs))


class Tracker:
    """Traqueur multi-couloirs à vitesse commune estimée en continu."""

    def __init__(self, geo: SceneGeometry, params: TrackingParams) -> None:
        self.geo = geo
        self.p = params
        self.mm = geo.mm_per_px
        self.active: list[TrackState] = []
        self.finished: list[TrackState] = []
        self._next_no = 0
        self._speed_hist: deque[float] = deque(maxlen=max(1, params.speed_history))
        self.speed: float | None = (params.initial_speed_mm_s / self.mm) if params.initial_speed_mm_s else None
        self.speed_trace: list[tuple[float, float]] = []  # (ts, vitesse px/s) pour diagnostic
        self.frames_seen = 0
        self._last_ts: float | None = None
        self._dt_hist: deque[float] = deque(maxlen=50)  # intervalles entre images (cadence réelle)

    # -- outils ------------------------------------------------------------
    def _gate_s(self, dt: float) -> float:
        base = self.p.gate_mm / self.mm
        if self.speed is None:
            return base + self.p.max_speed_mm_s / self.mm * dt
        # Fenêtre élargie proportionnellement au déplacement prédit au-delà d'une image (images manquantes).
        frame_dt = statistics.median(self._dt_hist) if self._dt_hist else dt
        return base + self.p.gate_speed_frac * abs(self.speed) * max(0.0, dt - frame_dt)

    def _predict(self, t: TrackState, ts: float) -> float:
        dt = ts - t.last.ts
        return t.last.s + (self.speed or 0.0) * dt

    def _exited(self, t: TrackState, ts: float) -> bool:
        """La position prédite a entièrement quitté l'image par le bord de sortie."""
        if self.speed is None:
            return False
        d = t.last.det
        half = self.geo.half_extent_along(d["major"], d["minor"], d["angle"])
        s_pred = self._predict(t, ts)
        return s_pred - half > self.geo.s_range[1]

    # -- mise à jour -----------------------------------------------------------
    def update(self, frame_index: int, ts: float, detections: Iterable[dict[str, Any]], payloads: list[Any] | None = None) -> list[TrackState]:
        """Intègre les détections d'une image ; retourne les pistes closes à cette occasion.

        Les images doivent arriver dans l'ordre chronologique (``ts`` croissant).
        """
        dets = list(detections)
        if self._last_ts is not None and ts <= self._last_ts:
            raise ValueError("Horodatages non croissants : impossible de suivre les œufs.")
        if self._last_ts is not None:
            self._dt_hist.append(ts - self._last_ts)
        self._last_ts = ts
        self.frames_seen += 1
        obs = [
            Observation(frame_index, ts, d, self.geo.along(d["cx"], d["cy"]), self.geo.lateral(d["cx"], d["cy"]),
                        payloads[i] if payloads is not None else None)
            for i, d in enumerate(dets)
            if d.get("lane", -1) >= 0
        ]
        gate_l = self.p.lateral_gate_mm / self.mm
        back_tol = self.p.gate_mm / self.mm

        # Coûts de toutes les paires plausibles (piste, détection).
        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(self.active):
            dt = ts - t.last.ts
            pred = self._predict(t, ts)
            gate_s = self._gate_s(dt)
            for oi, o in enumerate(obs):
                dl = abs(o.l - t.last.l)
                if dl > gate_l:
                    continue
                if self.speed is None:
                    # Vitesse inconnue : on n'accepte qu'un déplacement vers l'aval, borné par la vitesse maximale.
                    ds = o.s - t.last.s
                    if ds < -back_tol or ds > gate_s:
                        continue
                    cost = abs(ds) / max(gate_s, 1e-6) + dl / gate_l
                else:
                    ds = o.s - pred
                    if abs(ds) > gate_s:
                        continue
                    cost = (ds / gate_s) ** 2 + (dl / gate_l) ** 2
                pairs.append((cost, ti, oi))
        pairs.sort()
        used_t: set[int] = set()
        used_o: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, ti, oi in pairs:
            if ti in used_t or oi in used_o:
                continue
            used_t.add(ti)
            used_o.add(oi)
            matches.append((ti, oi))

        # Estimation de la vitesse : déplacements des centres fiables (détections complètes).
        # Base de mesure longue (jusqu'à SPEED_BASELINE détections en arrière) : moins sensible au bruit de
        # position et à la quantification des positions qu'un simple déplacement d'une image à la suivante.
        samples = []
        for ti, oi in matches:
            t, o = self.active[ti], obs[oi]
            if o.det.get("partial"):
                continue
            ref = None
            for prev in t.obs[-SPEED_BASELINE:]:
                if not prev.det.get("partial"):
                    ref = prev
                    break
            if ref is not None and ts - ref.ts > 0:
                samples.append((o.s - ref.s) / (ts - ref.ts))
        if samples:
            self._speed_hist.append(statistics.median(samples))
            self.speed = statistics.median(self._speed_hist)
            self.speed_trace.append((ts, self.speed))

        for ti, oi in matches:
            t = self.active[ti]
            t.obs.append(obs[oi])
            t.missed = 0

        closed: list[TrackState] = []
        still: list[TrackState] = []
        for ti, t in enumerate(self.active):
            if ti not in used_t:
                t.missed += 1
                t.total_missed += 1
                if t.missed > self.p.max_missed or self._exited(t, ts):
                    closed.append(t)
                    continue
            still.append(t)
        for oi, o in enumerate(obs):
            if oi not in used_o:
                still.append(TrackState(self._next_no, [o]))
                self._next_no += 1
        self.active = still
        self.finished.extend(closed)
        return closed

    def flush(self) -> list[TrackState]:
        """Clôt toutes les pistes encore actives (fin du lot)."""
        closed = self.active
        self.active = []
        self.finished.extend(closed)
        return closed

    @property
    def speed_mm_s(self) -> float | None:
        return None if self.speed is None else self.speed * self.mm


# ---------------------------------------------------------------------------
# Qualité d'une piste
# ---------------------------------------------------------------------------
FLAG_LABELS = {
    "courte": "Trop peu de vues exploitables",
    "incomplete": "N'a pas traversé toute l'image",
    "partielle": "Trop de détections tronquées",
    "lacunes": "Trop de détections manquées",
    "vitesse_incoherente": "Vitesse différente de celle de la table",
    "taille_incoherente": "Taille instable (fusion ou séparation d'œufs ?)",
    "couloir_ambigu": "Couloir incertain",
    "floue": "Vues floues",
}


def _slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def analyse_track(t: TrackState, geo: SceneGeometry, p: TrackingParams, table_speed: float | None) -> dict[str, Any]:
    """Statistiques et drapeaux de qualité d'une piste (hors flou, évalué à l'échelle du lot).

    Retourne aussi ``candidates`` : rangs (dans ``t.obs``) des détections éligibles comme vues
    (complètes et suffisamment nettes), dans l'ordre chronologique.
    """
    obs = t.obs
    mm = geo.mm_per_px
    complete = [i for i, o in enumerate(obs) if not o.det.get("partial")]
    sharp_vals = [obs[i].det.get("sharpness", 0.0) for i in complete]
    med_sharp = statistics.median(sharp_vals) if sharp_vals else 0.0
    candidates = [i for i in complete if obs[i].det.get("sharpness", 0.0) >= p.view_min_sharpness_ratio * med_sharp]
    lane, agreement = t.lane()
    s_min, s_max = geo.s_range
    margin = p.edge_margin_mm / mm
    first, last = obs[0], obs[-1]
    first_half = geo.half_extent_along(first.det["major"], first.det["minor"], first.det["angle"])
    last_half = geo.half_extent_along(last.det["major"], last.det["minor"], last.det["angle"])
    # Complète = vue dès son entrée par le bord amont et jusqu'à sa sortie par le bord aval.
    entered = (first.s - first_half) <= s_min + margin
    exited = (last.s + last_half) >= s_max - margin

    speed = _slope([obs[i].ts for i in complete], [obs[i].s for i in complete]) if len(complete) >= 3 else None
    minors = [obs[i].det["minor"] for i in complete]
    majors = [obs[i].det["major"] for i in complete]
    size_cv = (statistics.pstdev(minors) / statistics.mean(minors)) if len(minors) >= 3 else 0.0
    span_frames = max(1, len(obs) + t.total_missed)
    gap_ratio = t.total_missed / span_frames
    partial_ratio = 1.0 - len(complete) / max(1, len(obs))

    flags: list[str] = []
    if len(candidates) < p.min_views:
        flags.append("courte")
    if not (entered and exited):
        flags.append("incomplete")
    if partial_ratio > p.max_partial_ratio:
        flags.append("partielle")
    if gap_ratio > p.max_gap_ratio:
        flags.append("lacunes")
    if speed is not None and table_speed and abs(speed - table_speed) > p.speed_tolerance * abs(table_speed):
        flags.append("vitesse_incoherente")
    if size_cv > p.size_cv_max:
        flags.append("taille_incoherente")
    if agreement < p.min_lane_agreement or lane < 0:
        flags.append("couloir_ambigu")
    return {
        "lane": lane,
        "lane_agreement": round(agreement, 3),
        "complete": complete,
        "candidates": candidates,
        "entered": entered,
        "exited": exited,
        "speed_px_s": float(speed) if speed is not None else float(table_speed or 0.0),
        "partial_ratio": round(partial_ratio, 4),
        "gap_ratio": round(gap_ratio, 4),
        "size_cv": round(size_cv, 4),
        "mean_major": float(statistics.mean(majors)) if majors else float(statistics.mean(o.det["major"] for o in obs)),
        "mean_minor": float(statistics.mean(minors)) if minors else float(statistics.mean(o.det["minor"] for o in obs)),
        "median_sharpness": float(med_sharp),
        "flags": flags,
    }


def robust_speed(values: list[float]) -> float | None:
    vals = [v for v in values if v and math.isfinite(v)]
    return statistics.median(vals) if vals else None
