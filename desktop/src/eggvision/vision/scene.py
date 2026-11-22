"""Calibration de la scène : couloirs, sens de défilement, échelle.

Schéma (``config.yaml`` → ``scene``) ::

    scene:
      image_width: 1456
      image_height: 1088
      direction: down            # up | down | left | right
      mm_per_px: 0.27
      lanes:
        - {index: 0, polygon: [[x, y], [x, y], [x, y], [x, y]]}
      reference: {lot_id: "...", frame_index: 12}   # facultatif : image de référence

Coordonnées utilisées par le suivi :
  * ``s`` : position le long du sens de défilement (croît dans le sens du mouvement) ;
  * ``l`` : position latérale (perpendiculaire au défilement).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from eggvision_shared.constants import DEFAULT_N_LANES, SENSOR_WIDTH
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DIRECTIONS: dict[str, tuple[float, float]] = {"down": (0.0, 1.0), "up": (0.0, -1.0), "right": (1.0, 0.0), "left": (-1.0, 0.0)}
DIRECTION_LABELS = {"down": "vers le bas", "up": "vers le haut", "right": "vers la droite", "left": "vers la gauche"}
# Échelle du capteur de référence (IMX296, champ de vue du prototype) : 0,27 mm/px à 1456 px de large.
REFERENCE_MM_PER_PX = 0.27


class SceneError(ValueError):
    """Calibration invalide (message en français)."""


class LaneModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int = Field(ge=0, le=63)
    polygon: list[tuple[float, float]]

    @field_validator("polygon")
    @classmethod
    def _poly(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("un couloir doit avoir au moins 3 sommets")
        pts = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            raise ValueError("coordonnées non finies")
        if abs(cv2.contourArea(pts.astype(np.float32))) < 1.0:
            raise ValueError("polygone de surface nulle")
        return [(round(float(x), 2), round(float(y), 2)) for x, y in v]


class SceneModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image_width: int = Field(gt=0, le=20000)
    image_height: int = Field(gt=0, le=20000)
    direction: Literal["up", "down", "left", "right"] = "down"
    mm_per_px: float = Field(gt=0.0, le=20.0)
    lanes: list[LaneModel] = Field(default_factory=list)
    reference: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> SceneModel:
        idx = [lane.index for lane in self.lanes]
        if len(set(idx)) != len(idx):
            raise ValueError("indices de couloir en double")
        return self


def validate_scene(data: dict[str, Any]) -> dict[str, Any]:
    """Valide une calibration ; retourne sa forme normalisée (couloirs triés par indice). Lève :class:`SceneError`."""
    try:
        model = SceneModel.model_validate(data)
    except ValidationError as exc:
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msgs.append(f"{loc} : {err.get('msg')}" if loc else str(err.get("msg")))
        raise SceneError("Calibration invalide — " + " ; ".join(msgs)) from exc
    out = model.model_dump(mode="json")
    out["lanes"] = sorted(out["lanes"], key=lambda lane: lane["index"])
    for lane in out["lanes"]:
        lane["polygon"] = [list(p) for p in lane["polygon"]]
    if out.get("reference") is None:
        out.pop("reference", None)
    return out


def default_scene(width: int, height: int, n_lanes: int = DEFAULT_N_LANES, direction: str = "down") -> dict[str, Any]:
    """Calibration par défaut : couloirs réguliers, échelle du capteur de référence ramenée à la largeur."""
    return {
        "image_width": int(width),
        "image_height": int(height),
        "direction": direction,
        "mm_per_px": round(REFERENCE_MM_PER_PX * SENSOR_WIDTH / max(1, width), 5),
        "lanes": regular_lanes(width, height, n_lanes, direction),
    }


def regular_lanes(width: int, height: int, n_lanes: int = DEFAULT_N_LANES, direction: str = "down") -> list[dict[str, Any]]:
    """Découpage régulier de l'image en ``n_lanes`` bandes parallèles au défilement."""
    if direction not in DIRECTIONS:
        raise SceneError(f"Sens de défilement inconnu : {direction}")
    n_lanes = max(1, int(n_lanes))
    lanes = []
    vertical = direction in ("up", "down")
    span = width if vertical else height
    edges = np.linspace(0, span, n_lanes + 1)
    for i in range(n_lanes):
        a, b = round(float(edges[i]), 1), round(float(edges[i + 1]), 1)
        if vertical:
            poly = [[a, 0.0], [b, 0.0], [b, float(height)], [a, float(height)]]
        else:
            poly = [[0.0, a], [float(width), a], [float(width), b], [0.0, b]]
        lanes.append({"index": i, "polygon": poly})
    return _order_lanes(lanes, direction)


def _order_lanes(lanes: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    """Numérote les couloirs de gauche à droite par rapport au sens de défilement (vue de dessus)."""
    geo = SceneGeometry.from_scene({"image_width": 1, "image_height": 1, "direction": direction, "mm_per_px": 1.0, "lanes": []})
    def key(lane: dict[str, Any]) -> float:
        c = np.mean(np.asarray(lane["polygon"], dtype=np.float64), axis=0)
        return -geo.lateral(float(c[0]), float(c[1]))
    ordered = sorted(lanes, key=key)
    return [{"index": i, "polygon": lane["polygon"]} for i, lane in enumerate(ordered)]


def lanes_from_positions(
    points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    direction: str,
    n_lanes: int = DEFAULT_N_LANES,
    min_gap_px: float | None = None,
) -> list[dict[str, Any]]:
    """Propose des couloirs à partir de centres d'œufs détectés (une ou plusieurs images).

    Les positions latérales sont regroupées (k-moyennes 1D initialisées sur un découpage
    régulier) ; les limites des couloirs passent à mi-chemin entre groupes voisins.
    Retombe sur un découpage régulier s'il n'y a pas assez de points.
    """
    if direction not in DIRECTIONS:
        raise SceneError(f"Sens de défilement inconnu : {direction}")
    vertical = direction in ("up", "down")
    span = float(width if vertical else height)
    coords = np.asarray([p[0] if vertical else p[1] for p in points], dtype=np.float64)
    coords = coords[(coords >= 0) & (coords <= span)]
    if len(coords) < n_lanes:
        return regular_lanes(width, height, n_lanes, direction)
    centers = (np.arange(n_lanes) + 0.5) * span / n_lanes
    for _ in range(50):
        assign = np.argmin(np.abs(coords[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for k in range(n_lanes):
            sel = coords[assign == k]
            if len(sel):
                new[k] = float(np.median(sel))
        if np.allclose(new, centers, atol=0.05):
            break
        centers = new
    centers = np.sort(centers)
    gaps = np.diff(centers)
    min_gap = min_gap_px if min_gap_px is not None else 0.3 * span / n_lanes
    if len(gaps) and gaps.min() < min_gap:  # groupes dégénérés (couloirs vides) : découpage régulier
        return regular_lanes(width, height, n_lanes, direction)
    bounds = [0.0] + [float((centers[i] + centers[i + 1]) / 2) for i in range(n_lanes - 1)] + [span]
    lanes = []
    for i in range(n_lanes):
        a, b = round(bounds[i], 1), round(bounds[i + 1], 1)
        if vertical:
            poly = [[a, 0.0], [b, 0.0], [b, float(height)], [a, float(height)]]
        else:
            poly = [[0.0, a], [float(width), a], [float(width), b], [0.0, b]]
        lanes.append({"index": i, "polygon": poly})
    return _order_lanes(lanes, direction)


def mm_per_px_from_ruler(p1: Sequence[float], p2: Sequence[float], length_mm: float) -> float:
    """Échelle (mm par pixel) à partir de deux points cliqués sur une règle et de la distance réelle."""
    if length_mm <= 0:
        raise SceneError("La longueur réelle doit être positive.")
    dist = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    if dist < 2.0:
        raise SceneError("Les deux points de la règle sont trop proches.")
    return float(length_mm) / dist


def rescale_scene(scene: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Adapte une calibration à une image d'une autre résolution (même champ de vue)."""
    sw, sh = scene.get("image_width") or width, scene.get("image_height") or height
    if (sw, sh) == (width, height):
        return scene
    fx, fy = width / sw, height / sh
    out = dict(scene)
    out["image_width"], out["image_height"] = int(width), int(height)
    out["mm_per_px"] = float(scene["mm_per_px"]) / ((fx + fy) / 2)
    out["lanes"] = [
        {"index": lane["index"], "polygon": [[p[0] * fx, p[1] * fy] for p in lane["polygon"]]} for lane in scene.get("lanes", [])
    ]
    return out


@dataclass
class SceneGeometry:
    """Géométrie précalculée d'une calibration (rattachement aux couloirs, axes de défilement)."""

    width: int
    height: int
    direction: str
    mm_per_px: float
    lanes: list[tuple[int, np.ndarray]]
    ux: float
    uy: float

    @classmethod
    def from_scene(cls, scene: dict[str, Any]) -> SceneGeometry:
        direction = scene.get("direction", "down")
        if direction not in DIRECTIONS:
            raise SceneError(f"Sens de défilement inconnu : {direction}")
        ux, uy = DIRECTIONS[direction]
        lanes = [(int(lane["index"]), np.asarray(lane["polygon"], dtype=np.float32).reshape(-1, 1, 2)) for lane in scene.get("lanes") or []]
        return cls(int(scene["image_width"]), int(scene["image_height"]), direction, float(scene["mm_per_px"]), lanes, ux, uy)

    # -- axes ----------------------------------------------------------------
    def along(self, x: float, y: float) -> float:
        """Position le long du défilement (croissante dans le sens du mouvement)."""
        return x * self.ux + y * self.uy

    def lateral(self, x: float, y: float) -> float:
        return -x * self.uy + y * self.ux

    @property
    def s_range(self) -> tuple[float, float]:
        """(bord d'entrée, bord de sortie) le long du défilement."""
        corners = [self.along(x, y) for x in (0.0, float(self.width)) for y in (0.0, float(self.height))]
        return min(corners), max(corners)

    def half_extent_along(self, major: float, minor: float, angle_deg: float) -> float:
        """Demi-étendue d'une ellipse (axes complets ``major``/``minor``, angle du grand axe) le long du défilement."""
        a, b = major / 2.0, minor / 2.0
        t = math.radians(angle_deg)
        e1 = (math.cos(t), math.sin(t))
        e2 = (-math.sin(t), math.cos(t))
        p1 = e1[0] * self.ux + e1[1] * self.uy
        p2 = e2[0] * self.ux + e2[1] * self.uy
        return math.sqrt((a * p1) ** 2 + (b * p2) ** 2)

    # -- couloirs ----------------------------------------------------------------
    def lane_of(self, x: float, y: float) -> int:
        """Couloir dont le polygone contient le point (ramené dans l'image) ; -1 sinon."""
        if not self.lanes:
            return -1
        px = min(max(float(x), 0.0), self.width - 1.0)
        py = min(max(float(y), 0.0), self.height - 1.0)
        for index, poly in self.lanes:
            if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                return index
        return -1

    def lane_mask(self, scale: float = 1.0) -> np.ndarray:
        """Image des indices de couloir (+1 ; 0 = hors couloir) à l'échelle donnée — pour l'affichage."""
        h, w = max(1, round(self.height * scale)), max(1, round(self.width * scale))
        mask = np.zeros((h, w), dtype=np.uint8)
        for index, poly in self.lanes:
            cv2.fillPoly(mask, [np.round(poly.reshape(-1, 2) * scale).astype(np.int32)], int(index) + 1)
        return mask
