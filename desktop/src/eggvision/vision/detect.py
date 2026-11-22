"""Détection des œufs dans une image complète.

Méthode classique (par défaut) :

1. sous-échantillonnage à la résolution de travail (``work_mm_per_px``) ;
2. indice de segmentation ``luminosité + poids × (rouge − bleu)`` : les œufs
   blancs sont clairs, les œufs bruns (même foncés) sont « chauds », alors que
   les rouleaux sont sombres et neutres ou bleutés ; seuil fixe ou automatique ;
3. morphologie (ouverture contre les reflets, fermeture contre les taches
   sombres des œufs souillés), remplissage des trous via les contours externes ;
4. séparation des œufs qui se touchent : cœurs = pixels éloignés du bord
   (transformée en distance), partition de Voronoï autour de chaque cœur ;
5. ajustement d'ellipse sur l'enveloppe convexe (robuste aux échancrures),
   en ignorant les points coupés par le bord de l'image pour les œufs tronqués ;
6. filtres de taille et de forme, rattachement au couloir (centre dans le
   polygone), drapeau ``partial`` et netteté locale.

Méthode « nn » : petit détecteur entraînable (voir :mod:`.nn`), même post-traitement.

Fonction publique stable : :func:`detect_frame`.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np
from eggvision_shared.preprocess import sharpness

from .params import DetectionParams, detection_params
from .scene import SceneGeometry, default_scene, rescale_scene

# Cache des géométries de scène (la conversion des polygones coûte peu, mais detect_frame
# est appelé à chaque image de l'aperçu en direct).
_GEO_CACHE: dict[str, SceneGeometry] = {}
_GEO_LOCK = threading.Lock()


def _geometry(scene: dict[str, Any] | None, width: int, height: int) -> SceneGeometry:
    if not scene:
        scene = default_scene(width, height)
    elif (scene.get("image_width"), scene.get("image_height")) != (width, height):
        scene = rescale_scene(scene, width, height)
    key = repr((scene.get("image_width"), scene.get("image_height"), scene.get("direction"), scene.get("mm_per_px"),
                [(lane.get("index"), lane.get("polygon")) for lane in scene.get("lanes") or []]))
    with _GEO_LOCK:
        geo = _GEO_CACHE.get(key)
        if geo is None:
            geo = SceneGeometry.from_scene(scene)
            if len(_GEO_CACHE) > 32:
                _GEO_CACHE.clear()
            _GEO_CACHE[key] = geo
    return geo


def _as_params(params: dict[str, Any] | DetectionParams | None) -> DetectionParams:
    if isinstance(params, DetectionParams):
        return params
    return detection_params(params)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def segmentation_index(small_bgr: np.ndarray, warm_weight: float) -> np.ndarray:
    """Indice ``gris + poids × (R − B)`` borné à [0, 255] (uint8)."""
    f = small_bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    idx = gray + float(warm_weight) * (r - b)
    return np.clip(idx, 0, 255).astype(np.uint8)


def _odd(n: float) -> int:
    k = max(1, int(round(n)))
    return k if k % 2 == 1 else k + 1


def segment(image_bgr: np.ndarray, p: DetectionParams, mm_per_px: float) -> tuple[np.ndarray, float, float]:
    """Masque binaire des œufs à la résolution de travail. Retourne (masque, échelle, seuil appliqué)."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, mm_per_px / p.work_mm_per_px)
    if scale < 0.999:
        small = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    else:
        small, scale = image_bgr, 1.0
    work_px_mm = 1.0 / (mm_per_px / scale)  # pixels de travail par mm
    idx = segmentation_index(small, p.warm_weight)
    if p.blur_mm > 0:
        k = _odd(p.blur_mm * work_px_mm)
        if k > 1:
            idx = cv2.GaussianBlur(idx, (k, k), 0)
    if p.threshold_mode == "auto":
        thr, _ = cv2.threshold(idx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(float(thr), p.auto_min_threshold)
    else:
        thr = float(p.threshold)
    mask = (idx > thr).astype(np.uint8) * 255
    if p.open_mm > 0:
        k = _odd(p.open_mm * work_px_mm)
        if k > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if p.close_mm > 0:
        k = _odd(p.close_mm * work_px_mm)
        if k > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Remplissage des trous (taches sombres intérieures) : on redessine les contours externes pleins.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled, scale, thr


def _split_component(comp: np.ndarray, core_frac: float, min_core_px: float) -> list[np.ndarray]:
    """Sépare une composante en plusieurs œufs si sa transformée en distance a plusieurs cœurs."""
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax <= 0:
        return [comp]
    cores = (dist > max(core_frac * dmax, min_core_px)).astype(np.uint8)
    n, labels = cv2.connectedComponents(cores, connectivity=8)
    if n <= 2:  # fond + un seul cœur
        return [comp]
    # Partition de Voronoï : chaque pixel est rattaché au cœur le plus proche.
    inv = (cores == 0).astype(np.uint8)
    _, vor = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    # ``vor`` numérote les composantes de pixels nuls de ``inv`` (= les cœurs) : correspondance via les cœurs.
    pieces = []
    for k in range(1, n):
        ys, xs = np.nonzero(labels == k)
        if len(xs) == 0:
            continue
        vlabel = vor[ys[0], xs[0]]
        piece = ((vor == vlabel) & (comp > 0)).astype(np.uint8)
        if piece.any():
            pieces.append(piece)
    return pieces or [comp]


def _fit(points: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Ellipse (cx, cy, grand axe, petit axe, angle du grand axe en degrés) ajustée sur des points."""
    if len(points) < 5:
        return None
    try:
        (cx, cy), (d1, d2), ang = cv2.fitEllipseDirect(points.astype(np.float32))
    except cv2.error:
        return None
    if not all(map(math.isfinite, (cx, cy, d1, d2, ang))) or d1 <= 0 or d2 <= 0:
        return None
    # OpenCV : d1 est l'axe orienté selon ``ang`` ; on normalise pour que ``angle`` soit celui du grand axe.
    if d1 >= d2:
        major, minor, angle = d1, d2, ang
    else:
        major, minor, angle = d2, d1, ang + 90.0
    angle = ((angle + 90.0) % 180.0) - 90.0  # dans [-90, 90)
    return float(cx), float(cy), float(major), float(minor), float(angle)


def ellipse_half_extents(major: float, minor: float, angle: float) -> tuple[float, float]:
    """Demi-étendues horizontale et verticale d'une ellipse (axes complets, angle du grand axe en degrés)."""
    a, b = major / 2, minor / 2
    t = math.radians(angle)
    hx = math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
    hy = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
    return hx, hy


def border_overflow(cx: float, cy: float, major: float, minor: float, angle: float, w: float, h: float) -> dict[str, float]:
    """Dépassement (px, positif = hors image) de l'ellipse au-delà de chaque bord de l'image."""
    hx, hy = ellipse_half_extents(major, minor, angle)
    return {"left": hx - cx, "top": hy - cy, "right": cx + hx - (w - 1), "bottom": cy + hy - (h - 1)}


# Taille typique d'un œuf (mm) : sert à estimer le centre d'un œuf très tronqué quand l'image
# ne contient aucun œuf complet pour servir de référence.
TYPICAL_EGG_MM = (58.0, 43.0)
SIDES = ("left", "top", "right", "bottom")


def classic_candidates(image_bgr: np.ndarray, p: DetectionParams, geo: SceneGeometry, want_mask: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Candidats (ellipses en coordonnées de l'image complète) issus de la segmentation."""
    mm_per_px = geo.mm_per_px
    mask, scale, thr = segment(image_bgr, p, mm_per_px)
    work_mm = mm_per_px / scale  # mm par pixel de travail
    mh, mw = mask.shape
    margin_work = max(1.0, p.partial_margin_mm / work_mm)
    min_minor_work = p.min_minor_mm / work_mm
    exp_area_min = math.pi / 4 * (p.min_major_mm / work_mm) * min_minor_work
    exp_area_max = math.pi / 4 * (p.max_major_mm / work_mm) * (p.max_minor_mm / work_mm)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    complete: list[dict[str, Any]] = []
    truncated: list[dict[str, Any]] = []
    for k in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[k])
        if area < p.partial_min_visible * exp_area_min * 0.5:
            continue
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1, y1 = min(mw, x + bw + 1), min(mh, y + bh + 1)
        comp = (labels[y0:y1, x0:x1] == k).astype(np.uint8)
        pieces = [comp]
        if p.split_touching:
            hull = cv2.convexHull(cv2.findNonZero(comp))
            solidity = area / max(1.0, cv2.contourArea(hull))
            if solidity < 0.96 or area > exp_area_max:
                pieces = _split_component(comp, p.split_core, 0.2 * min_minor_work)
        for piece in pieces:
            item = _measure_piece(piece, x0, y0, mw, mh, margin_work)
            if item is None or item["area"] < p.partial_min_visible * exp_area_min:
                continue
            (truncated if item["touch"] else complete).append(item)

    # Taille et orientation de référence pour les œufs tronqués : médiane des œufs complets de l'image.
    ok_complete = []
    for item in complete:
        det = _accept(item, p, work_mm, exp_area_min)
        if det is not None:
            ok_complete.append(det)
    if len(ok_complete) >= 2:
        typ_major = float(np.median([d["major"] for d in ok_complete]))
        typ_minor = float(np.median([d["minor"] for d in ok_complete]))
        typ_angle = float(np.median([d["angle"] for d in ok_complete]))
    else:
        typ_major, typ_minor = TYPICAL_EGG_MM[0] / work_mm, TYPICAL_EGG_MM[1] / work_mm
        # Sur une table à rouleaux, le grand axe est parallèle aux rouleaux, donc perpendiculaire au défilement.
        typ_angle = 0.0 if geo.direction in ("up", "down") else 90.0
    out = list(ok_complete)
    for item in truncated:
        det = _accept_truncated(item, p, work_mm, exp_area_min, (typ_major, typ_minor, typ_angle))
        if det is not None:
            out.append(det)
    for det in out:
        # Retour aux coordonnées de l'image complète (centres de pixels alignés).
        det["cx"] = (det["cx"] + 0.5) / scale - 0.5
        det["cy"] = (det["cy"] + 0.5) / scale - 0.5
        det["major"] /= scale
        det["minor"] /= scale
    info: dict[str, Any] = {"threshold": thr, "scale": scale}
    if want_mask:
        info["mask"] = mask
    return out, info


def _measure_piece(piece: np.ndarray, ox: int, oy: int, mw: int, mh: int, margin: float) -> dict[str, Any] | None:
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    pts = cnt.reshape(-1, 2).astype(np.float32) + np.array([ox, oy], dtype=np.float32)
    sides = {
        "left": pts[:, 0] <= margin,
        "top": pts[:, 1] <= margin,
        "right": pts[:, 0] >= mw - 1 - margin,
        "bottom": pts[:, 1] >= mh - 1 - margin,
    }
    touch = [side for side, sel in sides.items() if sel.any()]
    on_border = sides["left"] | sides["top"] | sides["right"] | sides["bottom"]
    hull = cv2.convexHull(pts)
    return {"pts": pts, "area": area, "hull": hull.reshape(-1, 2), "hull_area": float(cv2.contourArea(hull)),
            "touch": touch, "on_border": on_border}


def _accept(item: dict[str, Any], p: DetectionParams, work_mm: float, exp_area_min: float) -> dict[str, Any] | None:
    """Œuf entièrement visible : ajustement sur l'enveloppe convexe puis filtres de taille et de forme."""
    fit = _fit(item["hull"])
    if fit is None:
        return None
    cx, cy, major, minor, angle = fit
    # Les points du contour sont les centres des pixels de bord : l'ellipse réelle passe ~½ pixel plus loin.
    major, minor = major + 1.0, minor + 1.0
    major_mm, minor_mm = major * work_mm, minor * work_mm
    area = item["area"]
    solidity = area / max(1.0, item["hull_area"])
    fill = area / max(1.0, math.pi / 4 * major * minor)
    if not (p.min_major_mm <= major_mm <= p.max_major_mm) or not (p.min_minor_mm <= minor_mm <= p.max_minor_mm):
        return None
    if major / max(minor, 1e-6) > p.max_axis_ratio or solidity < p.min_solidity or fill < p.min_fill:
        return None
    score = float(min(1.0, solidity) * min(1.0, fill))
    return {"cx": cx, "cy": cy, "major": major, "minor": minor, "angle": angle, "touch": [], "score": round(score, 4)}


def _accept_truncated(item: dict[str, Any], p: DetectionParams, work_mm: float, exp_area_min: float,
                      typical: tuple[float, float, float]) -> dict[str, Any] | None:
    """Œuf tronqué par le bord : ajustement sur l'arc observé, ou à défaut estimation à partir de la taille typique."""
    typ_major, typ_minor, typ_angle = typical
    pts, on_border, touch = item["pts"], item["on_border"], item["touch"]
    arc = pts[~on_border]
    fit = None
    if len(arc) >= 8:
        fit = _fit(cv2.convexHull(arc).reshape(-1, 2))
        if fit is not None:
            _, _, fmaj, fmin, _ = fit
            if abs(fmaj / typ_major - 1.0) > 0.15 or abs(fmin / typ_minor - 1.0) > 0.15:
                fit = None  # arc trop court pour un ajustement fiable
    if fit is not None:
        cx, cy, major, minor, angle = fit
        major, minor = major + 1.0, minor + 1.0
    else:
        major, minor, angle = typ_major, typ_minor, typ_angle
        hx, hy = ellipse_half_extents(major, minor, angle)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        left, right = "left" in touch, "right" in touch
        top, bottom = "top" in touch, "bottom" in touch
        if left and not right:
            cx = float(xmax + 0.5 - hx)
        elif right and not left:
            cx = float(xmin - 0.5 + hx)
        else:
            cx = float((xmin + xmax) / 2)
        if top and not bottom:
            cy = float(ymax + 0.5 - hy)
        elif bottom and not top:
            cy = float(ymin - 0.5 + hy)
        else:
            cy = float((ymin + ymax) / 2)
    major_mm, minor_mm = major * work_mm, minor * work_mm
    if major_mm > 1.3 * p.max_major_mm or minor_mm > 1.3 * p.max_minor_mm:
        return None
    ellipse_area = math.pi / 4 * major * minor
    visible = min(1.0, item["area"] / max(1.0, ellipse_area))
    if item["area"] < p.partial_min_visible * min(ellipse_area, exp_area_min * 2):
        return None
    solidity = item["area"] / max(1.0, item["hull_area"])
    if solidity < p.min_solidity:
        return None
    return {"cx": cx, "cy": cy, "major": major, "minor": minor, "angle": angle, "touch": touch,
            "score": round(float(min(1.0, solidity) * visible), 4)}


# ---------------------------------------------------------------------------
# Post-traitement commun
# ---------------------------------------------------------------------------
def local_sharpness(gray: np.ndarray, cx: float, cy: float, major: float, minor: float, region: float) -> float:
    """Netteté (variance du Laplacien, :func:`preprocess.sharpness`) au centre de l'œuf."""
    h, w = gray.shape[:2]
    r = max(4.0, min(major, minor) * region / 2.0)
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    return sharpness(gray[y0:y1, x0:x1])


def finalize(image_bgr: np.ndarray, candidates: list[dict[str, Any]], geo: SceneGeometry, p: DetectionParams) -> list[dict[str, Any]]:
    """Rattachement au couloir, netteté, arrondis ; tri le long du défilement."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    margin_px = p.partial_margin_mm / geo.mm_per_px
    lateral_tol_px = p.lateral_partial_mm / geo.mm_per_px
    # Bords d'entrée / de sortie (perpendiculaires au défilement) et bords latéraux.
    flow_sides = ("top", "bottom") if geo.direction in ("up", "down") else ("left", "right")
    out = []
    for c in candidates:
        cx, cy, major, minor, angle = c["cx"], c["cy"], c["major"], c["minor"], c["angle"]
        over = border_overflow(cx, cy, major, minor, angle, w, h)
        touch = set(c.get("touch") or ())
        partial = bool(c.get("partial", False))
        for side in SIDES:
            if side in flow_sides:
                # Entrée / sortie : la moindre troncature compte (l'œuf entre ou sort du champ).
                partial |= side in touch or over[side] > -margin_px
            else:
                # Bords latéraux : un œuf de couloir extérieur qui frôle le bord reste exploitable.
                partial |= over[side] > lateral_tol_px
        out.append(
            {
                "cx": round(float(cx), 2),
                "cy": round(float(cy), 2),
                "major": round(float(major), 2),
                "minor": round(float(minor), 2),
                "angle": round(float(angle), 2),
                "lane": geo.lane_of(cx, cy),
                "partial": bool(partial),
                "sharpness": round(local_sharpness(gray, cx, cy, major, minor, p.sharpness_region), 3),
                "score": round(float(c.get("score", 1.0)), 4),
            }
        )
    out.sort(key=lambda d: (geo.along(d["cx"], d["cy"]), d["lane"]))
    return out


def detect_frame(
    image_bgr: np.ndarray,
    scene: dict[str, Any] | None,
    params: dict[str, Any] | DetectionParams | None = None,
    *,
    return_info: bool = False,
    want_mask: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Détecte les œufs d'une image BGR uint8.

    ``scene`` : calibration (``config.yaml`` → ``scene``) ; ``None`` = couloirs réguliers par défaut.
    ``params`` : paramètres de détection (dictionnaire partiel accepté, complété par les valeurs par défaut).

    Retourne une liste de dictionnaires ``{cx, cy, major, minor, angle, lane, partial, sharpness, score}`` :
    centre et longueurs complètes des axes en pixels de l'image, angle du grand axe en degrés,
    couloir (-1 si hors couloir), ``partial`` = tronqué par le bord de l'image.
    Avec ``return_info=True``, retourne ``(détections, infos)`` (seuil appliqué, masque si ``want_mask``).
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("detect_frame attend une image BGR à 3 canaux")
    p = _as_params(params)
    h, w = image_bgr.shape[:2]
    geo = _geometry(scene, w, h)
    if p.method == "nn":
        from .nn import nn_candidates  # import paresseux : PyTorch n'est chargé qu'à la demande

        candidates, info = nn_candidates(image_bgr, p, geo)
    else:
        candidates, info = classic_candidates(image_bgr, p, geo, want_mask=want_mask)
    dets = finalize(image_bgr, candidates, geo, p)
    if return_info:
        return dets, info
    return dets
