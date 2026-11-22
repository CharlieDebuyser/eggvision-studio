"""Analyse de l'aperçu : netteté, histogramme, pixels saturés, détections en direct (facultatif)."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from eggvision_shared.preprocess import sharpness

log = logging.getLogger(__name__)

SATURATION_LEVEL = 255
SATURATION_WARNING = 0.005  # au-delà de 0,5 % de pixels saturés : alerte


def analyze_image(image_bgr: np.ndarray, bins: int = 64) -> dict[str, Any]:
    """Indicateurs de réglage d'une image BGR.

    * ``sharpness`` : variance du Laplacien (plus grand = plus net) ;
    * ``histogram`` : ``{b, g, r, luma}``, ``bins`` classes, fractions de pixels ;
    * ``saturated`` : fraction de pixels saturés (≥ 255) par canal et toutes couches
      confondues (``any``) ; ``saturation_warning`` si ``any`` dépasse 0,5 % ;
    * ``mean`` : niveau moyen par canal ; ``dark_ratio`` : fraction de pixels très sombres.
    """
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    bins = int(min(max(bins, 8), 256))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    n = float(gray.size)
    hist: dict[str, list[float]] = {}
    for name, channel in (("b", image_bgr[:, :, 0]), ("g", image_bgr[:, :, 1]), ("r", image_bgr[:, :, 2]), ("luma", gray)):
        h = cv2.calcHist([channel], [0], None, [bins], [0, 256]).ravel() / n
        hist[name] = [round(float(x), 6) for x in h]
    sat = image_bgr >= SATURATION_LEVEL
    any_sat = float(sat.any(axis=2).mean())
    means = image_bgr.reshape(-1, 3).mean(axis=0)
    return {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "sharpness": round(sharpness(gray), 2),
        "histogram": {"bins": bins, **hist},
        "saturated": {
            "any": round(any_sat, 6),
            "b": round(float(sat[:, :, 0].mean()), 6),
            "g": round(float(sat[:, :, 1].mean()), 6),
            "r": round(float(sat[:, :, 2].mean()), 6),
        },
        "saturation_warning": any_sat > SATURATION_WARNING,
        "mean": {"b": round(float(means[0]), 2), "g": round(float(means[1]), 2), "r": round(float(means[2]), 2)},
        "dark_ratio": round(float((gray <= 5).mean()), 6),
    }


def detect_eggs(image_bgr: np.ndarray, scene: dict[str, Any] | None) -> tuple[list[dict[str, Any]] | None, str | None, tuple[int, int]]:
    """Détections d'œufs par le module de détection (``eggvision.vision.detect_frame``), s'il est présent.

    L'image est mise à la taille de la calibration de scène si besoin ; les
    coordonnées retournées sont dans ce repère, renvoyé en 3e position
    (largeur, hauteur). Retourne ``(None, message, taille)`` si la détection est indisponible.
    """
    h, w = image_bgr.shape[:2]
    try:
        from eggvision.vision import detect_frame  # type: ignore[attr-defined]
    except Exception as exc:  # module absent ou en chantier
        return None, f"Détection indisponible : {exc}", (w, h)
    frame = image_bgr
    if scene and scene.get("image_width") and scene.get("image_height"):
        sw, sh = int(scene["image_width"]), int(scene["image_height"])
        if (sw, sh) != (w, h):
            frame = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
            w, h = sw, sh
    try:
        dets = detect_frame(frame, scene, params=None)
    except Exception as exc:
        log.debug("Détection en direct en échec", exc_info=True)
        return None, f"Erreur de détection : {exc}", (w, h)
    keys = ("cx", "cy", "major", "minor", "angle", "lane", "partial")
    out = []
    for d in dets or []:
        item = {k: (d.get(k) if isinstance(d, dict) else getattr(d, k, None)) for k in keys}
        for k in ("cx", "cy", "major", "minor", "angle"):
            if item[k] is not None:
                item[k] = round(float(item[k]), 2)
        out.append(item)
    return out, None, (w, h)
