"""Module 2 — Détection et suivi des œufs.

API publique stable :

* :func:`detect_frame` — détection des œufs d'une image (utilisée aussi par l'aperçu en direct de la capture) ;
* :class:`DetectionParams`, :class:`TrackingParams`, :func:`load_params` — réglages (``config.yaml``) ;
* :class:`Tracker` — suivi par couloir sans codeur ;
* :func:`process_lot` — traitement d'un lot (images → détections → pistes → vues).
"""

from __future__ import annotations

from .detect import detect_frame
from .params import DetectionParams, TrackingParams, detection_params, load_params, tracking_params
from .scene import SceneGeometry, default_scene, validate_scene

__all__ = [
    "detect_frame",
    "DetectionParams",
    "TrackingParams",
    "detection_params",
    "tracking_params",
    "load_params",
    "SceneGeometry",
    "default_scene",
    "validate_scene",
    "Tracker",
    "process_lot",
]


def __getattr__(name: str):  # imports paresseux (évite de charger le pipeline pour un simple aperçu)
    if name == "Tracker":
        from .tracking import Tracker

        return Tracker
    if name == "process_lot":
        from .pipeline import process_lot

        return process_lot
    raise AttributeError(name)
