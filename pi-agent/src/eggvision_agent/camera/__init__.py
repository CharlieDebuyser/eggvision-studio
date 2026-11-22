"""Caméras de l'agent : réelle (Picamera2) ou simulée."""

from __future__ import annotations

import logging

from ..config import AgentConfig
from .base import CameraBackend
from .simulated import SimulatedBackend

log = logging.getLogger(__name__)

__all__ = ["CameraBackend", "SimulatedBackend", "make_backend"]


def make_backend(config: AgentConfig) -> CameraBackend:
    """Crée la caméra demandée (``auto`` : Picamera2 si une caméra est détectée, sinon simulée)."""
    choice = config.camera
    if choice in ("auto", "picamera2"):
        from .picamera2_backend import Picamera2Backend, picamera2_available

        if picamera2_available():
            return Picamera2Backend(camera_num=config.camera_num, lores_width=config.lores_width)
        if choice == "picamera2":
            raise RuntimeError("Picamera2 indisponible ou aucune caméra détectée (installer python3-picamera2, vérifier le câble CSI)")
        log.warning("Aucune caméra Picamera2 détectée : caméra simulée")
    return SimulatedBackend(scale=config.sim_scale, folder=config.sim_folder, cycle_s=config.sim_cycle_s)
