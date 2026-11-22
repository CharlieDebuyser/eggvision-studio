"""Interface commune des sources de caméra (``CameraSource``).

Trois implémentations interchangeables :
  * :class:`~eggvision.capture.pi.PiAgentSource` — agent Raspberry Pi (cible principale) ;
  * :class:`~eggvision.capture.usb.UsbCameraSource` — caméra UVC / USB3 branchée sur le PC ;
  * :class:`~eggvision.capture.simulated.SimulatedSource` — images synthétiques ou dossier d'images.

Toutes exposent les mêmes capacités : état, réglages, calibration de la balance
des blancs, aperçu, démarrage / arrêt d'un lot et statistiques en direct.
Les sources locales (USB, simulée) écrivent le lot directement dans le projet ;
l'agent Pi l'écrit sur son SSD, d'où il est ensuite synchronisé.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from eggvision_shared.formats import CameraSettings, LotManifest

# Rappel de fin de lot : (lot_id, manifeste final en dict, raison de l'arrêt)
LotFinishedCallback = Callable[[str, dict[str, Any], str], None]


class SourceError(Exception):
    """Erreur d'une source, avec le code HTTP à renvoyer à l'interface."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class CameraSource(ABC):
    kind: str = "base"  # pi | usb | simulated

    def __init__(self) -> None:
        self.on_lot_finished: LotFinishedCallback | None = None

    # -- cycle de vie -----------------------------------------------------
    @abstractmethod
    def open(self) -> None:
        """Ouvre la source (connexion à l'agent, ouverture de la caméra…)."""

    @abstractmethod
    def close(self) -> None:
        """Ferme la source. Une capture locale en cours est close (« interrompu »)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Description courte pour l'interface."""

    def describe(self) -> dict[str, Any]:
        """Informations figées dans ``lot.json`` → ``source``."""
        return {"kind": self.kind, "label": self.label}

    # -- état -------------------------------------------------------------
    @abstractmethod
    def status(self) -> dict[str, Any]:
        """``{kind, label, connected, ready, camera:{…}, actual_fps, device, storage, capturing, capture}``."""

    # -- réglages ---------------------------------------------------------
    @abstractmethod
    def get_settings(self) -> CameraSettings: ...

    @abstractmethod
    def set_settings(self, changes: dict[str, Any]) -> CameraSettings:
        """Applique des réglages complets ou partiels. ``SourceError(409)`` pendant une capture."""

    @abstractmethod
    def calibrate_awb(self, duration_s: float = 2.0) -> dict[str, Any]:
        """AWB ~2 s sur une carte grise puis gains figés : ``{red_gain, blue_gain, settings}``."""

    def output_size(self) -> tuple[int, int]:
        st = self.status()
        cam = st.get("camera") or {}
        return int(cam.get("output_width") or 0), int(cam.get("output_height") or 0)

    # -- aperçu -----------------------------------------------------------
    @abstractmethod
    def preview_jpeg(self, width: int | None = None, quality: int = 80) -> bytes: ...

    def preview_frame(self, width: int | None = None) -> np.ndarray:
        """Dernière image d'aperçu en BGR."""
        data = self.preview_jpeg(width, quality=90)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SourceError(502, "Aperçu illisible")
        return img

    # -- lots -------------------------------------------------------------
    @abstractmethod
    def start_lot(self, manifest: LotManifest, lot_dir: Path | None, min_free_bytes: int) -> dict[str, Any]:
        """Démarre la capture d'un lot. ``lot_dir`` : dossier local (ignoré par l'agent Pi)."""

    @abstractmethod
    def stop_lot(self) -> dict[str, Any]:
        """Arrête le lot en cours : ``{lot_id, manifest, capture}``."""

    @abstractmethod
    def capture_stats(self) -> dict[str, Any] | None:
        """Statistiques du lot en cours (mêmes clés que ``CaptureRecorder.stats()``), ou None."""

    @property
    @abstractmethod
    def capturing(self) -> bool: ...


def resize_to_width(img: np.ndarray, width: int | None) -> np.ndarray:
    if not width:
        return img
    h, w = img.shape[:2]
    if 0 < width < w:
        return cv2.resize(img, (int(width), max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return img


def encode_preview(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(min(max(quality, 10), 100))])
    if not ok:
        raise SourceError(500, "Échec de l'encodage de l'aperçu")
    return buf.tobytes()
