"""Caméra réelle : Raspberry Pi Global Shutter (IMX296) via Picamera2.

Ce module n'est importé que sur le Raspberry Pi (``picamera2`` est fourni par le
paquet apt ``python3-picamera2`` de Raspberry Pi OS) ; l'import de
``picamera2`` est fait à l'ouverture de la caméra, jamais au chargement de
l'agent.

Choix techniques :
  * configuration « vidéo » (flux continu), ``main`` en ``RGB888`` pleine
    résolution (1456 × 1088) et ``lores`` en ``YUV420`` (640 × 478) pour
    l'aperçu : un seul flux caméra sert les deux, l'aperçu ne coûte rien à la capture ;
  * cadence fixe : ``FrameDurationLimits = (T, T)`` avec ``T = 1e6 / fps`` µs ;
  * automatismes coupés : ``AeEnable = False``, ``AwbEnable = False`` ; réglages
    manuels ``ExposureTime``, ``AnalogueGain``, ``ColourGains`` ;
  * ROI : ``ScalerCrop`` + taille de sortie égale à la ROI (pas de mise à l'échelle) ;
  * horodatage : ``SensorTimestamp`` (ns) des métadonnées de chaque requête ; le
    numéro de séquence est déduit de cet horodatage à cadence fixe (un trou =
    une image perdue en amont) ;
  * aucun encodage vidéo : les images brutes partent vers le moteur
    d'enregistrement (JPEG qualité 95 image par image).

ATTENTION : ce code n'a pas pu être exécuté sur du matériel réel au moment de
son écriture ; il suit l'API documentée de Picamera2 (manuel « The Picamera2
Library »). À valider sur le Pi (voir ``docs/pi-agent.md``, section Validation).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from eggvision_shared.capture import CapturedFrame, nominal_seq
from eggvision_shared.constants import SENSOR_HEIGHT, SENSOR_WIDTH
from eggvision_shared.formats import CameraSettings

from .base import CameraBackend, FrameCallback

log = logging.getLogger(__name__)


def picamera2_available() -> bool:
    """True si Picamera2 est importable et qu'au moins une caméra est détectée."""
    try:
        from picamera2 import Picamera2  # type: ignore[import-not-found]

        return len(Picamera2.global_camera_info()) > 0
    except Exception:
        return False


def _even(x: int) -> int:
    return max(16, int(x) - int(x) % 2)


class Picamera2Backend(CameraBackend):
    name = "picamera2"

    def __init__(self, camera_num: int = 0, lores_width: int = 640, buffer_count: int = 6) -> None:
        super().__init__()
        self.camera_num = camera_num
        self.lores_width = lores_width
        self.buffer_count = buffer_count
        self._picam2: Any = None
        self._on_frame: FrameCallback | None = None
        self._settings = CameraSettings()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._running = False
        self._model = "imx296"
        self._lores_size: tuple[int, int] = (0, 0)
        self._sensor = (SENSOR_WIDTH, SENSOR_HEIGHT)
        self._output = (SENSOR_WIDTH, SENSOR_HEIGHT)
        self._last_metadata: dict[str, Any] = {}
        self._ts_base: int | None = None
        self._seq_base = 0
        self._last_seq = -1

    # ------------------------------------------------------------------ infos
    @property
    def model(self) -> str:
        return self._model

    @property
    def ready(self) -> bool:
        return self._running

    def sensor_size(self) -> tuple[int, int]:
        return self._sensor

    def output_size(self, settings: CameraSettings) -> tuple[int, int]:
        w, h = settings.output_size(*self._sensor)
        return _even(w), _even(h)

    def preview_to_bgr(self, preview: Any) -> Any:
        import cv2

        # lores YUV420 (I420) : tableau (h × 3/2, largeur) → BGR. Une forme inattendue (pas de
        # ligne alignée) lève ValueError : le service se rabat alors sur l'image principale réduite.
        lw, lh = self._lores_size
        if preview.shape != (lh * 3 // 2, lw):
            raise ValueError(f"lores inattendu : {preview.shape} pour {lw}×{lh}")
        return cv2.cvtColor(preview, cv2.COLOR_YUV2BGR_I420)

    def info(self) -> dict[str, Any]:
        out = super().info()
        md = self._last_metadata
        out["metadata"] = {
            k: md.get(k)
            for k in ("ExposureTime", "AnalogueGain", "ColourGains", "FrameDuration", "SensorTemperature", "Lux")
            if k in md
        }
        return out

    # ------------------------------------------------------------------ configuration
    def _controls(self, s: CameraSettings) -> dict[str, Any]:
        frame_us = int(round(1e6 / s.fps))
        controls: dict[str, Any] = {
            "FrameDurationLimits": (frame_us, frame_us),
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": int(min(s.exposure_us, frame_us)),
            "AnalogueGain": float(s.analogue_gain),
            "ColourGains": (float(s.red_gain), float(s.blue_gain)),
        }
        if s.roi is not None:
            x, y, w, h = s.roi
            controls["ScalerCrop"] = (int(x), int(y), _even(w), _even(h))
        return controls

    def _configure(self, s: CameraSettings) -> None:
        picam2 = self._picam2
        w, h = self.output_size(s)
        # largeur multiple de 64 : pas de remplissage de ligne (stride = largeur) dans le tableau YUV420
        lw = max(64, min(self.lores_width, w) // 64 * 64)
        lh = int(round(lw * h / w))
        lh -= lh % 2
        config = picam2.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},  # RGB888 Picamera2 = octets B, G, R (ordre OpenCV)
            lores={"size": (lw, lh), "format": "YUV420"},
            buffer_count=self.buffer_count,
            controls=self._controls(s),
            display=None,  # pas de fenêtre d'aperçu locale (service sans écran)
        )
        picam2.configure(config)
        self._output = tuple(config["main"]["size"])  # type: ignore[assignment]
        self._lores_size = tuple(config["lores"]["size"])  # type: ignore[assignment]
        self._ts_base = None
        log.info("Caméra configurée : main %s, lores %s, %.1f images/s", config["main"]["size"], config["lores"]["size"], s.fps)

    # ------------------------------------------------------------------ cycle de vie
    def start(self, on_frame: FrameCallback, settings: CameraSettings) -> None:
        from picamera2 import Picamera2  # type: ignore[import-not-found]

        with self._lock:
            self._on_frame = on_frame
            self._settings = settings
            info = Picamera2.global_camera_info()
            if not info:
                raise RuntimeError("Aucune caméra détectée (câble CSI, dtoverlay ?)")
            self._picam2 = Picamera2(self.camera_num)
            props = self._picam2.camera_properties
            self._model = str(props.get("Model", "imx296"))
            size = props.get("PixelArraySize")
            if size:
                self._sensor = (int(size[0]), int(size[1]))
            self._configure(settings)
            self._picam2.start()
            self._running = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="picamera2-capture", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(5.0)
                self._thread = None
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception:
                    log.exception("Fermeture de la caméra")
                self._picam2 = None
            self._running = False

    def _restart(self, settings: CameraSettings) -> None:
        """Reconfiguration complète (changement de ROI) : arrêt du flux, configuration, redémarrage."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
        self._picam2.stop()
        self._configure(settings)
        self._picam2.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="picamera2-capture", daemon=True)
        self._thread.start()

    def apply_settings(self, settings: CameraSettings) -> None:
        with self._lock:
            previous = self._settings
            self._settings = settings
            if self._picam2 is None:
                return
            if settings.roi != previous.roi:
                self._restart(settings)
                return
            controls = self._controls(settings)
            controls.pop("ScalerCrop", None)
            self._picam2.set_controls(controls)
            if settings.fps != previous.fps:
                self._ts_base = None  # nouvelle base pour le calcul de seq

    def calibrate_awb(self, duration_s: float, settings: CameraSettings) -> tuple[float, float]:
        """AWB automatique pendant ``duration_s`` (carte grise devant la caméra), puis gains figés."""
        with self._lock:
            if self._picam2 is None:
                raise RuntimeError("Caméra non démarrée")
            self._picam2.set_controls({"AwbEnable": True})
        time.sleep(max(0.5, duration_s))
        gains = self._last_metadata.get("ColourGains")
        with self._lock:
            if not gains:
                self._picam2.set_controls({"AwbEnable": False, "ColourGains": (settings.red_gain, settings.blue_gain)})
                raise RuntimeError("La caméra n'a pas renvoyé de gains de couleur")
            red, blue = float(gains[0]), float(gains[1])
            self._picam2.set_controls({"AwbEnable": False, "ColourGains": (red, blue)})
        return round(red, 4), round(blue, 4)

    # ------------------------------------------------------------------ acquisition
    def _loop(self) -> None:
        picam2 = self._picam2
        while not self._stop.is_set():
            try:
                request = picam2.capture_request()
            except Exception as exc:
                self.error = f"Erreur d'acquisition : {exc}"
                log.exception("capture_request")
                time.sleep(0.1)
                continue
            try:
                main = request.make_array("main")  # copie : le tampon est rendu tout de suite
                lores = request.make_array("lores")
                md = request.get_metadata()
            finally:
                request.release()
            self._last_metadata = md
            if not main.flags["C_CONTIGUOUS"]:
                main = np.ascontiguousarray(main)
            ts = int(md.get("SensorTimestamp") or time.monotonic_ns())
            if self._ts_base is None:
                self._ts_base = ts
                self._seq_base = self._last_seq + 1
            seq = self._seq_base + nominal_seq(ts, self._ts_base, self._settings.fps)
            if seq <= self._last_seq:
                seq = self._last_seq + 1
            self._last_seq = seq
            self.error = None
            callback = self._on_frame
            if callback is not None:
                callback(CapturedFrame(image=main, sensor_ts_ns=ts, system_ts=time.time(), seq=seq, preview=lores, metadata={}))
