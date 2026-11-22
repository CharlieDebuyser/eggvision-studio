"""Source USB : caméra UVC / USB3 branchée sur le PC, via OpenCV ``VideoCapture``.

Réglages manuels « dans la mesure du possible » : OpenCV ne donne accès qu'à
ce que le pilote expose, avec des unités propres à chaque système :

* exposition : V4L2 (Linux) en unités de 100 µs ; DirectShow / Media Foundation
  (Windows) en log2(secondes) ; non réglable sous macOS (AVFoundation) ;
* gain analogique : transmis à ``CAP_PROP_GAIN`` (échelle propre à la caméra :
  ``(gain - 1) × 10``) ;
* balance des blancs automatique coupée (``CAP_PROP_AUTO_WB = 0``) ; les gains
  rouge / bleu sont appliqués en logiciel (table de correspondance) ;
* ROI appliquée en logiciel (recadrage).

``status()["camera"]["controls"]`` indique quels réglages le pilote a acceptés.
Les horodatages sont pris à la réception de l'image (horloge monotone du PC).
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
from eggvision_shared.capture import CapturedFrame, apply_roi, gray_world_gains, settings_lut
from eggvision_shared.constants import SENSOR_HEIGHT, SENSOR_WIDTH
from eggvision_shared.formats import CameraSettings

from .base import SourceError
from .local import LocalSource

log = logging.getLogger(__name__)

CaptureFactory = Callable[..., Any]


def _default_api() -> int:
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def list_usb_cameras(max_index: int = 6, factory: CaptureFactory | None = None) -> list[dict[str, Any]]:
    """Caméras présentes (indices 0…max_index-1) ; ouvre chacune brièvement."""
    factory = factory or cv2.VideoCapture
    out = []
    for i in range(max_index):
        cap = factory(i, _default_api()) if factory is cv2.VideoCapture else factory(i)
        try:
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                out.append({"device": i, "label": f"Caméra USB n° {i}", "width": w, "height": h})
        finally:
            if cap is not None:
                cap.release()
    return out


class UsbCameraSource(LocalSource):
    kind = "usb"

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = SENSOR_WIDTH,
        height: int = SENSOR_HEIGHT,
        capture_factory: CaptureFactory | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.req_width, self.req_height = width, height
        self._factory = capture_factory
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._backend_name = ""
        self.controls: dict[str, bool] = {}
        self._color_lut = None

    @property
    def label(self) -> str:
        return f"Caméra USB {self.device}" + (f" ({self._backend_name})" if self._backend_name else "")

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "device": self.device, "backend": self._backend_name}

    @property
    def ready(self) -> bool:
        return self._thread is not None and self._latest is not None

    def _camera_info(self) -> dict[str, Any]:
        w = h = 0
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {"backend": f"opencv-{self._backend_name or '?'}", "model": self.label, "sensor_width": w, "sensor_height": h, "controls": self.controls, "ready": self.ready}

    # -- ouverture ----------------------------------------------------------
    def _start_stream(self) -> None:
        if self._factory is not None:
            cap = self._factory(self.device)
        else:
            cap = cv2.VideoCapture(self.device, _default_api())
        if cap is None or not cap.isOpened():
            raise SourceError(503, f"Impossible d'ouvrir la caméra USB {self.device}")
        self._cap = cap
        try:
            self._backend_name = str(cap.getBackendName())
        except Exception:
            self._backend_name = ""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        self._apply_settings(self.settings)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="usb-camera", daemon=True)
        self._thread.start()

    def _stop_stream(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- réglages ------------------------------------------------------------
    def _exposure_value(self, exposure_us: int) -> float | None:
        name = self._backend_name.upper()
        if "V4L" in name:
            return max(1.0, exposure_us / 100.0)  # exposure_absolute : unités de 100 µs
        if name in ("DSHOW", "MSMF"):
            return float(round(math.log2(exposure_us / 1e6)))  # log2(secondes)
        return None

    def _apply_settings(self, settings: CameraSettings) -> None:
        cap = self._cap
        if cap is not None:
            controls: dict[str, bool] = {}
            name = self._backend_name.upper()
            manual = 1 if "V4L" in name else 0.25  # V4L2 : 1 = manuel ; DSHOW/MSMF : 0,25 = manuel
            controls["auto_exposure_off"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual))
            value = self._exposure_value(settings.exposure_us)
            controls["exposure"] = bool(cap.set(cv2.CAP_PROP_EXPOSURE, value)) if value is not None else False
            controls["gain"] = bool(cap.set(cv2.CAP_PROP_GAIN, (settings.analogue_gain - 1.0) * 10.0))
            controls["auto_wb_off"] = bool(cap.set(cv2.CAP_PROP_AUTO_WB, 0))
            controls["fps"] = bool(cap.set(cv2.CAP_PROP_FPS, settings.fps))
            controls["colour_gains"] = True  # appliqués en logiciel
            controls["roi"] = True  # appliquée en logiciel
            self.controls = controls
        # gains couleur logiciels (exposition et gain : matériels)
        self._color_lut = settings_lut(CameraSettings(red_gain=settings.red_gain, blue_gain=settings.blue_gain), (1.0, 1.0, 1.0))

    def _awb_gains(self, duration_s: float) -> tuple[float, float]:
        time.sleep(max(0.0, duration_s))
        frame = self._latest
        if frame is None:
            raise SourceError(503, "Aucune image disponible.")
        return gray_world_gains(frame.image, self.settings)

    def output_size(self) -> tuple[int, int]:
        frame = self._latest
        if frame is not None:
            h, w = frame.image.shape[:2]
            return w, h
        return self.settings.output_size(self.req_width, self.req_height)

    # -- acquisition -------------------------------------------------------------
    def _loop(self) -> None:
        seq = 0
        failures = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, img = cap.read()
            ts = time.monotonic_ns()
            if not ok or img is None:
                failures += 1
                if failures in (1, 50):
                    log.warning("Lecture de la caméra USB %s en échec", self.device)
                self.error = "Lecture de la caméra impossible" if failures > 10 else None
                time.sleep(0.02)
                continue
            failures = 0
            self.error = None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = apply_roi(img, self.settings.roi)
            lut = self._color_lut
            if lut is not None:
                img = cv2.LUT(img, lut)
            self._on_frame(CapturedFrame(image=img, sensor_ts_ns=ts, system_ts=time.time(), seq=seq))
            seq += 1
