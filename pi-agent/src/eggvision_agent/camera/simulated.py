"""Caméra simulée : images synthétiques pré-rendues ou relecture d'un dossier, à cadence fixe."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from eggvision_shared.capture import CapturedFrame, PacedLoop, SyntheticFrameSource, gray_world_gains
from eggvision_shared.formats import CameraSettings

from .base import CameraBackend, FrameCallback

log = logging.getLogger(__name__)


class SimulatedBackend(CameraBackend):
    """Imite l'IMX296 : flux continu à cadence fixe, horodatages « capteur », images perdues si retard.

    Le rendu synthétique pleine résolution (~45 ms/image) est trop lent pour
    30 images/s : un cycle de ``cycle_s`` secondes est pré-rendu en mémoire au
    démarrage (en arrière-plan ; ``ready`` passe à True ensuite).
    """

    name = "simulated"

    def __init__(self, *, scale: float = 1.0, folder: str | Path | None = None, cycle_s: float = 1.0, seed: int = 0) -> None:
        super().__init__()
        self.source = SyntheticFrameSource(scale=scale, folder=folder, cycle_s=cycle_s, seed=seed)
        self._settings = CameraSettings()
        self._on_frame: FrameCallback | None = None
        self._loop: PacedLoop | None = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    @property
    def model(self) -> str:
        return "simulée (dossier)" if self.source.folder else "simulée (synthétique)"

    @property
    def ready(self) -> bool:
        return self.source.ready.is_set() and self.source.error is None and self._loop is not None

    def sensor_size(self) -> tuple[int, int]:
        return self.source.size

    def start(self, on_frame: FrameCallback, settings: CameraSettings) -> None:
        self._on_frame = on_frame
        self._settings = settings
        self._stopped.clear()
        threading.Thread(target=self._prepare_and_run, name="sim-camera-prepare", daemon=True).start()

    def _prepare_and_run(self) -> None:
        t = time.perf_counter()
        self.source.prepare()
        if self.source.error:
            self.error = self.source.error
            return
        log.info("Caméra simulée prête : %d images %dx%d en %.1f s", len(self.source.frames), *self.source.size, time.perf_counter() - t)
        with self._lock:
            if self._stopped.is_set():
                return
            self._loop = PacedLoop(self._tick, self._settings.fps, name="sim-camera")
            self._loop.start()

    def _tick(self, seq: int, ts_ns: int) -> None:
        img = self.source.frame(seq, self._settings)
        if self._on_frame is not None:
            self._on_frame(CapturedFrame(image=img, sensor_ts_ns=ts_ns, system_ts=time.time(), seq=seq))

    def stop(self) -> None:
        with self._lock:
            self._stopped.set()
            if self._loop is not None:
                self._loop.stop()
                self._loop = None

    def apply_settings(self, settings: CameraSettings) -> None:
        self._settings = settings
        if self._loop is not None:
            self._loop.set_fps(settings.fps)

    def calibrate_awb(self, duration_s: float, settings: CameraSettings) -> tuple[float, float]:
        if not self.source.ready.wait(30) or not self.source.frames:
            raise RuntimeError("Caméra simulée non prête")
        time.sleep(max(0.0, duration_s))
        return gray_world_gains(self.source.frame(0, settings), settings)
