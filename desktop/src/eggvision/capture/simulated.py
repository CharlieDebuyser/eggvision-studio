"""Source simulée : images synthétiques (pré-rendues) ou relecture d'un dossier d'images."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from eggvision_shared.capture import CapturedFrame, PacedLoop, SyntheticFrameSource, gray_world_gains
from eggvision_shared.formats import CameraSettings

from .base import SourceError
from .local import LocalSource


class SimulatedSource(LocalSource):
    """Caméra simulée à cadence fixe ; le lot est écrit directement dans le projet.

    ``scale`` réduit la résolution (1 = 1456 × 1088) ; ``folder`` relit un dossier
    d'images (ou un lot existant : ``frames/`` est pris en compte) au lieu du rendu
    synthétique.
    """

    kind = "simulated"

    def __init__(
        self,
        *,
        scale: float = 0.5,
        folder: str | Path | None = None,
        cycle_s: float = 1.0,
        lot_type: str = "production",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.frames_source = SyntheticFrameSource(scale=scale, folder=folder, cycle_s=cycle_s, lot_type=lot_type, seed=seed)
        self._loop: PacedLoop | None = None
        self._stopped = threading.Event()

    @property
    def label(self) -> str:
        if self.frames_source.folder:
            return f"Simulation (dossier {self.frames_source.folder.name})"
        w, h = self.frames_source.size
        return f"Simulation synthétique {w}×{h}"

    def describe(self) -> dict[str, Any]:
        out = {**super().describe(), "scale": self.frames_source.scale}
        if self.frames_source.folder:
            out["folder"] = str(self.frames_source.folder)
        return out

    @property
    def ready(self) -> bool:
        return self.frames_source.ready.is_set() and self.frames_source.error is None and self._loop is not None

    def _camera_info(self) -> dict[str, Any]:
        w, h = self.frames_source.size
        return {"backend": "simulated", "model": self.label, "sensor_width": w, "sensor_height": h, "ready": self.ready}

    def output_size(self) -> tuple[int, int]:
        return self.settings.output_size(*self.frames_source.size)

    def _start_stream(self) -> None:
        self._stopped.clear()
        threading.Thread(target=self._prepare_and_run, name="sim-source-prepare", daemon=True).start()

    def _prepare_and_run(self) -> None:
        self.frames_source.prepare()
        if self.frames_source.error:
            self.error = self.frames_source.error
            return
        with self._lock:
            if self._stopped.is_set():
                return
            self._loop = PacedLoop(self._tick, self.settings.fps, name="sim-source")
            self._loop.start()

    def _tick(self, seq: int, ts_ns: int) -> None:
        img = self.frames_source.frame(seq, self.settings)
        self._on_frame(CapturedFrame(image=img, sensor_ts_ns=ts_ns, system_ts=time.time(), seq=seq))

    def _stop_stream(self) -> None:
        self._stopped.set()
        if self._loop is not None:
            self._loop.stop()
            self._loop = None

    def _apply_settings(self, settings: CameraSettings) -> None:
        if self._loop is not None:
            self._loop.set_fps(settings.fps)

    def _awb_gains(self, duration_s: float) -> tuple[float, float]:
        if not self.frames_source.frames:
            raise SourceError(503, "Source simulée non prête.")
        time.sleep(max(0.0, duration_s))
        return gray_world_gains(self.frames_source.frame(0, self.settings), self.settings)

    def wait_ready(self, timeout: float = 60.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ready and self._latest is not None:
                return True
            if self.frames_source.error:
                return False
            time.sleep(0.02)
        return False
