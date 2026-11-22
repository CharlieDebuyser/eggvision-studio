"""Base des sources locales (USB, simulée) : le lot est écrit directement dans le projet."""

from __future__ import annotations

import collections
import logging
import threading
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from eggvision_shared.capture import (
    CapturedFrame,
    CaptureRecorder,
    DiskFullError,
    disk_free_bytes,
    fps_from_timestamps,
)
from eggvision_shared.formats import CameraSettings, LotManifest
from pydantic import ValidationError

from .base import CameraSource, SourceError, encode_preview, resize_to_width

log = logging.getLogger(__name__)


class LocalSource(CameraSource):
    """Acquisition dans un fil dédié + :class:`CaptureRecorder` (file bornée, jamais bloquant)."""

    def __init__(self, *, queue_size: int = 45, n_writers: int = 3, preview_width: int = 728) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.n_writers = n_writers
        self.preview_width = preview_width
        self.settings = CameraSettings()
        self._lock = threading.RLock()
        self._latest: CapturedFrame | None = None
        self._ts: collections.deque[int] = collections.deque(maxlen=128)
        self._frames_seen = 0
        self._recorder: CaptureRecorder | None = None
        self._lot_dir: Path | None = None
        self.last_capture: dict[str, Any] | None = None
        self.error: str | None = None
        self._opened = False

    # -- à fournir par les sous-classes -------------------------------------
    @abstractmethod
    def _start_stream(self) -> None: ...

    @abstractmethod
    def _stop_stream(self) -> None: ...

    @abstractmethod
    def _apply_settings(self, settings: CameraSettings) -> None: ...

    @abstractmethod
    def _awb_gains(self, duration_s: float) -> tuple[float, float]: ...

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    def _camera_info(self) -> dict[str, Any]:
        return {}

    # -- cycle de vie -------------------------------------------------------
    def open(self) -> None:
        with self._lock:
            if not self._opened:
                self._start_stream()
                self._opened = True

    def close(self) -> None:
        rec = self._recorder
        if rec is not None and rec.running:
            rec.stop(reason="shutdown")
        with self._lock:
            if self._opened:
                self._stop_stream()
                self._opened = False

    # -- images -------------------------------------------------------------
    def _on_frame(self, frame: CapturedFrame) -> None:
        """Appelé par le fil d'acquisition pour chaque image : rend la main immédiatement."""
        self._latest = frame
        self._ts.append(frame.sensor_ts_ns)
        self._frames_seen += 1
        rec = self._recorder
        if rec is not None:
            rec.submit(frame.image, frame.sensor_ts_ns, frame.system_ts, frame.seq)

    def actual_fps(self) -> float:
        ts = list(self._ts)
        if ts and time.monotonic_ns() - ts[-1] > 2_000_000_000:
            return 0.0
        return round(fps_from_timestamps(ts), 2)

    def preview_frame(self, width: int | None = None) -> np.ndarray:
        frame = self._latest
        if frame is None:
            raise SourceError(503, "Aucune image disponible (caméra en cours de démarrage ?)")
        return resize_to_width(frame.image, width or self.preview_width)

    def preview_jpeg(self, width: int | None = None, quality: int = 80) -> bytes:
        return encode_preview(self.preview_frame(width), quality)

    # -- réglages -----------------------------------------------------------
    def get_settings(self) -> CameraSettings:
        return self.settings

    def set_settings(self, changes: dict[str, Any]) -> CameraSettings:
        with self._lock:
            if self.capturing:
                raise SourceError(409, "Réglages verrouillés pendant la capture.")
            try:
                new = CameraSettings.model_validate({**self.settings.model_dump(), **changes})
            except ValidationError as exc:
                raise SourceError(422, f"Réglages invalides : {exc.errors()[0].get('msg', exc)}") from exc
            if new.exposure_us > 1e6 / new.fps:
                raise SourceError(422, f"Exposition ({new.exposure_us} µs) supérieure à la période d'image ({1e6 / new.fps:.0f} µs)")
            self._apply_settings(new)
            self.settings = new
            return new

    def calibrate_awb(self, duration_s: float = 2.0) -> dict[str, Any]:
        if self.capturing:
            raise SourceError(409, "Calibration impossible pendant la capture.")
        if not self.ready:
            raise SourceError(503, "Caméra non prête.")
        red, blue = self._awb_gains(duration_s)
        settings = self.set_settings({"red_gain": red, "blue_gain": blue})
        return {"red_gain": red, "blue_gain": blue, "duration_s": duration_s, "settings": settings.model_dump()}

    # -- état ---------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        rec = self._recorder
        w, h = self.output_size()
        storage = None
        if self._lot_dir is not None:
            try:
                storage = {"free_bytes": disk_free_bytes(self._lot_dir)}
            except OSError:
                storage = None
        return {
            "kind": self.kind,
            "label": self.label,
            "connected": self._opened,
            "ready": self.ready,
            "error": self.error,
            "camera": {
                **self._camera_info(),
                "output_width": w,
                "output_height": h,
                "frames_seen": self._frames_seen,
                "settings": self.settings.model_dump(),
            },
            "actual_fps": self.actual_fps(),
            "device": None,
            "storage": storage,
            "capturing": self.capturing,
            "capture": rec.stats() if rec is not None else None,
        }

    def output_size(self) -> tuple[int, int]:
        frame = self._latest
        if frame is not None:
            h, w = frame.image.shape[:2]
            return w, h
        return self.settings.output_size()

    # -- lots ---------------------------------------------------------------
    @property
    def capturing(self) -> bool:
        rec = self._recorder
        return rec is not None and rec.running

    def start_lot(self, manifest: LotManifest, lot_dir: Path | None, min_free_bytes: int) -> dict[str, Any]:
        if lot_dir is None:
            raise SourceError(500, "Dossier du lot manquant")
        with self._lock:
            if self.capturing:
                raise SourceError(409, "Une capture est déjà en cours.")
            if not self.ready:
                raise SourceError(503, "Caméra non prête.")
            w, h = self.output_size()
            manifest.camera = self.settings.model_copy()
            manifest.width, manifest.height = w, h
            manifest.source = {**self.describe(), **manifest.source}
            rec = CaptureRecorder(
                lot_dir,
                manifest,
                queue_size=self.queue_size,
                n_writers=self.n_writers,
                min_free_bytes=min_free_bytes,
                on_finished=self._finished,
            )
            try:
                rec.start()
            except DiskFullError as exc:
                raise SourceError(507, str(exc)) from exc
            self._lot_dir = lot_dir
            self._recorder = rec
            return {"lot_id": manifest.lot_id, "manifest": manifest.model_dump(), "capture": rec.stats()}

    def _finished(self, manifest: LotManifest, reason: str) -> None:
        rec = self._recorder
        stats = rec.stats() if rec is not None else {}
        self.last_capture = {**stats, "lot_id": manifest.lot_id, "status": manifest.status, "stop_reason": reason}
        callback = self.on_lot_finished
        if callback is not None:
            try:
                callback(manifest.lot_id, manifest.model_dump(), reason)
            except Exception:
                log.exception("Traitement de fin du lot %s", manifest.lot_id)

    def stop_lot(self) -> dict[str, Any]:
        rec = self._recorder
        if rec is None or not rec.running:
            raise SourceError(409, "Aucune capture en cours.")
        manifest = rec.stop()
        return {"lot_id": manifest.lot_id, "manifest": manifest.model_dump(), "capture": rec.stats()}

    def capture_stats(self) -> dict[str, Any] | None:
        rec = self._recorder
        if rec is None:
            return None
        return {"active": rec.running, **rec.stats()}
