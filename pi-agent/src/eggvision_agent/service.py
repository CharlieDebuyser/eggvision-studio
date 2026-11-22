"""Cœur de l'agent : caméra, aperçu, lots (capture et fichiers), réglages, banc de mesure."""

from __future__ import annotations

import collections
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from eggvision_shared.capture import CapturedFrame, CaptureRecorder, DiskFullError, fps_from_timestamps
from eggvision_shared.constants import LOT_STATUS_RUNNING
from eggvision_shared.formats import (
    LOT_MANIFEST,
    CameraSettings,
    LotManifest,
    atomic_write_text,
    iter_lot_dirs,
    lot_file_list,
    mark_interrupted_lots,
    new_lot_id,
)
from pydantic import BaseModel, Field, ValidationError

from . import API_VERSION, __version__, system
from .camera import CameraBackend, make_backend
from .config import AgentConfig

log = logging.getLogger(__name__)

LOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SETTINGS_FILE = "agent_settings.json"


class AgentError(Exception):
    """Erreur présentable au client : code HTTP + message en français."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class StartLotRequest(BaseModel):
    """Manifeste fourni par l'application de bureau pour démarrer un lot."""

    lot_id: str | None = Field(None, description="Identifiant (généré si absent)")
    name: str = Field(..., min_length=1, max_length=200)
    type: str
    farm: str = ""
    operator: str = ""
    comment: str = ""
    created_at: str | None = None
    camera: CameraSettings | None = Field(None, description="Réglages à appliquer et figer pour ce lot")
    min_free_bytes: int | None = Field(None, ge=0, description="Seuil d'arrêt automatique (octets libres)")
    source: dict[str, Any] = Field(default_factory=dict, description="Informations complémentaires (PC…)")


def validate_lot_id(lot_id: str) -> str:
    if not LOT_ID_RE.match(lot_id or ""):
        raise AgentError(400, f"Identifiant de lot invalide : {lot_id!r}")
    return lot_id


def check_settings(settings: CameraSettings) -> None:
    frame_us = 1e6 / settings.fps
    if settings.exposure_us > frame_us:
        raise AgentError(422, f"Exposition ({settings.exposure_us} µs) supérieure à la période d'image ({frame_us:.0f} µs à {settings.fps:g} images/s)")


class AgentService:
    def __init__(self, config: AgentConfig, backend: CameraBackend | None = None) -> None:
        config.validate()
        self.config = config
        self.storage = Path(config.storage)
        self.lots_root = config.lots_root
        self.lots_root.mkdir(parents=True, exist_ok=True)
        self.interrupted_at_start = mark_interrupted_lots(self.lots_root)
        if self.interrupted_at_start:
            log.warning("Lots interrompus (arrêt brutal précédent) : %s", ", ".join(self.interrupted_at_start))
        self.backend = backend or make_backend(config)
        self.settings = self._load_settings()
        self._lock = threading.RLock()
        self._recorder: CaptureRecorder | None = None
        self._last_capture: dict[str, Any] | None = None
        self._latest: CapturedFrame | None = None
        self._frame_ts: collections.deque[int] = collections.deque(maxlen=128)
        self._frames_seen = 0
        self._preview_cache: tuple[Any, bytes] | None = None
        self._preview_lock = threading.Lock()
        self._bench_lock = threading.Lock()
        self._awb_running = False
        self.started_at = time.time()
        self.closing = threading.Event()
        self._camera_started = False

    # ------------------------------------------------------------------ cycle de vie
    def start(self) -> None:
        try:
            self.backend.start(self._on_frame, self.settings)
            self._camera_started = True
        except Exception as exc:
            self.backend.error = f"Caméra indisponible : {exc}"
            log.exception("Démarrage de la caméra")

    def shutdown(self) -> None:
        """Arrêt du service : un lot en cours est clos proprement (statut « interrompu »)."""
        self.closing.set()
        rec = self._recorder
        if rec is not None and rec.running:
            log.warning("Arrêt du service pendant la capture du lot %s : clôture", rec.manifest.lot_id)
            rec.stop(reason="shutdown")
        if self._camera_started:
            try:
                self.backend.stop()
            except Exception:
                log.exception("Arrêt de la caméra")
            self._camera_started = False

    # ------------------------------------------------------------------ images
    def _on_frame(self, frame: CapturedFrame) -> None:
        """Appelé par le fil de la caméra pour CHAQUE image : doit rendre la main immédiatement."""
        self._latest = frame
        self._frame_ts.append(frame.sensor_ts_ns)
        self._frames_seen += 1
        rec = self._recorder
        if rec is not None:
            rec.submit(frame.image, frame.sensor_ts_ns, frame.system_ts, frame.seq)

    def actual_fps(self) -> float:
        ts = list(self._frame_ts)
        if ts and time.monotonic_ns() - ts[-1] > 2_000_000_000:
            return 0.0  # plus d'images depuis 2 s
        return round(fps_from_timestamps(ts), 2)

    def preview_image(self, width: int | None = None):
        frame = self._latest
        if frame is None:
            raise AgentError(503, "Aucune image disponible (caméra en cours de démarrage ?)")
        img = frame.image
        if frame.preview is not None:
            try:
                img = self.backend.preview_to_bgr(frame.preview)
            except Exception:  # flux basse résolution inutilisable : image principale réduite
                img = frame.image
        width = int(width or self.config.preview_width)
        h, w = img.shape[:2]
        if 0 < width < w:
            img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
        return frame, img

    def preview_jpeg(self, width: int | None = None, quality: int = 80) -> tuple[bytes, int]:
        """Dernière image en JPEG basse résolution ; retourne (données, numéro de séquence)."""
        frame = self._latest
        key = (id(frame), frame.seq if frame else None, width, quality)
        with self._preview_lock:
            if self._preview_cache is not None and self._preview_cache[0] == key:
                return self._preview_cache[1], frame.seq if frame else -1
        frame, img = self.preview_image(width)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(min(max(quality, 10), 100))])
        if not ok:
            raise AgentError(500, "Échec de l'encodage de l'aperçu")
        data = buf.tobytes()
        with self._preview_lock:
            self._preview_cache = (key, data)
        return data, frame.seq

    # ------------------------------------------------------------------ réglages
    def _settings_path(self) -> Path:
        return self.storage / SETTINGS_FILE

    def _load_settings(self) -> CameraSettings:
        try:
            return CameraSettings.model_validate_json(self._settings_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return CameraSettings()
        except (OSError, ValidationError, ValueError):
            log.warning("Réglages enregistrés illisibles : valeurs par défaut")
            return CameraSettings()

    def _save_settings(self) -> None:
        try:
            atomic_write_text(self._settings_path(), self.settings.model_dump_json(indent=2))
        except OSError:
            log.exception("Enregistrement des réglages")

    @property
    def capturing(self) -> bool:
        rec = self._recorder
        return rec is not None and rec.running

    def _require_idle(self, what: str) -> None:
        if self.capturing:
            raise AgentError(409, f"{what} impossible pendant une capture : réglages verrouillés.")

    def update_settings(self, changes: dict[str, Any]) -> CameraSettings:
        """Applique des réglages (complets ou partiels). Refusé (409) pendant une capture."""
        with self._lock:
            self._require_idle("Modification des réglages")
            if self._awb_running:
                raise AgentError(409, "Calibration de la balance des blancs en cours.")
            try:
                new = CameraSettings.model_validate({**self.settings.model_dump(), **changes})
            except ValidationError as exc:
                raise AgentError(422, f"Réglages invalides : {exc.errors()[0].get('msg', exc)}") from exc
            check_settings(new)
            w, h = self.backend.sensor_size()
            if new.roi is not None and (new.roi[0] + 16 > w or new.roi[1] + 16 > h):
                raise AgentError(422, "ROI hors du capteur")
            if self._camera_started:
                self.backend.apply_settings(new)
            self.settings = new
            self._save_settings()
            return new

    def calibrate_awb(self, duration_s: float = 2.0) -> dict[str, Any]:
        with self._lock:
            self._require_idle("Calibration")
            if self._awb_running:
                raise AgentError(409, "Calibration déjà en cours.")
            if not self.backend.ready:
                raise AgentError(503, "Caméra non prête.")
            self._awb_running = True
        try:
            red, blue = self.backend.calibrate_awb(float(duration_s), self.settings)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(500, f"Échec de la calibration : {exc}") from exc
        finally:
            self._awb_running = False
        with self._lock:
            self.settings = self.settings.model_copy(update={"red_gain": red, "blue_gain": blue})
            if self._camera_started:
                self.backend.apply_settings(self.settings)
            self._save_settings()
        return {"red_gain": red, "blue_gain": blue, "duration_s": duration_s, "settings": self.settings.model_dump()}

    # ------------------------------------------------------------------ état
    def status(self) -> dict[str, Any]:
        rec = self._recorder
        capture = None
        if rec is not None:
            capture = {**rec.stats(), "name": rec.manifest.name, "type": rec.manifest.type}
        w, h = self.backend.output_size(self.settings)
        return {
            "agent": "eggvision-agent",
            "version": __version__,
            "api_version": API_VERSION,
            "hostname": system.hostname(),
            "device": system.device_info(),
            "time": time.time(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "storage": {**system.disk_info(self.storage), "min_free_bytes": self.config.min_free_bytes},
            "camera": {
                **self.backend.info(),
                "output_width": w,
                "output_height": h,
                "actual_fps": self.actual_fps(),
                "frames_seen": self._frames_seen,
                "settings": self.settings.model_dump(),
                "awb_running": self._awb_running,
            },
            "capturing": capture is not None and capture["running"],
            "capture": capture,
            "last_capture": self._last_capture,
            "bench_running": self._bench_lock.locked(),
            "interrupted_at_start": self.interrupted_at_start,
        }

    def capture_status(self) -> dict[str, Any]:
        rec = self._recorder
        if rec is None:
            return {"active": False, "last": self._last_capture}
        return {"active": rec.running, **rec.stats(), "name": rec.manifest.name, "type": rec.manifest.type}

    # ------------------------------------------------------------------ lots : capture
    def lot_dir(self, lot_id: str) -> Path:
        return self.lots_root / validate_lot_id(lot_id)

    def start_lot(self, req: StartLotRequest) -> dict[str, Any]:
        with self._lock:
            if self.capturing:
                raise AgentError(409, f"Une capture est déjà en cours (lot {self._recorder.manifest.lot_id}).")  # type: ignore[union-attr]
            if self._bench_lock.locked():
                raise AgentError(409, "Banc de mesure en cours : capture impossible.")
            if self._awb_running:
                raise AgentError(409, "Calibration de la balance des blancs en cours.")
            if not self.backend.ready:
                raise AgentError(503, "Caméra non prête : " + (self.backend.error or "démarrage en cours"))
            if req.camera is not None:
                self.update_settings(req.camera.model_dump())
            lot_id = validate_lot_id(req.lot_id) if req.lot_id else new_lot_id(req.name)
            lot_dir = self.lot_dir(lot_id)
            if lot_dir.exists():
                raise AgentError(409, f"Le lot {lot_id} existe déjà sur l'agent.")
            w, h = self.backend.output_size(self.settings)
            try:
                manifest = LotManifest(
                    lot_id=lot_id,
                    name=req.name,
                    type=req.type,
                    farm=req.farm,
                    operator=req.operator,
                    comment=req.comment,
                    source={
                        **req.source,
                        "kind": "pi",
                        "agent_version": __version__,
                        "hostname": system.hostname(),
                        "device": system.pi_model(),
                        "camera_backend": self.backend.name,
                        "camera_model": self.backend.model,
                    },
                    camera=self.settings.model_copy(),
                    width=w,
                    height=h,
                    **({"created_at": req.created_at} if req.created_at else {}),
                )
            except ValidationError as exc:
                raise AgentError(422, f"Manifeste invalide : {exc.errors()[0].get('msg', exc)}") from exc
            min_free = self.config.min_free_bytes if req.min_free_bytes is None else max(req.min_free_bytes, 0)
            rec = CaptureRecorder(
                lot_dir,
                manifest,
                queue_size=self.config.queue_size,
                n_writers=self.config.n_writers,
                min_free_bytes=min_free,
                disk_check_interval=self.config.disk_check_interval,
                on_finished=self._on_lot_finished,
            )
            try:
                rec.start()
            except DiskFullError as exc:
                shutil.rmtree(lot_dir, ignore_errors=True)
                raise AgentError(507, str(exc)) from exc
            self._recorder = rec
            log.info("Capture démarrée : lot %s (%s, %s)", lot_id, req.name, req.type)
            return {"lot_id": lot_id, "manifest": manifest.model_dump(), "capture": rec.stats()}

    def _on_lot_finished(self, manifest: LotManifest, reason: str) -> None:
        rec = self._recorder
        stats = rec.stats() if rec is not None and rec.manifest.lot_id == manifest.lot_id else {}
        self._last_capture = {**stats, "lot_id": manifest.lot_id, "status": manifest.status, "stop_reason": reason, "name": manifest.name}
        if rec is not None and rec.manifest.lot_id == manifest.lot_id:
            self._recorder = None  # plus de capture en cours
        log.info("Lot %s clos (%s, %s) : %d images, %d perdues", manifest.lot_id, manifest.status, reason, manifest.stats.frames_written, manifest.stats.frames_dropped)

    def stop_lot(self, lot_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            rec = self._recorder
            if rec is None or (lot_id and rec.manifest.lot_id != lot_id):
                raise AgentError(409, "Aucune capture en cours" + (f" pour le lot {lot_id}." if lot_id else "."))
        manifest = rec.stop()
        stats = rec.stats()
        return {"lot_id": manifest.lot_id, "manifest": manifest.model_dump(), "capture": stats}

    # ------------------------------------------------------------------ lots : fichiers
    def _current_lot_id(self) -> str | None:
        rec = self._recorder
        return rec.manifest.lot_id if rec is not None and rec.running else None

    def list_lots(self, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        items = []
        current = self._current_lot_id()
        for d in iter_lot_dirs(self.lots_root):
            try:
                m = LotManifest.load(d)
            except Exception:
                items.append({"lot_id": d.name, "status": "illisible"})
                continue
            items.append(
                {
                    "lot_id": m.lot_id,
                    "name": m.name,
                    "type": m.type,
                    "farm": m.farm,
                    "operator": m.operator,
                    "status": m.status,
                    "created_at": m.created_at,
                    "started_at": m.started_at,
                    "ended_at": m.ended_at,
                    "width": m.width,
                    "height": m.height,
                    "stats": m.stats.model_dump(),
                    "capturing": m.lot_id == current,
                }
            )
        items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        total = len(items)
        return {"total": total, "items": items[offset : offset + max(1, min(limit, 1000))]}

    def _existing_lot_dir(self, lot_id: str) -> Path:
        d = self.lot_dir(lot_id)
        if not (d / LOT_MANIFEST).is_file():
            raise AgentError(404, f"Lot {lot_id} introuvable sur l'agent.")
        return d

    def lot_info(self, lot_id: str) -> dict[str, Any]:
        d = self._existing_lot_dir(lot_id)
        manifest = LotManifest.load(d).model_dump()
        rec = self._recorder
        live = rec.stats() if rec is not None and rec.manifest.lot_id == lot_id else None
        return {"manifest": manifest, "capture": live}

    def lot_files(self, lot_id: str) -> dict[str, Any]:
        d = self._existing_lot_dir(lot_id)
        if self._current_lot_id() == lot_id:
            raise AgentError(409, "Lot en cours de capture : synchronisation possible à la fin du lot.")
        manifest = LotManifest.load(d)
        if manifest.status == LOT_STATUS_RUNNING:
            raise AgentError(409, "Lot en cours d'écriture.")
        files = lot_file_list(d)
        return {
            "lot_id": lot_id,
            "status": manifest.status,
            "n_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "files": files,
        }

    def lot_file_path(self, lot_id: str, relpath: str) -> Path:
        d = self._existing_lot_dir(lot_id)
        if self._current_lot_id() == lot_id:
            raise AgentError(409, "Lot en cours de capture.")
        target = (d / relpath).resolve()
        if d.resolve() not in target.parents or not target.is_file():
            raise AgentError(404, f"Fichier introuvable : {relpath}")
        return target

    def delete_lot(self, lot_id: str) -> dict[str, Any]:
        d = self._existing_lot_dir(lot_id)
        with self._lock:
            if self._current_lot_id() == lot_id:
                raise AgentError(409, "Impossible de supprimer le lot en cours de capture.")
            shutil.rmtree(d)
        log.info("Lot %s supprimé de l'agent", lot_id)
        return {"deleted": lot_id, "storage": system.disk_info(self.storage)}

    # ------------------------------------------------------------------ banc de mesure
    def run_bench(self, package_zip: Path, variant: str, n_columns: int, threads: int | None) -> dict[str, Any]:
        from .bench import run_bench

        if self.capturing:
            raise AgentError(409, "Banc de mesure impossible pendant une capture.")
        if not self._bench_lock.acquire(blocking=False):
            raise AgentError(409, "Un banc de mesure est déjà en cours.")
        try:
            if self.capturing:
                raise AgentError(409, "Banc de mesure impossible pendant une capture.")
            return run_bench(package_zip, variant=variant, n_columns=n_columns, threads=threads)
        finally:
            self._bench_lock.release()

