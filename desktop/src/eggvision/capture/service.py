"""Service de prise de vue : source active, lots, synchronisation, préréglages, repères, diagnostic.

Un seul service par application (:data:`service`) et une seule source active à
la fois. Les lots d'une source locale (USB, simulée) sont écrits directement
dans ``lots/<id>/`` du projet ; ceux de l'agent Pi sont écrits sur le SSD du Pi
puis synchronisés (tâche de fond ``kind="sync"``) à la fin du lot.

Configuration du projet (``config.yaml``) ::

    capture:
      options:  {min_free_gb: 2.0, auto_sync: true}
      overlays: {lane_lines: [{x1, y1, x2, y2}, …], grid: {enabled, cols, rows},
                 show_lanes: true, show_detections: false}

Coordonnées des repères normalisées dans [0, 1] (indépendantes de la résolution).
"""

from __future__ import annotations

import base64
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from eggvision_shared.capture import disk_free_bytes
from eggvision_shared.constants import LOT_STATUS_RUNNING, parse_lot_type
from eggvision_shared.formats import CameraSettings, LotManifest, new_lot_id
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select

from .. import settings as app_settings
from .. import state
from ..db import models as m
from ..jobs import JobCancelled, JobContext, JobState
from ..project import Project
from .analysis import analyze_image, detect_eggs
from .base import CameraSource, SourceError, encode_preview
from .pi import PiAgentSource
from .simulated import SimulatedSource
from .sync import SyncError, sync_lot
from .usb import UsbCameraSource, list_usb_cameras

log = logging.getLogger(__name__)

SOURCE_KINDS = ("pi", "usb", "simulated")


# ---------------------------------------------------------------------------
# Modèles de configuration
# ---------------------------------------------------------------------------
class LaneLine(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class GridOverlay(BaseModel):
    enabled: bool = False
    cols: int = Field(8, ge=1, le=64)
    rows: int = Field(6, ge=1, le=64)


def _default_lines(n_lanes: int = 6) -> list[LaneLine]:
    return [LaneLine(x1=i / n_lanes, y1=0.0, x2=i / n_lanes, y2=1.0) for i in range(n_lanes + 1)]


class Overlays(BaseModel):
    """Repères superposés à l'aperçu (coordonnées normalisées)."""

    lane_lines: list[LaneLine] = Field(default_factory=_default_lines, max_length=64)
    grid: GridOverlay = Field(default_factory=GridOverlay)
    show_lanes: bool = True
    show_detections: bool = False


class CaptureOptions(BaseModel):
    min_free_gb: float = Field(2.0, ge=0, le=10_000, description="Arrêt automatique sous ce seuil d'espace libre (Go)")
    auto_sync: bool = Field(True, description="Synchroniser automatiquement les lots de l'agent à la fin de la capture")


class LotCreate(BaseModel):
    """Formulaire de création d'un lot (le lot démarre aussitôt)."""

    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(..., description="viable | declasse | declasse:<défaut> | production")
    farm: str = Field("", max_length=200)
    operator: str = Field("", max_length=200)
    comment: str = Field("", max_length=5000)
    settings: dict[str, Any] | None = Field(None, description="Réglages à appliquer juste avant le départ")

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        parse_lot_type(v)
        return v.strip().lower()


def _iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class CaptureService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.source: CameraSource | None = None
        self.current: dict[str, Any] | None = None  # lot en cours : {lot_id, kind, project, name, type}
        self.last_lot: dict[str, Any] | None = None
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._finalize_lock = threading.Lock()

    # ------------------------------------------------------------------ source
    def _require_source(self) -> CameraSource:
        src = self.source
        if src is None:
            raise SourceError(409, "Aucune source de prise de vue connectée.")
        return src

    def connect(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        """Active une source (remplace la précédente, sauf pendant une capture)."""
        if kind not in SOURCE_KINDS:
            raise SourceError(400, f"Source inconnue : {kind!r}")
        with self._lock:
            if self.current is not None:
                raise SourceError(409, "Capture en cours : arrêter le lot avant de changer de source.")
            if self.source is not None:
                self.disconnect()
            src = self._make_source(kind, params)
            try:
                src.open()
            except SourceError:
                src.close()
                raise
            except Exception as exc:
                src.close()
                raise SourceError(503, f"Ouverture de la source impossible : {exc}") from exc
            src.on_lot_finished = self._on_local_lot_finished
            self.source = src
            if isinstance(src, PiAgentSource):
                state.agent_url = src.url
                app_settings.update_state(last_agent_url=src.url)
                self._resume_pi_capture(src)
            log.info("Source de prise de vue : %s", src.label)
            return self.status()

    def _make_source(self, kind: str, params: dict[str, Any]) -> CameraSource:
        if kind == "pi":
            address = params.get("url") or params.get("address") or params.get("host")
            if not address:
                raise SourceError(400, "Adresse de l'agent manquante (champ « url »).")
            return PiAgentSource(str(address), timeout=float(params.get("timeout", 5.0)))
        if kind == "usb":
            device = params.get("device", 0)
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            return UsbCameraSource(device)
        return SimulatedSource(
            scale=float(params.get("scale", 0.5)),
            folder=params.get("folder") or None,
            cycle_s=float(params.get("cycle_s", 1.0)),
            lot_type=str(params.get("lot_type", "production")),
        )

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            src = self.source
            if src is None:
                return self.status()
            if self.current is not None and self.current["kind"] != "pi":
                raise SourceError(409, "Capture en cours : arrêter le lot avant de déconnecter la source.")
            self._stop_monitor()
            self.current = None
            self.source = None
            if isinstance(src, PiAgentSource):
                state.agent_url = None
            try:
                src.close()
            except Exception:
                log.exception("Fermeture de la source")
            return self.status()

    def status(self) -> dict[str, Any]:
        src = self.source
        out: dict[str, Any] = {
            "connected": src is not None,
            "kind": src.kind if src else None,
            "source": None,
            "capturing": self.current is not None,
            "locked": self.current is not None,
            "current_lot": self._current_public(),
            "last_lot": self.last_lot,
            "agent_url": state.agent_url,
            "last_agent_url": app_settings.load_state().get("last_agent_url"),
        }
        if src is not None:
            try:
                out["source"] = src.status()
                project = state.projects.current
                if src.kind != "pi" and project is not None:
                    # source locale : le lot s'écrit dans le projet → espace libre de ce disque
                    try:
                        out["source"]["storage"] = {"free_bytes": disk_free_bytes(project.lots_dir), "path": str(project.lots_dir)}
                    except OSError:
                        pass
            except SourceError as exc:
                out["source"] = {"kind": src.kind, "label": src.label, "connected": False, "error": exc.message}
        return out

    def _current_public(self) -> dict[str, Any] | None:
        cur = self.current
        if cur is None:
            return None
        return {k: v for k, v in cur.items() if k != "project"}

    @staticmethod
    def available_sources() -> list[dict[str, Any]]:
        return [
            {"kind": "pi", "label": "Agent Raspberry Pi", "description": "Caméra Global Shutter pilotée par l'agent Pi (découverte mDNS ou adresse IP)."},
            {"kind": "usb", "label": "Caméra USB", "description": "Caméra UVC / USB3 branchée sur ce PC (OpenCV)."},
            {"kind": "simulated", "label": "Simulation", "description": "Images synthétiques ou dossier d'images relu en boucle."},
        ]

    @staticmethod
    def usb_devices(max_index: int = 6) -> list[dict[str, Any]]:
        return list_usb_cameras(max_index)

    # ------------------------------------------------------------------ réglages
    def get_settings(self) -> CameraSettings:
        return self._require_source().get_settings()

    def set_settings(self, changes: dict[str, Any]) -> CameraSettings:
        if self.current is not None:
            raise SourceError(409, "Réglages verrouillés pendant la capture.")
        return self._require_source().set_settings(changes)

    def calibrate_awb(self, duration_s: float) -> dict[str, Any]:
        if self.current is not None:
            raise SourceError(409, "Calibration impossible pendant la capture.")
        return self._require_source().calibrate_awb(duration_s)

    # ------------------------------------------------------------------ préréglages
    @staticmethod
    def list_presets(project: Project) -> list[dict[str, Any]]:
        with project.session() as s:
            rows = s.scalars(select(m.CameraPreset).order_by(m.CameraPreset.name)).all()
            return [{"name": r.name, "settings": r.settings, "updated_at": (r.updated_at if r.updated_at.tzinfo else r.updated_at.replace(tzinfo=UTC)).isoformat() if r.updated_at else None} for r in rows]

    def save_preset(self, project: Project, name: str, settings: dict[str, Any] | None) -> dict[str, Any]:
        name = (name or "").strip()
        if not name or len(name) > 100:
            raise SourceError(400, "Nom de préréglage invalide (1 à 100 caractères).")
        if settings is None:
            cs = self.get_settings()
        else:
            try:
                cs = CameraSettings.model_validate(settings)
            except ValidationError as exc:
                raise SourceError(422, f"Réglages invalides : {exc.errors()[0].get('msg', exc)}") from exc
        with project.session() as s:
            row = s.get(m.CameraPreset, name)
            if row is None:
                row = m.CameraPreset(name=name)
                s.add(row)
            row.settings = cs.model_dump()
            row.updated_at = m.utcnow()
        return {"name": name, "settings": cs.model_dump()}

    @staticmethod
    def delete_preset(project: Project, name: str) -> None:
        with project.session() as s:
            row = s.get(m.CameraPreset, name)
            if row is None:
                raise SourceError(404, f"Préréglage « {name} » introuvable.")
            s.delete(row)

    def apply_preset(self, project: Project, name: str) -> CameraSettings:
        with project.session() as s:
            row = s.get(m.CameraPreset, name)
            if row is None:
                raise SourceError(404, f"Préréglage « {name} » introuvable.")
            values = dict(row.settings or {})
        return self.set_settings(values)

    # ------------------------------------------------------------------ configuration du projet
    @staticmethod
    def get_overlays(project: Project) -> Overlays:
        raw = (project.get_config("capture", {}) or {}).get("overlays")
        try:
            return Overlays.model_validate(raw or {})
        except ValidationError:
            return Overlays()

    @staticmethod
    def set_overlays(project: Project, overlays: Overlays) -> Overlays:
        project.update_config({"capture": {"overlays": overlays.model_dump()}})
        return overlays

    @staticmethod
    def get_options(project: Project) -> CaptureOptions:
        raw = (project.get_config("capture", {}) or {}).get("options")
        try:
            return CaptureOptions.model_validate(raw or {})
        except ValidationError:
            return CaptureOptions()

    @staticmethod
    def set_options(project: Project, options: CaptureOptions) -> CaptureOptions:
        project.update_config({"capture": {"options": options.model_dump()}})
        return options

    # ------------------------------------------------------------------ aperçu
    def preview_jpeg(self, width: int | None, quality: int) -> bytes:
        return self._require_source().preview_jpeg(width, quality)

    def analysis(
        self, project: Project | None, width: int | None = None, detections: bool = False, bins: int = 64, image: bool = False
    ) -> dict[str, Any]:
        src = self._require_source()
        img = src.preview_frame(width)
        out = analyze_image(img, bins)
        if image:
            # image analysée elle-même : les détections s'affichent sur la bonne image (scène en mouvement)
            out["image"] = "data:image/jpeg;base64," + base64.b64encode(encode_preview(img, 75)).decode("ascii")
        if detections:
            scene = project.get_config("scene") if project is not None else None
            dets, error, (fw, fh) = detect_eggs(img, scene)
            out["detections"] = dets
            out["detections_error"] = error
            out["detections_frame"] = {"width": fw, "height": fh}
        return out

    # ------------------------------------------------------------------ lots
    def start_lot(self, project: Project, req: LotCreate) -> dict[str, Any]:
        with self._lock:
            src = self._require_source()
            if self.current is not None or src.capturing:
                raise SourceError(409, "Une capture est déjà en cours.")
            if req.settings:
                src.set_settings(req.settings)
            options = self.get_options(project)
            lot_id = new_lot_id(req.name)
            while project.lot_dir(lot_id).exists():
                lot_id = new_lot_id(req.name)
            settings = src.get_settings()
            manifest = LotManifest(
                lot_id=lot_id,
                name=req.name.strip(),
                type=req.type,
                farm=req.farm.strip(),
                operator=(req.operator or project.user_name).strip(),
                comment=req.comment,
                source=src.describe(),
                camera=settings,
            )
            min_free = int(options.min_free_gb * 1e9)
            if isinstance(src, PiAgentSource):
                result = src.start_lot(manifest, None, min_free)
                agent_manifest = result.get("manifest") or manifest.model_dump()
                self._insert_pi_lot(project, agent_manifest)
                self.current = {"lot_id": lot_id, "kind": "pi", "project": project, "name": manifest.name, "type": manifest.type, "agent_url": src.url}
                self._start_monitor(src)
            else:
                result = src.start_lot(manifest, project.lot_dir(lot_id), min_free)
                self.current = {"lot_id": lot_id, "kind": src.kind, "project": project, "name": manifest.name, "type": manifest.type}
                try:
                    project.register_lot(lot_id, src.kind, sync_status="done")
                except Exception:
                    log.exception("Enregistrement du lot %s au démarrage", lot_id)
            log.info("Lot %s démarré (%s)", lot_id, src.kind)
            return self.live()

    def _insert_pi_lot(self, project: Project, manifest: dict[str, Any]) -> None:
        """Ligne ``Lot`` d'un lot capturé par l'agent, avant sa synchronisation (``sync_status = pending``)."""
        camera = manifest.get("camera") or {}
        started = _iso_to_dt(manifest.get("started_at")) or _iso_to_dt(manifest.get("created_at")) or m.utcnow()
        with project.session() as s:
            lot = s.get(m.Lot, manifest["lot_id"])
            if lot is None:
                lot = m.Lot(id=manifest["lot_id"], name=manifest.get("name", ""), type=manifest.get("type", "production"))
                s.add(lot)
            lot.name = manifest.get("name", lot.name)
            lot.type = manifest.get("type", lot.type)
            lot.farm = manifest.get("farm", "")
            lot.operator = manifest.get("operator", "")
            lot.comment = manifest.get("comment", "")
            lot.status = manifest.get("status") or LOT_STATUS_RUNNING
            lot.source_kind = "pi"
            lot.created_at = _iso_to_dt(manifest.get("created_at")) or m.utcnow()
            lot.started_at = started
            lot.ended_at = _iso_to_dt(manifest.get("ended_at"))
            lot.session_date = started.astimezone().strftime("%Y-%m-%d")
            lot.width = int(manifest.get("width") or 0)
            lot.height = int(manifest.get("height") or 0)
            lot.fps = float(camera.get("fps") or 0.0)
            lot.camera = camera
            stats = manifest.get("stats") or {}
            lot.dropped_frames = int(stats.get("frames_dropped") or 0)
            lot.duration_s = float(stats.get("duration_s") or 0.0)
            if lot.sync_status != "done":
                lot.sync_status = "pending"
            lot.extra = {**(lot.extra or {}), "agent_stats": stats}

    def stop_lot(self) -> dict[str, Any]:
        with self._lock:
            src = self._require_source()
            cur = self.current
            if cur is None and not src.capturing:
                raise SourceError(409, "Aucune capture en cours.")
            result = src.stop_lot()
            if isinstance(src, PiAgentSource):
                self._finalize_pi_lot(result.get("manifest") or {}, "user")
            # sources locales : la fin est traitée par _on_local_lot_finished (appelé par l'enregistreur)
            return {**self.live(), "result": {"lot_id": result.get("lot_id"), "manifest": result.get("manifest")}}

    def live(self) -> dict[str, Any]:
        """État en direct de la capture : durée, images, cadence réelle, pertes, disque, file."""
        src = self.source
        cur = self.current
        stats = None
        error = None
        if src is not None and cur is not None:
            try:
                stats = src.capture_stats()
            except SourceError as exc:
                error = exc.message
        return {
            "active": cur is not None,
            "lot": self._current_public(),
            "stats": stats,
            "error": error,
            "locked": cur is not None,
            "last": self.last_lot,
        }

    # -- fin de lot : sources locales ----------------------------------------
    def _on_local_lot_finished(self, lot_id: str, manifest: dict[str, Any], reason: str) -> None:
        cur = self.current
        project: Project | None = cur["project"] if cur and cur.get("lot_id") == lot_id else state.projects.current
        stats = (self.source.capture_stats() if self.source else None) or {}
        if project is not None:
            try:
                project.register_lot(lot_id, manifest.get("source", {}).get("kind"), sync_status="done")
                with project.session() as s:
                    lot = s.get(m.Lot, lot_id)
                    if lot is not None:
                        lot.extra = {**(lot.extra or {}), "stop_reason": reason, "capture_stats": _public_stats(stats)}
            except Exception:
                log.exception("Enregistrement du lot %s", lot_id)
        self.last_lot = {"lot_id": lot_id, "status": manifest.get("status"), "stop_reason": reason, "stats": manifest.get("stats"), "kind": manifest.get("source", {}).get("kind")}
        if cur and cur.get("lot_id") == lot_id:
            self.current = None
        log.info("Lot %s terminé (%s)", lot_id, reason)

    # -- fin de lot : agent Pi -----------------------------------------------
    def _resume_pi_capture(self, src: PiAgentSource) -> None:
        """Reconnexion à un agent en pleine capture (application relancée) : on reprend le suivi."""
        try:
            cap = src.capture_stats() or {}
        except SourceError:
            return
        project = state.projects.current
        if cap.get("active") and project is not None:
            lot_id = cap.get("lot_id")
            try:
                info = src.lot_info(lot_id)
                self._insert_pi_lot(project, info["manifest"])
            except SourceError:
                pass
            self.current = {"lot_id": lot_id, "kind": "pi", "project": project, "name": cap.get("name"), "type": cap.get("type"), "agent_url": src.url}
            self._start_monitor(src)

    def _start_monitor(self, src: PiAgentSource) -> None:
        self._stop_monitor()
        self._monitor_stop = threading.Event()
        stop = self._monitor_stop
        self._monitor = threading.Thread(target=self._monitor_loop, args=(src, stop), name="pi-capture-monitor", daemon=True)
        self._monitor.start()

    def _stop_monitor(self) -> None:
        self._monitor_stop.set()
        t = self._monitor
        if t is not None and t is not threading.current_thread():
            t.join(5.0)
        self._monitor = None

    def _monitor_loop(self, src: PiAgentSource, stop: threading.Event) -> None:
        """Suit le lot sur l'agent ; détecte une fin côté agent (arrêt automatique disque, redémarrage…)."""
        failures = 0
        while not stop.wait(1.0):
            cur = self.current
            if cur is None or cur.get("kind") != "pi":
                return
            try:
                cap = src.capture_stats() or {}
                failures = 0
            except SourceError:
                failures += 1
                continue  # l'agent continue sa capture seul ; on réessaie
            if not cap.get("active") or cap.get("lot_id") != cur["lot_id"]:
                try:
                    manifest = src.lot_info(cur["lot_id"]).get("manifest") or {}
                except SourceError:
                    manifest = {"lot_id": cur["lot_id"]}
                last = cap.get("last") or {}
                reason = last.get("stop_reason", "agent") if last.get("lot_id") == cur["lot_id"] else "agent"
                self._finalize_pi_lot(manifest, reason)
                return

    def _finalize_pi_lot(self, manifest: dict[str, Any], reason: str) -> None:
        with self._finalize_lock:
            cur = self.current
            lot_id = manifest.get("lot_id") or (cur or {}).get("lot_id")
            if cur is None or cur.get("lot_id") != lot_id:
                return  # déjà traité
            project: Project = cur["project"]
            self.current = None
            self._monitor_stop.set()
            if manifest.get("status"):
                try:
                    self._insert_pi_lot(project, manifest)
                    with project.session() as s:
                        lot = s.get(m.Lot, lot_id)
                        if lot is not None:
                            lot.extra = {**(lot.extra or {}), "stop_reason": reason}
                except Exception:
                    log.exception("Mise à jour du lot %s", lot_id)
            self.last_lot = {"lot_id": lot_id, "status": manifest.get("status"), "stop_reason": reason, "stats": manifest.get("stats"), "kind": "pi"}
            log.info("Lot %s terminé sur l'agent (%s)", lot_id, reason)
            if self.get_options(project).auto_sync and cur.get("agent_url"):
                try:
                    job = self.submit_sync(project, lot_id, cur["agent_url"])
                    self.last_lot["sync_job_id"] = job.id
                except Exception:
                    log.exception("Lancement de la synchronisation du lot %s", lot_id)

    # ------------------------------------------------------------------ lots de l'agent / synchronisation
    def _agent(self) -> PiAgentSource:
        src = self.source
        if isinstance(src, PiAgentSource):
            return src
        raise SourceError(409, "Aucun agent Pi connecté.")

    def agent_lots(self, project: Project | None, only_unsynced: bool = False) -> dict[str, Any]:
        agent = self._agent()
        listing = agent.list_lots()
        local: dict[str, str] = {}
        if project is not None:
            ids = [it["lot_id"] for it in listing.get("items", [])]
            with project.session() as s:
                for i in range(0, len(ids), 500):
                    for lot_id, sync_status in s.execute(select(m.Lot.id, m.Lot.sync_status).where(m.Lot.id.in_(ids[i : i + 500]))):
                        local[lot_id] = sync_status
        active = {j.params.get("lot_id"): j.id for j in state.jobs.active(kind="sync")}
        items = []
        for it in listing.get("items", []):
            sync_status = local.get(it["lot_id"])
            synced = sync_status == "done"
            item = {**it, "in_project": it["lot_id"] in local, "sync_status": sync_status, "synced": synced, "sync_job_id": active.get(it["lot_id"])}
            if only_unsynced and synced:
                continue
            items.append(item)
        unsynced = sum(1 for it in items if not it["synced"] and not it.get("capturing"))
        return {"agent_url": agent.url, "total": len(items), "unsynced": unsynced, "items": items}

    def submit_sync(self, project: Project, lot_id: str, agent_url: str | None = None) -> JobState:
        url = agent_url or (self.source.url if isinstance(self.source, PiAgentSource) else None) or state.agent_url
        if not url:
            raise SourceError(409, "Aucun agent Pi connecté : synchronisation impossible.")
        for job in state.jobs.active(kind="sync"):
            if job.params.get("lot_id") == lot_id:
                return job  # déjà en cours ou en attente
        cur = self.current
        if cur is not None and cur.get("lot_id") == lot_id:
            raise SourceError(409, "Lot en cours de capture : synchronisation possible à la fin du lot.")
        self._mark_sync(project, lot_id, "pending", agent_url=url)

        def _run(ctx: JobContext) -> dict[str, Any]:
            return self._sync_job(ctx, project, lot_id, url)

        return state.jobs.submit("sync", f"Synchronisation du lot {lot_id}", _run, params={"lot_id": lot_id, "agent_url": url}, project=project, group="sync")

    def _mark_sync(self, project: Project, lot_id: str, status: str, agent_url: str | None = None) -> None:
        """Met à jour ``Lot.sync_status`` ; crée la ligne du lot (manifeste de l'agent) s'il n'est pas encore en base."""
        with project.session() as s:
            lot = s.get(m.Lot, lot_id)
            if lot is not None:
                lot.sync_status = status
                return
        if agent_url is None:
            return
        client = PiAgentSource(agent_url)
        try:
            manifest = client.lot_info(lot_id)["manifest"]
        except SourceError:
            return
        finally:
            client.close()
        self._insert_pi_lot(project, manifest)
        with project.session() as s:
            lot = s.get(m.Lot, lot_id)
            if lot is not None:
                lot.sync_status = status

    def _sync_job(self, ctx: JobContext, project: Project, lot_id: str, url: str) -> dict[str, Any]:
        self._mark_sync(project, lot_id, "syncing", agent_url=url)

        def progress(p: dict[str, Any]) -> None:
            frac = p["bytes_done"] / p["bytes_total"] if p["bytes_total"] else 0.0
            ctx.progress(
                min(0.99, frac),
                f"{p['files_done']}/{p['files_total']} fichiers — {p['bytes_done'] / 1e6:.0f}/{p['bytes_total'] / 1e6:.0f} Mo ({p['rate_mb_s']:.1f} Mo/s)",
            )
            ctx.publish(**p)

        try:
            result = sync_lot(url, lot_id, project.lot_dir(lot_id), progress=progress, check_cancelled=ctx.check_cancelled, log_fn=ctx.log)
            ctx.message("Enregistrement du lot dans le projet…")
            project.register_lot(lot_id, "pi", sync_status="done")
            ctx.message(f"Lot synchronisé et vérifié : {result['files']} fichiers, {result['bytes'] / 1e6:.0f} Mo")
        except JobCancelled:
            self._mark_sync(project, lot_id, "pending")
            raise
        except SyncError:
            self._mark_sync(project, lot_id, "error")
            raise
        except Exception:
            self._mark_sync(project, lot_id, "error")
            raise
        ctx.log(
            f"Synchronisation vérifiée : {result['files']} fichiers, {result['bytes'] / 1e6:.1f} Mo "
            f"({result['files_skipped']} déjà présents, {result['retries']} reprises, {result['corrupted']} fichiers corrompus retéléchargés)"
        )
        return result

    def delete_agent_lot(self, project: Project | None, lot_id: str, force: bool = False) -> dict[str, Any]:
        agent = self._agent()
        if not force:
            synced = False
            if project is not None:
                with project.session() as s:
                    lot = s.get(m.Lot, lot_id)
                    synced = lot is not None and lot.sync_status == "done"
            if not synced:
                raise SourceError(409, "Ce lot n'est pas synchronisé dans le projet : suppression refusée (utiliser force=true pour forcer).")
        return agent.delete_lot(lot_id)

    # ------------------------------------------------------------------ diagnostic / arrêt
    def diagnostic(self) -> dict[str, Any]:
        """Fournisseur de l'écran Diagnostic : ``{connected, url, version, detail}``."""
        src = self.source
        if isinstance(src, PiAgentSource):
            try:
                st = src.raw_status(max_age=0, timeout=1.5)
            except SourceError as exc:
                return {"connected": False, "url": src.url, "version": src.version, "detail": exc.message}
            cam = st.get("camera") or {}
            dev = st.get("device") or {}
            storage = st.get("storage") or {}
            temp = dev.get("temperature_c")
            parts = [
                dev.get("model") or "?",
                f"{temp:.1f} °C" if isinstance(temp, (int, float)) else "température inconnue",
                f"{(storage.get('free_bytes') or 0) / 1e9:.1f} Go libres",
                f"caméra {cam.get('model')} {'prête' if cam.get('ready') else 'non prête'}",
                f"{cam.get('actual_fps', 0):.1f} images/s",
            ]
            if st.get("capturing"):
                parts.append("capture en cours")
            return {
                "connected": True,
                "url": src.url,
                "version": st.get("version"),
                "detail": " — ".join(parts),
                "hostname": st.get("hostname"),
                "temperature_c": temp,
                "disk_free_bytes": storage.get("free_bytes"),
                "actual_fps": cam.get("actual_fps"),
                "capturing": st.get("capturing"),
            }
        last = app_settings.load_state().get("last_agent_url")
        detail = "Aucun agent Pi connecté"
        if src is not None:
            detail += f" (source active : {src.label})"
        return {"connected": False, "url": state.agent_url or last, "version": None, "detail": detail}

    def shutdown(self) -> None:
        """Arrêt de l'application : capture locale close (« interrompu »), source fermée.

        Une capture de l'agent Pi continue sur le Pi (elle sera synchronisée plus tard).
        """
        with self._lock:
            self._stop_monitor()
            src = self.source
            cur = self.current
            if src is not None and cur is not None and cur["kind"] != "pi":
                try:
                    src.close()  # clôt le lot local : _on_local_lot_finished l'enregistre
                except Exception:
                    log.exception("Arrêt de la capture")
            self.current = None
            self.source = None
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass
            if isinstance(src, PiAgentSource):
                state.agent_url = None


def _public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keys = ("frames_written", "frames_dropped", "drop_ratio", "measured_fps", "duration_s", "bytes_written", "queue_max_fill", "stop_reason")
    return {k: stats.get(k) for k in keys if k in stats}


service = CaptureService()

