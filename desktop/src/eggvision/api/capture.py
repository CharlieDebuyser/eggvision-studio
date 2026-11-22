"""API du module « Prise de vue » (``/api/capture/...``). Référence : ``docs/api/capture.md``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
from anyio import to_thread
from fastapi import APIRouter, Body, Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from .. import state
from ..capture.base import SourceError
from ..capture.discovery import discover_agents
from ..capture.pi import PiAgentSource
from ..capture.service import CaptureOptions, LotCreate, Overlays, service
from ..db import models as m
from ..project import Project

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/capture", tags=["capture"])


def setup(app: FastAPI) -> None:
    """Crochet appelé par ``app.py`` : erreurs des sources, diagnostic, arrêt propre."""

    @app.exception_handler(SourceError)
    async def _source_error(_: Request, exc: SourceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    state.diagnostic_providers["agent"] = service.diagnostic
    app.state.shutdown_hooks.append(service.shutdown)


# ---------------------------------------------------------------------------
# Sources et connexion
# ---------------------------------------------------------------------------
class ConnectRequest(BaseModel):
    kind: str = Field(..., description="pi | usb | simulated")
    url: str | None = Field(None, description="pi : adresse de l'agent (IP, hôte[:port] ou URL)")
    device: int | str | None = Field(None, description="usb : indice ou chemin du périphérique")
    scale: float | None = Field(None, gt=0, le=1, description="simulated : échelle de l'image (1 = 1456 × 1088)")
    folder: str | None = Field(None, description="simulated : dossier d'images à relire")
    lot_type: str | None = Field(None, description="simulated : type de scène synthétique")


@router.get("/sources")
def sources() -> list[dict[str, Any]]:
    return service.available_sources()


@router.get("/usb/devices")
def usb_devices(max_index: int = Query(6, ge=1, le=16)) -> list[dict[str, Any]]:
    return service.usb_devices(max_index)


@router.get("/discover")
def discover(timeout: float = Query(2.0, ge=0.2, le=10.0)) -> dict[str, Any]:
    return {"agents": discover_agents(timeout)}


@router.post("/connect")
def connect(req: ConnectRequest) -> dict[str, Any]:
    params = {k: v for k, v in req.model_dump().items() if v is not None and k != "kind"}
    return service.connect(req.kind, params)


@router.post("/disconnect")
def disconnect() -> dict[str, Any]:
    return service.disconnect()


@router.get("/status")
def status() -> dict[str, Any]:
    return service.status()


# ---------------------------------------------------------------------------
# Réglages et préréglages
# ---------------------------------------------------------------------------
@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return service.get_settings().model_dump()


@router.put("/settings")
def put_settings(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return service.set_settings(changes).model_dump()


@router.post("/awb")
def awb(body: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
    duration = float((body or {}).get("duration_s", 2.0))
    if not 0 <= duration <= 30:
        raise SourceError(422, "duration_s doit être compris entre 0 et 30 s.")
    return service.calibrate_awb(duration)


class PresetBody(BaseModel):
    settings: dict[str, Any] | None = Field(None, description="Réglages ; absent = réglages actuels de la source")


@router.get("/presets")
def list_presets(project: Project = Depends(state.get_project)) -> list[dict[str, Any]]:
    return service.list_presets(project)


@router.put("/presets/{name}")
def save_preset(name: str, body: PresetBody | None = Body(None), project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.save_preset(project, name, body.settings if body else None)


@router.delete("/presets/{name}")
def delete_preset(name: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    service.delete_preset(project, name)
    return {"deleted": name}


@router.post("/presets/{name}/apply")
def apply_preset(name: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.apply_preset(project, name).model_dump()


# ---------------------------------------------------------------------------
# Aperçu, analyse, repères, options
# ---------------------------------------------------------------------------
@router.get("/preview.jpg")
def preview_jpg(width: int | None = Query(None, ge=32, le=4096), quality: int = Query(80, ge=10, le=100)) -> Response:
    data = service.preview_jpeg(width, quality)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/preview.mjpeg")
async def preview_mjpeg(
    fps: float = Query(10.0, gt=0, le=30),
    width: int | None = Query(None, ge=32, le=4096),
    quality: int = Query(75, ge=10, le=100),
) -> StreamingResponse:
    src = service.source
    if src is None:
        raise SourceError(409, "Aucune source de prise de vue connectée.")
    headers = {"Cache-Control": "no-store"}
    if isinstance(src, PiAgentSource):
        params: dict[str, Any] = {"fps": fps, "quality": quality}
        if width:
            params["width"] = width
        url = src.mjpeg_url()

        async def relay() -> AsyncIterator[bytes]:
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=30.0)) as client:
                    async with client.stream("GET", url, params=params) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
            except httpx.HTTPError as exc:
                log.info("Flux d'aperçu de l'agent interrompu : %s", exc)

        return StreamingResponse(relay(), media_type="multipart/x-mixed-replace; boundary=frame", headers=headers)

    async def local() -> AsyncIterator[bytes]:
        period = 1.0 / fps
        while service.source is src:
            try:
                data = await to_thread.run_sync(src.preview_jpeg, width, quality)
            except SourceError:
                await asyncio.sleep(0.5)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            await asyncio.sleep(period)

    return StreamingResponse(local(), media_type="multipart/x-mixed-replace; boundary=frame", headers=headers)


@router.get("/preview/analysis")
def preview_analysis(
    width: int | None = Query(None, ge=32, le=4096),
    detections: bool = Query(False, description="Ajouter les détections d'œufs (module de détection)"),
    bins: int = Query(64, ge=8, le=256),
    image: bool = Query(False, description="Joindre l'image analysée (data URL JPEG)"),
) -> dict[str, Any]:
    return service.analysis(state.projects.current, width, detections, bins, image)


@router.get("/overlays")
def get_overlays(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.get_overlays(project).model_dump()


@router.put("/overlays")
def put_overlays(overlays: Overlays, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.set_overlays(project, overlays).model_dump()


@router.get("/options")
def get_options(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.get_options(project).model_dump()


@router.put("/options")
def put_options(options: CaptureOptions, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.set_options(project, options).model_dump()


# ---------------------------------------------------------------------------
# Lots
# ---------------------------------------------------------------------------
@router.post("/lots")
def start_lot(req: LotCreate, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return service.start_lot(project, req)


@router.post("/lots/stop")
def stop_lot() -> dict[str, Any]:
    return service.stop_lot()


@router.get("/lots/current")
def current_lot() -> dict[str, Any]:
    return service.live()


@router.get("/lots")
def list_lots(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    source_kind: str | None = Query(None, description="pi | usb | simulated | import"),
    sync_status: str | None = Query(None, description="pending | syncing | done | error"),
    project: Project = Depends(state.get_project),
) -> dict[str, Any]:
    with project.session() as s:
        q = select(m.Lot)
        if source_kind:
            q = q.where(m.Lot.source_kind == source_kind)
        if sync_status:
            q = q.where(m.Lot.sync_status == sync_status)
        total = s.scalar(select(func.count()).select_from(q.subquery())) or 0
        rows = s.scalars(q.order_by(m.Lot.created_at.desc()).limit(limit).offset(offset)).all()
        items = [
            {
                "id": r.id,
                "name": r.name,
                "type": r.type,
                "farm": r.farm,
                "operator": r.operator,
                "status": r.status,
                "source_kind": r.source_kind,
                "sync_status": r.sync_status,
                "frame_count": r.frame_count,
                "dropped_frames": r.dropped_frames,
                "duration_s": r.duration_s,
                "fps": r.fps,
                "width": r.width,
                "height": r.height,
                "created_at": _iso_utc(r.created_at),
                "stop_reason": (r.extra or {}).get("stop_reason"),
            }
            for r in rows
        ]
    return {"total": total, "items": items}


def _iso_utc(dt: datetime | None) -> str | None:
    """SQLite rend des dates sans fuseau : elles sont en UTC."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).isoformat()


@router.get("/agent/lots")
def agent_lots(only_unsynced: bool = Query(False)) -> dict[str, Any]:
    return service.agent_lots(state.projects.current, only_unsynced)


@router.delete("/agent/lots/{lot_id}")
def delete_agent_lot(lot_id: str, force: bool = Query(False)) -> dict[str, Any]:
    return service.delete_agent_lot(state.projects.current, lot_id, force)


@router.post("/lots/{lot_id}/sync")
def sync_lot(lot_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    job = service.submit_sync(project, lot_id)
    return job.to_dict()


@router.get("/sync")
def sync_jobs(limit: int = Query(20, ge=1, le=200), project: Project = Depends(state.get_project)) -> list[dict[str, Any]]:
    return state.jobs.list(project, kind="sync", limit=limit)
