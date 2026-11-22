"""API HTTP de l'agent (FastAPI). Référence : ``docs/pi-agent.md`` (section API)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from anyio import to_thread
from fastapi import Body, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .config import AgentConfig
from .service import AgentError, AgentService, StartLotRequest

log = logging.getLogger(__name__)

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK = 1 << 20


def _file_response(path: Path, range_header: str | None) -> Response:
    """Fichier complet (200) ou partiel (206) selon l'en-tête ``Range`` (reprise de téléchargement)."""
    size = path.stat().st_size
    headers = {"Accept-Ranges": "bytes"}
    start, end = 0, size - 1
    status = 200
    if range_header:
        m = RANGE_RE.match(range_header.strip())
        if not m or (not m.group(1) and not m.group(2)):
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:  # suffixe : bytes=-N
            n = int(m.group(2))
            start = max(0, size - n)
        if start >= size or start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    length = end - start + 1 if size else 0
    headers["Content-Length"] = str(length)
    media = "image/jpeg" if path.suffix.lower() == ".jpg" else "application/octet-stream"
    if length <= 16 * CHUNK:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)
        return Response(content=data, status_code=status, headers=headers, media_type=media)

    def _iter():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                block = f.read(min(CHUNK, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block

    return StreamingResponse(_iter(), status_code=status, headers=headers, media_type=media)


def create_app(config: AgentConfig | None = None, service: AgentService | None = None) -> FastAPI:
    config = config or (service.config if service else AgentConfig.from_env())
    svc = service or AgentService(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start()
        announcer = None
        if config.mdns:
            from .mdns import MdnsAnnouncer

            announcer = MdnsAnnouncer(config.port, config.host)
            announcer.start()
        log.info("Agent EggVision %s prêt (stockage : %s)", __version__, config.storage)
        try:
            yield
        finally:
            svc.shutdown()
            if announcer is not None:
                await to_thread.run_sync(announcer.stop)

    app = FastAPI(title="EggVision — agent Raspberry Pi", version=__version__, lifespan=lifespan)
    app.state.service = svc

    @app.exception_handler(AgentError)
    async def _agent_error(_: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Erreur interne sur %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Erreur interne de l'agent : {exc}"})

    # ------------------------------------------------------------------ état
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return svc.status()

    # ------------------------------------------------------------------ caméra
    @app.get("/api/camera/settings")
    def get_settings() -> dict[str, Any]:
        return svc.settings.model_dump()

    @app.put("/api/camera/settings")
    def put_settings(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return svc.update_settings(changes).model_dump()

    @app.post("/api/camera/awb")
    def calibrate_awb(body: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        duration = float((body or {}).get("duration_s", 2.0))
        if not (0.0 <= duration <= 30.0):
            raise AgentError(422, "duration_s doit être compris entre 0 et 30 s.")
        return svc.calibrate_awb(duration)

    # ------------------------------------------------------------------ aperçu
    @app.get("/api/preview.jpg")
    def preview_jpg(width: int | None = Query(None, ge=32, le=4096), quality: int = Query(80, ge=10, le=100)) -> Response:
        data, seq = svc.preview_jpeg(width, quality)
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)})

    @app.get("/api/preview.mjpeg")
    async def preview_mjpeg(
        fps: float = Query(10.0, gt=0, le=30),
        width: int | None = Query(None, ge=32, le=4096),
        quality: int = Query(75, ge=10, le=100),
    ) -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            period = 1.0 / fps
            last_seq = None
            while not svc.closing.is_set():
                try:
                    data, seq = await to_thread.run_sync(svc.preview_jpeg, width, quality)
                except AgentError:
                    await asyncio.sleep(0.5)
                    continue
                if seq != last_seq:
                    last_seq = seq
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ capture / lots
    @app.get("/api/capture")
    def capture() -> dict[str, Any]:
        return svc.capture_status()

    @app.post("/api/lots")
    def start_lot(req: StartLotRequest) -> dict[str, Any]:
        return svc.start_lot(req)

    @app.post("/api/lots/stop")
    def stop_current() -> dict[str, Any]:
        return svc.stop_lot()

    @app.post("/api/lots/{lot_id}/stop")
    def stop_lot(lot_id: str) -> dict[str, Any]:
        return svc.stop_lot(lot_id)

    @app.get("/api/lots")
    def list_lots(limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
        return svc.list_lots(limit, offset)

    @app.get("/api/lots/{lot_id}")
    def lot_info(lot_id: str) -> dict[str, Any]:
        return svc.lot_info(lot_id)

    @app.get("/api/lots/{lot_id}/files")
    def lot_files(lot_id: str) -> dict[str, Any]:
        return svc.lot_files(lot_id)

    @app.get("/api/lots/{lot_id}/files/{relpath:path}")
    def lot_file(lot_id: str, relpath: str, request: Request) -> Response:
        return _file_response(svc.lot_file_path(lot_id, relpath), request.headers.get("range"))

    @app.delete("/api/lots/{lot_id}")
    def delete_lot(lot_id: str) -> dict[str, Any]:
        return svc.delete_lot(lot_id)

    # ------------------------------------------------------------------ banc de mesure
    @app.post("/api/bench")
    def bench(
        package: UploadFile = File(..., description="Paquet de déploiement (zip)"),
        variant: str = Form("fp32"),
        n_columns: int = Form(50),
        threads: str | None = Form(None),
    ) -> dict[str, Any]:
        n_threads = None
        if threads not in (None, ""):
            try:
                n_threads = int(threads)  # type: ignore[arg-type]
            except ValueError as exc:
                raise AgentError(422, "threads doit être un entier.") from exc
        if svc.capturing:
            raise AgentError(409, "Banc de mesure impossible pendant une capture.")
        fd, tmp = tempfile.mkstemp(prefix="eggvision-package-", suffix=".zip")
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    block = package.file.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
            return svc.run_bench(Path(tmp), variant, n_columns, n_threads)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return app
