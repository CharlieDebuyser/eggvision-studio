"""Application FastAPI : API + frontend compilé servi en local."""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, settings, state
from .jobs import JobCancelled
from .logging_setup import setup_logging
from .project import ProjectError

log = logging.getLogger(__name__)

API_MODULES = ("project", "jobs", "diagnostics", "media", "capture", "detection", "labeling", "training", "export", "simulation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("EggVision Studio %s démarré", __version__)
    yield
    state.jobs.cancel_all(wait=5.0)
    for hook in list(app.state.shutdown_hooks):
        try:
            hook()
        except Exception:
            log.exception("Erreur à l'arrêt")
    state.projects.close()


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="EggVision Studio", version=__version__, lifespan=lifespan)
    app.state.shutdown_hooks = []  # les modules peuvent y ajouter des fonctions d'arrêt

    @app.exception_handler(ProjectError)
    async def _project_error(_: Request, exc: ProjectError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(JobCancelled)
    async def _cancelled(_: Request, exc: JobCancelled) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": "Opération annulée."})

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Erreur interne sur %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"Erreur interne : {exc}"})

    for name in API_MODULES:
        module = importlib.import_module(f"eggvision.api.{name}")
        app.include_router(module.router)
        setup = getattr(module, "setup", None)  # crochet facultatif : setup(app)
        if callable(setup):
            setup(app)

    web = settings.web_dir()
    if (web / "index.html").is_file():
        if (web / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def _spa(path: str):
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Ressource inconnue."})
            candidate = (web / path).resolve()
            if path and candidate.is_file() and web.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(web / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        async def _no_frontend() -> JSONResponse:
            return JSONResponse({"detail": "Frontend non compilé : lancer « npm run build » dans desktop/frontend."})

    return app
