"""API du module « Export vers Raspberry Pi » (référence : docs/api/export.md)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import state
from ..ml import store
from ..project import Project

router = APIRouter(prefix="/api/export", tags=["export"])


@contextmanager
def _errors() -> Iterator[None]:
    from ..ml.bench import BenchError
    from ..ml.export import ExportError

    try:
        yield
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(404, str(exc).strip("'\"") or "Ressource introuvable.") from exc
    except (ExportError, BenchError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


class ExportBody(BaseModel):
    variants: list[Literal["fp32", "int8"]] = Field(default_factory=lambda: ["fp32", "int8"])
    n_reference: int = Field(6, ge=1, le=50)
    n_calibration_eggs: int = Field(32, ge=1, le=1000)
    n_parity_eggs: int = Field(16, ge=1, le=500)


class BenchBody(BaseModel):
    target: Literal["local", "pi"] = "local"
    variant: Literal["fp32", "int8"] = "fp32"
    n_columns: int = Field(50, ge=1, le=10_000)
    threads: int | None = Field(None, ge=1, le=64)
    agent_url: str | None = None


@router.post("/models/{model_id}")
def export_model(model_id: str, body: ExportBody | None = None, project: Project = Depends(state.get_project)):
    from ..ml.export import start_export

    with _errors():
        return start_export(project, model_id, (body or ExportBody()).model_dump())


@router.get("/models/{model_id}")
def export_status(model_id: str, project: Project = Depends(state.get_project)):
    from ..ml.export import export_status as status

    with _errors():
        return status(project, model_id)


@router.get("/models/{model_id}/download")
def download(model_id: str, project: Project = Depends(state.get_project)):
    with _errors(), project.session() as s:
        export = dict(store.get_model_row(s, model_id).export or {})
    if not export.get("zip"):
        raise HTTPException(404, "Ce modèle n'a pas été exporté.")
    path = project.abspath(export["zip"])
    if not path.is_file():
        raise HTTPException(404, "Paquet introuvable sur le disque : relancer l'export.")
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.post("/models/{model_id}/bench")
def bench(model_id: str, body: BenchBody, project: Project = Depends(state.get_project)):
    from ..ml.bench import start_bench

    with _errors():
        return start_bench(project, model_id, body.target, body.variant, body.n_columns, body.threads, body.agent_url)


@router.get("/models/{model_id}/bench")
def bench_history(model_id: str, limit: int = Query(50, ge=1, le=50), project: Project = Depends(state.get_project)):
    from ..ml.bench import bench_history as history

    with _errors():
        return {"agent_url": state.agent_url, "items": history(project, model_id)[:limit]}
