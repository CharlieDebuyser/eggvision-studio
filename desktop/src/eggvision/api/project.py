"""API : projet ouvert, projets récents, configuration, sélecteur de dossiers."""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from .. import state
from ..project import Project, ProjectError, is_project_dir

router = APIRouter(prefix="/api", tags=["projet"])


class CreateBody(BaseModel):
    parent_dir: str
    name: str


class OpenBody(BaseModel):
    path: str


class DemoBody(BaseModel):
    parent_dir: str
    name: str = "Demo EggVision"
    size: str = "mini"


def _info() -> dict[str, Any]:
    project = state.projects.current
    return project.info() if project is not None else {"open": False}


def _ensure_idle() -> None:
    current = state.projects.current
    if current is not None and state.jobs.active(project=current):
        raise HTTPException(409, "Des tâches sont en cours sur le projet ouvert : attendez leur fin ou annulez-les.")


@router.get("/project")
def get_project_info() -> dict[str, Any]:
    return _info()


@router.get("/project/recent")
def recent_projects() -> list[dict[str, Any]]:
    return state.projects.recent()


@router.post("/project/create")
def create_project(body: CreateBody) -> dict[str, Any]:
    _ensure_idle()
    try:
        state.projects.create(body.parent_dir, body.name)
    except ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _info()


@router.post("/project/open")
def open_project(body: OpenBody) -> dict[str, Any]:
    _ensure_idle()
    try:
        project = state.projects.open(body.path)
    except ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.jobs.mark_stale(project)
    return _info()


@router.post("/project/close")
def close_project() -> dict[str, Any]:
    _ensure_idle()
    state.projects.close()
    return _info()


@router.post("/project/demo")
def create_demo(body: DemoBody) -> dict[str, Any]:
    from ..demo import DEMO_SIZES, build_demo_project

    if body.size not in DEMO_SIZES:
        raise HTTPException(400, f"Taille inconnue : {body.size}")
    parent = Path(body.parent_dir).expanduser()
    if not parent.is_dir():
        raise HTTPException(400, f"Dossier parent introuvable : {parent}")
    target = parent / body.name.strip()
    if target.exists() and any(target.iterdir()):
        raise HTTPException(400, f"Le dossier existe déjà et n'est pas vide : {target}")

    def run(ctx):  # type: ignore[no-untyped-def]
        project = build_demo_project(target, name=body.name.strip(), progress=ctx.progress, check_cancelled=ctx.check_cancelled, log=ctx.log, **DEMO_SIZES[body.size])
        project.close()
        return {"path": str(target)}

    job = state.jobs.submit("demo", f"Création du projet de démonstration « {body.name} »", run, params=body.model_dump())
    return {"job_id": job.id}


@router.get("/project/config")
def get_config(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return project.load_config()


@router.put("/project/config")
def put_config(changes: dict[str, Any] = Body(...), project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return project.update_config(changes)


# -- sélecteur de dossiers ---------------------------------------------------
@router.get("/fs/home")
def fs_home() -> dict[str, str]:
    home = Path.home()
    documents = home / "Documents"
    return {"home": str(home), "documents": str(documents if documents.is_dir() else home)}


@router.get("/fs/list")
def fs_list(path: str = "") -> dict[str, Any]:
    """Liste les sous-dossiers de ``path`` (dossier personnel si vide). ``path = "/"`` liste les lecteurs sous Windows."""
    if sys.platform == "win32" and path in ("/", "\\"):
        drives = [f"{letter}:\\" for letter in string.ascii_uppercase if os.path.exists(f"{letter}:\\")]
        return {"path": "/", "parent": None, "entries": [{"name": d, "path": d, "is_dir": True, "is_project": False} for d in drives]}
    base = Path(path).expanduser() if path else Path.home()
    if not base.is_dir():
        raise HTTPException(400, f"Dossier introuvable : {base}")
    base = base.resolve()
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError as exc:
        raise HTTPException(403, f"Accès refusé : {base}") from exc
    for child in children:
        if child.name.startswith(".") or child.name.startswith("$"):
            continue
        try:
            if not child.is_dir():
                continue
        except OSError:
            continue
        entries.append({"name": child.name, "path": str(child), "is_dir": True, "is_project": is_project_dir(child)})
    parent: str | None = str(base.parent) if base.parent != base else ("/" if sys.platform == "win32" else None)
    return {"path": str(base), "parent": parent, "entries": entries}


@router.post("/fs/mkdir")
def fs_mkdir(body: CreateBody) -> dict[str, str]:
    name = body.name.strip()
    if not name or any(c in name for c in '/\\:*?"<>|'):
        raise HTTPException(400, "Nom de dossier invalide.")
    target = Path(body.parent_dir).expanduser() / name
    try:
        target.mkdir(parents=False, exist_ok=True)
    except OSError as exc:
        raise HTTPException(400, f"Impossible de créer le dossier : {exc}") from exc
    return {"path": str(target)}
