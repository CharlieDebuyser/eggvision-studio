"""Singletons de l'application et dépendances FastAPI communes."""

from __future__ import annotations

from fastapi import HTTPException

from .jobs import JobManager
from .project import Project, ProjectManager

projects = ProjectManager()
jobs = JobManager()


def get_project() -> Project:
    """Dépendance FastAPI : le projet ouvert, sinon erreur 409."""
    project = projects.current
    if project is None:
        raise HTTPException(status_code=409, detail="Aucun projet n'est ouvert.")
    return project


def not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} introuvable.")


def bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)

# Fournisseurs d'informations pour l'écran Diagnostic, enregistrés par les modules
# (ex. le module de prise de vue enregistre "agent" → état de l'agent Pi).
diagnostic_providers: dict[str, object] = {}

# URL de l'agent Pi actuellement connecté (ex. "http://192.168.4.1:8770"), ou None.
# Tenue à jour par le module de prise de vue ; lue par l'export (banc de mesure sur le Pi).
agent_url: str | None = None
