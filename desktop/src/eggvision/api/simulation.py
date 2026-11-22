"""API du module « simulation du tri avec trappes ».

Référence : ``docs/api/simulation.md``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .. import state
from ..project import Project
from ..simulation import service
from ..simulation.composition import SimulationError
from ..simulation.engine import OUTCOME_LABELS, OUTCOMES
from ..simulation.params import SimulationParams, default_params
from ..simulation.video import PALETTE, VideoOptions

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


class CreateRequest(SimulationParams):
    """Paramètres de la simulation + nom facultatif."""

    name: str | None = Field(None, max_length=200)


class DuplicateRequest(BaseModel):
    """Duplication : mêmes paramètres (même graine), modèle / seuil / variante modifiables."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(None, max_length=200)
    model_id: str | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    variant: Literal["fp32", "int8"] | None = None
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class TimingRequest(SimulationParams):
    """Paramètres partiels pour le calcul en direct du budget de temps (modèle facultatif)."""

    model_id: str = "-"


def _bad(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc))


def _not_found() -> HTTPException:
    return HTTPException(404, "Simulation introuvable.")


@router.get("/defaults")
def defaults() -> dict[str, Any]:
    """Valeurs par défaut des paramètres, grandeurs dérivées, issues possibles et palette."""
    out = default_params()
    out["params"].pop("model_id", None)
    out["outcomes"] = [{"key": k, "label": OUTCOME_LABELS[k], "color": PALETTE[k]} for k in OUTCOMES]
    out["palette"] = PALETTE
    return out


@router.post("/timing")
def timing(body: TimingRequest) -> dict[str, Any]:
    """Grandeurs temporelles dérivées (vitesse, budget, colonnes en vol…) pour des paramètres donnés."""
    return body.timing()


@router.get("/models")
def models(project: Project = Depends(state.get_project)) -> list[dict[str, Any]]:
    """Modèles exportés, avec la taille de leur jeu test et leur éligibilité."""
    return service.eligible_models(project)


@router.post("")
def create(body: CreateRequest, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Crée une simulation et lance la tâche de fond (``kind="simulation"``)."""
    data = body.model_dump(exclude={"name"})
    try:
        return service.create(project, SimulationParams(**data), name=body.name)
    except SimulationError as exc:
        raise _bad(exc) from exc


@router.get("")
def list_(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model_id: str | None = None,
    project: Project = Depends(state.get_project),
) -> dict[str, Any]:
    """Liste paginée des simulations (plus récentes d'abord)."""
    return service.list_simulations(project, limit=limit, offset=offset, model_id=model_id)


@router.get("/compare")
def compare(ids: list[str] = Query(..., min_length=1, max_length=12), project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Comparaison côte à côte : ``?ids=a&ids=b…`` (paramètres qui diffèrent, métriques, par défaut)."""
    from ..simulation.results import compare as do_compare

    sims = []
    for sim_id in ids:
        try:
            sims.append(service.raw(project, sim_id))
        except KeyError:
            raise HTTPException(404, f"Simulation introuvable : {sim_id}") from None
    return do_compare(sims)


@router.get("/{sim_id}")
def detail(sim_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Paramètres, grandeurs dérivées, résultats chiffrés et état des tâches."""
    try:
        return service.get(project, sim_id)
    except KeyError:
        raise _not_found() from None


@router.delete("/{sim_id}")
def delete(sim_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    try:
        service.delete(project, sim_id)
    except KeyError:
        raise _not_found() from None
    except SimulationError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"deleted": sim_id}


@router.post("/{sim_id}/cancel")
def cancel(sim_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    try:
        job = service.cancel(project, sim_id)
    except KeyError:
        raise _not_found() from None
    return {"job": job}


@router.post("/{sim_id}/duplicate")
def duplicate(sim_id: str, body: DuplicateRequest, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Relance avec les mêmes paramètres ; seuls les champs fournis sont modifiés (``threshold: null`` = seuil du modèle)."""
    overrides = {k: getattr(body, k) for k in body.model_fields_set if k != "name"}
    try:
        return service.duplicate(project, sim_id, overrides, name=body.name)
    except KeyError:
        raise _not_found() from None
    except (SimulationError, ValueError) as exc:
        raise _bad(exc) from exc


@router.get("/{sim_id}/timeline")
def timeline(
    sim_id: str,
    start: int = Query(0, ge=0),
    count: int = Query(200, ge=1, le=2000),
    project: Project = Depends(state.get_project),
) -> dict[str, Any]:
    """Tranche de la chronologie (colonnes ``start`` … ``start + count − 1``) pour l'animation."""
    try:
        return service.timeline_slice(project, sim_id, start, count)
    except KeyError:
        raise _not_found() from None
    except SimulationError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{sim_id}/egg/{column}/{lane}")
def egg(sim_id: str, column: int, lane: int, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Détail d'un œuf : vues (présentées ou non), scores par vue, décision, latence, issue."""
    try:
        return service.egg_detail(project, sim_id, column, lane)
    except KeyError:
        raise HTTPException(404, "Œuf introuvable.") from None
    except SimulationError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{sim_id}/journal.csv")
def journal(sim_id: str, project: Project = Depends(state.get_project)):
    """Journal CSV (une ligne par œuf, séparateur « ; »)."""
    try:
        path = service.journal_path(project, sim_id)
    except KeyError:
        raise _not_found() from None
    except SimulationError as exc:
        raise HTTPException(409, str(exc)) from exc
    return FileResponse(path, media_type="text/csv; charset=utf-8", filename=f"{sim_id}-journal.csv")


@router.post("/{sim_id}/video")
def make_video(sim_id: str, body: VideoOptions | None = None, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Lance le rendu de la vidéo MP4 (``kind="video"``)."""
    try:
        return {"job": service.start_video(project, sim_id, body or VideoOptions())}
    except KeyError:
        raise _not_found() from None
    except SimulationError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{sim_id}/video")
def get_video(sim_id: str, download: bool = False, project: Project = Depends(state.get_project)):
    """Fichier MP4 (``?download=1`` pour un téléchargement)."""
    try:
        path = service.video_path(project, sim_id)
    except KeyError:
        raise _not_found() from None
    if path is None:
        raise HTTPException(404, "Aucune vidéo : la générer d'abord.")
    return FileResponse(path, media_type="video/mp4", filename=f"{sim_id}.mp4" if download else None)
