"""API du module « labellisation » (relecture des pistes). Référence : ``docs/api/labeling.md``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .. import labeling as lab
from .. import state
from ..labeling.active import DEFAULT_MARGIN
from ..labeling.common import UNKNOWN_USER, UNSET
from ..labeling.dashboard import MIN_EXAMPLES
from ..project import Project

router = APIRouter(prefix="/api/labeling", tags=["labellisation"])


def get_labeling_project() -> Project:
    """Projet ouvert, avec les index de labellisation garantis."""
    project = state.get_project()
    lab.ensure_indexes(project.engine)
    return project


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except lab.NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except lab.LabelingError as exc:
        raise HTTPException(400, str(exc)) from exc


def _user(project: Project) -> str:
    return project.user_name.strip() or UNKNOWN_USER


def _split(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for v in values or []:
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out


def _filters(
    lot: list[str] | None = Query(None, description="Identifiant(s) de lot (paramètre répétable)"),
    lot_type: list[str] | None = Query(None, description="Type(s) de lot ; « declasse » couvre « declasse:<défaut> »"),
    verdict: list[str] | None = Query(None, description="viable, non_viable, incertain, exclu, aucun (sans label)"),
    defect: list[str] | None = Query(None, description="Type(s) de défaut ; « aucun » = non viable sans type"),
    reviewed: bool | None = Query(None, description="true : relues ; false : à relire"),
    quality: list[str] | None = Query(None, description="ok, douteuse"),
    lane: list[str] | None = Query(None, description="Couloir(s)"),
    source: str | None = Query(None, description="auto | humain"),
    score_min: float | None = Query(None, ge=0, le=1),
    score_max: float | None = Query(None, ge=0, le=1),
    reason: str | None = Query(None, description="File d'apprentissage actif : file | contredit | hesite"),
    track_id: list[int] | None = Query(None, description="Restreindre à ces pistes"),
) -> lab.TrackFilters:
    try:
        lanes = [int(x) for x in _split(lane)]
    except ValueError as exc:
        raise HTTPException(400, "Couloir invalide.") from exc
    return lab.TrackFilters(
        lots=_split(lot),
        lot_types=_split(lot_type),
        verdicts=_split(verdict),
        defects=_split(defect),
        reviewed=reviewed,
        quality=_split(quality),
        lanes=lanes,
        source=source or None,
        score_min=score_min,
        score_max=score_max,
        reason=reason or None,
        track_ids=list(track_id or []),
    )


# -- vocabulaire, relecteur, modèles, lots ----------------------------------
@router.get("/meta")
def get_meta(project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    """Verdicts, défauts et leurs touches, actions de l'historique, nom du relecteur."""
    return {**lab.meta(), "user": project.user_name, "min_examples": MIN_EXAMPLES, "default_margin": DEFAULT_MARGIN}


@router.get("/models")
def get_models(project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    """Modèles ayant des prédictions, et modèle retenu par défaut pour l'apprentissage actif."""
    with project.session() as s:
        default = lab.resolve_model(s, None)
        return {"items": lab.list_models(s), "default": default.to_dict() if default else None}


@router.get("/lots")
def get_lots(
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    """Lots ayant des pistes (pour le panneau de filtres), du plus récent au plus ancien."""
    from sqlalchemy import func, select

    from ..db import models as m

    with project.session() as s:
        total = s.execute(select(func.count()).select_from(m.Lot)).scalar_one()
        rows = s.execute(
            select(m.Lot.id, m.Lot.name, m.Lot.type, m.Lot.farm, m.Lot.session_date, m.Lot.track_count)
            .order_by(m.Lot.created_at.desc(), m.Lot.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    return {
        "items": [
            {"id": r.id, "name": r.name, "type": r.type, "farm": r.farm, "session_date": r.session_date, "track_count": r.track_count}
            for r in rows
        ],
        "total": total,
    }


# -- recherche et détail -----------------------------------------------------
@router.get("/tracks")
def list_tracks(
    filters: lab.TrackFilters = Depends(_filters),
    sort: str = Query("naturel", description="naturel | recent | priorite"),
    limit: int = Query(100, ge=1, le=500),
    cursor: str | None = Query(None, description="next_cursor de la page précédente"),
    offset: int = Query(0, ge=0),
    anchor: int | None = Query(None, description="Commencer à cette piste (incluse)"),
    model_id: str | None = Query(None, description="Modèle pour score/priorité ; défaut : dernier modèle entraîné ; « aucun »"),
    margin: float = Query(DEFAULT_MARGIN, gt=0, le=0.5),
    with_total: bool = Query(True),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    with _errors(), project.session() as s:
        page = lab.search_tracks(
            s, filters, sort=sort, limit=limit, cursor=cursor, offset=offset, anchor=anchor,
            model_id=model_id, margin=margin, with_total=with_total,
        )
        return page.to_dict()


@router.get("/active")
def active_queue(
    filters: lab.TrackFilters = Depends(_filters),
    limit: int = Query(100, ge=1, le=500),
    cursor: str | None = Query(None),
    model_id: str | None = Query(None),
    margin: float = Query(DEFAULT_MARGIN, gt=0, le=0.5),
    with_total: bool = Query(True),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    """File d'apprentissage actif : pistes où le modèle contredit le label ou hésite, non relues d'abord."""
    with _errors(), project.session() as s:
        model = lab.resolve_model(s, model_id)
        if model is None:
            return {
                "items": [], "next_cursor": None, "total": 0, "sort": "priorite", "model": None,
                "message": "Aucun modèle entraîné avec des prédictions : la file d'apprentissage actif est vide.",
            }
        if filters.reason is None:
            filters.reason = "file"
        page = lab.search_tracks(
            s, filters, sort="priorite", limit=limit, cursor=cursor, model_id=model.id, margin=margin, with_total=with_total
        ).to_dict()
        page["message"] = None if page["items"] or cursor else "Le modèle ne contredit aucun label et n'hésite sur aucune piste."
        return page


@router.get("/tracks/{track_id}")
def get_track(
    track_id: int,
    model_id: str | None = Query(None),
    margin: float = Query(DEFAULT_MARGIN, gt=0, le=0.5),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    with _errors(), project.session() as s:
        return lab.track_detail(s, track_id, model_id=model_id, margin=margin)


# -- écritures ---------------------------------------------------------------
class LabelBody(BaseModel):
    """Champs absents = inchangés. ``defect`` renseigné ⇒ verdict « non viable »."""

    verdict: str | None = None
    defect: str | None = None
    reviewed: bool = True
    note: str | None = Field(None, max_length=2000)


@router.put("/tracks/{track_id}/label")
def put_label(track_id: int, body: LabelBody, project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    given = body.model_fields_set
    kwargs: dict[str, Any] = {
        "verdict": body.verdict if "verdict" in given else UNSET,
        "defect": body.defect if "defect" in given else UNSET,
        "note": body.note if "note" in given else UNSET,
        "reviewed": body.reviewed,
    }
    with _errors(), project.session() as s:
        return lab.set_label(s, track_id, _user(project), **kwargs)


@router.post("/tracks/{track_id}/confirm")
def post_confirm(track_id: int, project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    with _errors(), project.session() as s:
        return lab.confirm(s, track_id, _user(project))


@router.post("/tracks/{track_id}/reopen")
def post_reopen(track_id: int, project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    with _errors(), project.session() as s:
        return lab.reopen(s, track_id, _user(project))


class ViewMarkBody(BaseModel):
    visible: bool | None = None  # absent = inverser


@router.post("/tracks/{track_id}/views/{view_index}/defect-visible")
def post_view_mark(track_id: int, view_index: int, body: ViewMarkBody | None = None, project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    with _errors(), project.session() as s:
        return lab.set_view_defect(s, track_id, view_index, _user(project), None if body is None else body.visible)


class BulkBody(BaseModel):
    track_ids: list[int] = Field(..., min_length=1)
    verdict: str | None = None
    defect: str | None = None
    reviewed: bool = True
    confirm: bool = False  # confirmer le label existant de chaque piste


@router.post("/bulk")
def post_bulk(body: BulkBody, project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    given = body.model_fields_set
    with _errors(), project.session() as s:
        return lab.bulk(
            s, body.track_ids, _user(project),
            verdict=body.verdict if "verdict" in given else UNSET,
            defect=body.defect if "defect" in given else UNSET,
            reviewed=body.reviewed,
            confirm_only=body.confirm,
        )


@router.post("/undo")
def post_undo(project: Project = Depends(get_labeling_project)) -> dict[str, Any]:
    """Annule la dernière action du relecteur courant (une saisie en masse d'un bloc)."""
    with _errors(), project.session() as s:
        return lab.undo_last(s, _user(project))


# -- historique et tableau de bord --------------------------------------------
@router.get("/history")
def get_history(
    track_id: int | None = Query(None),
    user: str | None = Query(None),
    action: str | None = Query(None),
    lot: str | None = Query(None),
    before_id: int | None = Query(None, description="next_before_id de la page précédente"),
    limit: int = Query(50, ge=1, le=500),
    with_total: bool = Query(False),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    with project.session() as s:
        return lab.history_rows(s, track_id=track_id, user=user, action=action, lot_id=lot, before_id=before_id, limit=limit, with_total=with_total)


@router.get("/history/users")
def get_history_users(project: Project = Depends(get_labeling_project)) -> list[str]:
    with project.session() as s:
        return lab.history_users(s)


@router.get("/dashboard")
def get_dashboard(
    min_examples: int = Query(MIN_EXAMPLES, ge=1, le=100_000),
    model_id: str | None = Query(None),
    project: Project = Depends(get_labeling_project),
) -> dict[str, Any]:
    with _errors(), project.session() as s:
        return lab.dashboard(s, min_examples=min_examples, model_id=model_id)
