"""API du module « Entraînement et évaluation » (référence : docs/api/training.md).

Les modules lourds (PyTorch, timm) sont importés à la demande pour garder un
démarrage rapide de l'application.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import state
from ..device import detect_device
from ..ml import evaluation as ev
from ..ml import metrics as mt
from ..ml import splits as sp
from ..ml import store
from ..project import Project

router = APIRouter(prefix="/api/training", tags=["entraînement"])

Subset = Literal["train", "val", "test"]


@contextmanager
def _errors() -> Iterator[None]:
    """Traduit les exceptions métier en réponses HTTP (messages en français)."""
    from ..ml.training import TrainingError

    try:
        yield
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(404, str(exc).strip("'\"") or "Ressource introuvable.") from exc
    except (sp.SplitError, TrainingError, mt.ObjectiveError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
@router.get("/options")
def options(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Architectures (avec état des poids pré-entraînés), préréglages, choix possibles, appareil."""
    from ..ml.anomaly import AnomalyParams
    from ..ml.network import AGGREGATIONS, ARCHITECTURES, pretrained_status
    from ..ml.training import IMBALANCE, MONITORS, TrainParams, presets_payload

    return {
        "architectures": [{"id": k, **v, "pretrained": pretrained_status(k)} for k, v in ARCHITECTURES.items()],
        "presets": presets_payload(),
        "defaults": TrainParams().to_dict(),
        "anomaly_defaults": AnomalyParams().__dict__,
        "choices": {
            "aggregation": list(AGGREGATIONS),
            "imbalance": list(IMBALANCE),
            "monitor": list(MONITORS),
            "augment": ["none", "light", "standard"],
            "device": ["auto", "cuda", "mps", "cpu"],
        },
        "device": detect_device(),
        "split": sp.split_options(project),
    }


# ---------------------------------------------------------------------------
# Découpages
# ---------------------------------------------------------------------------
class SplitParamsBody(BaseModel):
    group_by: Literal["session", "lot"] = "session"
    ratios: list[float] = Field(default_factory=lambda: [0.70, 0.15, 0.15], min_length=3, max_length=3)
    holdout_farms: list[str] = Field(default_factory=list)
    holdout_dates: list[str] = Field(default_factory=list)
    seed: int = 0
    reviewed_only: bool = False
    exclude_lots: list[str] = Field(default_factory=list)

    def to_params(self) -> sp.SplitParams:
        return sp.SplitParams.from_dict(self.model_dump())


class SplitCreateBody(SplitParamsBody):
    name: str = "Découpage"


class AssignmentsBody(BaseModel):
    changes: dict[str, Subset | None]
    move_group: bool = True


@router.get("/splits")
def list_splits(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), project: Project = Depends(state.get_project)):
    return sp.list_splits(project, limit, offset)


@router.post("/splits/preview")
def preview_split(body: SplitParamsBody, project: Project = Depends(state.get_project)):
    with _errors():
        return sp.preview_split(project, body.to_params())


@router.post("/splits")
def create_split(body: SplitCreateBody, project: Project = Depends(state.get_project)):
    with _errors():
        return sp.create_split(project, body.name, body.to_params())


@router.get("/splits/{split_id}")
def get_split(split_id: int, project: Project = Depends(state.get_project)):
    with _errors():
        return sp.split_detail(project, split_id)


@router.patch("/splits/{split_id}/assignments")
def patch_assignments(split_id: int, body: AssignmentsBody, project: Project = Depends(state.get_project)):
    with _errors():
        return sp.update_assignments(project, split_id, dict(body.changes), body.move_group)


@router.delete("/splits/{split_id}")
def delete_split(split_id: int, project: Project = Depends(state.get_project)):
    with _errors():
        sp.delete_split(project, split_id)
    return {"deleted": split_id}


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
class TrainBody(BaseModel):
    split_id: int
    preset: Literal["rapide", "standard", "int8"] | None = "standard"
    hparams: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class AnomalyBody(BaseModel):
    split_id: int
    params: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class ResumeBody(BaseModel):
    mode: Literal["continue", "extend"] = "continue"
    epochs: int | None = Field(None, ge=1, le=1000)
    hparams: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


@router.post("/train")
def train(body: TrainBody, project: Project = Depends(state.get_project)):
    from ..ml.training import start_training

    with _errors():
        return start_training(project, body.split_id, body.preset, body.hparams, body.name)


@router.post("/anomaly")
def anomaly(body: AnomalyBody, project: Project = Depends(state.get_project)):
    from ..ml.anomaly import start_anomaly

    with _errors():
        return start_anomaly(project, body.split_id, body.params, body.name)


@router.post("/models/{model_id}/resume")
def resume(model_id: str, body: ResumeBody, project: Project = Depends(state.get_project)):
    from ..ml.training import resume_training

    with _errors():
        return resume_training(project, model_id, body.mode, body.epochs, body.hparams, body.name)


@router.post("/models/{model_id}/cancel")
def cancel(model_id: str, project: Project = Depends(state.get_project)):
    job = store.active_job_for_model(model_id)
    if job is None:
        raise HTTPException(409, "Aucune tâche en cours pour ce modèle.")
    state.jobs.cancel(job.id)
    return {"job_id": job.id, "status": "cancelling"}


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
class RenameBody(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@router.get("/models")
def list_models(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    kind: str | None = None,
    status: str | None = None,
    project: Project = Depends(state.get_project),
):
    return store.list_models(project, limit, offset, kind, status)


@router.get("/models/{model_id}")
def get_model(model_id: str, project: Project = Depends(state.get_project)):
    with _errors(), project.session() as s:
        row = store.get_model_row(s, model_id)
        store.reconcile_status(row)
        return store.model_detail(row, project)


@router.patch("/models/{model_id}")
def rename_model(model_id: str, body: RenameBody, project: Project = Depends(state.get_project)):
    with _errors(), project.session() as s:
        row = store.get_model_row(s, model_id)
        row.name = body.name.strip()
        return store.model_summary(row, project)


@router.delete("/models/{model_id}")
def delete_model(model_id: str, project: Project = Depends(state.get_project)):
    with _errors():
        store.delete_model(project, model_id)
    return {"deleted": model_id}


@router.get("/models/{model_id}/history")
def history(model_id: str, project: Project = Depends(state.get_project)):
    """Courbes d'apprentissage (en direct pendant l'entraînement)."""
    with _errors(), project.session() as s:
        row = store.get_model_row(s, model_id)
        hist = list(row.history or [])
        status = row.status
        epochs = (row.hparams or {}).get("epochs")
    job = store.active_job_for_model(model_id)
    if job is not None and job.result and len(job.result.get("history") or []) >= len(hist):
        hist = job.result["history"]
    return {
        "model_id": model_id,
        "status": status,
        "epochs": epochs,
        "history": hist,
        "job": job.to_dict() if job is not None else None,
    }


@router.get("/models/{model_id}/report")
def report(model_id: str, project: Project = Depends(state.get_project)):
    rep = store.read_report(project, model_id)
    if rep is None:
        raise HTTPException(404, "Rapport indisponible : le modèle n'a pas encore été évalué.")
    return rep


@router.post("/models/{model_id}/report/refresh")
def refresh(model_id: str, project: Project = Depends(state.get_project)):
    """Recalcule le rapport depuis les prédictions et les labels actuels (après relecture)."""
    with _errors():
        return ev.refresh_report(project, model_id)


class SuggestBody(BaseModel):
    objective: Literal["recall", "false_reject"]
    target: float = Field(ge=0.0, le=1.0)
    subset: Literal["val", "test"] = "val"


class ThresholdBody(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


@router.post("/models/{model_id}/threshold/suggest")
def suggest(model_id: str, body: SuggestBody, project: Project = Depends(state.get_project)):
    with _errors():
        return ev.suggest_threshold(project, model_id, body.objective, body.target, body.subset)


@router.get("/models/{model_id}/threshold/evaluate")
def evaluate_threshold(model_id: str, threshold: float = Query(..., ge=0.0, le=1.0), project: Project = Depends(state.get_project)):
    """Métriques val / test à un seuil donné (curseur de seuil), sans modifier le modèle."""
    with _errors(), project.session() as s:
        store.get_model_row(s, model_id)
    with _errors():
        return ev.evaluate_threshold(project, model_id, threshold)


@router.put("/models/{model_id}/threshold")
def set_threshold(model_id: str, body: ThresholdBody, project: Project = Depends(state.get_project)):
    with _errors():
        rep = ev.set_threshold(project, model_id, body.threshold)
        with project.session() as s:
            row = store.get_model_row(s, model_id)
            return {"threshold": row.threshold, "metrics": row.metrics, "export_outdated": store.model_summary(row)["export_outdated"], "report_generated_at": rep["generated_at"]}


@router.get("/models/{model_id}/errors")
def errors(
    model_id: str,
    kind: Literal["fn", "fp"] = "fn",
    subset: Subset = "test",
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    project: Project = Depends(state.get_project),
):
    with _errors():
        return ev.list_errors(project, model_id, kind, subset, threshold, limit, offset)


@router.get("/models/{model_id}/predictions")
def predictions(
    model_id: str,
    subset: Subset | None = None,
    order: Literal["uncertain", "score_desc", "track"] = "uncertain",
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    project: Project = Depends(state.get_project),
):
    with _errors():
        return ev.list_predictions(project, model_id, subset, limit, offset, order)


@router.get("/models/{model_id}/gradcam/{track_id}")
def gradcam_meta(model_id: str, track_id: int, project: Project = Depends(state.get_project)):
    """Cartes de chaleur de toutes les vues d'une piste (calculées au premier appel, puis en cache)."""
    from ..ml.gradcam import track_cams

    with _errors():
        meta = track_cams(project, model_id, track_id)
    return {
        **meta,
        "views": [{**v, "url": f"/api/training/models/{model_id}/gradcam/{track_id}/{v['index']}"} for v in meta["views"]],
    }


@router.get("/models/{model_id}/gradcam/{track_id}/{view_index}")
def gradcam_image(model_id: str, track_id: int, view_index: int, project: Project = Depends(state.get_project)):
    from ..ml.gradcam import cam_image_path

    with _errors():
        path = cam_image_path(project, model_id, track_id, view_index)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})


@router.get("/compare")
def compare(
    ids: list[str] = Query(..., min_length=1, max_length=10),
    objective: Literal["recall", "false_reject"] = "recall",
    target: float = Query(0.95, ge=0.0, le=1.0),
    subset: Literal["val", "test"] = "val",
    project: Project = Depends(state.get_project),
):
    with _errors():
        return ev.compare_models(project, ids, objective, target, subset)
