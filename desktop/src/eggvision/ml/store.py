"""Modèles enregistrés : identifiants, dossiers, points de contrôle, sérialisation.

Dossier d'un modèle ``models/<id>/`` ::

    best.pt          meilleurs poids (sélection sur la validation)
    last.pt          état complet de la dernière époque (reprise : optimiseur, planificateur, historique…)
    anomaly.pt       (modèle « anomaly ») banque mémoire + calibration
    report.json      rapport d'évaluation complet
    cam/             cartes de chaleur (cache)
    package/  *.zip  paquet de déploiement (export)
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

from .. import state
from ..db import models as m
from ..project import Project

CHECKPOINT_FORMAT = 1
MODEL_STATUSES = ("created", "training", "trained", "failed", "cancelled")


def new_model_id(arch: str) -> str:
    short = {"mobilenetv3_small_100": "mnv3s", "mobilenetv3_large_100": "mnv3l", "efficientnet_lite0": "efflite0"}.get(arch, arch[:10])
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{short}-{uuid.uuid4().hex[:4]}"


def next_model_number(session: Any) -> int:
    """Numéro de version lisible (1, 2, 3…) attribué dans l'ordre de création des modèles du projet."""
    numbers = [int((x or {}).get("number", 0)) for x in session.scalars(select(m.Model.extra))]
    return max(numbers, default=0) + 1


def model_dir(project: Project, model_id: str) -> Path:
    d = project.model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_path(project: Project, model_id: str) -> Path:
    return project.model_dir(model_id) / "report.json"


def read_report(project: Project, model_id: str) -> dict[str, Any] | None:
    p = report_path(project, model_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Statut
# ---------------------------------------------------------------------------
def active_job_for_model(model_id: str) -> Any:
    for job in state.jobs.active():
        if (job.params or {}).get("model_id") == model_id:
            return job
    return None


def reconcile_status(row: m.Model) -> bool:
    """Un modèle resté « training » sans tâche active (arrêt brutal) passe « cancelled » (reprise possible)."""
    if row.status == "training" and active_job_for_model(row.id) is None:
        row.status = "cancelled"
        row.extra = {**(row.extra or {}), "interrupted": True}
        return True
    return False


def model_summary(row: m.Model, project: Project | None = None) -> dict[str, Any]:
    extra = row.extra or {}
    export = row.export or {}
    job = active_job_for_model(row.id)
    resumable = False
    if project is not None and row.kind == "classifier" and row.status in ("cancelled", "failed"):
        resumable = (project.model_dir(row.id) / "last.pt").is_file()
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "arch": row.arch,
        "status": row.status,
        "number": extra.get("number"),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "split_id": row.split_id,
        "parent_id": row.parent_id,
        "threshold": row.threshold,
        "n_views": row.n_views,
        "input_size": row.input_size,
        "metrics": row.metrics or {},
        "epochs_done": len(row.history or []),
        "best_epoch": extra.get("best_epoch"),
        "device": extra.get("device"),
        "duration_s": extra.get("duration_s"),
        "warnings": extra.get("warnings", []),
        "pretrained": extra.get("pretrained"),
        "job_id": row.job_id,
        "job_active": job is not None,
        "resumable": resumable,
        "exported": bool(export.get("zip")),
        "export_outdated": bool(export.get("zip")) and abs(float(export.get("threshold", row.threshold)) - row.threshold) > 1e-12,
    }


def model_detail(row: m.Model, project: Project) -> dict[str, Any]:
    d = model_summary(row, project)
    d.update(
        {
            "hparams": row.hparams or {},
            "data_summary": row.data_summary or {},
            "history": row.history or [],
            "export": row.export or {},
            "extra": row.extra or {},
        }
    )
    return d


def get_model_row(session: Any, model_id: str) -> m.Model:
    row = session.get(m.Model, model_id)
    if row is None:
        raise LookupError("Modèle introuvable.")
    return row


def list_models(project: Project, limit: int = 50, offset: int = 0, kind: str | None = None, status: str | None = None) -> dict[str, Any]:
    with project.session() as s:
        q = select(m.Model)
        cq = select(func.count()).select_from(m.Model)
        if kind:
            q, cq = q.where(m.Model.kind == kind), cq.where(m.Model.kind == kind)
        if status:
            q, cq = q.where(m.Model.status == status), cq.where(m.Model.status == status)
        rows = s.scalars(q.order_by(m.Model.created_at.desc()).limit(limit).offset(offset)).all()
        for r in rows:
            reconcile_status(r)
        return {"total": s.scalar(cq) or 0, "items": [model_summary(r, project) for r in rows]}


def delete_model(project: Project, model_id: str) -> None:
    if active_job_for_model(model_id) is not None:
        raise RuntimeError("Une tâche est en cours sur ce modèle : l'annuler d'abord.")
    with project.session() as s:
        row = get_model_row(s, model_id)
        s.execute(delete(m.Prediction).where(m.Prediction.model_id == model_id))
        s.delete(row)
    forget_cached(model_id)
    shutil.rmtree(project.model_dir(model_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# Points de contrôle et chargement
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save({"format": CHECKPOINT_FORMAT, **data}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Point de contrôle introuvable : {path.name}")
    return torch.load(path, map_location="cpu", weights_only=False)


def classifier_from_checkpoint(ckpt: dict[str, Any]):
    from .network import MultiViewClassifier

    model = MultiViewClassifier(ckpt["arch"], aggregation=ckpt.get("aggregation", "max"), dropout=float(ckpt.get("dropout", 0.2)))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


_cache_lock = threading.Lock()
_cache: OrderedDict[tuple[str, str, float], Any] = OrderedDict()


def load_classifier(project: Project, model_id: str, which: str = "best"):
    """Modèle (CPU, mode évaluation) depuis ``best.pt`` / ``last.pt`` ; petit cache LRU (2 modèles)."""
    path = project.model_dir(model_id) / f"{which}.pt"
    key = (str(path), which, path.stat().st_mtime if path.is_file() else 0.0)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    model = classifier_from_checkpoint(load_checkpoint(path))
    with _cache_lock:
        _cache[key] = model
        while len(_cache) > 2:
            _cache.popitem(last=False)
    return model


def forget_cached(model_id: str) -> None:
    with _cache_lock:
        for key in [k for k in _cache if f"/{model_id}/" in k[0].replace("\\", "/")]:
            _cache.pop(key, None)
