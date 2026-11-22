"""Évaluation : prédictions, rapport, seuil par objectif, explorateur d'erreurs, comparaison.

Les prédictions de toutes les pistes utilisables du découpage sont écrites dans
la table ``Prediction`` (avec leur sous-ensemble et les scores par vue) : elles
alimentent l'apprentissage actif d'un autre module. Le rapport est ensuite
**recalculé à partir de ces prédictions et des labels actuels** : après une
relecture, ``refresh_report`` le met à jour sans réévaluer le modèle.

Convention de décision : éjecter si ``score >= seuil`` (comme ``EggClassifier``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from eggvision_shared.constants import VERDICT_NON_VIABLE, VERDICT_VIABLE
from eggvision_shared.preprocess import PreprocessConfig
from sqlalchemy import and_, delete, func, select

from ..db import models as m
from ..project import Project
from . import metrics as mt
from .dataset import EggRecord, EggViewsDataset, ViewCache, pick_view_indices
from .store import get_model_row, read_report, report_path, write_json

REPORT_FORMAT = 1
EVAL_SUBSETS = ("val", "test")
DEFAULT_SUGGESTIONS = (("recall", 0.95), ("recall", 0.99), ("false_reject", 0.02), ("false_reject", 0.05))


def _noop(*_: Any, **__: Any) -> None:
    return None


# ---------------------------------------------------------------------------
# Inférence PyTorch sur des œufs du projet
# ---------------------------------------------------------------------------
def predict_records(
    model: Any,
    records: Sequence[EggRecord],
    n_views: int,
    input_size: int,
    device: Any,
    batch_size: int = 16,
    num_workers: int = 0,
    check_cancelled: Callable[[], None] = _noop,
    progress: Callable[[float], None] = _noop,
    cache: ViewCache | None = None,
) -> tuple[np.ndarray, list[list[float]]]:
    """Scores (œuf, vues) de chaque enregistrement, vues choisies comme au déploiement.

    Les scores par vue renvoyés correspondent aux vues ``pick_view_indices(n, n_views)``,
    sans les répétitions de complément.
    """
    import torch
    from torch.utils.data import DataLoader

    ds = EggViewsDataset(records, n_views, PreprocessConfig(input_size=input_size), augment=None, cache=cache)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores = np.zeros(len(records), dtype=np.float64)
    views: list[list[float]] = [[] for _ in records]
    model.eval()
    done = 0
    with torch.no_grad():
        for x, _, idx in loader:
            check_cancelled()
            egg, vs = model(x.to(device))
            egg = egg.float().cpu().numpy()
            vs = vs.float().cpu().numpy()
            for k, i in enumerate(idx.tolist()):
                scores[i] = float(egg[k])
                n_unique = min(len(records[i].view_paths), n_views)
                views[i] = [float(v) for v in vs[k][:n_unique]]
            done += len(idx)
            progress(done / max(1, len(records)))
    return scores, views


def write_predictions(project: Project, model_id: str, records: Sequence[EggRecord], scores: np.ndarray, view_scores: list[list[float]]) -> int:
    """Remplace les prédictions du modèle (écriture par paquets)."""
    with project.session() as s:
        s.execute(delete(m.Prediction).where(m.Prediction.model_id == model_id))
    rows = [
        {
            "model_id": model_id,
            "track_id": r.track_id,
            "subset": r.subset,
            "score": float(scores[i]),
            "view_scores": [round(v, 6) for v in view_scores[i]],
        }
        for i, r in enumerate(records)
    ]
    for i in range(0, len(rows), 2000):
        with project.session() as s:
            s.execute(m.Prediction.__table__.insert(), rows[i : i + 2000])
    return len(rows)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def load_labeled_predictions(project: Project, model_id: str, subsets: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    """Prédictions jointes aux labels ACTUELS (œufs utilisables uniquement), par sous-ensemble."""
    q = (
        select(m.Prediction.subset, m.Prediction.track_id, m.Prediction.score, m.Label.verdict, m.Label.defect)
        .join(m.Label, m.Label.track_id == m.Prediction.track_id)
        .where(m.Prediction.model_id == model_id, m.Label.verdict.in_((VERDICT_VIABLE, VERDICT_NON_VIABLE)))
    )
    if subsets:
        q = q.where(m.Prediction.subset.in_(list(subsets)))
    out: dict[str, dict[str, list]] = {}
    with project.session() as s:
        for subset, tid, score, verdict, defect in s.execute(q.order_by(m.Prediction.track_id)):
            e = out.setdefault(subset, {"track_ids": [], "y": [], "s": [], "defects": []})
            e["track_ids"].append(int(tid))
            e["y"].append(1 if verdict == VERDICT_NON_VIABLE else 0)
            e["s"].append(float(score))
            e["defects"].append(defect if verdict == VERDICT_NON_VIABLE else None)
    return {
        k: {"track_ids": np.asarray(v["track_ids"]), "y": np.asarray(v["y"], dtype=np.int64), "s": np.asarray(v["s"]), "defects": v["defects"]}
        for k, v in out.items()
    }


def tracks_fingerprint(track_ids: np.ndarray) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in sorted(track_ids.tolist())).encode()).hexdigest()[:16]


def _suggestions(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for objective, target in DEFAULT_SUGGESTIONS:
        try:
            out.append(suggest_from_data(data, objective, target, "val" if "val" in data else "test"))
        except mt.ObjectiveError as exc:
            out.append({"objective": objective, "target": target, "error": str(exc)})
    return out


def build_report(project: Project, model_id: str, threshold: float) -> dict[str, Any]:
    data = load_labeled_predictions(project, model_id)
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "model_id": model_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": float(threshold),
        "decision_rule": "éjecter si score ≥ seuil",
        "subsets": {},
    }
    for sub in ("train", *EVAL_SUBSETS):
        if sub not in data:
            continue
        d = data[sub]
        block = mt.binary_report(d["y"], d["s"], threshold, defects=d["defects"])
        if sub == "train":  # courbes inutiles pour l'entraînement : résumé seulement
            block = {k: block[k] for k in ("n", "n_pos", "n_neg", "auc_roc", "average_precision", "at_threshold")}
        report["subsets"][sub] = block
    if "test" in data:
        report["test_set"] = {"n": int(len(data["test"]["y"])), "fingerprint": tracks_fingerprint(data["test"]["track_ids"])}
    report["threshold_suggestions"] = _suggestions(data) if any(k in data for k in EVAL_SUBSETS) else []
    return report


def metrics_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Résumé stocké dans ``Model.metrics`` (et repris dans le paquet de déploiement)."""
    out: dict[str, Any] = {"threshold": report.get("threshold"), "generated_at": report.get("generated_at")}
    for sub, block in (report.get("subsets") or {}).items():
        at = block.get("at_threshold") or {}
        e = {
            "n": block.get("n"),
            "n_pos": block.get("n_pos"),
            "n_neg": block.get("n_neg"),
            "auc_roc": _r(block.get("auc_roc")),
            "average_precision": _r(block.get("average_precision")),
            "recall": _r(at.get("recall")),
            "false_reject_rate": _r(at.get("false_reject_rate")),
            "precision": _r(at.get("precision")),
            "tp": at.get("tp"),
            "fn": at.get("fn"),
            "fp": at.get("fp"),
            "tn": at.get("tn"),
        }
        if sub == "test" and block.get("recall_by_defect"):
            e["recall_by_defect"] = {k: _r(v.get("recall")) for k, v in block["recall_by_defect"].items()}
        out[sub] = e
    if report.get("test_set"):
        out["test_set"] = report["test_set"]
    return out


def _r(x: Any, nd: int = 4) -> Any:
    return None if x is None else round(float(x), nd)


def refresh_report(project: Project, model_id: str, threshold: float | None = None) -> dict[str, Any]:
    """Recalcule le rapport depuis les prédictions et les labels actuels ; met à jour ``Model.metrics``."""
    with project.session() as s:
        row = get_model_row(s, model_id)
        thr = float(row.threshold if threshold is None else threshold)
    report = build_report(project, model_id, thr)
    previous = read_report(project, model_id) or {}
    for key in ("training", "anomaly", "model"):
        if key in previous and key not in report:
            report[key] = previous[key]
    write_json(report_path(project, model_id), report)
    with project.session() as s:
        row = get_model_row(s, model_id)
        row.threshold = thr
        row.metrics = metrics_summary(report)
    return report


# ---------------------------------------------------------------------------
# Seuil par objectif
# ---------------------------------------------------------------------------
def suggest_from_data(data: dict[str, dict[str, Any]], objective: str, target: float, subset: str) -> dict[str, Any]:
    if subset not in data:
        raise mt.ObjectiveError(f"Aucune prédiction sur le jeu « {subset} ».")
    d = data[subset]
    thr = mt.threshold_for_objective(d["y"], d["s"], objective, float(target))
    out: dict[str, Any] = {"objective": objective, "target": float(target), "computed_on": subset, "threshold": thr}
    for sub in EVAL_SUBSETS:
        if sub in data:
            c = mt.confusion(data[sub]["y"], data[sub]["s"], thr)
            c["objective_met"] = mt.objective_met(c, objective, float(target))
            out[sub] = c
    return out


def suggest_threshold(project: Project, model_id: str, objective: str, target: float, subset: str = "val") -> dict[str, Any]:
    if subset not in EVAL_SUBSETS:
        raise mt.ObjectiveError("Le seuil se calcule sur la validation (« val ») ou le test (« test »).")
    return suggest_from_data(load_labeled_predictions(project, model_id, EVAL_SUBSETS), objective, target, subset)


def evaluate_threshold(project: Project, model_id: str, threshold: float) -> dict[str, Any]:
    """Matrice de confusion (val, test) et rappel par défaut à un seuil quelconque, sans rien modifier."""
    data = load_labeled_predictions(project, model_id, EVAL_SUBSETS)
    out: dict[str, Any] = {"threshold": float(threshold)}
    for sub in EVAL_SUBSETS:
        if sub in data:
            d = data[sub]
            out[sub] = {**mt.confusion(d["y"], d["s"], threshold), "recall_by_defect": mt.recall_by_defect(d["y"], d["s"], d["defects"], threshold)}
    return out


def set_threshold(project: Project, model_id: str, threshold: float) -> dict[str, Any]:
    if not 0.0 <= float(threshold) <= 1.0 + 1e-6:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")
    return refresh_report(project, model_id, float(threshold))


# ---------------------------------------------------------------------------
# Explorateur d'erreurs
# ---------------------------------------------------------------------------
def list_errors(
    project: Project,
    model_id: str,
    kind: str = "fn",
    subset: str = "test",
    threshold: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Faux négatifs (non viables laissés passer, score le plus bas d'abord) ou faux positifs
    (viables éjectés, score le plus haut d'abord), paginés."""
    if kind not in ("fn", "fp"):
        raise ValueError("Type d'erreur inconnu : « fn » (faux négatifs) ou « fp » (faux positifs).")
    with project.session() as s:
        row = get_model_row(s, model_id)
        thr = float(row.threshold if threshold is None else threshold)
        n_views = row.n_views
        cond = [m.Prediction.model_id == model_id]
        if subset:
            cond.append(m.Prediction.subset == subset)
        if kind == "fn":
            cond += [m.Label.verdict == VERDICT_NON_VIABLE, m.Prediction.score < thr]
            order = (m.Prediction.score.asc(), m.Prediction.track_id)
        else:
            cond += [m.Label.verdict == VERDICT_VIABLE, m.Prediction.score >= thr]
            order = (m.Prediction.score.desc(), m.Prediction.track_id)
        base = select(m.Prediction, m.Label, m.Track, m.Lot).join(m.Label, m.Label.track_id == m.Prediction.track_id)
        base = base.join(m.Track, m.Track.id == m.Prediction.track_id).join(m.Lot, m.Lot.id == m.Track.lot_id)
        total = s.scalar(
            select(func.count()).select_from(m.Prediction).join(m.Label, m.Label.track_id == m.Prediction.track_id).where(and_(*cond))
        )
        items = []
        for pred, label, track, lot in s.execute(base.where(and_(*cond)).order_by(*order).limit(limit).offset(offset)):
            n_avail = track.n_views or len(pred.view_scores or [])
            idx = pick_view_indices(max(1, n_avail), n_views)[: len(pred.view_scores or [])]
            items.append(
                {
                    "track_id": track.id,
                    "lot_id": lot.id,
                    "lot_name": lot.name,
                    "lane": track.lane,
                    "subset": pred.subset,
                    "score": pred.score,
                    "severity": (thr - pred.score) if kind == "fn" else (pred.score - thr),
                    "verdict": label.verdict,
                    "defect": label.defect,
                    "reviewed": label.reviewed,
                    "n_views": track.n_views,
                    "views": [{"index": int(i), "score": float(v)} for i, v in zip(idx, pred.view_scores or [], strict=False)],
                }
            )
    return {"kind": kind, "subset": subset, "threshold": thr, "total": int(total or 0), "limit": limit, "offset": offset, "items": items}


def list_predictions(project: Project, model_id: str, subset: str | None = None, limit: int = 100, offset: int = 0, order: str = "uncertain") -> dict[str, Any]:
    """Prédictions paginées (``order`` : ``uncertain`` = plus proche du seuil, ``score_desc``, ``track``)."""
    with project.session() as s:
        row = get_model_row(s, model_id)
        thr = row.threshold
        q = select(m.Prediction).where(m.Prediction.model_id == model_id)
        cq = select(func.count()).select_from(m.Prediction).where(m.Prediction.model_id == model_id)
        if subset is not None:
            q, cq = q.where(m.Prediction.subset == subset), cq.where(m.Prediction.subset == subset)
        if order == "uncertain":
            q = q.order_by(func.abs(m.Prediction.score - thr), m.Prediction.track_id)
        elif order == "score_desc":
            q = q.order_by(m.Prediction.score.desc(), m.Prediction.track_id)
        else:
            q = q.order_by(m.Prediction.track_id)
        rows = s.scalars(q.limit(limit).offset(offset)).all()
        return {
            "total": int(s.scalar(cq) or 0),
            "threshold": thr,
            "items": [{"track_id": r.track_id, "subset": r.subset, "score": r.score, "view_scores": r.view_scores} for r in rows],
        }


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------
def compare_models(project: Project, model_ids: Sequence[str], objective: str = "recall", target: float = 0.95, subset: str = "val") -> dict[str, Any]:
    """Métriques test côte à côte : au seuil propre de chaque modèle et à un objectif commun."""
    rows_out = []
    fingerprints = set()
    for mid in model_ids:
        with project.session() as s:
            row = get_model_row(s, mid)
            info = {"id": row.id, "name": row.name, "kind": row.kind, "arch": row.arch, "split_id": row.split_id, "number": (row.extra or {}).get("number"), "threshold": row.threshold, "status": row.status}
        data = load_labeled_predictions(project, mid, EVAL_SUBSETS)
        entry: dict[str, Any] = {**info}
        if "test" not in data:
            entry["error"] = "Aucune prédiction sur le jeu de test."
            rows_out.append(entry)
            continue
        t = data["test"]
        fp = tracks_fingerprint(t["track_ids"])
        fingerprints.add(fp)
        entry["test_set"] = {"n": int(len(t["y"])), "n_pos": int(t["y"].sum()), "fingerprint": fp}
        entry["auc_roc"] = mt.roc_auc(t["y"], t["s"])
        entry["average_precision"] = mt.average_precision(t["y"], t["s"])
        entry["own_threshold"] = mt.confusion(t["y"], t["s"], info["threshold"])
        entry["recall_by_defect"] = mt.recall_by_defect(t["y"], t["s"], t["defects"], info["threshold"])
        try:
            sub = subset if subset in data else "test"
            sug = suggest_from_data(data, objective, target, sub)
            entry["common_objective"] = {"threshold": sug["threshold"], "computed_on": sub, "test": sug.get("test")}
        except mt.ObjectiveError as exc:
            entry["common_objective"] = {"error": str(exc)}
        rows_out.append(entry)
    split_ids = {r.get("split_id") for r in rows_out}
    warnings = []
    if len(fingerprints) > 1:
        warnings.append("Les jeux de test diffèrent entre ces modèles : la comparaison n'est qu'indicative.")
    if len(split_ids) > 1:
        warnings.append("Ces modèles n'utilisent pas le même découpage.")
    return {
        "objective": {"objective": objective, "target": target, "computed_on": subset},
        "same_test_set": len(fingerprints) <= 1,
        "models": rows_out,
        "warnings": warnings,
    }
