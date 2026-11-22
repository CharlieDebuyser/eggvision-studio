"""Apprentissage actif : quelles pistes relire en priorité une fois un modèle entraîné ?

Le score d'un modèle (:class:`~eggvision.db.models.Prediction`) est la probabilité
de défaut de l'œuf, comparée au seuil du modèle (``Model.threshold``).

Priorité d'une piste (plus grand = plus urgent) ::

    contradiction  label viable et score ≥ seuil      → c = (score − seuil) / (1 − seuil)
                   label non viable et score < seuil  → c = (seuil − score) / seuil
    hésitation     |score − seuil| < marge             → h = 1 − |score − seuil| / marge

    base     = 1 + c   si contradiction   (] 1 ; 2 ])
             = h       sinon, si hésitation (] 0 ; 1 ])
             = 0       sinon (modèle confiant et d'accord)
    priorité = base + 2 si la piste n'est pas encore relue

Les pistes non relues passent donc toujours en premier ; parmi elles, les
contradictions (d'autant plus prioritaires que l'écart est grand) précèdent
les hésitations (d'autant plus prioritaires que le score est proche du seuil).
Les pistes exclues ne sont jamais proposées.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import and_, case, exists, func, literal, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..db import models as m
from .common import NotFoundError, iso

DEFAULT_MARGIN = 0.15
REASON_CONTRADICTION = "contredit"
REASON_HESITATION = "hesite"
REASON_LABELS = {REASON_CONTRADICTION: "Contredit le label", REASON_HESITATION: "Hésite"}
UNREVIEWED_BONUS = 2.0


@dataclass(frozen=True)
class ActiveModel:
    id: str
    name: str
    threshold: float
    created_at: str | None

    @property
    def t(self) -> float:
        """Seuil borné (évite les divisions par zéro)."""
        return min(0.99, max(0.01, float(self.threshold)))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "threshold": self.threshold, "created_at": self.created_at}


def list_models(session: Session) -> list[dict[str, Any]]:
    """Modèles entraînés ayant des prédictions (du plus récent au plus ancien)."""
    counts = (
        select(m.Prediction.model_id, func.count().label("n"))
        .group_by(m.Prediction.model_id)
        .subquery()
    )
    rows = session.execute(
        select(m.Model.id, m.Model.name, m.Model.threshold, m.Model.created_at, m.Model.status, counts.c.n)
        .join(counts, counts.c.model_id == m.Model.id)
        .order_by(m.Model.created_at.desc())
        .limit(200)
    ).all()
    return [
        {"id": r.id, "name": r.name, "threshold": r.threshold, "created_at": iso(r.created_at), "status": r.status, "n_predictions": r.n}
        for r in rows
    ]


def resolve_model(session: Session, model_id: str | None) -> ActiveModel | None:
    """Modèle demandé, ou par défaut le dernier modèle ``trained`` ayant des prédictions.

    ``model_id`` vaut ``None``/``""``/``"auto"`` pour le choix par défaut, ``"aucun"`` pour aucun modèle.
    """
    if model_id in ("aucun", "none"):
        return None
    if model_id and model_id != "auto":
        row = session.get(m.Model, model_id)
        if row is None:
            raise NotFoundError(f"Modèle « {model_id} » introuvable.")
        return ActiveModel(row.id, row.name, float(row.threshold), iso(row.created_at))
    has_pred = exists().where(m.Prediction.model_id == m.Model.id)
    row = session.execute(
        select(m.Model.id, m.Model.name, m.Model.threshold, m.Model.created_at)
        .where(m.Model.status == "trained", has_pred)
        .order_by(m.Model.created_at.desc(), m.Model.id.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return ActiveModel(row.id, row.name, float(row.threshold), iso(row.created_at))


# -- version SQL (tri et file) -----------------------------------------------
@dataclass
class PriorityExprs:
    priority: ColumnElement[Any]
    reason: ColumnElement[Any]
    in_queue: ColumnElement[Any]


def priority_exprs(model: ActiveModel, margin: float = DEFAULT_MARGIN) -> PriorityExprs:
    """Expressions SQL de priorité et de raison (requête jointe sur ``labels`` et ``predictions``)."""
    # Expressions « à plat » (un seul CASE chacune) : SQLite ne factorise pas les sous-expressions,
    # et la file est évaluée sur toutes les prédictions du modèle.
    tv = model.t
    margin = max(1e-3, float(margin))
    t = literal(tv)
    s = m.Prediction.score
    verdict = m.Label.verdict
    is_contra_v = and_(verdict == "viable", s >= t)
    is_contra_nv = and_(verdict == "non_viable", s < t)
    is_hesit = and_(s > tv - margin, s < tv + margin)
    excluded = verdict == "exclu"
    bonus = case((m.Label.reviewed.is_(True), 0.0), else_=UNREVIEWED_BONUS)
    priority = case(
        (s.is_(None), -1.0),
        (excluded, -1.0),
        (is_contra_v, 1.0 + (s - t) / (1.0 - tv) + bonus),
        (is_contra_nv, 1.0 + (t - s) / tv + bonus),
        (is_hesit, 1.0 - func.abs(s - t) / margin + bonus),
        else_=bonus,
    )
    reason = case(
        (s.is_(None), None),
        (excluded, None),
        (or_(is_contra_v, is_contra_nv), REASON_CONTRADICTION),
        (is_hesit, REASON_HESITATION),
        else_=None,
    )
    in_queue = and_(
        s.is_not(None),
        or_(verdict.is_(None), verdict != "exclu"),
        or_(is_hesit, is_contra_v, is_contra_nv),
    )
    return PriorityExprs(priority=priority, reason=reason, in_queue=in_queue)


# -- version Python (détail d'une piste, tests) ------------------------------
def evaluate(score: float | None, verdict: str | None, reviewed: bool, model: ActiveModel, margin: float = DEFAULT_MARGIN) -> dict[str, Any]:
    """Même calcul que :func:`priority_exprs`, pour une piste isolée."""
    if score is None:
        return {"priority": -1.0, "reason": None, "predicted": None}
    t = model.t
    predicted = "non_viable" if score >= model.t else "viable"
    if verdict == "exclu":
        return {"priority": -1.0, "reason": None, "predicted": predicted}
    contra = None
    if verdict == "viable" and score >= t:
        contra = (score - t) / (1.0 - t)
    elif verdict == "non_viable" and score < t:
        contra = (t - score) / t
    margin = max(1e-3, float(margin))
    dist = abs(score - t)
    hesit = 1.0 - dist / margin if dist < margin else None
    if contra is not None:
        base, reason = 1.0 + contra, REASON_CONTRADICTION
    elif hesit is not None:
        base, reason = hesit, REASON_HESITATION
    else:
        base, reason = 0.0, None
    return {"priority": base + (0.0 if reviewed else UNREVIEWED_BONUS), "reason": reason, "predicted": predicted}


def queue_summary(session: Session, model: ActiveModel, margin: float = DEFAULT_MARGIN) -> dict[str, int]:
    """Effectifs de la file d'apprentissage actif (requête agrégée)."""
    ex = priority_exprs(model, margin)
    reviewed = func.coalesce(m.Label.reviewed, False)
    rows = session.execute(
        select(ex.reason, reviewed, func.count())
        .select_from(m.Prediction)
        .outerjoin(m.Label, m.Label.track_id == m.Prediction.track_id)
        .where(m.Prediction.model_id == model.id, ex.in_queue)
        .group_by(ex.reason, reviewed)
    ).all()
    out = {"contradictions": 0, "hesitations": 0, "unreviewed": 0, "total": 0}
    for reason, rev, n in rows:
        out["total"] += n
        if reason == REASON_CONTRADICTION:
            out["contradictions"] += n
        elif reason == REASON_HESITATION:
            out["hesitations"] += n
        if not rev:
            out["unreviewed"] += n
    return out
