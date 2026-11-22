"""Recherche de pistes (filtres combinables, tris, pagination par clé) et détail d'une piste."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from eggvision_shared.constants import DEFECT_TYPES, VERDICTS
from sqlalchemy import Select, and_, false, func, or_, select, text, tuple_
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ..db import models as m
from . import active
from .common import MAX_LIMIT, NONE_KEY, LabelingError, NotFoundError, iso, parse_iso, prelabel_of

SORTS = ("naturel", "recent", "priorite")
QUALITIES = ("ok", "douteuse")


@dataclass
class TrackFilters:
    """Filtres combinables (ET entre critères, OU entre valeurs d'un même critère)."""

    lots: list[str] = field(default_factory=list)
    lot_types: list[str] = field(default_factory=list)  # « declasse » couvre « declasse:<défaut> »
    verdicts: list[str] = field(default_factory=list)  # + « aucun » : sans label
    defects: list[str] = field(default_factory=list)  # + « aucun » : non viable sans type
    reviewed: bool | None = None
    quality: list[str] = field(default_factory=list)
    lanes: list[int] = field(default_factory=list)
    source: str | None = None  # auto | humain
    score_min: float | None = None
    score_max: float | None = None
    reason: str | None = None  # contredit | hesite (avec modèle) — file d'apprentissage actif
    track_ids: list[int] = field(default_factory=list)

    def validate(self) -> None:
        for v in self.verdicts:
            if v != NONE_KEY and v not in VERDICTS:
                raise LabelingError(f"Verdict inconnu : « {v} ».")
        for d in self.defects:
            if d != NONE_KEY and d not in DEFECT_TYPES:
                raise LabelingError(f"Type de défaut inconnu : « {d} ».")
        for q in self.quality:
            if q not in QUALITIES:
                raise LabelingError(f"Qualité inconnue : « {q} » (attendu : ok, douteuse).")
        if self.source not in (None, "auto", "humain"):
            raise LabelingError(f"Source inconnue : « {self.source} » (attendu : auto, humain).")
        for bound in (self.score_min, self.score_max):
            if bound is not None and not 0.0 <= bound <= 1.0:
                raise LabelingError("La plage de confiance doit être comprise entre 0 et 1.")
        if self.score_min is not None and self.score_max is not None and self.score_min > self.score_max:
            raise LabelingError("Plage de confiance vide : minimum supérieur au maximum.")
        if self.reason not in (None, active.REASON_CONTRADICTION, active.REASON_HESITATION, "file"):
            raise LabelingError(f"Raison inconnue : « {self.reason} » (attendu : contredit, hesite, file).")

    @property
    def needs_prediction(self) -> bool:
        return self.score_min is not None or self.score_max is not None or self.reason is not None


def _conditions(f: TrackFilters, model: active.ActiveModel | None, margin: float) -> list[ColumnElement[Any]]:
    T, L, LO, P = m.Track, m.Label, m.Lot, m.Prediction
    cond: list[ColumnElement[Any]] = []
    if f.track_ids:
        cond.append(T.id.in_(f.track_ids))
    if f.lots:
        cond.append(T.lot_id.in_(f.lots))
    if f.lot_types:
        cond.append(or_(*[or_(LO.type == t, LO.type.like(f"{t}:%")) for t in f.lot_types]))
    if f.verdicts:
        named = [v for v in f.verdicts if v != NONE_KEY]
        parts: list[ColumnElement[Any]] = []
        if named:
            parts.append(L.verdict.in_(named))
        if NONE_KEY in f.verdicts:
            parts.append(L.verdict.is_(None))
        cond.append(or_(*parts))
    if f.defects:
        named = [d for d in f.defects if d != NONE_KEY]
        parts = []
        if named:
            parts.append(L.defect.in_(named))
        if NONE_KEY in f.defects:
            parts.append(and_(L.verdict == "non_viable", L.defect.is_(None)))
        cond.append(or_(*parts))
    if f.reviewed is True:
        cond.append(L.reviewed.is_(True))
    elif f.reviewed is False:
        cond.append(or_(L.reviewed.is_(False), L.reviewed.is_(None)))
    if f.quality:
        cond.append(T.quality.in_(f.quality))
    if f.lanes:
        cond.append(T.lane.in_(f.lanes))
    if f.source:
        cond.append(L.source == f.source)
    if f.needs_prediction:
        if model is None:
            cond.append(false())
        else:
            if f.score_min is not None:
                cond.append(P.score >= f.score_min)
            if f.score_max is not None:
                cond.append(P.score <= f.score_max)
            if f.reason is not None:
                ex = active.priority_exprs(model, margin)
                cond.append(ex.in_queue)
                if f.reason != "file":
                    cond.append(ex.reason == f.reason)
    return cond


# -- curseur (pagination par clé) -------------------------------------------
def encode_cursor(sort: str, values: list[Any]) -> str:
    raw = json.dumps({"s": sort, "k": values}, separators=(",", ":"), default=str).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str, sort: str) -> list[Any]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
        values = data["k"]
        assert data["s"] == sort and isinstance(values, list)
    except Exception as exc:  # noqa: BLE001
        raise LabelingError("Curseur de pagination invalide (les filtres ou le tri ont changé ?).") from exc
    return values


def _after(keys: list[tuple[ColumnElement[Any], bool]], values: list[Any], inclusive: bool) -> ColumnElement[Any]:
    """Condition « strictement après (ou à partir de) ``values`` » pour un ordre lexicographique.

    ``keys`` : [(expression, descendant ?)].
    """
    if all(not desc for _, desc in keys):
        row = tuple_(*[k for k, _ in keys])
        return row >= tuple_(*values) if inclusive else row > tuple_(*values)
    clauses = []
    for i, (expr, desc) in enumerate(keys):
        eq = [keys[j][0] == values[j] for j in range(i)]
        last = i == len(keys) - 1
        if last and inclusive:
            cmp = expr <= values[i] if desc else expr >= values[i]
        else:
            cmp = expr < values[i] if desc else expr > values[i]
        clauses.append(and_(*eq, cmp))
    return or_(*clauses)


@dataclass
class Page:
    items: list[dict[str, Any]]
    next_cursor: str | None
    total: int | None
    model: active.ActiveModel | None
    sort: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "next_cursor": self.next_cursor,
            "total": self.total,
            "sort": self.sort,
            "model": self.model.to_dict() if self.model else None,
        }


def search_tracks(
    session: Session,
    filters: TrackFilters,
    *,
    sort: str = "naturel",
    limit: int = 100,
    cursor: str | None = None,
    offset: int = 0,
    anchor: int | None = None,
    model_id: str | None = None,
    margin: float = active.DEFAULT_MARGIN,
    with_total: bool = True,
) -> Page:
    """Une page de pistes, sous forme compacte.

    * ``cursor`` : suite de la page précédente (``next_cursor``) — pagination par clé, en temps constant ;
    * ``anchor`` : identifiant de piste à partir de laquelle commencer (incluse), selon le tri courant ;
    * ``offset`` : saut direct (moins efficace sur de gros volumes, réservé aux sauts de page).
    """
    filters.validate()
    if sort not in SORTS:
        raise LabelingError(f"Tri inconnu : « {sort} » (attendu : {', '.join(SORTS)}).")
    limit = max(1, min(MAX_LIMIT, int(limit)))
    model = active.resolve_model(session, model_id)
    if sort == "priorite" and model is None:
        raise LabelingError("Tri par priorité impossible : aucun modèle entraîné avec des prédictions.")

    T, L, LO, P = m.Track, m.Label, m.Lot, m.Prediction
    ex = active.priority_exprs(model, margin) if model is not None else None

    if sort == "naturel":
        keys: list[tuple[ColumnElement[Any], bool]] = [(T.lot_id, False), (T.start_frame, False), (T.lane, False), (T.id, False)]
    elif sort == "recent":
        keys = [(L.updated_at, True), (T.id, True)]
    else:
        assert ex is not None
        keys = [(ex.priority, True), (T.id, False)]

    label_filters = bool(filters.verdicts or filters.defects or filters.reviewed is not None or filters.source or filters.reason)

    def base(cols: list[Any], counting: bool = False) -> Select[Any]:
        # Pour le comptage, on n'ajoute que les jointures dont dépendent les filtres.
        q = select(*cols).select_from(T)
        if not counting or filters.lot_types:
            q = q.join(LO, LO.id == T.lot_id)
        if sort == "recent":
            q = q.join(L, L.track_id == T.id)
        elif not counting or label_filters:
            q = q.outerjoin(L, L.track_id == T.id)
        if model is not None and (not counting or filters.needs_prediction):
            if filters.reason is not None or (sort == "priorite" and filters.score_min is None and filters.score_max is None):
                # Toutes les prédictions du modèle sont évaluées : le « + » empêche SQLite de partir de
                # l'index (model_id, score) — peu sélectif ici — et de faire trois accès aléatoires par ligne.
                model_cond: ColumnElement[Any] = text("+predictions.model_id = :lbl_model_id").bindparams(lbl_model_id=model.id)
            else:
                model_cond = P.model_id == model.id
            join_cond = and_(P.track_id == T.id, model_cond)
            q = q.join(P, join_cond) if filters.needs_prediction else q.outerjoin(P, join_cond)
        conds = _conditions(filters, model, margin)
        return q.where(*conds) if conds else q

    cols: list[Any] = [
        T.id, T.lot_id, T.lane, T.start_frame, T.start_ts, T.n_views, T.quality, T.flags,
        LO.type.label("lot_type"),
        L.verdict, L.defect, L.reviewed, L.source, L.reviewer, L.updated_at, L.note,
    ]
    if model is not None:
        assert ex is not None
        cols += [P.score.label("score"), ex.priority.label("priority"), ex.reason.label("reason")]

    total = None
    if with_total:
        total = session.execute(base([func.count()], counting=True)).scalar_one()

    q = base(cols)
    start_values: list[Any] | None = None
    inclusive = False
    if cursor:
        start_values = decode_cursor(cursor, sort)
        if sort == "recent":
            start_values[0] = parse_iso(start_values[0])
    elif anchor is not None:
        start_values = _anchor_values(session, anchor, sort, model, margin)
        inclusive = True
    if start_values is not None:
        q = q.where(_after(keys, start_values, inclusive))
    q = q.order_by(*[k.desc() if desc else k.asc() for k, desc in keys])
    if offset and start_values is None:
        q = q.offset(max(0, int(offset)))
    rows = session.execute(q.limit(limit + 1)).all()

    has_more = len(rows) > limit
    rows = rows[:limit]
    prelabels: dict[str, dict[str, Any]] = {}
    items = []
    for r in rows:
        if r.lot_type not in prelabels:
            prelabels[r.lot_type] = prelabel_of(r.lot_type)
        item: dict[str, Any] = {
            "id": r.id,
            "lot_id": r.lot_id,
            "lot_type": r.lot_type,
            "lane": r.lane,
            "start_frame": r.start_frame,
            "start_ts": r.start_ts,
            "n_views": r.n_views,
            "quality": r.quality,
            "flags": r.flags or [],
            "prelabel": prelabels[r.lot_type],
            "verdict": r.verdict,
            "defect": r.defect,
            "reviewed": bool(r.reviewed),
            "source": r.source,
            "reviewer": r.reviewer or "",
            "updated_at": iso(r.updated_at),
            "note": r.note or "",
        }
        if model is not None:
            item["score"] = r.score
            item["priority"] = r.priority
            item["reason"] = r.reason
        items.append(item)

    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        if sort == "naturel":
            values: list[Any] = [last.lot_id, last.start_frame, last.lane, last.id]
        elif sort == "recent":
            values = [last.updated_at.isoformat() if last.updated_at else None, last.id]
        else:
            values = [last.priority, last.id]
        next_cursor = encode_cursor(sort, values)
    return Page(items=items, next_cursor=next_cursor, total=total, model=model, sort=sort)


def _anchor_values(session: Session, track_id: int, sort: str, model: active.ActiveModel | None, margin: float) -> list[Any]:
    T, L, P = m.Track, m.Label, m.Prediction
    row = session.execute(
        select(T.id, T.lot_id, T.start_frame, T.lane, L.updated_at).outerjoin(L, L.track_id == T.id).where(T.id == track_id)
    ).first()
    if row is None:
        raise NotFoundError(f"Piste {track_id} introuvable.")
    if sort == "naturel":
        return [row.lot_id, row.start_frame, row.lane, row.id]
    if sort == "recent":
        return [row.updated_at, row.id]
    assert model is not None
    ex = active.priority_exprs(model, margin)
    prio = session.execute(
        select(ex.priority)
        .select_from(T)
        .outerjoin(L, L.track_id == T.id)
        .outerjoin(P, and_(P.track_id == T.id, P.model_id == model.id))
        .where(T.id == track_id)
    ).scalar_one()
    return [prio, row.id]


# -- détail d'une piste ------------------------------------------------------
def track_detail(session: Session, track_id: int, *, model_id: str | None = None, margin: float = active.DEFAULT_MARGIN, history_limit: int = 30) -> dict[str, Any]:
    from .history import history_rows

    track = session.get(m.Track, track_id)
    if track is None:
        raise NotFoundError(f"Piste {track_id} introuvable.")
    lot = session.get(m.Lot, track.lot_id)
    label = session.get(m.Label, track_id)
    views = session.execute(
        select(m.View.index, m.View.frame_index, m.View.sharpness, m.View.partial, m.View.defect_visible)
        .where(m.View.track_id == track_id)
        .order_by(m.View.index)
    ).all()
    model = active.resolve_model(session, model_id)
    prediction = None
    if model is not None:
        pred = session.execute(
            select(m.Prediction.score, m.Prediction.view_scores, m.Prediction.subset).where(
                m.Prediction.model_id == model.id, m.Prediction.track_id == track_id
            )
        ).first()
        if pred is not None:
            ev = active.evaluate(pred.score, label.verdict if label else None, bool(label and label.reviewed), model, margin)
            prediction = {
                "model": model.to_dict(),
                "score": pred.score,
                "view_scores": pred.view_scores or [],
                "subset": pred.subset,
                "predicted": ev["predicted"],
                "priority": ev["priority"],
                "reason": ev["reason"],
            }
    extra = track.extra or {}
    return {
        "id": track.id,
        "lot": {
            "id": lot.id,
            "name": lot.name,
            "type": lot.type,
            "farm": lot.farm,
            "session_date": lot.session_date,
        } if lot else None,
        "lane": track.lane,
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "start_ts": track.start_ts,
        "end_ts": track.end_ts,
        "n_detections": track.n_detections,
        "n_views": track.n_views,
        "quality": track.quality,
        "flags": track.flags or [],
        "mean_sharpness": track.mean_sharpness,
        "prelabel": prelabel_of(lot.type if lot else None),
        "label": {
            "verdict": label.verdict if label else None,
            "defect": label.defect if label else None,
            "reviewed": bool(label.reviewed) if label else False,
            "source": label.source if label else None,
            "reviewer": (label.reviewer or "") if label else "",
            "note": (label.note or "") if label else "",
            "updated_at": iso(label.updated_at) if label else None,
        },
        "views": [
            {
                "index": v.index,
                "frame_index": v.frame_index,
                "sharpness": v.sharpness,
                "partial": bool(v.partial),
                "defect_visible": bool(v.defect_visible),
                "url": f"/api/media/crop/{track_id}/{v.index}",
            }
            for v in views
        ],
        "prediction": prediction,
        # Vérité terrain des données synthétiques : information de contrôle, jamais un label.
        "synthetic_truth": extra.get("truth"),
        "history": history_rows(session, track_id=track_id, limit=history_limit)["items"],
    }
