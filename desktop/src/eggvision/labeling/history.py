"""Historique des modifications de labels (qui, quoi, quand), paginé par clé."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import models as m
from .common import ACTION_LABELS, MAX_LIMIT, iso


def history_rows(
    session: Session,
    *,
    track_id: int | None = None,
    user: str | None = None,
    action: str | None = None,
    lot_id: str | None = None,
    before_id: int | None = None,
    limit: int = 50,
    with_total: bool = False,
) -> dict[str, Any]:
    """Entrées d'historique, de la plus récente à la plus ancienne.

    ``before_id`` : curseur (``next_before_id`` de la page précédente).
    """
    H, T = m.LabelHistory, m.Track
    limit = max(1, min(MAX_LIMIT, int(limit)))
    conds = []
    if track_id is not None:
        conds.append(H.track_id == track_id)
    if user:
        conds.append(H.user == user)
    if action:
        conds.append(H.action == action)
    if lot_id:
        conds.append(T.lot_id == lot_id)
    q = select(H.id, H.track_id, H.user, H.at, H.action, H.before, H.after, T.lot_id, T.lane).join(T, T.id == H.track_id)
    if conds:
        q = q.where(*conds)
    page_q = q.where(H.id < before_id) if before_id is not None else q
    rows = session.execute(page_q.order_by(H.id.desc()).limit(limit + 1)).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    total = None
    if with_total:
        cq = select(func.count()).select_from(H)
        if lot_id:
            cq = cq.join(T, T.id == H.track_id)
        total = session.execute(cq.where(*conds) if conds else cq).scalar_one()
    return {
        "items": [
            {
                "id": r.id,
                "track_id": r.track_id,
                "lot_id": r.lot_id,
                "lane": r.lane,
                "user": r.user,
                "at": iso(r.at),
                "action": r.action,
                "action_label": ACTION_LABELS.get(r.action, r.action),
                "before": r.before or {},
                "after": r.after or {},
            }
            for r in rows
        ],
        "next_before_id": rows[-1].id if has_more and rows else None,
        "total": total,
    }


def history_users(session: Session, limit: int = 200) -> list[str]:
    """Auteurs distincts de l'historique (index sur ``user``)."""
    H = m.LabelHistory
    return list(session.scalars(select(H.user).distinct().order_by(H.user).limit(limit)))
