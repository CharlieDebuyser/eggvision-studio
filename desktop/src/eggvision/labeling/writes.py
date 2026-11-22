"""Écriture des labels : saisie, confirmation, réouverture, masse, marquage des vues, annulation.

Règles de cohérence :
  * un type de défaut implique le verdict « non viable » ;
  * les verdicts viable / incertain / exclu (ou l'absence de verdict) vident le type de défaut ;
  * toute écriture humaine pose ``source="humain"`` et le nom du relecteur ;
  * chaque modification ajoute une entrée d'historique (instantané avant / après).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import models as m
from .common import (
    ACTION_BULK,
    ACTION_CONFIRM,
    ACTION_LABEL,
    ACTION_REOPEN,
    ACTION_UNDO,
    ACTION_VIEW,
    MAX_BULK,
    UNDOABLE_ACTIONS,
    UNSET,
    LabelingError,
    NotFoundError,
    iso,
    snapshot,
    validate_defect,
    validate_verdict,
)

_CHUNK = 500


def resolve(current: dict[str, Any], verdict: Any = UNSET, defect: Any = UNSET) -> tuple[str | None, str | None]:
    """Applique les règles de cohérence et retourne le couple (verdict, défaut) final."""
    cur_verdict = current.get("verdict")
    cur_defect = current.get("defect")
    if defect is not UNSET:
        validate_defect(defect)
    if verdict is not UNSET:
        validate_verdict(verdict)
    if defect is not UNSET and defect is not None:
        if verdict is not UNSET and verdict != "non_viable":
            raise LabelingError("Un type de défaut implique le verdict « non viable ».")
        return "non_viable", defect
    if verdict is UNSET:
        if defect is None:  # effacement explicite du type de défaut
            return cur_verdict, None
        return cur_verdict, cur_defect
    if verdict == "non_viable":
        if defect is None:
            return verdict, None
        return verdict, cur_defect if cur_verdict == "non_viable" else None
    return verdict, None


def _new_label(track_id: int) -> m.Label:
    return m.Label(track_id=track_id, verdict=None, defect=None, source="auto", reviewed=False, reviewer="", note="")


def _get_label(session: Session, track_id: int) -> tuple[m.Label, bool]:
    label = session.get(m.Label, track_id)
    if label is not None:
        return label, False
    if session.get(m.Track, track_id) is None:
        raise NotFoundError(f"Piste {track_id} introuvable.")
    label = _new_label(track_id)
    session.add(label)
    return label, True


def _apply(
    label: m.Label,
    user: str,
    *,
    verdict: Any = UNSET,
    defect: Any = UNSET,
    reviewed: Any = UNSET,
    note: Any = UNSET,
    human: bool = True,
) -> None:
    v, d = resolve(snapshot(label), verdict, defect)
    label.verdict, label.defect = v, d
    if reviewed is not UNSET:
        label.reviewed = bool(reviewed)
    if note is not UNSET:
        label.note = str(note or "")
    if human:
        label.source = "humain"
        label.reviewer = user
    label.updated_at = m.utcnow()


def _label_out(label: m.Label) -> dict[str, Any]:
    return {**snapshot(label), "track_id": label.track_id, "updated_at": iso(label.updated_at)}


def set_label(
    session: Session,
    track_id: int,
    user: str,
    *,
    verdict: Any = UNSET,
    defect: Any = UNSET,
    reviewed: Any = True,
    note: Any = UNSET,
    action: str = ACTION_LABEL,
) -> dict[str, Any]:
    """Saisit le label d'une piste (marquée relue par défaut) et l'historise."""
    label, created = _get_label(session, track_id)
    before = {} if created else snapshot(label)
    _apply(label, user, verdict=verdict, defect=defect, reviewed=reviewed, note=note)
    after = snapshot(label)
    entry = m.LabelHistory(track_id=track_id, user=user, at=label.updated_at, action=action, before=before, after=after)
    session.add(entry)
    session.flush()
    return {"label": _label_out(label), "history_id": entry.id, "changed": before != after}


def confirm(session: Session, track_id: int, user: str) -> dict[str, Any]:
    """Confirme le label affiché (pré-label compris) tel quel : relu, source humaine."""
    label = session.get(m.Label, track_id)
    if label is None or label.verdict is None:
        if session.get(m.Track, track_id) is None:
            raise NotFoundError(f"Piste {track_id} introuvable.")
        raise LabelingError("Rien à confirmer : cette piste n'a pas encore de label.")
    return set_label(session, track_id, user, action=ACTION_CONFIRM)


def reopen(session: Session, track_id: int, user: str) -> dict[str, Any]:
    """Remet la piste « à relire » (le verdict est conservé)."""
    label, created = _get_label(session, track_id)
    before = {} if created else snapshot(label)
    label.reviewed = False
    label.updated_at = m.utcnow()
    entry = m.LabelHistory(track_id=track_id, user=user, at=label.updated_at, action=ACTION_REOPEN, before=before, after=snapshot(label))
    session.add(entry)
    session.flush()
    return {"label": _label_out(label), "history_id": entry.id, "changed": before != snapshot(label)}


def bulk(
    session: Session,
    track_ids: Iterable[int],
    user: str,
    *,
    verdict: Any = UNSET,
    defect: Any = UNSET,
    confirm_only: bool = False,
    reviewed: Any = True,
) -> dict[str, Any]:
    """Écriture en masse (mode grille), en une seule transaction, annulable d'un bloc.

    * ``confirm_only`` : confirme le label existant de chaque piste (les pistes sans label sont ignorées) ;
    * sinon applique ``verdict`` / ``defect`` à toutes les pistes.
    """
    ids = list(dict.fromkeys(int(i) for i in track_ids))
    if not ids:
        raise LabelingError("Aucune piste sélectionnée.")
    if len(ids) > MAX_BULK:
        raise LabelingError(f"Trop de pistes à la fois ({len(ids)} ; maximum {MAX_BULK}).")
    if not confirm_only and verdict is UNSET and defect is UNSET and reviewed is UNSET:
        raise LabelingError("Indiquez un verdict, un défaut ou une confirmation.")
    resolve({}, verdict, defect)  # validation anticipée (lève avant toute écriture)

    existing: dict[int, m.Label] = {}
    known: set[int] = set()
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i : i + _CHUNK]
        known.update(session.scalars(select(m.Track.id).where(m.Track.id.in_(chunk))))
        for label in session.scalars(select(m.Label).where(m.Label.track_id.in_(chunk))):
            existing[label.track_id] = label
    missing = [i for i in ids if i not in known]
    if missing:
        raise NotFoundError(f"Pistes introuvables : {', '.join(map(str, missing[:10]))}{'…' if len(missing) > 10 else ''}")

    batch = uuid.uuid4().hex[:12]
    at = m.utcnow()
    history: list[dict[str, Any]] = []
    skipped: list[int] = []
    updated = 0
    for tid in ids:
        label = existing.get(tid)
        if confirm_only and (label is None or label.verdict is None):
            skipped.append(tid)
            continue
        if label is None:
            label = _new_label(tid)
            session.add(label)
            before: dict[str, Any] = {}
        else:
            before = snapshot(label)
        if confirm_only:
            _apply(label, user, reviewed=True)
        else:
            _apply(label, user, verdict=verdict, defect=defect, reviewed=reviewed)
        label.updated_at = at
        after = snapshot(label)
        history.append({"track_id": tid, "user": user, "at": at, "action": ACTION_BULK, "before": before, "after": {**after, "batch": batch}})
        updated += 1
    session.flush()
    if history:
        session.execute(m.LabelHistory.__table__.insert(), history)
    return {"updated": updated, "skipped": skipped, "batch": batch, "at": iso(at)}


def set_view_defect(session: Session, track_id: int, view_index: int, user: str, visible: bool | None = None) -> dict[str, Any]:
    """Marque (ou démarque) une vue où le défaut est visible ; ``visible=None`` inverse l'état."""
    view = session.execute(select(m.View).where(m.View.track_id == track_id, m.View.index == view_index)).scalar_one_or_none()
    if view is None:
        raise NotFoundError(f"Vue {view_index} de la piste {track_id} introuvable.")
    before = bool(view.defect_visible)
    view.defect_visible = (not before) if visible is None else bool(visible)
    entry = m.LabelHistory(
        track_id=track_id,
        user=user,
        at=m.utcnow(),
        action=ACTION_VIEW,
        before={"view_index": view_index, "defect_visible": before},
        after={"view_index": view_index, "defect_visible": view.defect_visible},
    )
    session.add(entry)
    session.flush()
    return {"track_id": track_id, "view_index": view_index, "defect_visible": view.defect_visible, "history_id": entry.id}


# -- annulation ---------------------------------------------------------------
_COMPARED = ("verdict", "defect", "reviewed")


def _restore_label(label: m.Label, state: dict[str, Any]) -> None:
    if not state:  # le label n'existait pas : retour à l'état vierge
        state = {"verdict": None, "defect": None, "reviewed": False, "source": "auto", "reviewer": "", "note": ""}
    label.verdict = state.get("verdict")
    label.defect = state.get("defect")
    label.reviewed = bool(state.get("reviewed", False))
    label.source = state.get("source") or "auto"
    label.reviewer = state.get("reviewer") or ""
    label.note = state.get("note") or ""
    label.updated_at = m.utcnow()


def undo_last(session: Session, user: str, scan: int = 1000) -> dict[str, Any]:
    """Annule la dernière action non encore annulée de ``user`` (une saisie en masse est annulée d'un bloc).

    Une piste modifiée entre-temps (état actuel ≠ état laissé par l'action) n'est pas restaurée : elle est
    comptée dans ``conflicts``.
    """
    H = m.LabelHistory
    undone: set[int] = set()
    target: m.LabelHistory | None = None
    for h in session.scalars(select(H).where(H.user == user).order_by(H.id.desc()).limit(scan)):
        if h.action == ACTION_UNDO:
            undone.update(int(i) for i in (h.after or {}).get("undo_of", []))
            continue
        if h.id in undone or h.action not in UNDOABLE_ACTIONS:
            continue
        target = h
        break
    if target is None:
        return {"undone": 0, "conflicts": 0, "track_ids": [], "action": None, "message": "Rien à annuler."}

    entries = [target]
    batch = (target.after or {}).get("batch")
    if target.action == ACTION_BULK and batch:
        entries = [
            h
            for h in session.scalars(select(H).where(H.user == user, H.at == target.at, H.action == ACTION_BULK))
            if (h.after or {}).get("batch") == batch and h.id not in undone
        ]
    consumed = [h.id for h in entries]
    restored: list[int] = []
    conflicts: list[int] = []
    new_entries: list[m.LabelHistory] = []
    for h in entries:
        if h.action == ACTION_VIEW:
            idx = int(h.before.get("view_index", -1))
            view = session.execute(select(m.View).where(m.View.track_id == h.track_id, m.View.index == idx)).scalar_one_or_none()
            if view is None or bool(view.defect_visible) != bool(h.after.get("defect_visible")):
                conflicts.append(h.track_id)
                continue
            view.defect_visible = bool(h.before.get("defect_visible"))
            new_entries.append(
                H(track_id=h.track_id, user=user, at=m.utcnow(), action=ACTION_UNDO,
                  before=dict(h.after), after={**h.before})
            )
            restored.append(h.track_id)
            continue
        label = session.get(m.Label, h.track_id)
        current = snapshot(label)
        expected = h.after or {}
        if label is None or any(current.get(k) != expected.get(k) for k in _COMPARED):
            conflicts.append(h.track_id)
            continue
        _restore_label(label, h.before or {})
        new_entries.append(H(track_id=h.track_id, user=user, at=label.updated_at, action=ACTION_UNDO, before=current, after=snapshot(label)))
        restored.append(h.track_id)
    if new_entries:
        new_entries[0].after = {**new_entries[0].after, "undo_of": consumed}
    else:  # tout est en conflit : on consomme quand même l'action pour ne pas bloquer les annulations suivantes
        new_entries.append(H(track_id=target.track_id, user=user, at=m.utcnow(), action=ACTION_UNDO, before={}, after={"undo_of": consumed, "skipped": True}))
    session.add_all(new_entries)
    session.flush()
    n = len(restored)
    if n == 0:
        message = "Annulation impossible : la piste a été modifiée depuis."
    elif target.action == ACTION_BULK:
        message = f"Saisie en masse annulée ({n} piste{'s' if n > 1 else ''})."
    else:
        message = f"Action annulée sur la piste {restored[0]}."
    if conflicts and n:
        message += f" {len(conflicts)} piste(s) modifiée(s) depuis n'ont pas été restaurée(s)."
    labels = {}
    for tid in restored:
        lab = session.get(m.Label, tid)
        if lab is not None:
            labels[tid] = _label_out(lab)
    return {
        "undone": n,
        "conflicts": len(conflicts),
        "track_ids": restored,
        "action": target.action,
        "labels": list(labels.values()),
        "message": message,
    }
