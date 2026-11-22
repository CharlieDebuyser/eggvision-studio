"""Pré-label automatique : le type du lot donne le label initial de chacune de ses pistes."""

from __future__ import annotations

from eggvision_shared.constants import prelabel_for_lot_type
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import models as m


def apply_prelabels(session: Session, lot_id: str, overwrite_auto: bool = True) -> int:
    """Crée le label ``auto`` des pistes du lot qui n'en ont pas (ou rafraîchit ceux encore ``auto`` non relus).

    Ne modifie jamais un label saisi ou relu par un humain. Retourne le nombre de labels créés ou mis à jour.
    """
    lot = session.get(m.Lot, lot_id)
    if lot is None:
        return 0
    verdict, defect = prelabel_for_lot_type(lot.type)
    changed = 0
    rows = session.execute(
        select(m.Track.id, m.Label).outerjoin(m.Label, m.Label.track_id == m.Track.id).where(m.Track.lot_id == lot_id)
    ).all()
    for track_id, label in rows:
        if label is None:
            session.add(m.Label(track_id=track_id, verdict=verdict, defect=defect, source="auto", reviewed=False))
            session.add(m.LabelHistory(track_id=track_id, user="auto", action="prelabel", before={}, after={"verdict": verdict, "defect": defect}))
            changed += 1
        elif overwrite_auto and label.source == "auto" and not label.reviewed and (label.verdict, label.defect) != (verdict, defect):
            before = {"verdict": label.verdict, "defect": label.defect}
            label.verdict, label.defect = verdict, defect
            session.add(m.LabelHistory(track_id=track_id, user="auto", action="prelabel", before=before, after={"verdict": verdict, "defect": defect}))
            changed += 1
    return changed
