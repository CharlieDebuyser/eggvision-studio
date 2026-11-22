"""Tableau de bord du jeu de données : effectifs, avancement de la relecture, alertes.

Une seule requête agrégée (GROUP BY lot × verdict × défaut × relu × qualité) alimente
l'ensemble : quelques centaines de lignes au plus, quel que soit le nombre de pistes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eggvision_shared.constants import DEFECT_LABELS, DEFECT_TYPES, VERDICTS
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import models as m
from . import active
from .common import VERDICT_LABELS

MIN_EXAMPLES = 50


def _progress(reviewed: int, total: int) -> float:
    return round(reviewed / total, 4) if total else 0.0


def dashboard(session: Session, *, min_examples: int = MIN_EXAMPLES, model_id: str | None = None, lots_limit: int = 1000) -> dict[str, Any]:
    T, L, LO = m.Track, m.Label, m.Lot
    reviewed_col = func.coalesce(L.reviewed, False)
    rows = session.execute(
        select(T.lot_id, L.verdict, L.defect, reviewed_col, T.quality, func.count())
        .select_from(T)
        .outerjoin(L, L.track_id == T.id)
        .group_by(T.lot_id, L.verdict, L.defect, reviewed_col, T.quality)
    ).all()
    lots_meta = {
        r.id: r
        for r in session.execute(
            select(LO.id, LO.name, LO.type, LO.farm, LO.session_date, LO.status, LO.created_at)
        ).all()
    }

    totals = {"tracks": 0, "labeled": 0, "reviewed": 0, "usable": 0, "usable_reviewed": 0, "doubtful": 0, "doubtful_unreviewed": 0, "defect_untyped": 0}
    verdicts: dict[str | None, dict[str, int]] = {v: {"total": 0, "reviewed": 0} for v in (*VERDICTS, None)}
    defects: dict[str | None, dict[str, int]] = {d: {"total": 0, "reviewed": 0} for d in (*DEFECT_TYPES, None)}
    per_lot: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "reviewed": 0, "doubtful": 0, "verdicts": defaultdict(int), "defects": defaultdict(int)}
    )
    for lot_id, verdict, defect, reviewed, quality, n in rows:
        reviewed = bool(reviewed)
        totals["tracks"] += n
        lot = per_lot[lot_id]
        lot["total"] += n
        lot["verdicts"][verdict or "aucun"] += n
        if verdict is not None:
            totals["labeled"] += n
        if reviewed:
            totals["reviewed"] += n
            lot["reviewed"] += n
        if verdict in ("viable", "non_viable"):
            totals["usable"] += n
            if reviewed:
                totals["usable_reviewed"] += n
        if quality == "douteuse":
            totals["doubtful"] += n
            lot["doubtful"] += n
            if not reviewed:
                totals["doubtful_unreviewed"] += n
        bucket = verdicts.get(verdict)
        if bucket is not None:
            bucket["total"] += n
            bucket["reviewed"] += n if reviewed else 0
        if verdict == "non_viable":
            dk = defect if defect in defects else None
            defects[dk]["total"] += n
            defects[dk]["reviewed"] += n if reviewed else 0
            lot["defects"][defect or "aucun"] += n
            if defect is None:
                totals["defect_untyped"] += n
    totals["unreviewed"] = totals["tracks"] - totals["reviewed"]
    totals["progress"] = _progress(totals["reviewed"], totals["tracks"])

    lots_out = []
    sessions: dict[tuple[str, str], dict[str, Any]] = {}
    ordered = sorted(per_lot.items(), key=lambda kv: (getattr(lots_meta.get(kv[0]), "created_at", None) is None, str(getattr(lots_meta.get(kv[0]), "created_at", "")), kv[0]))
    for lot_id, agg in ordered:
        meta = lots_meta.get(lot_id)
        farm = meta.farm if meta else ""
        day = meta.session_date if meta else ""
        lots_out.append({
            "lot_id": lot_id,
            "name": meta.name if meta else lot_id,
            "type": meta.type if meta else "",
            "farm": farm,
            "session_date": day,
            "status": meta.status if meta else "",
            "total": agg["total"],
            "reviewed": agg["reviewed"],
            "progress": _progress(agg["reviewed"], agg["total"]),
            "doubtful": agg["doubtful"],
            "verdicts": dict(agg["verdicts"]),
            "defects": dict(agg["defects"]),
        })
        s = sessions.setdefault((farm, day), {"farm": farm, "session_date": day, "lots": 0, "total": 0, "reviewed": 0})
        s["lots"] += 1
        s["total"] += agg["total"]
        s["reviewed"] += agg["reviewed"]
    sessions_out = sorted(sessions.values(), key=lambda s: (s["session_date"], s["farm"]))
    for s in sessions_out:
        s["progress"] = _progress(s["reviewed"], s["total"])

    # -- alertes ------------------------------------------------------------
    alerts: list[dict[str, Any]] = []
    classes = [("viable", VERDICT_LABELS["viable"], verdicts["viable"]), ("non_viable", VERDICT_LABELS["non_viable"], verdicts["non_viable"])]
    classes += [(f"defaut:{d}", f"Défaut « {DEFECT_LABELS[d]} »", defects[d]) for d in DEFECT_TYPES]
    for key, label, c in classes:
        if c["total"] < min_examples:
            alerts.append({
                "level": "warning",
                "code": "classe_sous_representee",
                "class": key,
                "count": c["total"],
                "reviewed": c["reviewed"],
                "message": f"{label} : {c['total']} exemple{'s' if c['total'] > 1 else ''} (dont {c['reviewed']} relu{'s' if c['reviewed'] > 1 else ''}), moins de {min_examples}.",
            })
    if totals["defect_untyped"]:
        n = totals["defect_untyped"]
        alerts.append({
            "level": "warning",
            "code": "defaut_sans_type",
            "count": n,
            "message": f"{n} piste{'s' if n > 1 else ''} non viable{'s' if n > 1 else ''} sans type de défaut.",
        })
    unreviewed_lots = [lo for lo in lots_out if lo["total"] and lo["reviewed"] == 0]
    if unreviewed_lots:
        n = len(unreviewed_lots)
        alerts.append({
            "level": "info",
            "code": "lots_non_relus",
            "count": n,
            "lots": [lo["lot_id"] for lo in unreviewed_lots[:50]],
            "message": f"{n} lot{'s' if n > 1 else ''} sans aucune piste relue : " + ", ".join(lo["name"] for lo in unreviewed_lots[:5]) + ("…" if n > 5 else ""),
        })
    if totals["doubtful_unreviewed"]:
        n = totals["doubtful_unreviewed"]
        alerts.append({
            "level": "info",
            "code": "pistes_douteuses",
            "count": n,
            "message": f"{n} piste{'s' if n > 1 else ''} de qualité douteuse à relire.",
        })

    model = active.resolve_model(session, model_id)
    active_out = None
    if model is not None:
        active_out = {"model": model.to_dict(), **active.queue_summary(session, model)}
        if active_out["contradictions"]:
            n = active_out["contradictions"]
            alerts.append({
                "level": "info",
                "code": "contradictions_modele",
                "count": n,
                "message": f"Le modèle « {model.name} » contredit {n} label{'s' if n > 1 else ''} : voir la file d'apprentissage actif.",
            })

    return {
        "totals": totals,
        "verdicts": [
            {"verdict": v, "label": VERDICT_LABELS.get(v, "Sans label") if v else "Sans label", **verdicts[v]}
            for v in (*VERDICTS, None)
        ],
        "defects": [
            {"defect": d, "label": DEFECT_LABELS[d] if d else "Sans type", **defects[d]}
            for d in (*DEFECT_TYPES, None)
        ],
        "lots": lots_out[: max(1, int(lots_limit))],
        "lots_total": len(lots_out),
        "sessions": sessions_out,
        "alerts": alerts,
        "min_examples": min_examples,
        "active": active_out,
    }
