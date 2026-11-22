"""Découpage entraînement / validation / test — toujours par lot (ou par session).

Principe : aucune image ni aucune piste d'un même lot ne peut se retrouver dans
deux sous-ensembles, puisque l'unité d'affectation est le lot (``group_by="lot"``)
ou l'ensemble des lots d'une même journée de prise de vue (``group_by="session"``,
clé ``Lot.session_date``). Des élevages (``Lot.farm``) ou des journées entières
peuvent être réservés au test.

La répartition cherche à approcher les ratios demandés **pour chaque classe**
(viables et chaque type de défaut) : recherche gloutonne randomisée, répétée
avec plusieurs ordres (graine fixée → résultat reproductible), qui retient la
répartition de coût minimal (écart aux effectifs cibles + forte pénalité si
une classe présente dans plusieurs groupes manque dans un sous-ensemble).

Un découpage enregistré (:class:`~eggvision.db.models.DatasetSplit`) est
versionné : dès qu'un modèle l'utilise, il devient immuable et toute
modification crée une nouvelle version (même nom, ``version + 1``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from eggvision_shared.constants import DEFECT_TYPES, VERDICT_NON_VIABLE, VERDICT_VIABLE
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import models as m
from ..project import Project

SUBSETS = ("train", "val", "test")
SUBSET_LABELS = {"train": "entraînement", "val": "validation", "test": "test"}
UNKNOWN_DEFECT = "non_precise"


class SplitError(ValueError):
    """Erreur de découpage présentable à l'utilisateur."""


@dataclass
class SplitParams:
    """Paramètres de création d'un découpage."""

    group_by: str = "session"  # session | lot
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    holdout_farms: list[str] = field(default_factory=list)
    holdout_dates: list[str] = field(default_factory=list)  # AAAA-MM-JJ
    seed: int = 0
    reviewed_only: bool = False
    exclude_lots: list[str] = field(default_factory=list)
    n_restarts: int = 300

    def validate(self) -> SplitParams:
        if self.group_by not in ("session", "lot"):
            raise SplitError("Regroupement inconnu : choisir « session » ou « lot ».")
        r = [float(x) for x in self.ratios]
        if len(r) != 3 or any(x < 0 for x in r) or sum(r) <= 0:
            raise SplitError("Les ratios doivent être trois nombres positifs (entraînement, validation, test).")
        if r[0] <= 0:
            raise SplitError("Le ratio d'entraînement doit être strictement positif.")
        total = sum(r)
        self.ratios = (r[0] / total, r[1] / total, r[2] / total)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ratios"] = [round(x, 6) for x in self.ratios]
        d.pop("n_restarts", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SplitParams:
        d = dict(d or {})
        known = {k: d[k] for k in ("group_by", "holdout_farms", "holdout_dates", "seed", "reviewed_only", "exclude_lots", "n_restarts") if k in d and d[k] is not None}
        p = cls(**known)
        if d.get("ratios") is not None:
            p.ratios = tuple(d["ratios"])  # type: ignore[assignment]
        return p.validate()


# ---------------------------------------------------------------------------
# Inventaire des lots
# ---------------------------------------------------------------------------
@dataclass
class LotInfo:
    id: str
    name: str
    type: str
    farm: str
    session_date: str
    counts: dict[str, int]  # classe → nombre d'œufs utilisables (viable, <défaut>, non_precise)

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        viable = self.counts.get("viable", 0)
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "farm": self.farm,
            "session_date": self.session_date,
            "n_eggs": self.n,
            "viable": viable,
            "non_viable": self.n - viable,
            "defects": {k: v for k, v in self.counts.items() if k != "viable"},
        }


def class_key(verdict: str | None, defect: str | None) -> str:
    if verdict == VERDICT_VIABLE:
        return "viable"
    return defect if defect in DEFECT_TYPES else UNKNOWN_DEFECT


def usable_verdict_filter(reviewed_only: bool) -> list[Any]:
    conds: list[Any] = [m.Label.verdict.in_((VERDICT_VIABLE, VERDICT_NON_VIABLE))]
    if reviewed_only:
        conds.append(m.Label.reviewed.is_(True))
    return conds


def lot_inventory(session: Session, reviewed_only: bool = False) -> list[LotInfo]:
    """Lots possédant au moins un œuf utilisable, avec leurs effectifs par classe."""
    rows = session.execute(
        select(m.Track.lot_id, m.Label.verdict, m.Label.defect, func.count())
        .join(m.Label, m.Label.track_id == m.Track.id)
        .where(*usable_verdict_filter(reviewed_only))
        .group_by(m.Track.lot_id, m.Label.verdict, m.Label.defect)
    ).all()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for lot_id, verdict, defect, n in rows:
        counts[lot_id][class_key(verdict, defect)] += int(n)
    if not counts:
        return []
    lots = session.scalars(select(m.Lot).where(m.Lot.id.in_(list(counts)))).all()
    out = [
        LotInfo(lot.id, lot.name, lot.type, lot.farm or "", lot.session_date or "", dict(counts[lot.id]))
        for lot in lots
    ]
    out.sort(key=lambda li: (li.session_date, li.id))
    return out


# ---------------------------------------------------------------------------
# Répartition
# ---------------------------------------------------------------------------
def _group_key(lot: LotInfo, group_by: str) -> str:
    if group_by == "session":
        return f"session:{lot.session_date or 'sans_date'}"
    return f"lot:{lot.id}"


def _assign_groups(
    groups: dict[str, list[LotInfo]],
    forced: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
    n_restarts: int,
) -> dict[str, str]:
    """Affecte chaque groupe libre à un sous-ensemble (recherche gloutonne randomisée)."""
    classes = sorted({c for lots in groups.values() for li in lots for c in li.counts})
    keys = sorted(groups)
    if not classes:
        return {k: forced.get(k, "train") for k in keys}
    cidx = {c: i for i, c in enumerate(classes)}
    vec = {}
    for k in keys:
        v = np.zeros(len(classes))
        for li in groups[k]:
            for c, n in li.counts.items():
                v[cidx[c]] += n
        vec[k] = v
    total = np.sum([vec[k] for k in keys], axis=0)
    # nombre de groupes contenant chaque classe : une classe dans ≥ 3 groupes doit être partout
    n_groups_with = np.sum([vec[k] > 0 for k in keys], axis=0)
    ratio = np.asarray(ratios)
    target = ratio[:, None] * total[None, :]  # [3, C]
    target_size = ratio * total.sum()
    weight = 1.0 / np.maximum(total, 1.0)
    free = [k for k in keys if k not in forced]
    base = np.zeros((3, len(classes)))
    for k, sub in forced.items():
        base[SUBSETS.index(sub)] += vec[k]
    active = ratio > 0

    def cost(cur: np.ndarray) -> float:
        dev = ((cur - target) * weight[None, :]) ** 2
        c = float(dev[active].sum())
        size_dev = (cur.sum(axis=1) - target_size) / max(total.sum(), 1.0)
        c += float((size_dev[active] ** 2).sum())
        for s in range(3):
            if not active[s]:
                continue
            missing = (cur[s] == 0) & (n_groups_with >= 3)
            c += 10.0 * float(missing.sum())
            if cur[s].sum() == 0:
                c += 100.0
        return c

    rng = np.random.default_rng(seed)
    best: tuple[float, dict[str, str]] | None = None
    # budget borné : beaucoup d'essais pour quelques dizaines de groupes, moins pour des milliers
    n_iter = max(1, min(n_restarts, max(10, 30_000 // max(1, len(free))))) if free else 1
    for it in range(n_iter):
        order = list(free)
        if it == 0:
            order.sort(key=lambda k: -vec[k].sum())  # premier essai : plus gros groupes d'abord
        else:
            rng.shuffle(order)
        cur = base.copy()
        assign = dict(forced)
        for k in order:
            best_s, best_c = None, None
            for s in range(3):
                if not active[s]:
                    continue
                trial = cur.copy()
                trial[s] += vec[k]
                c = cost(trial)
                if best_c is None or c < best_c - 1e-12:
                    best_s, best_c = s, c
            assert best_s is not None
            cur[best_s] += vec[k]
            assign[k] = SUBSETS[best_s]
        c = cost(cur)
        if best is None or c < best[0] - 1e-12:
            best = (c, assign)
    assert best is not None
    return best[1]


def compute_split(lots: list[LotInfo], params: SplitParams) -> dict[str, Any]:
    """Calcule un découpage (sans l'enregistrer) : ``{"assignments", "summary", "lots"}``."""
    params.validate()
    excluded = set(params.exclude_lots)
    lots = [li for li in lots if li.id not in excluded]
    if not lots:
        raise SplitError("Aucun œuf utilisable (verdict viable / non viable) : impossible de découper.")
    groups: dict[str, list[LotInfo]] = defaultdict(list)
    for li in lots:
        groups[_group_key(li, params.group_by)].append(li)
    farms = {f.strip() for f in params.holdout_farms if f and f.strip()}
    dates = {d.strip() for d in params.holdout_dates if d and d.strip()}
    forced: dict[str, str] = {}
    pulled: list[str] = []
    for k, members in groups.items():
        hit = [li for li in members if li.farm in farms or li.session_date in dates]
        if hit:
            forced[k] = "test"
            pulled += [li.id for li in members if li not in hit]
    assign_groups = _assign_groups(groups, forced, params.ratios, params.seed, params.n_restarts)
    assignments = {li.id: assign_groups[_group_key(li, params.group_by)] for li in lots}
    summary = summarize(lots, assignments, params)
    if pulled:
        summary["warnings"].append(
            f"{len(pulled)} lot(s) placé(s) en test car ils partagent une session avec un élevage ou une journée réservé(e)."
        )
    unknown_farms = farms - {li.farm for li in lots}
    unknown_dates = dates - {li.session_date for li in lots}
    if unknown_farms:
        summary["warnings"].append("Élevage(s) réservé(s) sans œuf utilisable : " + ", ".join(sorted(unknown_farms)))
    if unknown_dates:
        summary["warnings"].append("Journée(s) réservée(s) sans œuf utilisable : " + ", ".join(sorted(unknown_dates)))
    return {"assignments": assignments, "summary": summary, "lots": [lot_row(li, assignments) for li in lots]}


def lot_row(li: LotInfo, assignments: dict[str, str]) -> dict[str, Any]:
    return {**li.to_dict(), "subset": assignments.get(li.id)}


def _empty_counts() -> dict[str, Any]:
    return {"n_lots": 0, "n_eggs": 0, "viable": 0, "non_viable": 0, "defects": {}}


def summarize(lots: Iterable[LotInfo], assignments: dict[str, str], params: SplitParams | None = None) -> dict[str, Any]:
    """Effectifs par sous-ensemble, contrôle d'étanchéité et avertissements."""
    lots = list(lots)
    subsets = {s: _empty_counts() for s in SUBSETS}
    all_classes: set[str] = set()
    session_subsets: dict[str, set[str]] = defaultdict(set)
    farm_subsets: dict[str, set[str]] = defaultdict(set)
    for li in lots:
        sub = assignments.get(li.id)
        if sub not in subsets:
            continue
        e = subsets[sub]
        e["n_lots"] += 1
        e["n_eggs"] += li.n
        for c, n in li.counts.items():
            all_classes.add(c)
            if c == "viable":
                e["viable"] += n
            else:
                e["non_viable"] += n
                e["defects"][c] = e["defects"].get(c, 0) + n
        session_subsets[li.session_date].add(sub)
        farm_subsets[li.farm].add(sub)
    total = sum(e["n_eggs"] for e in subsets.values()) or 1
    for e in subsets.values():
        e["fraction"] = round(e["n_eggs"] / total, 4)
        e["defects"] = dict(sorted(e["defects"].items()))

    warnings: list[str] = []
    for sub in SUBSETS:
        e = subsets[sub]
        ratio = params.ratios[SUBSETS.index(sub)] if params else None
        if e["n_eggs"] == 0:
            if ratio is None or ratio > 0:
                warnings.append(f"Le jeu de {SUBSET_LABELS[sub]} est vide.")
            continue
        for c in sorted(all_classes):
            n = e["viable"] if c == "viable" else e["defects"].get(c, 0)
            if n == 0:
                what = "œuf viable" if c == "viable" else f"œuf « {c} »"
                warnings.append(f"Aucun {what} dans le jeu de {SUBSET_LABELS[sub]}.")

    if params is not None and any(subsets[s]["n_eggs"] for s in SUBSETS):
        gaps = [
            f"{SUBSET_LABELS[s]} {subsets[s]['fraction']:.0%} (demandé {params.ratios[i]:.0%})"
            for i, s in enumerate(SUBSETS)
            if abs(subsets[s]["fraction"] - params.ratios[i]) > 0.10
        ]
        if gaps:
            warnings.append(
                "Ratios non atteints — " + ", ".join(gaps)
                + " : trop peu de groupes pour concilier les ratios et la présence de chaque classe dans chaque sous-ensemble."
            )

    # Étanchéité : vérifiée et non supposée.
    lot_ids = [li.id for li in lots]
    duplicated = sorted({x for x in lot_ids if lot_ids.count(x) > 1})
    shared_sessions = sorted(d for d, subs in session_subsets.items() if len(subs) > 1)
    by_session = params is not None and params.group_by == "session"
    leakage = {
        "unit": "session" if by_session else "lot",
        "lots_in_one_subset": not duplicated and all(assignments.get(i) in SUBSETS for i in lot_ids),
        "shared_sessions": shared_sessions,
        "ok": not duplicated and (not by_session or not shared_sessions),
    }
    if by_session and shared_sessions:
        warnings.append(
            "Fuite possible : des lots d'une même session sont répartis dans plusieurs sous-ensembles ("
            + ", ".join(shared_sessions)
            + ")."
        )
    if params is not None:
        farms = {f for f in params.holdout_farms if f}
        dates = {d for d in params.holdout_dates if d}
        bad = [li.id for li in lots if (li.farm in farms or li.session_date in dates) and assignments.get(li.id) != "test"]
        leakage["holdout_respected"] = not bad
        if bad:
            leakage["ok"] = False
            warnings.append("Lots réservés au test placés ailleurs : " + ", ".join(bad))
    return {
        "subsets": subsets,
        "total_eggs": sum(e["n_eggs"] for e in subsets.values()),
        "total_lots": sum(e["n_lots"] for e in subsets.values()),
        "classes": sorted(all_classes),
        "leakage_check": leakage,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------
def split_used_by(session: Session, split_id: int) -> list[str]:
    return list(session.scalars(select(m.Model.id).where(m.Model.split_id == split_id)))


def split_to_dict(session: Session, split: m.DatasetSplit, with_assignments: bool = True) -> dict[str, Any]:
    used = split_used_by(session, split.id)
    d = {
        "id": split.id,
        "name": split.name,
        "version": split.version,
        "created_at": split.created_at.isoformat() if split.created_at else None,
        "strategy": split.strategy or {},
        "summary": split.summary or {},
        "used_by": used,
        "locked": bool(used),
    }
    if with_assignments:
        d["assignments"] = split.assignments or {}
    return d


def preview_split(project: Project, params: SplitParams) -> dict[str, Any]:
    with project.session() as s:
        lots = lot_inventory(s, params.reviewed_only)
    return compute_split(lots, params)


def create_split(project: Project, name: str, params: SplitParams) -> dict[str, Any]:
    """Calcule et enregistre un découpage. Si le nom existe déjà, crée la version suivante."""
    name = (name or "").strip() or "Découpage"
    result = preview_split(project, params)
    with project.session() as s:
        version = (s.scalar(select(func.max(m.DatasetSplit.version)).where(m.DatasetSplit.name == name)) or 0) + 1
        row = m.DatasetSplit(
            name=name,
            version=version,
            strategy=params.to_dict(),
            assignments=result["assignments"],
            summary=result["summary"],
        )
        s.add(row)
        s.flush()
        return split_to_dict(s, row)


def update_assignments(
    project: Project, split_id: int, changes: dict[str, str], move_group: bool = True
) -> dict[str, Any]:
    """Modifie l'affectation de lots. Découpage utilisé par un modèle → nouvelle version.

    ``move_group`` : en regroupement par session, déplace aussi les autres lots de la
    même session (sinon le contrôle d'étanchéité signale la fuite).
    """
    for lot_id, sub in changes.items():
        if sub not in SUBSETS and sub is not None:
            raise SplitError(f"Sous-ensemble inconnu pour {lot_id} : {sub!r}")
    with project.session() as s:
        split = s.get(m.DatasetSplit, split_id)
        if split is None:
            raise LookupError("Découpage introuvable.")
        params = SplitParams.from_dict(split.strategy)
        lots = {li.id: li for li in lot_inventory(s, params.reviewed_only)}
        assignments = dict(split.assignments or {})
        for lot_id, sub in changes.items():
            if lot_id not in lots and lot_id not in assignments:
                raise SplitError(f"Lot inconnu ou sans œuf utilisable : {lot_id}")
            targets = [lot_id]
            if move_group and params.group_by == "session" and lot_id in lots:
                date = lots[lot_id].session_date
                targets = [i for i, li in lots.items() if li.session_date == date]
            for t in targets:
                if sub is None:
                    assignments.pop(t, None)
                else:
                    assignments[t] = sub
        summary = summarize([lots[i] for i in assignments if i in lots], assignments, params)
        strategy = {**(split.strategy or {}), "manual_changes": int((split.strategy or {}).get("manual_changes", 0)) + len(changes)}
        if split_used_by(s, split.id):
            row = m.DatasetSplit(
                name=split.name,
                version=(s.scalar(select(func.max(m.DatasetSplit.version)).where(m.DatasetSplit.name == split.name)) or 0) + 1,
                strategy={**strategy, "derived_from": split.id},
                assignments=assignments,
                summary=summary,
            )
            s.add(row)
            s.flush()
            out = split_to_dict(s, row)
            out["new_version"] = True
            return out
        split.assignments = assignments
        split.summary = summary
        split.strategy = strategy
        s.flush()
        out = split_to_dict(s, split)
        out["new_version"] = False
        return out


def split_detail(project: Project, split_id: int) -> dict[str, Any]:
    """Découpage enregistré + état actuel (labels modifiés depuis, lots non affectés)."""
    with project.session() as s:
        split = s.get(m.DatasetSplit, split_id)
        if split is None:
            raise LookupError("Découpage introuvable.")
        params = SplitParams.from_dict(split.strategy)
        lots = lot_inventory(s, params.reviewed_only)
        assignments = split.assignments or {}
        out = split_to_dict(s, split)
        out["current_summary"] = summarize([li for li in lots if li.id in assignments], assignments, params)
        out["lots"] = [lot_row(li, assignments) for li in lots if li.id in assignments]
        out["unassigned_lots"] = [li.to_dict() for li in lots if li.id not in assignments]
        return out


def list_splits(project: Project, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    with project.session() as s:
        total = s.scalar(select(func.count()).select_from(m.DatasetSplit)) or 0
        rows = s.scalars(
            select(m.DatasetSplit).order_by(m.DatasetSplit.created_at.desc(), m.DatasetSplit.id.desc()).limit(limit).offset(offset)
        ).all()
        return {"total": total, "items": [split_to_dict(s, r, with_assignments=False) for r in rows]}


def delete_split(project: Project, split_id: int) -> None:
    with project.session() as s:
        split = s.get(m.DatasetSplit, split_id)
        if split is None:
            raise LookupError("Découpage introuvable.")
        if split_used_by(s, split_id):
            raise SplitError("Ce découpage est utilisé par au moins un modèle : suppression impossible.")
        s.delete(split)


def split_options(project: Project) -> dict[str, Any]:
    """Élevages et journées disponibles (pour la réservation au test)."""
    with project.session() as s:
        lots = lot_inventory(s, False)
    farms: dict[str, int] = defaultdict(int)
    dates: dict[str, int] = defaultdict(int)
    for li in lots:
        farms[li.farm] += li.n
        dates[li.session_date] += li.n
    return {
        "farms": [{"farm": f, "n_eggs": n} for f, n in sorted(farms.items())],
        "dates": [{"date": d, "n_eggs": n} for d, n in sorted(dates.items())],
        "lots": [li.to_dict() for li in lots],
        "defaults": SplitParams().to_dict(),
    }
