"""Constitution des colonnes d'une simulation : tirage déterministe dans le jeu de TEST.

Règles (cahier des charges § 8.3) :

* seules les pistes des lots affectés à ``"test"`` dans le découpage du modèle, avec
  un verdict ``viable`` ou ``non_viable`` et au moins une vue, sont tirables — jamais
  une piste d'entraînement ou de validation (garde-fou vérifié après le tirage) ;
* la proportion de non viables demandée est respectée exactement (arrondi à l'œuf près) ;
* tirage sans remise tant que le jeu test suffit, sinon avec remise équilibrée
  (permutations successives : chaque œuf est réutilisé le même nombre de fois, à un près),
  signalée dans les résultats (œufs distincts, taux de réutilisation) ;
* couloirs vides optionnels, vues manquantes optionnelles ;
* tout dépend uniquement de la graine : même graine + même jeu test ⇒ même composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from eggvision_shared.constants import VERDICT_NON_VIABLE, VERDICT_VIABLE
from eggvision_shared.preprocess import select_view_indices
from sqlalchemy import select

from ..db import models as m
from ..project import Project
from .params import SimulationParams


class SimulationError(Exception):
    """Erreur présentable à l'utilisateur (message en français)."""


@dataclass
class PoolEgg:
    """Un œuf tirable du jeu test."""

    track_id: int
    lot_id: str
    verdict: str
    defect: str | None
    view_indices: list[int]  # View.index, triés
    view_paths: list[str]  # relatifs au projet, même ordre


@dataclass
class TestPool:
    """Jeu test d'un modèle (pistes tirables)."""

    split_id: int
    test_lots: list[str]
    viable: list[PoolEgg]
    non_viable: list[PoolEgg]
    excluded_no_verdict: int = 0
    excluded_no_views: int = 0

    def summary(self) -> dict[str, Any]:
        by_defect: dict[str, int] = {}
        for e in self.non_viable:
            key = e.defect or "non_precise"
            by_defect[key] = by_defect.get(key, 0) + 1
        return {
            "split_id": self.split_id,
            "n_test_lots": len(self.test_lots),
            "n_viable": len(self.viable),
            "n_non_viable": len(self.non_viable),
            "n_total": len(self.viable) + len(self.non_viable),
            "by_defect": dict(sorted(by_defect.items())),
            "excluded_no_verdict": self.excluded_no_verdict,
            "excluded_no_views": self.excluded_no_views,
        }


@dataclass
class Slot:
    """Un œuf placé dans une colonne."""

    egg: PoolEgg
    presented: list[int]  # View.index réellement présentés (après sélection et vues manquantes)
    presented_paths: list[str]
    n_selected: int  # vues qu'aurait vues la caméra sans perte


@dataclass
class Composition:
    columns: list[list[Slot | None]]
    jitter_ms: np.ndarray  # gigue tirée pour chaque colonne
    info: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> list[list[tuple[int, tuple[int, ...]] | None]]:
        """Représentation comparable (pistes et vues présentées), utile aux tests de reproductibilité."""
        return [[None if s is None else (s.egg.track_id, tuple(s.presented)) for s in col] for col in self.columns]


# ---------------------------------------------------------------------------
# Jeu test
# ---------------------------------------------------------------------------
def model_split(project: Project, model_id: str) -> tuple[m.Model, m.DatasetSplit]:
    """Modèle + découpage, avec les erreurs claires attendues par l'interface."""
    with project.session() as s:
        model = s.get(m.Model, model_id)
        if model is None:
            raise SimulationError(f"Modèle introuvable : {model_id}")
        if model.split_id is None:
            raise SimulationError(f"Le modèle « {model.name} » n'a pas de découpage entraînement / validation / test.")
        split = s.get(m.DatasetSplit, model.split_id)
        if split is None:
            raise SimulationError(f"Le découpage n° {model.split_id} du modèle « {model.name} » est introuvable.")
        s.expunge(model)
        s.expunge(split)
    return model, split


def load_test_pool(project: Project, split: m.DatasetSplit) -> TestPool:
    """Charge les pistes tirables du jeu test (verdict viable / non viable, au moins une vue)."""
    assignments = split.assignments or {}
    test_lots = sorted(lot for lot, subset in assignments.items() if subset == "test")
    pool = TestPool(split_id=split.id, test_lots=test_lots, viable=[], non_viable=[])
    if not test_lots:
        return pool
    with project.session() as s:
        rows = s.execute(
            select(m.Track.id, m.Track.lot_id, m.Label.verdict, m.Label.defect)
            .outerjoin(m.Label, m.Label.track_id == m.Track.id)
            .where(m.Track.lot_id.in_(test_lots))
            .order_by(m.Track.id)
        ).all()
        views: dict[int, list[tuple[int, str]]] = {}
        view_rows = s.execute(
            select(m.View.track_id, m.View.index, m.View.path)
            .join(m.Track, m.Track.id == m.View.track_id)
            .where(m.Track.lot_id.in_(test_lots))
            .order_by(m.View.track_id, m.View.index)
        )
        for track_id, index, path in view_rows:
            views.setdefault(track_id, []).append((index, path))
    for track_id, lot_id, verdict, defect in rows:
        if verdict not in (VERDICT_VIABLE, VERDICT_NON_VIABLE):
            pool.excluded_no_verdict += 1
            continue
        tv = views.get(track_id) or []
        if not tv:
            pool.excluded_no_views += 1
            continue
        egg = PoolEgg(
            track_id=int(track_id),
            lot_id=str(lot_id),
            verdict=verdict,
            defect=defect if verdict == VERDICT_NON_VIABLE else None,
            view_indices=[int(i) for i, _ in tv],
            view_paths=[p for _, p in tv],
        )
        (pool.viable if verdict == VERDICT_VIABLE else pool.non_viable).append(egg)
    return pool


# ---------------------------------------------------------------------------
# Tirage
# ---------------------------------------------------------------------------
def _draw(rng: np.random.Generator, n_pool: int, k: int) -> np.ndarray:
    """``k`` indices parmi ``n_pool`` : sans remise si possible, sinon permutations successives."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k <= n_pool:
        return rng.choice(n_pool, size=k, replace=False)
    parts = []
    remaining = k
    while remaining > 0:
        perm = rng.permutation(n_pool)
        parts.append(perm[: min(remaining, n_pool)])
        remaining -= n_pool
    return np.concatenate(parts)


def build_composition(pool: TestPool, params: SimulationParams, n_views_model: int) -> Composition:
    """Constitue les colonnes (déterministe pour une graine donnée)."""
    if params.seed is None:
        raise ValueError("La graine doit être fixée avant la constitution des colonnes")
    if not pool.test_lots:
        raise SimulationError("Le découpage du modèle ne contient aucun lot de test.")
    if not pool.viable and not pool.non_viable:
        raise SimulationError("Le jeu test ne contient aucun œuf labellisé viable ou non viable.")

    # Flux aléatoires indépendants : modifier un paramètre de robustesse ne change pas la composition.
    ss_lanes, ss_eggs, ss_views, ss_jitter = np.random.SeedSequence(params.seed).spawn(4)
    rng_lanes = np.random.default_rng(ss_lanes)
    rng_eggs = np.random.default_rng(ss_eggs)
    rng_views = np.random.default_rng(ss_views)
    rng_jitter = np.random.default_rng(ss_jitter)

    n_cols, n_lanes = params.n_columns, params.n_lanes
    occupied = rng_lanes.random((n_cols, n_lanes)) >= params.empty_lane_rate
    n_eggs = int(occupied.sum())
    if n_eggs == 0:
        raise SimulationError("Tous les couloirs sont vides : réduire la proportion de couloirs vides.")
    n_def = int(round(params.defect_rate * n_eggs))
    n_ok = n_eggs - n_def
    if n_def > 0 and not pool.non_viable:
        raise SimulationError("Le jeu test ne contient aucun œuf non viable : impossible de respecter la proportion de défauts.")
    if n_ok > 0 and not pool.viable:
        raise SimulationError("Le jeu test ne contient aucun œuf viable : impossible de respecter la proportion de défauts.")

    picked_def = _draw(rng_eggs, len(pool.non_viable), n_def)
    picked_ok = _draw(rng_eggs, len(pool.viable), n_ok)
    eggs = [pool.non_viable[i] for i in picked_def] + [pool.viable[i] for i in picked_ok]
    order = rng_eggs.permutation(len(eggs))

    test_lots = set(pool.test_lots)
    columns: list[list[Slot | None]] = []
    it = iter(order)
    n_missing = 0
    n_selected_total = 0
    for c in range(n_cols):
        col: list[Slot | None] = []
        for lane in range(n_lanes):
            if not occupied[c, lane]:
                col.append(None)
                continue
            egg = eggs[int(next(it))]
            sel = select_view_indices(len(egg.view_indices), n_views_model)
            keep = rng_views.random(len(sel)) >= params.missing_view_rate if params.missing_view_rate > 0 else np.ones(len(sel), bool)
            presented = [egg.view_indices[j] for j, k in zip(sel, keep, strict=True) if k]
            paths = [egg.view_paths[j] for j, k in zip(sel, keep, strict=True) if k]
            n_selected_total += len(sel)
            n_missing += len(sel) - len(presented)
            col.append(Slot(egg=egg, presented=presented, presented_paths=paths, n_selected=len(sel)))
        columns.append(col)

    # Garde-fou : aucune piste hors du jeu test (entraînement / validation) ne doit apparaître.
    leaks = {s.egg.lot_id for col in columns for s in col if s is not None and s.egg.lot_id not in test_lots}
    if leaks:
        raise SimulationError(f"Fuite de données détectée : pistes hors jeu test ({sorted(leaks)}).")

    jitter = rng_jitter.random(n_cols) * params.latency_jitter_ms if params.latency_jitter_ms > 0 else np.zeros(n_cols)

    distinct = len({e.track_id for e in eggs})
    info = {
        "n_eggs": n_eggs,
        "n_slots": n_cols * n_lanes,
        "n_empty_slots": n_cols * n_lanes - n_eggs,
        "n_non_viable": n_def,
        "n_viable": n_ok,
        "defect_rate_requested": params.defect_rate,
        "defect_rate_actual": round(n_def / n_eggs, 6),
        "n_distinct_eggs": distinct,
        "reuse_rate": round(1.0 - distinct / n_eggs, 6),
        "with_replacement": bool(n_def > len(pool.non_viable) or n_ok > len(pool.viable)),
        "n_views_model": n_views_model,
        "n_views_selected": n_selected_total,
        "n_views_missing": n_missing,
        "n_eggs_without_views": sum(1 for col in columns for s in col if s is not None and not s.presented),
        "pool": pool.summary(),
    }
    return Composition(columns=columns, jitter_ms=np.asarray(jitter, dtype=np.float64), info=info)
