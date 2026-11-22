"""Moteur de simulation à horloge virtuelle.

La simulation ne tourne PAS en temps réel : chaque colonne reçoit des instants
virtuels calculés à partir de la cadence et de la géométrie ; seul l'appel au
modèle ONNX est réellement exécuté (et chronométré).

Modèle temporel, pour la colonne ``k`` (``P`` = période, ``v`` = vitesse) ::

    t_entree   = k · P                                 entrée dans la zone caméra
    vue j/N    = t_entree + (j + ½) / N · T_cam        présentation des vues (régulière)
    t_acq      = t_entree + T_cam                      fin d'acquisition (colonne prête)
    calcul     = latence_mesurée · facteur + latence_injectée + gigue
    début      = max(t_acq, première unité d'inférence libre)   (file FIFO, ``workers`` unités)
    t_decision = début + calcul
    t_trappes  = t_acq + B                              B = distance caméra→trappes / v (budget)
    échéance   = t_trappes − temps d'ouverture des trappes
    en retard  ⇔ t_decision > échéance

Quand l'inférence est plus lente que la cadence, la file s'allonge et l'attente
s'accumule de colonne en colonne jusqu'à dépasser le budget.

Une décision en retard n'est pas appliquée : l'automate ignore une commande
arrivée après l'échéance (sinon la trappe s'ouvrirait sous l'œuf suivant) ; un œuf
non viable dont la décision est en retard passe donc dans le bac des viables
(issue « manqué (retard) »). Un œuf sans aucune vue ne reçoit pas de décision et
passe dans les viables.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from eggvision_shared.constants import VERDICT_NON_VIABLE
from eggvision_shared.preprocess import load_view
from eggvision_shared.runtime import CONFIG_FILENAME, EggClassifier

from ..project import Project
from .composition import Composition, SimulationError
from .params import SimulationParams

log = logging.getLogger(__name__)

# Issues possibles d'un œuf (identifiants stables : journal, chronologie, interface).
EJECTED_OK = "ejected_ok"
MISSED = "missed"
MISSED_LATE = "missed_late"
FALSE_REJECT = "false_reject"
ACCEPTED_OK = "accepted_ok"
OUTCOMES = (EJECTED_OK, MISSED, MISSED_LATE, FALSE_REJECT, ACCEPTED_OK)
OUTCOME_LABELS = {
    EJECTED_OK: "Éjecté à raison",
    MISSED: "Manqué",
    MISSED_LATE: "Manqué (retard)",
    FALSE_REJECT: "Éjecté à tort",
    ACCEPTED_OK: "Accepté à raison",
}


# ---------------------------------------------------------------------------
# Cache des vues décodées
# ---------------------------------------------------------------------------
class ViewCache:
    """Petit cache LRU d'images décodées (mémoire bornée : ``capacity`` vues)."""

    def __init__(self, root: Path, capacity: int = 768) -> None:
        self.root = root
        self.capacity = max(1, capacity)
        self._items: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.failures = 0

    def get(self, relpath: str) -> np.ndarray | None:
        if relpath in self._items:
            self._items.move_to_end(relpath)
            self.hits += 1
            return self._items[relpath]
        self.misses += 1
        try:
            img: np.ndarray | None = load_view(str(self.root / relpath))
        except Exception:  # fichier absent ou illisible : la vue est considérée manquante
            img = None
        if img is None:
            self.failures += 1
        self._items[relpath] = img
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return img

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "failures": self.failures, "capacity": self.capacity}


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
def package_dir_of(project: Project, export: dict[str, Any] | None) -> Path | None:
    """Dossier du paquet de déploiement d'un modèle exporté, ou ``None``."""
    rel = (export or {}).get("package_dir")
    if not rel:
        return None
    path = project.abspath(rel)
    return path if (path / CONFIG_FILENAME).is_file() else None


def load_classifier(project: Project, model_export: dict[str, Any] | None, params: SimulationParams) -> EggClassifier:
    pkg = package_dir_of(project, model_export)
    if pkg is None:
        raise SimulationError("Le modèle n'est pas exporté (paquet de déploiement introuvable).")
    try:
        return EggClassifier.load(str(pkg), variant=params.variant, threads=params.inference_threads, threshold=params.threshold)
    except (ValueError, NotImplementedError, OSError) as exc:
        raise SimulationError(f"Impossible de charger le modèle : {exc}") from exc


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
@dataclass
class RunOutput:
    columns: list[dict[str, Any]]  # une entrée par colonne (chronologie)
    eggs: list[dict[str, Any]]  # une entrée par œuf (journal)
    wall_s: float
    cache: dict[str, int]
    threshold: float
    variant: str | None


def _outcome(verdict: str, decision: int | None, late: bool) -> str:
    fired = decision == 1 and not late
    if verdict == VERDICT_NON_VIABLE:
        if fired:
            return EJECTED_OK
        return MISSED_LATE if (decision == 1 and late) else MISSED
    return FALSE_REJECT if fired else ACCEPTED_OK


def run(
    project: Project,
    params: SimulationParams,
    composition: Composition,
    classifier: EggClassifier,
    *,
    check_cancelled: Callable[[], None] = lambda: None,
    progress: Callable[[int, int, dict[str, Any]], None] = lambda *_: None,
    cache_capacity: int = 768,
) -> RunOutput:
    """Déroule la simulation colonne par colonne (inférence réelle, horloge virtuelle)."""
    period = params.period_s
    cam_t = params.camera_time_s
    budget = params.budget_s
    actuation = params.trap_actuation_ms / 1000.0
    cache = ViewCache(project.root, cache_capacity)
    free_at = [0.0] * params.workers  # instants où chaque unité d'inférence se libère
    heapq.heapify(free_at)

    # Préchauffage (première exécution ONNX plus lente : ne pas la compter dans la latence).
    first = next((col for col in composition.columns if any(s is not None and s.presented for s in col)), None)
    if first is not None:
        warm = []
        for s in first:
            imgs = [img for img in (cache.get(p) for p in s.presented_paths) if img is not None] if s is not None else []
            warm.append(imgs or None)
        if any(v is not None for v in warm):
            for _ in range(2):
                classifier.predict_column(warm)

    columns_out: list[dict[str, Any]] = []
    eggs_out: list[dict[str, Any]] = []
    n_cols = len(composition.columns)
    step = max(1, n_cols // 200)
    n_late = 0
    cum = [0, 0, 0, 0]  # cumuls : bac rebut, dont à tort, bac viables, dont non viables
    t_wall = time.perf_counter()
    for k, col in enumerate(composition.columns):
        check_cancelled()
        t_enter = k * period
        t_acq = t_enter + cam_t
        t_trap = t_acq + budget
        deadline = t_trap - actuation

        lanes: list[list[np.ndarray] | None] = []
        presented_idx: list[list[int] | None] = []
        for slot in col:
            if slot is None:
                lanes.append(None)
                presented_idx.append(None)
                continue
            imgs, idxs = [], []
            for vi, path in zip(slot.presented, slot.presented_paths, strict=True):
                img = cache.get(path)
                if img is not None:
                    imgs.append(img)
                    idxs.append(vi)
            lanes.append(imgs or None)
            presented_idx.append(idxs)

        needs_decision = any(v is not None for v in lanes)
        if needs_decision:
            t0 = time.perf_counter()
            result = classifier.predict_column(lanes)
            measured_ms = (time.perf_counter() - t0) * 1000.0
            injected_ms = params.injected_latency_ms + float(composition.jitter_ms[k])
            compute_ms = measured_ms * params.latency_scale + injected_ms
            start = max(t_acq, heapq.heappop(free_at))
            t_decision = start + compute_ms / 1000.0
            heapq.heappush(free_at, t_decision)
            wait_ms = (start - t_acq) * 1000.0
            decisions: list[int | None] = list(result.decisions)
            scores = result.scores
            view_scores = result.view_scores
        else:  # colonne sans aucun œuf à juger : aucune décision à prendre
            measured_ms = injected_ms = compute_ms = wait_ms = 0.0
            t_decision = t_acq
            decisions = [None] * len(col)
            scores = [None] * len(col)
            view_scores = [None] * len(col)

        late = needs_decision and t_decision > deadline + 1e-12
        n_late += int(late)
        lanes_out: list[dict[str, Any] | None] = []
        for lane, slot in enumerate(col):
            if slot is None:
                lanes_out.append(None)
                continue
            has_views = lanes[lane] is not None
            decision = int(decisions[lane]) if (has_views and decisions[lane] is not None) else None
            outcome = _outcome(slot.egg.verdict, decision, late)
            score = scores[lane] if has_views else None
            egg = {
                "column": k,
                "lane": lane,
                "track_id": slot.egg.track_id,
                "lot_id": slot.egg.lot_id,
                "truth": slot.egg.verdict,
                "defect": slot.egg.defect,
                "n_views": len(presented_idx[lane] or []),
                "n_views_expected": slot.n_selected,
                "views": presented_idx[lane] or [],
                "score": None if score is None else round(float(score), 6),
                "view_scores": None if view_scores[lane] is None else [round(float(x), 5) for x in view_scores[lane]],
                "decision": decision,
                "late": bool(late and decision is not None),
                "ejected": outcome in (EJECTED_OK, FALSE_REJECT),
                "outcome": outcome,
            }
            eggs_out.append(egg)
            if egg["ejected"]:
                cum[0] += 1
                cum[1] += int(outcome == FALSE_REJECT)
            else:
                cum[2] += 1
                cum[3] += int(outcome in (MISSED, MISSED_LATE))
            lanes_out.append(egg)
        columns_out.append(
            {
                "k": k,
                "t_enter": round(t_enter, 6),
                "t_acq": round(t_acq, 6),
                "t_decision": round(t_decision, 6),
                "t_deadline": round(deadline, 6),
                "t_trap": round(t_trap, 6),
                "measured_ms": round(measured_ms, 4),
                "injected_ms": round(injected_ms, 4),
                "compute_ms": round(compute_ms, 4),
                "wait_ms": round(wait_ms, 4),
                "response_ms": round((t_decision - t_acq) * 1000.0, 4),
                "margin_ms": round((deadline - t_decision) * 1000.0, 4),
                "decided": needs_decision,
                "late": late,
                "cum": list(cum),
                "lanes": lanes_out,
            }
        )
        if (k + 1) % step == 0 or k + 1 == n_cols:
            progress(k + 1, n_cols, {"late_columns": n_late})
    return RunOutput(
        columns=columns_out,
        eggs=eggs_out,
        wall_s=time.perf_counter() - t_wall,
        cache=cache.stats(),
        threshold=float(classifier.threshold),
        variant=getattr(classifier, "variant", None),
    )
