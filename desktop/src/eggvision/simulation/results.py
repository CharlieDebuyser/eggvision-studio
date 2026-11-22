"""Résultats d'une simulation : agrégats, journal CSV, chronologie, comparaison.

Fichiers écrits dans ``simulations/<id>/`` :

* ``params.json``   paramètres complets + grandeurs dérivées + composition ;
* ``journal.csv``   une ligne par œuf (séparateur ``;``, en-têtes en français, UTF-8 avec BOM) ;
* ``timeline.json`` chronologie compacte (une entrée par colonne) pour l'animation et la vidéo ;
* ``video.mp4``     (facultatif) animation rendue côté serveur.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from eggvision_shared.constants import DEFECT_LABELS, VERDICT_NON_VIABLE, VERDICT_VIABLE

from .engine import ACCEPTED_OK, EJECTED_OK, FALSE_REJECT, MISSED, MISSED_LATE, OUTCOME_LABELS, RunOutput
from .params import SimulationParams

TIMELINE_VERSION = 1
PARAMS_FILE = "params.json"
JOURNAL_FILE = "journal.csv"
TIMELINE_FILE = "timeline.json"
VIDEO_FILE = "video.mp4"

JOURNAL_HEADERS = [
    ("column", "colonne"),
    ("lane", "couloir"),
    ("track_id", "piste"),
    ("lot_id", "lot"),
    ("truth", "verite"),
    ("defect", "defaut"),
    ("n_views", "vues_presentees"),
    ("n_views_expected", "vues_attendues"),
    ("score", "score"),
    ("decision", "decision"),
    ("late", "en_retard"),
    ("outcome_label", "issue"),
    ("measured_ms", "latence_mesuree_ms"),
    ("compute_ms", "latence_calcul_ms"),
    ("wait_ms", "attente_file_ms"),
    ("response_ms", "delai_reponse_ms"),
    ("margin_ms", "marge_ms"),
    ("t_enter", "t_entree_s"),
    ("t_decision", "t_decision_s"),
    ("t_deadline", "t_echeance_s"),
    ("t_trap", "t_trappes_s"),
]


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "p99": None, "max": None, "min": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "max": round(float(a.max()), 4),
        "min": round(float(a.min()), 4),
    }


def _histogram(values: list[float], budget_ms: float, bins: int = 40) -> dict[str, Any]:
    """Histogramme des délais de réponse. Si le budget dépasse largement les délais observés,
    l'échelle s'arrête aux délais (``budget_in_range = False``) pour rester lisible."""
    if not values:
        return {"edges": [], "counts": [], "budget_ms": budget_ms, "budget_in_range": False}
    vmax = max(values)
    in_range = budget_ms <= 3.0 * vmax
    hi = (max(vmax, budget_ms) * 1.05 if in_range else vmax * 1.1) or 1.0
    counts, edges = np.histogram(np.asarray(values), bins=bins, range=(0.0, hi))
    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "budget_ms": budget_ms,
        "budget_in_range": bool(in_range),
    }


def _counts(eggs: list[dict[str, Any]]) -> dict[str, int]:
    c = {o: 0 for o in (EJECTED_OK, MISSED, MISSED_LATE, FALSE_REJECT, ACCEPTED_OK)}
    for e in eggs:
        c[e["outcome"]] += 1
    return c


def aggregate(run: RunOutput, params: SimulationParams, composition_info: dict[str, Any]) -> dict[str, Any]:
    """Résumé chiffré stocké dans ``Simulation.results``."""
    eggs = run.eggs
    nv = [e for e in eggs if e["truth"] == VERDICT_NON_VIABLE]
    vi = [e for e in eggs if e["truth"] == VERDICT_VIABLE]
    c = _counts(eggs)
    n_nv, n_vi = len(nv), len(vi)
    late_eggs = sum(1 for e in eggs if e["late"])
    missed_traps = sum(1 for e in eggs if e["late"] and e["decision"] == 1)
    no_decision = sum(1 for e in eggs if e["decision"] is None)
    model_flagged_nv = sum(1 for e in nv if e["decision"] == 1)
    model_flagged_vi = sum(1 for e in vi if e["decision"] == 1)
    ejected = c[EJECTED_OK] + c[FALSE_REJECT]
    passed = len(eggs) - ejected
    defects_in_viable_bin = c[MISSED] + c[MISSED_LATE]

    counts = {
        "eggs": len(eggs),
        "non_viable": n_nv,
        "viable": n_vi,
        EJECTED_OK: c[EJECTED_OK],
        MISSED: c[MISSED],
        MISSED_LATE: c[MISSED_LATE],
        "missed_total": c[MISSED] + c[MISSED_LATE],
        FALSE_REJECT: c[FALSE_REJECT],
        ACCEPTED_OK: c[ACCEPTED_OK],
        "late_decisions": late_eggs,
        "missed_traps": missed_traps,
        "no_decision": no_decision,
        "bin_reject": ejected,
        "bin_viable": passed,
    }
    rates = {
        "recall": _rate(c[EJECTED_OK], n_nv),  # non viables correctement éjectés (rappel effectif)
        "miss_rate": _rate(c[MISSED] + c[MISSED_LATE], n_nv),
        "miss_late_rate": _rate(c[MISSED_LATE], n_nv),
        "false_reject_rate": _rate(c[FALSE_REJECT], n_vi),
        "late_rate": _rate(late_eggs, len(eggs)),
        "model_recall": _rate(model_flagged_nv, n_nv),  # rappel du modèle seul (sans contrainte de temps)
        "model_false_reject_rate": _rate(model_flagged_vi, n_vi),
        "residual_defect_rate": _rate(defects_in_viable_bin, passed),  # défauts restant dans le bac viables
        "reject_bin_purity": _rate(c[EJECTED_OK], ejected),
        "accuracy": _rate(c[EJECTED_OK] + c[ACCEPTED_OK], len(eggs)),
    }

    by_defect: dict[str, dict[str, Any]] = {}
    for e in nv:
        key = e["defect"] or "non_precise"
        d = by_defect.setdefault(
            key, {"label": DEFECT_LABELS.get(key, "Non précisé"), "n": 0, EJECTED_OK: 0, MISSED: 0, MISSED_LATE: 0}
        )
        d["n"] += 1
        d[e["outcome"]] += 1
    for d in by_defect.values():
        d["recall"] = _rate(d[EJECTED_OK], d["n"])
    by_defect = dict(sorted(by_defect.items()))
    by_defect_viable = {"label": "Viables", "n": n_vi, FALSE_REJECT: c[FALSE_REJECT], ACCEPTED_OK: c[ACCEPTED_OK], "false_reject_rate": rates["false_reject_rate"]}

    decided = [col for col in run.columns if col["decided"]]
    measured = [col["measured_ms"] for col in decided]
    compute = [col["compute_ms"] for col in decided]
    wait = [col["wait_ms"] for col in decided]
    response = [col["response_ms"] for col in decided]
    margins = [col["margin_ms"] for col in decided]
    budget_ms = params.budget_s * 1000.0
    eff_budget_ms = params.effective_budget_s * 1000.0
    mean_compute = float(np.mean(compute)) if compute else 0.0
    capacity = (params.workers * 1000.0 / mean_compute) if mean_compute > 0 else None
    late_columns = sum(1 for col in decided if col["late"])
    latency = {
        "budget_ms": round(budget_ms, 4),
        "effective_budget_ms": round(eff_budget_ms, 4),
        "trap_actuation_ms": params.trap_actuation_ms,
        "measured_ms": _stats(measured),
        "compute_ms": _stats(compute),
        "wait_ms": _stats(wait),
        "response_ms": _stats(response),
        "min_margin_ms": round(min(margins), 4) if margins else None,
        "late_columns": late_columns,
        "late_column_rate": _rate(late_columns, len(decided)),
        "histogram": _histogram(response, eff_budget_ms),
    }
    throughput = {
        "required_columns_per_s": round(1.0 / params.period_s, 4),
        "capacity_columns_per_s": None if capacity is None else round(capacity, 4),
        "capacity_ratio": None if capacity is None else round(capacity * params.period_s, 4),
        "measured_columns_per_s": round(len(measured) / (sum(measured) / 1000.0), 3) if measured and sum(measured) > 0 else None,
        "wall_s": round(run.wall_s, 3),
        "wall_columns_per_s": round(len(run.columns) / run.wall_s, 3) if run.wall_s > 0 else None,
        "virtual_duration_s": round(run.columns[-1]["t_trap"], 3) if run.columns else 0.0,
    }
    return {
        "n_columns": len(run.columns),
        "threshold": run.threshold,
        "variant": run.variant,
        "counts": counts,
        "rates": rates,
        "by_defect": by_defect,
        "viable": by_defect_viable,
        "latency": latency,
        "throughput": throughput,
        "composition": composition_info,
        "cache": run.cache,
        "timing": params.timing(),
    }


# ---------------------------------------------------------------------------
# Fichiers
# ---------------------------------------------------------------------------
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_params(sim_dir: Path, payload: dict[str, Any]) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(sim_dir / PARAMS_FILE, json.dumps(payload, indent=2, ensure_ascii=False))


def write_journal(sim_dir: Path, run: RunOutput) -> Path:
    path = sim_dir / JOURNAL_FILE
    tmp = path.with_suffix(".csv.tmp")
    cols = {c["k"]: c for c in run.columns}
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow([h for _, h in JOURNAL_HEADERS])
        for e in run.eggs:
            col = cols[e["column"]]
            row = {
                **e,
                "defect": e["defect"] or "",
                "score": "" if e["score"] is None else f"{e['score']:.6f}",
                "decision": "" if e["decision"] is None else e["decision"],
                "late": "oui" if e["late"] else "non",
                "outcome_label": OUTCOME_LABELS[e["outcome"]],
                **{k: col[k] for k in ("measured_ms", "compute_ms", "wait_ms", "response_ms", "margin_ms", "t_enter", "t_decision", "t_deadline", "t_trap")},
            }
            if not col["decided"]:
                for k in ("measured_ms", "compute_ms", "wait_ms", "response_ms", "margin_ms"):
                    row[k] = ""
            w.writerow([row[k] for k, _ in JOURNAL_HEADERS])
    os.replace(tmp, path)
    return path


def geometry(params: SimulationParams) -> dict[str, Any]:
    return {
        "n_lanes": params.n_lanes,
        "period_s": params.period_s,
        "pitch_mm": params.pitch_mm,
        "speed_mm_s": round(params.speed, 6),
        "camera_zone_mm": params.camera_zone_mm,
        "camera_to_trap_mm": params.camera_to_trap_mm,
        "trap_x_mm": params.trap_x_mm,
        "trap_actuation_ms": params.trap_actuation_ms,
        "camera_time_s": round(params.camera_time_s, 6),
        "budget_s": round(params.budget_s, 6),
        "effective_budget_s": round(params.effective_budget_s, 6),
    }


def write_timeline(sim_dir: Path, run: RunOutput, params: SimulationParams) -> Path:
    payload = {"version": TIMELINE_VERSION, "geometry": geometry(params), "threshold": run.threshold, "columns": run.columns}
    path = sim_dir / TIMELINE_FILE
    _atomic_write_text(path, json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return path


class TimelineCache:
    """Garde en mémoire les quelques dernières chronologies lues (rechargées si le fichier change)."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def load(self, path: Path) -> dict[str, Any]:
        key = str(path)
        mtime = path.stat().st_mtime
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and hit[0] == mtime:
                self._items.move_to_end(key)
                return hit[1]
        data = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._items[key] = (mtime, data)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return data

    def forget(self, path: Path) -> None:
        with self._lock:
            self._items.pop(str(path), None)


timelines = TimelineCache()


def slim_column(col: dict[str, Any]) -> dict[str, Any]:
    """Colonne allégée pour l'animation (sans scores par vue)."""
    out = {k: v for k, v in col.items() if k != "lanes"}
    out["lanes"] = [
        None
        if e is None
        else {k: e[k] for k in ("track_id", "truth", "defect", "n_views", "score", "decision", "late", "ejected", "outcome")}
        for e in col["lanes"]
    ]
    return out


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------
COMPARE_METRICS: list[tuple[str, str, str, str]] = [
    # (chemin, libellé, format, sens du meilleur)
    ("counts.eggs", "Œufs simulés", "int", ""),
    ("counts.non_viable", "Non viables", "int", ""),
    ("rates.recall", "Non viables éjectés (rappel effectif)", "pct", "max"),
    ("rates.miss_rate", "Non viables manqués", "pct", "min"),
    ("rates.miss_late_rate", "dont manqués par retard", "pct", "min"),
    ("rates.false_reject_rate", "Viables éjectés à tort", "pct", "min"),
    ("rates.late_rate", "Décisions en retard", "pct", "min"),
    ("rates.residual_defect_rate", "Défauts résiduels (bac viables)", "pct", "min"),
    ("rates.model_recall", "Rappel du modèle (sans contrainte de temps)", "pct", "max"),
    ("counts.ejected_ok", "Éjectés à raison", "int", "max"),
    ("counts.missed_total", "Manqués", "int", "min"),
    ("counts.false_reject", "Éjectés à tort", "int", "min"),
    ("counts.late_decisions", "Décisions en retard (œufs)", "int", "min"),
    ("latency.compute_ms.mean", "Latence de calcul moyenne (ms)", "ms", "min"),
    ("latency.compute_ms.p99", "Latence de calcul p99 (ms)", "ms", "min"),
    ("latency.response_ms.max", "Délai de réponse max (ms)", "ms", "min"),
    ("latency.effective_budget_ms", "Budget effectif (ms)", "ms", ""),
    ("latency.min_margin_ms", "Marge minimale (ms)", "ms", "max"),
    ("throughput.capacity_ratio", "Capacité / cadence requise", "ratio", "max"),
]


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def compare(sims: list[dict[str, Any]]) -> dict[str, Any]:
    """Tableau côte à côte : paramètres qui diffèrent + métriques (+ rappel par défaut).

    ``sims`` : dictionnaires ``{id, name, model_id, model_name, status, params, results}``.
    """
    keys: list[str] = []
    for s in sims:
        for k in (s.get("params") or {}):
            if k not in keys:
                keys.append(k)
    # Valeurs effectives (seuil / variante réellement utilisés) pour comparer à bon escient.
    def eff(s: dict[str, Any], k: str) -> Any:
        p = s.get("params") or {}
        r = s.get("results") or {}
        if k == "threshold" and p.get(k) is None:
            return r.get("threshold")
        if k == "variant" and p.get(k) is None:
            return r.get("variant")
        return p.get(k)

    params_rows = []
    for k in keys:
        values = [eff(s, k) for s in sims]
        params_rows.append({"key": k, "values": values, "differs": len({json.dumps(v) for v in values}) > 1})
    metric_rows = []
    for path, label, fmt, better in COMPARE_METRICS:
        values = [_get(s.get("results") or {}, path) for s in sims]
        nums = [v for v in values if isinstance(v, (int, float))]
        best = None
        if better and len(nums) > 1 and len(set(nums)) > 1:
            target = max(nums) if better == "max" else min(nums)
            best = [isinstance(v, (int, float)) and v == target for v in values]
        metric_rows.append({"key": path, "label": label, "format": fmt, "better": better, "values": values, "best": best})
    defects: list[str] = []
    for s in sims:
        for d in ((s.get("results") or {}).get("by_defect") or {}):
            if d not in defects:
                defects.append(d)
    defect_rows = []
    for d in sorted(defects):
        cells = []
        for s in sims:
            bd = ((s.get("results") or {}).get("by_defect") or {}).get(d)
            cells.append(None if bd is None else {"n": bd["n"], "recall": bd.get("recall"), "ejected_ok": bd.get(EJECTED_OK)})
        defect_rows.append({"defect": d, "label": DEFECT_LABELS.get(d, "Non précisé"), "values": cells})
    return {
        "simulations": [{k: s.get(k) for k in ("id", "name", "model_id", "model_name", "status", "created_at")} for s in sims],
        "params": params_rows,
        "metrics": metric_rows,
        "by_defect": defect_rows,
    }
