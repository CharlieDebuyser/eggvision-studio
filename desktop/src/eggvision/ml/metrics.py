"""Métriques de classification binaire en NumPy (sans scikit-learn).

Conventions :
  * ``y`` : 1 = œuf non viable (positif, à éjecter), 0 = viable ;
  * ``s`` : score de défaut dans [0, 1] ;
  * décision : éjecter si ``s >= seuil`` (même règle que ``EggClassifier``) ;
  * « faux rejets » = part des œufs viables éjectés à tort (taux de faux positifs).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _arrays(y: Sequence[int] | np.ndarray, s: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    s_arr = np.asarray(s, dtype=np.float64).reshape(-1)
    if y_arr.shape != s_arr.shape:
        raise ValueError("y et s doivent avoir la même longueur")
    return y_arr, s_arr


def _desc_cumulative(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seuils distincts décroissants et nombres cumulés de VP / FP (décision ``s >= seuil``)."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # dernier indice de chaque groupe de scores égaux
    last = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[last]
    fps = (last + 1) - tps
    return s_sorted[last], tps.astype(np.float64), fps.astype(np.float64)


def roc_curve(y: Sequence[int], s: Sequence[float]) -> dict[str, np.ndarray]:
    """Courbe ROC : ``fpr``, ``tpr``, ``thresholds`` (le premier seuil vaut +inf : rien n'est éjecté)."""
    y_arr, s_arr = _arrays(y, s)
    n_pos = float(y_arr.sum())
    n_neg = float(len(y_arr) - n_pos)
    if len(y_arr) == 0:
        return {"fpr": np.array([0.0]), "tpr": np.array([0.0]), "thresholds": np.array([np.inf])}
    thr, tps, fps = _desc_cumulative(y_arr, s_arr)
    tpr = np.r_[0.0, tps / n_pos] if n_pos > 0 else np.r_[0.0, np.zeros_like(tps)]
    fpr = np.r_[0.0, fps / n_neg] if n_neg > 0 else np.r_[0.0, np.zeros_like(fps)]
    return {"fpr": fpr, "tpr": tpr, "thresholds": np.r_[np.inf, thr]}


def roc_auc(y: Sequence[int], s: Sequence[float]) -> float | None:
    """Aire sous la courbe ROC (``None`` si une seule classe est présente)."""
    y_arr, _ = _arrays(y, s)
    if y_arr.sum() == 0 or y_arr.sum() == len(y_arr):
        return None
    c = roc_curve(y, s)
    return float(np.trapezoid(c["tpr"], c["fpr"]))


def precision_recall_curve(y: Sequence[int], s: Sequence[float]) -> dict[str, np.ndarray]:
    """Courbe précision / rappel par seuil décroissant (précédée du point rappel 0, précision 1)."""
    y_arr, s_arr = _arrays(y, s)
    n_pos = float(y_arr.sum())
    if len(y_arr) == 0 or n_pos == 0:
        return {"precision": np.array([1.0]), "recall": np.array([0.0]), "thresholds": np.array([np.inf])}
    thr, tps, fps = _desc_cumulative(y_arr, s_arr)
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return {"precision": np.r_[1.0, precision], "recall": np.r_[0.0, recall], "thresholds": np.r_[np.inf, thr]}


def average_precision(y: Sequence[int], s: Sequence[float]) -> float | None:
    """Précision moyenne : somme des (R_n − R_{n−1}) × P_n (définition de scikit-learn)."""
    y_arr, _ = _arrays(y, s)
    if y_arr.sum() == 0:
        return None
    c = precision_recall_curve(y, s)
    return float(np.sum(np.diff(c["recall"]) * c["precision"][1:]))


def confusion(y: Sequence[int], s: Sequence[float], threshold: float) -> dict[str, Any]:
    """Matrice de confusion et taux usuels au seuil donné."""
    y_arr, s_arr = _arrays(y, s)
    pred = s_arr >= threshold
    pos = y_arr == 1
    tp = int(np.sum(pred & pos))
    fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos))
    tn = int(np.sum(~pred & ~pos))
    n_pos, n_neg = tp + fn, fp + tn
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n": n_pos + n_neg,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "recall": tp / n_pos if n_pos else None,  # rappel sur les non viables
        "false_reject_rate": fp / n_neg if n_neg else None,  # viables éjectés à tort
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "specificity": tn / n_neg if n_neg else None,
        "accuracy": (tp + tn) / (n_pos + n_neg) if (n_pos + n_neg) else None,
    }


# ---------------------------------------------------------------------------
# Seuil par objectif
# ---------------------------------------------------------------------------
class ObjectiveError(ValueError):
    """Objectif impossible à évaluer (classe absente, valeur hors bornes)."""


def threshold_for_recall(y: Sequence[int], s: Sequence[float], target: float) -> float:
    """Seuil le plus HAUT tel que le rappel des non viables soit ≥ ``target``.

    Le rappel ne change qu'aux scores des œufs non viables : il faut au moins
    ``k = ⌈target × n_pos⌉`` positifs au-dessus du seuil, donc le seuil est le
    k-ième plus grand score positif. ``target = 0`` → seuil 1,0 (au-delà de 1,
    rien n'est éjecté ; on reste dans [0, 1]) sauf si un positif vaut exactement 1.
    """
    if not 0.0 <= target <= 1.0:
        raise ObjectiveError("L'objectif de rappel doit être compris entre 0 et 1.")
    y_arr, s_arr = _arrays(y, s)
    pos = np.sort(s_arr[y_arr == 1])[::-1]
    if len(pos) == 0:
        raise ObjectiveError("Aucun œuf non viable dans ce jeu : rappel indéfini.")
    k = int(math.ceil(target * len(pos) - 1e-9))
    if k <= 0:  # rappel nul exigé : seuil au-dessus de tous les scores positifs
        return 1.0 if pos[0] < 1.0 else float(np.nextafter(float(pos[0]), np.inf))
    return float(pos[k - 1])


def threshold_for_false_reject(y: Sequence[int], s: Sequence[float], target: float) -> float:
    """Seuil le plus BAS tel que le taux de faux rejets (viables éjectés) soit ≤ ``target``.

    On peut tolérer ``m = ⌊target × n_neg⌋`` viables au-dessus du seuil : le seuil
    est juste au-dessus du (m+1)-ième plus grand score viable (``nextafter``),
    ou 0 si tous les viables peuvent être éjectés.
    """
    if not 0.0 <= target <= 1.0:
        raise ObjectiveError("L'objectif de faux rejets doit être compris entre 0 et 1.")
    y_arr, s_arr = _arrays(y, s)
    neg = np.sort(s_arr[y_arr == 0])[::-1]
    if len(neg) == 0:
        raise ObjectiveError("Aucun œuf viable dans ce jeu : taux de faux rejets indéfini.")
    m = int(math.floor(target * len(neg) + 1e-9))
    if m >= len(neg):
        return 0.0
    return float(np.nextafter(float(neg[m]), np.inf))


def threshold_for_objective(y: Sequence[int], s: Sequence[float], objective: str, target: float) -> float:
    """``objective`` : ``"recall"`` (rappel ≥ target) ou ``"false_reject"`` (faux rejets ≤ target)."""
    if objective == "recall":
        return threshold_for_recall(y, s, target)
    if objective == "false_reject":
        return threshold_for_false_reject(y, s, target)
    raise ObjectiveError(f"Objectif inconnu : {objective!r} (attendu « recall » ou « false_reject »)")


def objective_met(metrics: dict[str, Any], objective: str, target: float) -> bool | None:
    if objective == "recall":
        r = metrics.get("recall")
        return None if r is None else bool(r >= target - 1e-12)
    fr = metrics.get("false_reject_rate")
    return None if fr is None else bool(fr <= target + 1e-12)


# ---------------------------------------------------------------------------
# Échantillonnage des courbes (affichage)
# ---------------------------------------------------------------------------
def _json_threshold(t: float) -> float | None:
    return None if not math.isfinite(t) else round(float(t), 6)


def sample_curve(x: np.ndarray, y: np.ndarray, thresholds: np.ndarray, max_points: int = 200) -> dict[str, list]:
    """Réduit une courbe à ``max_points`` points (extrémités conservées) pour l'affichage."""
    n = len(x)
    if n > max_points:
        idx = np.unique(np.r_[np.linspace(0, n - 1, max_points).round().astype(int), 0, n - 1])
    else:
        idx = np.arange(n)
    return {
        "x": [round(float(v), 6) for v in x[idx]],
        "y": [round(float(v), 6) for v in y[idx]],
        "thresholds": [_json_threshold(float(t)) for t in thresholds[idx]],
    }


def histogram(y: np.ndarray, s: np.ndarray, bins: int = 20) -> dict[str, list]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    clipped = np.clip(s, 0.0, 1.0)
    return {
        "edges": [round(float(e), 4) for e in edges],
        "viable": np.histogram(clipped[y == 0], edges)[0].astype(int).tolist(),
        "non_viable": np.histogram(clipped[y == 1], edges)[0].astype(int).tolist(),
    }


def binary_report(
    y: Sequence[int],
    s: Sequence[float],
    threshold: float,
    defects: Sequence[str | None] | None = None,
    max_points: int = 200,
) -> dict[str, Any]:
    """Bloc de rapport complet pour un sous-ensemble (courbes échantillonnées comprises)."""
    y_arr, s_arr = _arrays(y, s)
    roc = roc_curve(y_arr, s_arr)
    pr = precision_recall_curve(y_arr, s_arr)
    out: dict[str, Any] = {
        "n": int(len(y_arr)),
        "n_pos": int(y_arr.sum()),
        "n_neg": int(len(y_arr) - y_arr.sum()),
        "auc_roc": roc_auc(y_arr, s_arr),
        "average_precision": average_precision(y_arr, s_arr),
        "at_threshold": confusion(y_arr, s_arr, threshold),
        "roc": _named(sample_curve(roc["fpr"], roc["tpr"], roc["thresholds"], max_points), "fpr", "tpr"),
        "pr": _named(sample_curve(pr["recall"], pr["precision"], pr["thresholds"], max_points), "recall", "precision"),
        "histogram": histogram(y_arr, s_arr),
    }
    if defects is not None:
        out["recall_by_defect"] = recall_by_defect(y_arr, s_arr, defects, threshold)
    return out


def _named(c: dict[str, list], x_name: str, y_name: str) -> dict[str, list]:
    return {x_name: c["x"], y_name: c["y"], "thresholds": c["thresholds"]}


def recall_by_defect(y: np.ndarray, s: np.ndarray, defects: Sequence[str | None], threshold: float) -> dict[str, dict[str, Any]]:
    """Rappel des non viables par type de défaut (``"non_precise"`` si le défaut n'est pas renseigné)."""
    out: dict[str, dict[str, Any]] = {}
    for yi, si, d in zip(y, s, defects, strict=False):
        if yi != 1:
            continue
        key = d or "non_precise"
        e = out.setdefault(key, {"n": 0, "detected": 0})
        e["n"] += 1
        e["detected"] += int(si >= threshold)
    for e in out.values():
        e["recall"] = e["detected"] / e["n"] if e["n"] else None
    return dict(sorted(out.items()))
