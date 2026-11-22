#!/usr/bin/env python3
"""Moteur d'inférence EggVision — script autonome.

Ce fichier est utilisé tel quel :
  * par l'application de bureau (simulation du tri) ;
  * par l'agent Raspberry Pi (banc de mesure) ;
  * dans le paquet de déploiement, où il est copié sous le nom ``infer.py`` à
    côté de ``preprocess.py``, du modèle ONNX et de ``inference_config.json``.

Il ne dépend que de NumPy, OpenCV et ONNX Runtime.

Contrat du modèle ONNX
----------------------
  entrée  ``views``       float32 ``[B, V, 3, S, S]``  (B œufs, V vues, prétraitées)
  sortie  ``egg_scores``  float32 ``[B]``     score de défaut de l'œuf, dans [0, 1]
  sortie  ``view_scores`` float32 ``[B, V]``  score de défaut de chaque vue, dans [0, 1]
L'agrégation multi-vues fait partie du graphe : B et V sont dynamiques.

Utilisation en ligne de commande (dans le dossier du paquet) ::

    python infer.py --column chemin/vers/colonne     # sous-dossiers lane_0 … lane_5 contenant les vues
    python infer.py --bench 50                       # latence par colonne (6 œufs × N vues)
    python infer.py --verify                         # compare aux scores de référence (PyTorch) du paquet

Utilisation en Python ::

    from infer import EggClassifier
    clf = EggClassifier.load(".")
    result = clf.predict_column([vues_couloir_0, ..., vues_couloir_5])   # listes d'images BGR uint8, ou None
    result.decisions   # [0, 1, 0, 0, 0, 0]  (0 = viable, 1 = éjecter)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # utilisation dans le paquet Python du projet
    from .preprocess import PreprocessConfig, load_view, preprocess_views
except ImportError:  # utilisation autonome (paquet de déploiement) : preprocess.py est à côté
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import PreprocessConfig, load_view, preprocess_views  # type: ignore[no-redef]

CONFIG_FILENAME = "inference_config.json"
PACKAGE_FORMAT_VERSION = 1
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ---------------------------------------------------------------------------
# Moteurs
# ---------------------------------------------------------------------------
class InferenceEngine:
    """Interface d'un moteur d'inférence.

    Pour ajouter un moteur (Hailo, TFLite…) : dériver cette classe, implémenter
    ``run`` et l'enregistrer dans :data:`ENGINES`.
    """

    name = "abstrait"

    def __init__(self, model_path: str, threads: int | None = None) -> None:
        self.model_path = model_path
        self.threads = threads

    def run(self, views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``views`` float32 [B, V, 3, S, S] → (egg_scores [B], view_scores [B, V])."""
        raise NotImplementedError


class OnnxRuntimeEngine(InferenceEngine):
    """ONNX Runtime sur CPU."""

    name = "onnxruntime"

    def __init__(self, model_path: str, threads: int | None = None) -> None:
        super().__init__(model_path, threads)
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        names = [o.name for o in self.session.get_outputs()]
        if "egg_scores" not in names or "view_scores" not in names:
            raise ValueError(f"Sorties ONNX inattendues : {names} (attendu egg_scores, view_scores)")

    def run(self, views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        egg, view = self.session.run(["egg_scores", "view_scores"], {self.input_name: views})
        return np.asarray(egg, dtype=np.float32).reshape(-1), np.asarray(view, dtype=np.float32)


class _NotImplementedEngine(InferenceEngine):
    def __init__(self, model_path: str, threads: int | None = None) -> None:
        raise NotImplementedError(f"Le moteur « {self.name} » est prévu mais pas encore implémenté.")


class HailoEngine(_NotImplementedEngine):
    name = "hailo"


class TFLiteEngine(_NotImplementedEngine):
    name = "tflite"


ENGINES: dict[str, type[InferenceEngine]] = {
    OnnxRuntimeEngine.name: OnnxRuntimeEngine,
    HailoEngine.name: HailoEngine,
    TFLiteEngine.name: TFLiteEngine,
}


# ---------------------------------------------------------------------------
# Résultat
# ---------------------------------------------------------------------------
@dataclass
class ColumnResult:
    """Décision pour une colonne (un élément par couloir)."""

    decisions: list[int]  # 0 = viable (ou couloir vide), 1 = éjecter
    scores: list[float | None]  # score de défaut de l'œuf ; None si couloir vide
    view_scores: list[list[float] | None]
    n_views: list[int]
    latency_ms: float
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    threshold: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "scores": [None if s is None else round(float(s), 6) for s in self.scores],
            "view_scores": [None if v is None else [round(float(x), 6) for x in v] for v in self.view_scores],
            "n_views": self.n_views,
            "latency_ms": round(self.latency_ms, 3),
            "preprocess_ms": round(self.preprocess_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Classifieur
# ---------------------------------------------------------------------------
class EggClassifier:
    """Charge un paquet de déploiement et décide viable / éjecter pour chaque colonne."""

    def __init__(self, engine: InferenceEngine, config: dict[str, Any], threshold: float | None = None) -> None:
        self.engine = engine
        self.config = config
        self.preprocess = PreprocessConfig.from_dict(config.get("preprocess"))
        self.n_views = int(config.get("n_views", 12))
        self.n_lanes = int(config.get("n_lanes", 6))
        self.threshold = float(config.get("threshold", 0.5) if threshold is None else threshold)

    @classmethod
    def load(
        cls,
        package_dir: str,
        variant: str | None = None,
        threads: int | None = None,
        threshold: float | None = None,
        engine: str | None = None,
    ) -> EggClassifier:
        """Charge le paquet du dossier ``package_dir``.

        ``variant`` : ``"fp32"`` ou ``"int8"`` (défaut : ``default_variant`` de la configuration).
        """
        with open(os.path.join(package_dir, CONFIG_FILENAME), encoding="utf-8") as f:
            config = json.load(f)
        version = int(config.get("format_version", 0))
        if version != PACKAGE_FORMAT_VERSION:
            raise ValueError(f"Format de paquet non pris en charge : {version}")
        variant = variant or config.get("default_variant", "fp32")
        files = config.get("model_files", {})
        if variant not in files:
            raise ValueError(f"Variante « {variant} » absente du paquet (disponibles : {sorted(files)})")
        engine_name = engine or config.get("engine", "onnxruntime")
        if engine_name not in ENGINES:
            raise ValueError(f"Moteur inconnu : {engine_name}")
        eng = ENGINES[engine_name](os.path.join(package_dir, files[variant]), threads=threads)
        clf = cls(eng, config, threshold=threshold)
        clf.variant = variant  # type: ignore[attr-defined]
        return clf

    # -- scores ----------------------------------------------------------
    def score_tensor(self, views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scores à partir d'un tenseur déjà prétraité ``[B, V, 3, S, S]``."""
        return self.engine.run(np.ascontiguousarray(views, dtype=np.float32))

    def score_eggs(self, eggs: Sequence[Sequence[np.ndarray]]) -> tuple[list[float], list[list[float]]]:
        """Scores d'une liste d'œufs (chacun = liste de vues BGR uint8, au moins une)."""
        scores: list[float] = [0.0] * len(eggs)
        per_view: list[list[float]] = [[] for _ in eggs]
        # Les œufs ayant le même nombre de vues sont traités en un seul appel.
        groups: dict[int, list[int]] = {}
        for i, views in enumerate(eggs):
            if len(views) == 0:
                raise ValueError("Un œuf doit avoir au moins une vue")
            groups.setdefault(len(views), []).append(i)
        for _, idxs in sorted(groups.items()):
            batch = np.stack([preprocess_views(eggs[i], self.preprocess) for i in idxs], axis=0)
            egg_scores, view_scores = self.engine.run(batch)
            for k, i in enumerate(idxs):
                scores[i] = float(egg_scores[k])
                per_view[i] = [float(x) for x in view_scores[k]]
        return scores, per_view

    # -- décision par colonne -------------------------------------------
    def predict_column(self, lanes: Sequence[Sequence[np.ndarray] | None]) -> ColumnResult:
        """Décide pour une colonne.

        ``lanes`` : un élément par couloir — la liste des vues BGR uint8 de l'œuf,
        ou ``None`` / liste vide si le couloir est vide (décision 0, score None).
        """
        t0 = time.perf_counter()
        present = [i for i, v in enumerate(lanes) if v is not None and len(v) > 0]
        decisions = [0] * len(lanes)
        scores: list[float | None] = [None] * len(lanes)
        view_scores: list[list[float] | None] = [None] * len(lanes)
        n_views = [0 if (v is None) else len(v) for v in lanes]
        pre_ms = inf_ms = 0.0
        if present:
            groups: dict[int, list[int]] = {}
            for i in present:
                groups.setdefault(len(lanes[i]), []).append(i)  # type: ignore[arg-type]
            for _, idxs in sorted(groups.items()):
                t1 = time.perf_counter()
                batch = np.stack([preprocess_views(lanes[i], self.preprocess) for i in idxs], axis=0)  # type: ignore[arg-type]
                t2 = time.perf_counter()
                egg_scores, vs = self.engine.run(batch)
                t3 = time.perf_counter()
                pre_ms += (t2 - t1) * 1e3
                inf_ms += (t3 - t2) * 1e3
                for k, i in enumerate(idxs):
                    s = float(egg_scores[k])
                    scores[i] = s
                    view_scores[i] = [float(x) for x in vs[k]]
                    decisions[i] = 1 if s >= self.threshold else 0
        return ColumnResult(
            decisions=decisions,
            scores=scores,
            view_scores=view_scores,
            n_views=n_views,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            preprocess_ms=pre_ms,
            inference_ms=inf_ms,
            threshold=self.threshold,
        )

    # -- vérification de référence ------------------------------------------
    def verify_reference(self, package_dir: str) -> dict[str, Any] | None:
        """Rejoue les œufs de référence du paquet et compare aux scores PyTorch attendus.

        Le paquet peut contenir ``reference/expected.json`` ::

            {"tolerance": 1e-4,
             "eggs": [{"dir": "egg_00", "score": 0.93, "view_scores": [...]}, ...]}

        et, pour chaque œuf, ``reference/<dir>/<vue>.png`` (PNG : décodage identique
        sur toutes les plateformes). Retourne ``None`` si le paquet n'a pas de référence,
        sinon ``{"n_eggs", "max_abs_diff", "max_abs_diff_views", "tolerance", "ok"}``.
        La comparaison vaut pour la variante FP32 ; pour INT8 l'écart est informatif.
        """
        ref_dir = os.path.join(package_dir, "reference")
        expected_path = os.path.join(ref_dir, "expected.json")
        if not os.path.isfile(expected_path):
            return None
        with open(expected_path, encoding="utf-8") as f:
            expected = json.load(f)
        eggs, exp_scores, exp_views = [], [], []
        for item in expected.get("eggs", []):
            d = os.path.join(ref_dir, item["dir"])
            files = sorted(x for x in os.listdir(d) if x.lower().endswith(IMAGE_EXTENSIONS))
            eggs.append([load_view(os.path.join(d, x)) for x in files])
            exp_scores.append(float(item["score"]))
            exp_views.append([float(x) for x in item.get("view_scores", [])])
        if not eggs:
            return None
        scores, view_scores = self.score_eggs(eggs)
        diff = max(abs(a - b) for a, b in zip(scores, exp_scores))
        diff_views = max(
            (abs(a - b) for got, exp in zip(view_scores, exp_views) for a, b in zip(got, exp)),
            default=0.0,
        )
        tol = float(expected.get("tolerance", 1e-4))
        return {
            "n_eggs": len(eggs),
            "max_abs_diff": float(diff),
            "max_abs_diff_views": float(diff_views),
            "tolerance": tol,
            "ok": bool(diff <= tol and diff_views <= tol),
            "scores": [round(x, 6) for x in scores],
        }

    # -- banc de mesure ----------------------------------------------------
    def benchmark(self, n_columns: int = 50, warmup: int = 5, seed: int = 0) -> dict[str, Any]:
        """Latence par colonne (``n_lanes`` œufs × ``n_views`` vues), prétraitement compris."""
        rng = np.random.default_rng(seed)
        size = self.preprocess.input_size
        column = [
            [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(self.n_views)]
            for _ in range(self.n_lanes)
        ]
        for _ in range(max(0, warmup)):
            self.predict_column(column)
        total, pre, inf = [], [], []
        for _ in range(max(1, n_columns)):
            r = self.predict_column(column)
            total.append(r.latency_ms)
            pre.append(r.preprocess_ms)
            inf.append(r.inference_ms)
        arr = np.asarray(total)
        return {
            "n_columns": int(len(total)),
            "n_lanes": self.n_lanes,
            "n_views": self.n_views,
            "input_size": size,
            "variant": getattr(self, "variant", None),
            "engine": self.engine.name,
            "threads": self.engine.threads,
            "latency_ms": {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            },
            "preprocess_ms_mean": float(np.mean(pre)),
            "inference_ms_mean": float(np.mean(inf)),
            "columns_per_second": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
        }


# ---------------------------------------------------------------------------
# Ligne de commande
# ---------------------------------------------------------------------------
def load_column_dir(column_dir: str, n_lanes: int) -> list[list[np.ndarray] | None]:
    """Lit une colonne sur disque : sous-dossiers ``lane_<i>`` contenant les vues (triées par nom)."""
    lanes: list[list[np.ndarray] | None] = []
    for i in range(n_lanes):
        d = os.path.join(column_dir, f"lane_{i}")
        if not os.path.isdir(d):
            lanes.append(None)
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTENSIONS))
        lanes.append([load_view(os.path.join(d, f)) for f in files] or None)
    return lanes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inférence EggVision : décisions de tri pour une colonne d'œufs.")
    parser.add_argument("--package", default=os.path.dirname(os.path.abspath(__file__)), help="Dossier du paquet de déploiement")
    parser.add_argument("--variant", choices=["fp32", "int8"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Remplace le seuil de la configuration")
    parser.add_argument("--column", help="Dossier d'une colonne (sous-dossiers lane_0 … lane_N)")
    parser.add_argument("--bench", type=int, metavar="N", help="Mesure la latence sur N colonnes")
    parser.add_argument("--verify", action="store_true", help="Compare aux scores de référence du paquet (PyTorch)")
    args = parser.parse_args(argv)

    clf = EggClassifier.load(args.package, variant=args.variant, threads=args.threads, threshold=args.threshold)
    if args.verify:
        report = clf.verify_reference(args.package)
        print(json.dumps(report, indent=2))
        return 0 if (report is None or report["ok"] or getattr(clf, "variant", "fp32") != "fp32") else 1
    if args.bench:
        print(json.dumps(clf.benchmark(args.bench), indent=2))
        return 0
    if not args.column:
        parser.error("Indiquer --column ou --bench")
    result = clf.predict_column(load_column_dir(args.column, clf.n_lanes))
    print(json.dumps(result.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
