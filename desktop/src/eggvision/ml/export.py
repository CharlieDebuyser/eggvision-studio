"""Export ONNX (FP32 + INT8), vérification de parité et paquet de déploiement (tâche ``kind="export"``).

Paquet ``models/<id>/package/`` (et ``models/<id>/eggvision_<id>_v<version>.zip``) ::

    inference_config.json   cf. docs/ARCHITECTURE.md (seuil = Model.threshold, scène du projet…)
    model_fp32.onnx         opset 17, entrée « views » [B, V, 3, S, S], sorties « egg_scores » [B], « view_scores » [B, V]
    model_int8.onnx         quantification statique ONNX Runtime (QDQ), calibrée sur des vues d'entraînement
    preprocess.py infer.py  copies exactes de eggvision_shared/preprocess.py et runtime.py
    reference/              quelques œufs en PNG + expected.json (scores PyTorch calculés sur ces PNG)
    README.txt

Contrôles (l'export échoue si l'un des contrôles bloquants échoue) :
  * parité PyTorch ↔ ONNX FP32 sur des œufs réels du projet, pour V = n_views, 6 et 12
    (axes dynamiques) : écart absolu maximal ≤ 1e-4 (bloquant) ;
  * INT8 : écart à FP32 et métriques sur le jeu de test (informatif ; alerte si le rappel
    ou l'AUC se dégradent de plus de 2 points) ;
  * ``python infer.py --verify`` exécuté dans le dossier du paquet (bloquant).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from eggvision_shared import source_file
from eggvision_shared.constants import DEFAULT_N_LANES
from eggvision_shared.preprocess import PreprocessConfig, load_view, preprocess_views
from eggvision_shared.runtime import CONFIG_FILENAME, PACKAGE_FORMAT_VERSION

from .. import state
from ..jobs import JobContext
from ..project import Project
from . import metrics as mt
from .dataset import EggRecord, load_records, pick_view_indices
from .store import classifier_from_checkpoint, get_model_row, load_checkpoint

PARITY_TOLERANCE = 1e-4
OPSET = 17
INT8_ALERT_DROP = 0.02


class ExportError(RuntimeError):
    """Erreur d'export présentable à l'utilisateur."""


def start_export(project: Project, model_id: str, params: dict[str, Any] | None = None) -> dict[str, str]:
    params = dict(params or {})
    variants = params.get("variants") or ["fp32", "int8"]
    if "fp32" not in variants or not set(variants) <= {"fp32", "int8"}:
        raise ExportError("Variantes possibles : « fp32 » (obligatoire) et « int8 ».")
    with project.session() as s:
        row = get_model_row(s, model_id)
        if row.kind == "anomaly":
            raise ExportError("L'export ONNX du modèle de référence par anomalies n'est pas pris en charge.")
        if row.kind != "classifier":
            raise ExportError("Seuls les classifieurs peuvent être exportés.")
        if row.status != "trained" or not (project.model_dir(model_id) / "best.pt").is_file():
            raise ExportError("Le modèle doit être entraîné (statut « trained ») avant l'export.")
        name = row.name
    for job in state.jobs.active(kind="export"):
        if (job.params or {}).get("model_id") == model_id:
            raise ExportError("Un export de ce modèle est déjà en cours.")
    job = state.jobs.submit(
        "export", f"Export ONNX — {name}", lambda ctx: run_export(ctx, project, model_id, {**params, "variants": variants}),
        params={"model_id": model_id, **params, "variants": variants}, project=project, group="cpu",
    )
    return {"model_id": model_id, "job_id": job.id}


# ---------------------------------------------------------------------------
# Étapes
# ---------------------------------------------------------------------------
def export_onnx(model: Any, path: Path, n_views: int, input_size: int) -> None:
    """Export FP32 (exportateur TorchScript, axes dynamiques B et V)."""
    import warnings

    import onnx
    import torch

    model.eval()
    example = torch.zeros(2, n_views, 3, input_size, input_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (example,),
            str(path),
            input_names=["views"],
            output_names=["egg_scores", "view_scores"],
            dynamic_axes={"views": {0: "batch", 1: "n_views"}, "egg_scores": {0: "batch"}, "view_scores": {0: "batch", 1: "n_views"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)


def ort_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def ort_run(sess: Any, x: np.ndarray, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    eggs, views = [], []
    for i in range(0, len(x), batch):
        e, v = sess.run(["egg_scores", "view_scores"], {"views": np.ascontiguousarray(x[i : i + batch], dtype=np.float32)})
        eggs.append(np.asarray(e).reshape(-1))
        views.append(np.asarray(v))
    return np.concatenate(eggs), np.concatenate(views)


def torch_run(model: Any, x: np.ndarray, batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    import torch

    eggs, views = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            e, v = model(torch.from_numpy(np.ascontiguousarray(x[i : i + batch])))
            eggs.append(e.numpy().reshape(-1))
            views.append(v.numpy())
    return np.concatenate(eggs), np.concatenate(views)


def egg_tensor(records: Sequence[EggRecord], n_views: int, input_size: int) -> np.ndarray:
    """Tenseur ``[N, V, 3, S, S]`` (vues choisies comme au déploiement, prétraitement partagé)."""
    pre = PreprocessConfig(input_size=input_size)
    out = []
    for r in records:
        idx = pick_view_indices(len(r.view_paths), n_views)
        out.append(preprocess_views([load_view(r.view_paths[k]) for k in idx], pre))
    return np.stack(out) if out else np.zeros((0, n_views, 3, input_size, input_size), np.float32)


def parity_check(model: Any, onnx_path: Path, records: Sequence[EggRecord], view_counts: Sequence[int], input_size: int) -> dict[str, Any]:
    sess = ort_session(onnx_path)
    per_v = {}
    worst = 0.0
    for v in sorted(set(view_counts)):
        x = egg_tensor(records, v, input_size)
        te, tv = torch_run(model, x)
        oe, ov = ort_run(sess, x)
        d_egg = float(np.max(np.abs(te - oe))) if len(x) else 0.0
        d_view = float(np.max(np.abs(tv - ov))) if len(x) else 0.0
        per_v[str(v)] = {"n_eggs": int(len(x)), "max_abs_diff": d_egg, "max_abs_diff_views": d_view}
        worst = max(worst, d_egg, d_view)
    return {"tolerance": PARITY_TOLERANCE, "max_abs_diff": worst, "by_n_views": per_v, "ok": worst <= PARITY_TOLERANCE, "n_eggs": len(records)}


class _CalibrationReader:
    """Lecteur de calibration ONNX Runtime : un œuf ``[1, V, 3, S, S]`` par appel."""

    def __init__(self, tensor: np.ndarray) -> None:
        self._items = iter([{"views": tensor[i : i + 1]} for i in range(len(tensor))])

    def get_next(self):
        return next(self._items, None)

    def rewind(self) -> None:  # pragma: no cover - API ONNX Runtime
        pass


def quantize_int8(fp32: Path, out: Path, calib: np.ndarray, workdir: Path, exclude_ops: Sequence[str] = ()) -> dict[str, Any]:
    """Quantification statique QDQ (poids INT8 par canal, activations UINT8, calibration MinMax).

    Les opérateurs de la tête (après la mise en commun globale : produit matriciel final,
    sigmoïde, maximum) restent en flottant : ils pèsent peu en calcul et conditionnent la
    précision du score.

    Limite connue : MobileNetV3 (hard-swish, squeeze-excitation, plages d'activation très
    inégales après repli des BatchNorm) se quantifie mal sans réentraînement (QAT) ; les
    essais (MinMax / Percentile / Entropy, exclusion des convolutions en profondeur, des SE,
    du hard-swish) ne corrigent pas l'écart. EfficientNet-Lite0, conçu pour l'INT8, conserve
    ses performances. L'export mesure toujours l'écart et alerte ; la variante par défaut du
    paquet reste FP32.
    """
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = workdir / "model_fp32_pre.onnx"
    try:
        quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    except Exception:  # pré-traitement facultatif
        shutil.copyfile(fp32, pre)
    proto = onnx.load(str(pre))
    for i, node in enumerate(proto.graph.node):  # nœuds anonymes : impossibles à exclure sinon
        if not node.name:
            node.name = f"{node.op_type}_{i}"
    onnx.save(proto, str(pre))
    head_nodes = _head_nodes(proto)
    excluded = sorted(set(head_nodes) | {n.name for n in proto.graph.node if n.op_type in exclude_ops})
    quantize_static(
        str(pre),
        str(out),
        _CalibrationReader(calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=excluded,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    pre.unlink(missing_ok=True)
    return {"format": "QDQ", "activations": "uint8", "weights": "int8 par canal", "calibration": "MinMax", "n_calibration_eggs": int(len(calib)), "excluded_nodes": excluded}


def _head_nodes(proto: Any) -> list[str]:
    """Nœuds situés après la dernière mise en commun globale (tête de classification et agrégation)."""
    nodes = list(proto.graph.node)
    last_pool = max((i for i, n in enumerate(nodes) if n.op_type in ("GlobalAveragePool", "ReduceMean")), default=None)
    if last_pool is None:
        return []
    return [n.name for n in nodes[last_pool + 1 :] if n.name]


def write_reference(model: Any, ref_dir: Path, records: Sequence[EggRecord], n_views: int, input_size: int) -> dict[str, Any]:
    """Écrit les vues des œufs de référence en PNG puis calcule les scores PyTorch À PARTIR de ces PNG."""
    import cv2

    ref_dir.mkdir(parents=True, exist_ok=True)
    eggs = []
    pre = PreprocessConfig(input_size=input_size)
    for k, r in enumerate(records):
        d = ref_dir / f"egg_{k:02d}"
        d.mkdir(exist_ok=True)
        idx = pick_view_indices(len(r.view_paths), n_views)
        for j, vi in enumerate(idx):
            ok, buf = cv2.imencode(".png", load_view(r.view_paths[vi]))
            if not ok:
                raise ExportError("Écriture PNG impossible.")
            (d / f"{j:02d}.png").write_bytes(buf.tobytes())
        files = sorted(x for x in os.listdir(d) if x.endswith(".png"))
        x = preprocess_views([load_view(str(d / f)) for f in files], pre)[None]
        e, v = torch_run(model, x)
        eggs.append({"dir": d.name, "score": float(e[0]), "view_scores": [float(s) for s in v[0]], "track_id": r.track_id, "target": r.target, "defect": r.defect})
    expected = {"tolerance": PARITY_TOLERANCE, "source": "pytorch", "n_views": n_views, "eggs": eggs}
    (ref_dir / "expected.json").write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
    return expected


README = """EggVision — paquet de déploiement
=================================

Modèle : {model_name} (id {model_id}, version {version})
Exporté le : {exported_at}
Seuil de décision : {threshold}  (éjecter si score >= seuil)
Vues par œuf : {n_views} ; taille d'entrée : {input_size} px ; couloirs : {n_lanes}

Contenu
-------
inference_config.json   prétraitement, nombre de vues, seuil, couloirs, échelle, métriques
model_fp32.onnx         modèle de référence (float32)
model_int8.onnx         modèle quantifié (plus rapide sur CPU ARM), si présent
preprocess.py infer.py  prétraitement et moteur d'inférence autonomes
reference/              œufs de référence et scores PyTorch attendus

Prérequis sur le Raspberry Pi
-----------------------------
    python3 -m pip install numpy opencv-python-headless onnxruntime

Utilisation (dans ce dossier)
-----------------------------
    python3 infer.py --verify                    # parité avec PyTorch (FP32 : écart ≤ 1e-4)
    python3 infer.py --verify --variant int8     # écart de la variante INT8 (informatif)
    python3 infer.py --bench 50 --variant int8   # latence par colonne (6 œufs × N vues)
    python3 infer.py --column dossier_colonne    # sous-dossiers lane_0 … lane_5 contenant les vues

En Python :
    from infer import EggClassifier
    clf = EggClassifier.load(".", variant="int8")
    result = clf.predict_column([vues_couloir_0, ..., vues_couloir_5])   # listes d'images BGR uint8 ou None
    result.decisions   # [0, 1, 0, 0, 0, 0]  (1 = éjecter)
    result.scores      # scores de défaut par couloir
"""


def verify_package_subprocess(package_dir: Path, variant: str = "fp32", timeout: float = 300.0) -> dict[str, Any]:
    """Exécute ``infer.py --verify`` dans le dossier du paquet (processus séparé).

    Dans l'application empaquetée (PyInstaller), ``sys.executable`` n'est pas un
    interpréteur Python : la vérification est alors faite en chargeant ``infer.py``
    du paquet dans le processus courant.
    """
    if getattr(sys, "frozen", False):
        import importlib.util

        name = f"egg_infer_{int(time.time() * 1e6)}"
        spec = importlib.util.spec_from_file_location(name, package_dir / "infer.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[name] = mod  # requis par @dataclass pendant l'exécution du module
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            clf = mod.EggClassifier.load(str(package_dir), variant=variant)
            report = clf.verify_reference(str(package_dir))
        finally:
            sys.modules.pop(name, None)
        return {"mode": "in_process", "returncode": 0 if (report and report["ok"]) else 1, "report": report}
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    proc = subprocess.run(
        [sys.executable, "-I", "infer.py", "--verify", "--variant", variant],
        cwd=str(package_dir), capture_output=True, text=True, timeout=timeout, env=env,
    )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = None
    return {"mode": "subprocess", "returncode": proc.returncode, "report": report, "stderr": proc.stderr[-2000:]}


def _pick_eggs(records: Sequence[EggRecord], n: int, seed: int = 0) -> list[EggRecord]:
    """``n`` œufs, mélange de viables et de non viables, tirage reproductible."""
    rng = np.random.default_rng(seed)
    pos = [r for r in records if r.target]
    neg = [r for r in records if not r.target]
    rng.shuffle(pos)  # type: ignore[arg-type]
    rng.shuffle(neg)  # type: ignore[arg-type]
    out: list[EggRecord] = []
    while len(out) < n and (pos or neg):
        for pool in (pos, neg):
            if pool and len(out) < n:
                out.append(pool.pop())
    return out


def _decision_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, Any]:
    c = mt.confusion(y, s, thr)
    return {"auc_roc": mt.roc_auc(y, s), "recall": c["recall"], "false_reject_rate": c["false_reject_rate"], "tp": c["tp"], "fn": c["fn"], "fp": c["fp"], "tn": c["tn"]}


def run_export(ctx: JobContext, project: Project, model_id: str, params: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    variants = list(params.get("variants") or ["fp32", "int8"])
    n_reference = int(params.get("n_reference", 6))
    n_calibration = int(params.get("n_calibration_eggs", 32))
    n_parity = int(params.get("n_parity_eggs", 16))
    with project.session() as s:
        row = get_model_row(s, model_id)
        from ..db import models as m

        split = s.get(m.DatasetSplit, row.split_id) if row.split_id else None
        assignments = dict(split.assignments) if split else {}
        reviewed_only = bool((row.hparams or {}).get("reviewed_only", False))
        threshold, n_views, input_size = float(row.threshold), int(row.n_views), int(row.input_size)
        number = (row.extra or {}).get("number") or 0
        previous = dict(row.export or {})
        model_name = row.name
        metrics_summary = dict(row.metrics or {})
        hp = dict(row.hparams or {})
    export_count = int(previous.get("export_count", 0)) + 1
    version = f"{number}.{export_count}"
    mdir = project.model_dir(model_id)
    work = mdir / "package.tmp"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    warnings: list[str] = []

    model = classifier_from_checkpoint(load_checkpoint(mdir / "best.pt"))
    records = load_records(project, assignments, reviewed_only=reviewed_only)
    by_sub = {k: [r for r in records if r.subset == k] for k in ("train", "val", "test")}
    eval_pool = by_sub["test"] or by_sub["val"] or by_sub["train"]
    if not eval_pool:
        raise ExportError("Aucun œuf utilisable dans le découpage : parité impossible à vérifier.")

    # 1. FP32
    ctx.progress(0.05, "Export ONNX FP32")
    fp32 = work / "model_fp32.onnx"
    export_onnx(model, fp32, n_views, input_size)
    ctx.log(f"ONNX FP32 exporté (opset {OPSET}, {fp32.stat().st_size / 1e6:.1f} Mo).")
    ctx.check_cancelled()

    # 2. Parité
    ctx.progress(0.15, "Vérification de parité PyTorch ↔ ONNX")
    parity_eggs = _pick_eggs(eval_pool, n_parity)
    parity = parity_check(model, fp32, parity_eggs, [n_views, 6, 12], input_size)
    ctx.log(f"Parité FP32 : écart max {parity['max_abs_diff']:.2e} (tolérance {PARITY_TOLERANCE:g}) sur {len(parity_eggs)} œufs, V ∈ {list(parity['by_n_views'])}.")
    if not parity["ok"]:
        raise ExportError(f"Parité PyTorch ↔ ONNX non respectée : écart {parity['max_abs_diff']:.2e} > {PARITY_TOLERANCE:g}.")
    ctx.check_cancelled()

    # 3. INT8
    model_files = {"fp32": fp32.name}
    int8_info: dict[str, Any] | None = None
    if "int8" in variants:
        ctx.progress(0.3, "Quantification INT8")
        calib_pool = by_sub["train"] or eval_pool
        calib_eggs = _pick_eggs(calib_pool, n_calibration, seed=1)
        int8 = work / "model_int8.onnx"
        try:
            qinfo = quantize_int8(fp32, int8, egg_tensor(calib_eggs, n_views, input_size), work)
            ctx.check_cancelled()
            ctx.progress(0.5, "Évaluation de la variante INT8")
            s_fp = ort_session(fp32)
            s_q = ort_session(int8)
            xp = egg_tensor(parity_eggs, n_views, input_size)
            fe, fv = ort_run(s_fp, xp)
            qe, qv = ort_run(s_q, xp)
            int8_info = {
                **qinfo,
                "size_bytes": int8.stat().st_size,
                "vs_fp32": {"max_abs_diff": float(np.max(np.abs(fe - qe))), "mean_abs_diff": float(np.mean(np.abs(fe - qe))), "max_abs_diff_views": float(np.max(np.abs(fv - qv)))},
            }
            test = by_sub["test"]
            if test:
                y = np.asarray([r.target for r in test])
                xs_fp, xs_q = [], []
                for i in range(0, len(test), 32):
                    ctx.check_cancelled()
                    xt = egg_tensor(test[i : i + 32], n_views, input_size)
                    xs_fp.append(ort_run(s_fp, xt)[0])
                    xs_q.append(ort_run(s_q, xt)[0])
                sf, sq = np.concatenate(xs_fp), np.concatenate(xs_q)
                mf, mq = _decision_metrics(y, sf, threshold), _decision_metrics(y, sq, threshold)
                int8_info["test"] = {"n": int(len(y)), "fp32": mf, "int8": mq, "max_abs_diff": float(np.max(np.abs(sf - sq)))}
                drops = []
                for key, label in (("recall", "rappel"), ("auc_roc", "AUC")):
                    if mf.get(key) is not None and mq.get(key) is not None and mf[key] - mq[key] > INT8_ALERT_DROP:
                        drops.append(f"{label} {mf[key]:.3f} → {mq[key]:.3f}")
                if drops:
                    msg = "INT8 dégrade les performances sur le test (" + ", ".join(drops) + ") : préférer la variante FP32."
                    int8_info["alert"] = msg
                    warnings.append(msg)
                    ctx.log("ALERTE : " + msg)
            model_files["int8"] = int8.name
            ctx.log(f"INT8 : écart max à FP32 {int8_info['vs_fp32']['max_abs_diff']:.3g} ; {int8.stat().st_size / 1e6:.1f} Mo.")
        except Exception as exc:  # noqa: BLE001 — repli documenté : paquet FP32 seul
            if exc.__class__.__name__ == "JobCancelled":
                raise
            int8.unlink(missing_ok=True)
            msg = f"Quantification INT8 impossible ({exc}) : paquet FP32 uniquement."
            warnings.append(msg)
            ctx.log("AVERTISSEMENT : " + msg)
            int8_info = {"error": str(exc)}

    # 4. Référence, configuration, scripts
    ctx.progress(0.7, "Écriture du paquet")
    ref_eggs = _pick_eggs(eval_pool, n_reference, seed=2)
    write_reference(model, work / "reference", ref_eggs, n_views, input_size)
    scene = project.get_config("scene")
    n_lanes = len((scene or {}).get("lanes") or []) or DEFAULT_N_LANES
    exported_at = datetime.now(UTC).isoformat(timespec="seconds")
    test_m = metrics_summary.get("test") or {}
    config = {
        "format_version": PACKAGE_FORMAT_VERSION,
        "model_id": model_id,
        "model_name": model_name,
        "version": version,
        "exported_at": exported_at,
        "engine": "onnxruntime",
        "model_files": model_files,
        "default_variant": "fp32",
        "preprocess": PreprocessConfig(input_size=input_size).to_dict(),
        "n_views": n_views,
        "n_lanes": n_lanes,
        "aggregation": hp.get("aggregation", "max"),
        "arch": hp.get("arch"),
        "threshold": threshold,
        "scene": scene,
        "metrics": {
            "test": {k: test_m.get(k) for k in ("n", "n_pos", "n_neg", "auc_roc", "average_precision", "recall", "false_reject_rate", "recall_by_defect") if k in test_m},
            "threshold": threshold,
            "int8": (int8_info or {}).get("test"),
        },
        "io": {"input": "views [B, V, 3, S, S] float32", "outputs": ["egg_scores [B]", "view_scores [B, V]"], "opset": OPSET},
    }
    (work / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(source_file("preprocess.py"), work / "preprocess.py")
    shutil.copyfile(source_file("runtime.py"), work / "infer.py")
    (work / "README.txt").write_text(
        README.format(model_name=model_name, model_id=model_id, version=version, exported_at=exported_at, threshold=threshold, n_views=n_views, input_size=input_size, n_lanes=n_lanes),
        encoding="utf-8",
    )
    ctx.check_cancelled()

    # 5. Vérification du paquet en sous-processus
    ctx.progress(0.8, "Vérification du paquet (infer.py --verify)")
    verify = verify_package_subprocess(work, "fp32")
    vr = verify.get("report") or {}
    ctx.log(f"infer.py --verify (FP32) : code {verify['returncode']}, écart {vr.get('max_abs_diff')}.")
    if verify["returncode"] != 0 or not vr.get("ok"):
        raise ExportError(f"La vérification du paquet a échoué (infer.py --verify) : {verify.get('stderr') or vr}")
    verify_int8 = verify_package_subprocess(work, "int8") if "int8" in model_files else None

    # 6. Publication : dossier définitif + zip
    package_dir = mdir / "package"
    shutil.rmtree(package_dir, ignore_errors=True)
    work.rename(package_dir)
    for old in mdir.glob("eggvision_*.zip"):
        old.unlink(missing_ok=True)
    zip_path = mdir / f"eggvision_{model_id}_v{version}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(package_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(package_dir).as_posix())
    ctx.progress(0.97, "Paquet prêt")

    export = {
        "package_dir": project.relpath(package_dir),
        "zip": project.relpath(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "variants": sorted(model_files),
        "files": {k: {"file": v, "size_bytes": (package_dir / v).stat().st_size} for k, v in model_files.items()},
        "parity": {"fp32": parity, "int8": int8_info, "package_verify": {"fp32": vr, "int8": (verify_int8 or {}).get("report")}},
        "threshold": threshold,
        "version": version,
        "export_count": export_count,
        "opset": OPSET,
        "n_views": n_views,
        "input_size": input_size,
        "exported_at": exported_at,
        "duration_s": round(time.perf_counter() - t0, 1),
        "warnings": warnings,
        "bench": previous.get("bench", []),
    }
    with project.session() as s:
        get_model_row(s, model_id).export = export
    ctx.log(f"Paquet : {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} Mo).")
    return {"model_id": model_id, "zip": export["zip"], "version": version, "parity_max_abs_diff": parity["max_abs_diff"], "variants": export["variants"], "warnings": warnings}


def export_status(project: Project, model_id: str) -> dict[str, Any]:
    with project.session() as s:
        row = get_model_row(s, model_id)
        export = dict(row.export or {})
        threshold = row.threshold
        kind = row.kind
    jobs = [j.to_dict() for j in state.jobs.active() if j.kind in ("export", "bench") and (j.params or {}).get("model_id") == model_id]
    zip_ok = bool(export.get("zip")) and project.abspath(export["zip"]).is_file()
    return {
        "model_id": model_id,
        "exportable": kind == "classifier",
        "exported": zip_ok,
        "outdated": zip_ok and abs(float(export.get("threshold", threshold)) - threshold) > 1e-12,
        "current_threshold": threshold,
        "export": export,
        "active_jobs": jobs,
    }
