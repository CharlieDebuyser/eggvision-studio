"""Modèle de référence par détection d'anomalies (type PatchCore), tâche ``kind="anomaly"``.

* Caractéristiques intermédiaires d'un backbone pré-entraîné (gelé, aucun
  apprentissage) : sorties aux réductions 8 et 16, lissées localement (3 × 3),
  la plus profonde ré-échantillonnée à la résolution de la première, concaténées
  → un descripteur par patch.
* Banque mémoire construite **uniquement sur les œufs viables du jeu
  d'entraînement** : tirage aléatoire de ``patches_per_view`` patches par vue,
  puis sous-échantillonnage « coreset » glouton (k-centres, distances sur une
  projection aléatoire) ou aléatoire jusqu'à ``bank_size`` patches.
* Score d'une vue = distance maximale d'un de ses patches à son plus proche voisin
  dans la banque ; score de l'œuf = maximum sur ses vues.
* Calibration sur la validation (régression logistique 1-D, monotone : l'AUC est
  inchangée) pour ramener le score dans [0, 1] ; à défaut de non viables en
  validation, centrage sur le 99ᵉ centile des viables.

Le rapport d'évaluation est le même que pour le classifieur. L'export ONNX de ce
modèle n'est pas pris en charge.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .. import state
from ..db import models as m
from ..device import detect_device
from ..jobs import JobCancelled, JobContext
from ..project import Project
from .dataset import EggViewsDataset, ViewCache, count_records, load_records
from .evaluation import refresh_report, write_predictions
from .network import ARCHITECTURES, load_pretrained
from .store import (
    get_model_row,
    load_checkpoint,
    model_dir,
    new_model_id,
    next_model_number,
    report_path,
    save_checkpoint,
    write_json,
)
from .training import TrainingError, _split_for, labels_fingerprint, resolve_device, seed_everything


@dataclass
class AnomalyParams:
    arch: str = "mobilenetv3_large_100"
    input_size: int = 224
    n_views: int = 12
    patches_per_view: int = 48
    bank_size: int = 4000
    coreset: str = "greedy"  # greedy | random
    projection_dim: int = 64
    max_train_eggs: int = 400
    batch_size: int = 8
    seed: int = 42
    device: str = "auto"
    reviewed_only: bool = False
    cache_mb: int = 256

    def validate(self) -> AnomalyParams:
        if self.arch not in ARCHITECTURES:
            raise TrainingError(f"Architecture inconnue : {self.arch}")
        if self.coreset not in ("greedy", "random"):
            raise TrainingError("Sous-échantillonnage inconnu : « greedy » ou « random ».")
        if not 32 <= self.input_size <= 512 or not 1 <= self.n_views <= 64:
            raise TrainingError("Taille d'entrée ou nombre de vues invalide.")
        if self.bank_size < 10 or self.patches_per_view < 1:
            raise TrainingError("Banque mémoire trop petite.")
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AnomalyParams:
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names
        if unknown:
            raise TrainingError("Paramètre(s) inconnu(s) : " + ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in (d or {}).items() if k in names}).validate()


def build_extractor(arch: str):
    """Backbone gelé renvoyant les cartes aux réductions 8 et 16 ; + rapport des poids."""
    import timm

    net = timm.create_model(arch, pretrained=False, features_only=True, out_indices=(2, 3))
    info = load_pretrained(net, arch)
    return net.eval(), info


def patch_features(extractor: Any, x: Any) -> Any:
    """``x`` [N, 3, S, S] → descripteurs ``[N, D, h, w]``."""
    import torch
    import torch.nn.functional as F

    f2, f3 = extractor(x)
    f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
    f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3], dim=1)


def greedy_coreset(feats: Any, k: int, proj_dim: int, seed: int, check_cancelled=lambda: None) -> Any:
    """Indices d'un coreset k-centres glouton (distances calculées sur une projection aléatoire)."""
    import torch

    n = feats.shape[0]
    if k >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    if proj_dim and feats.shape[1] > proj_dim:
        proj = torch.randn(feats.shape[1], proj_dim, generator=g).to(feats.device) / math.sqrt(proj_dim)
        z = feats @ proj
    else:
        z = feats
    first = int(torch.randint(n, (1,), generator=g))
    chosen = [first]
    min_d = torch.cdist(z, z[first : first + 1]).squeeze(1)
    for i in range(1, k):
        if i % 200 == 0:
            check_cancelled()
        nxt = int(torch.argmax(min_d))
        chosen.append(nxt)
        min_d = torch.minimum(min_d, torch.cdist(z, z[nxt : nxt + 1]).squeeze(1))
    return torch.tensor(chosen)


def nn_distance(patches: Any, bank: Any, chunk: int = 4096) -> Any:
    """Distance de chaque patch ``[P, D]`` à son plus proche voisin dans ``bank`` ``[K, D]``."""
    import torch

    out = []
    for i in range(0, patches.shape[0], chunk):
        out.append(torch.cdist(patches[i : i + chunk], bank).min(dim=1).values)
    return torch.cat(out)


def fit_calibration(raw: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Paramètres ``(a, b)`` de ``sigmoid(a · raw + b)`` : régression logistique 1-D (Newton, L2 faible)."""
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    neg = raw[y == 0]
    if len(raw) == 0:
        return {"a": 1.0, "b": 0.0, "method": "identite"}
    if y.sum() == 0 or y.sum() == len(y):
        ref = neg if len(neg) else raw
        center = float(np.percentile(ref, 99))
        scale = float(np.std(ref)) or 1.0
        return {"a": 1.0 / scale, "b": -center / scale, "method": "centile_99_viables"}
    mu, sd = float(raw.mean()), float(raw.std()) or 1.0
    z = (raw - mu) / sd
    w = np.zeros(2)
    lam = 1e-3
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(w[0] * z + w[1])))
        g = np.array([np.sum((p - y) * z) + lam * w[0], np.sum(p - y)])
        r = p * (1 - p)
        h = np.array([[np.sum(r * z * z) + lam, np.sum(r * z)], [np.sum(r * z), np.sum(r) + 1e-9]])
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    a = float(w[0]) / sd
    b = float(w[1]) - float(w[0]) * mu / sd
    if a <= 0:  # sécurité : la calibration doit rester croissante
        a, b = 1.0 / sd, -float(np.percentile(neg, 99)) / sd if len(neg) else -mu / sd
    return {"a": a, "b": b, "method": "logistique_validation"}


def calibrate(raw: np.ndarray, cal: dict[str, float]) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(cal["a"] * np.asarray(raw, dtype=np.float64) + cal["b"])))


# ---------------------------------------------------------------------------
# Lancement
# ---------------------------------------------------------------------------
def start_anomaly(project: Project, split_id: int, params: dict[str, Any] | None = None, name: str | None = None) -> dict[str, str]:
    p = AnomalyParams.from_dict(params)
    with project.session() as s:
        split = _split_for(s, split_id)
        number = next_model_number(s)
        model_id = new_model_id(p.arch) + "-anom"
        s.add(
            m.Model(
                id=model_id,
                name=(name or "").strip() or f"#{number} Anomalies {ARCHITECTURES[p.arch]['label']} · {split.name} v{split.version}",
                kind="anomaly",
                arch=p.arch,
                status="created",
                split_id=split.id,
                hparams=asdict(p),
                n_views=p.n_views,
                input_size=p.input_size,
                threshold=0.5,
                extra={"number": number},
            )
        )
    model_dir(project, model_id)
    job = state.jobs.submit(
        "anomaly", f"Modèle d'anomalies — {model_id}", lambda ctx: run_anomaly(ctx, project, model_id),
        params={"model_id": model_id}, project=project, group="gpu",
    )
    with project.session() as s:
        get_model_row(s, model_id).job_id = job.id
    return {"model_id": model_id, "job_id": job.id}


def _embed_records(extractor, ds, device, batch_size, check_cancelled, progress=lambda f: None):
    """Itère ``(indices, descripteurs [B, V, D, h, w])`` sur un jeu d'œufs."""
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    done = 0
    with torch.no_grad():
        for x, _, idx in loader:
            check_cancelled()
            b, v = x.shape[:2]
            f = patch_features(extractor, x.to(device).flatten(0, 1))
            yield idx.tolist(), f.reshape(b, v, *f.shape[1:])
            done += b
            progress(done / max(1, len(ds)))


def run_anomaly(ctx: JobContext, project: Project, model_id: str) -> dict[str, Any]:
    import torch
    from eggvision_shared.preprocess import PreprocessConfig

    t_start = time.perf_counter()
    with project.session() as s:
        row = get_model_row(s, model_id)
        p = AnomalyParams.from_dict(row.hparams)
        split = _split_for(s, row.split_id)
        assignments = dict(split.assignments)
        split_info = {"split_id": split.id, "split_name": split.name, "split_version": split.version, "group_by": (split.strategy or {}).get("group_by")}
        row.status = "training"
        row.job_id = ctx.job_id
    warnings: list[str] = []
    try:
        records = load_records(project, assignments, reviewed_only=p.reviewed_only)
        bank_src = [r for r in records if r.subset == "train" and r.target == 0]
        if not bank_src:
            raise TrainingError("Aucun œuf viable dans le jeu d'entraînement : banque mémoire impossible.")
        rng = np.random.default_rng(p.seed)
        if len(bank_src) > p.max_train_eggs:
            bank_src = [bank_src[i] for i in sorted(rng.choice(len(bank_src), p.max_train_eggs, replace=False))]
        seed_everything(p.seed)
        device = resolve_device(p.device)
        extractor, info = build_extractor(p.arch)
        extractor.to(device)
        if info.get("warning"):
            warnings.append(info["warning"])
            ctx.log("AVERTISSEMENT : " + info["warning"])
        ctx.log(f"Banque mémoire : {len(bank_src)} œufs viables d'entraînement, appareil {device.type}.")
        pre = PreprocessConfig(input_size=p.input_size)
        cache = ViewCache(p.cache_mb * 1024 * 1024)

        # 1. patches des viables d'entraînement
        pool = []
        g = torch.Generator().manual_seed(p.seed)
        ds = EggViewsDataset(bank_src, p.n_views, pre, cache=cache)
        for _, f in _embed_records(extractor, ds, device, p.batch_size, ctx.check_cancelled, lambda fr: ctx.progress(0.3 * fr, "Extraction des patches viables")):
            flat = f.permute(0, 1, 3, 4, 2).reshape(-1, f.shape[2])  # [B*V*h*w, D]
            per_view = f.shape[3] * f.shape[4]
            n_views_batch = f.shape[0] * f.shape[1]
            k = min(p.patches_per_view, per_view)
            offs = torch.stack([torch.randperm(per_view, generator=g)[:k] + j * per_view for j in range(n_views_batch)]).reshape(-1)
            pool.append(flat[offs.to(flat.device)].float())
        pool_t = torch.cat(pool)
        ctx.log(f"{pool_t.shape[0]} patches candidats de dimension {pool_t.shape[1]}.")
        ctx.progress(0.35, "Sous-échantillonnage de la banque mémoire")
        if p.coreset == "greedy":
            idx = greedy_coreset(pool_t, p.bank_size, p.projection_dim, p.seed, ctx.check_cancelled)
        else:
            idx = torch.randperm(pool_t.shape[0], generator=g)[: p.bank_size]
        bank = pool_t[idx.to(pool_t.device)].contiguous()
        ctx.log(f"Banque mémoire : {bank.shape[0]} patches ({p.coreset}).")

        # 2. scores bruts de toutes les pistes
        raw_egg = np.zeros(len(records))
        raw_views: list[list[float]] = [[] for _ in records]
        ds_all = EggViewsDataset(records, p.n_views, pre, cache=cache)
        for idxs, f in _embed_records(extractor, ds_all, device, p.batch_size, ctx.check_cancelled, lambda fr: ctx.progress(0.45 + 0.45 * fr, "Calcul des scores")):
            b, v, d, h, w = f.shape
            dist = nn_distance(f.permute(0, 1, 3, 4, 2).reshape(-1, d).float(), bank).reshape(b, v, h * w)
            vmax = dist.max(dim=2).values.cpu().numpy()
            for k, i in enumerate(idxs):
                n_unique = min(len(records[i].view_paths), p.n_views)
                raw_views[i] = vmax[k][:n_unique].astype(float).tolist()
                raw_egg[i] = float(vmax[k].max())

        # 3. calibration sur la validation
        val_idx = [i for i, r in enumerate(records) if r.subset == "val"]
        if val_idx:
            cal = fit_calibration(raw_egg[val_idx], np.asarray([records[i].target for i in val_idx]))
        else:
            tr = [i for i, r in enumerate(records) if r.subset == "train" and r.target == 0]
            cal = fit_calibration(raw_egg[tr], np.zeros(len(tr)))
            warnings.append("Validation vide : calibration sur les viables d'entraînement.")
        ctx.log(f"Calibration : {cal['method']} (a = {cal['a']:.4g}, b = {cal['b']:.4g}).")
        scores = calibrate(raw_egg, cal)
        view_scores = [calibrate(np.asarray(v), cal).tolist() for v in raw_views]
        mdir = model_dir(project, model_id)
        save_checkpoint(
            mdir / "anomaly.pt",
            {"kind": "anomaly", "arch": p.arch, "input_size": p.input_size, "n_views": p.n_views, "bank": bank.cpu(), "calibration": cal, "out_indices": [2, 3]},
        )
        ctx.check_cancelled()
        n_pred = write_predictions(project, model_id, records, scores, view_scores)
        data_summary = {**split_info, "reviewed_only": p.reviewed_only, "counts": count_records(records), "n_eggs": len(records), "bank_eggs": len(bank_src), "labels_fingerprint": labels_fingerprint(records)}
        duration = round(time.perf_counter() - t_start, 1)
        with project.session() as s:
            row = get_model_row(s, model_id)
            row.status = "trained"
            row.data_summary = data_summary
            row.extra = {**(row.extra or {}), "pretrained": info, "device": {**detect_device(), "used": device.type}, "warnings": warnings, "duration_s": duration, "calibration": cal, "bank_size": int(bank.shape[0])}
        report = refresh_report(project, model_id)
        report["anomaly"] = {"bank_size": int(bank.shape[0]), "bank_eggs": len(bank_src), "calibration": cal, "duration_s": duration, "warnings": warnings}
        write_json(report_path(project, model_id), report)
        auc = (report.get("subsets") or {}).get("test", {}).get("auc_roc")
        ctx.log(f"Terminé : {n_pred} prédictions ; AUC test = {auc}.")
        return {"model_id": model_id, "test_auc": auc, "n_predictions": n_pred}
    except JobCancelled:
        with project.session() as s:
            get_model_row(s, model_id).status = "cancelled"
        raise
    except Exception:
        with project.session() as s:
            get_model_row(s, model_id).status = "failed"
        raise


# ---------------------------------------------------------------------------
# Cartes d'anomalie (explorateur d'erreurs)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_loaded: dict[str, tuple[float, Any, dict[str, Any]]] = {}


def _load(project: Project, model_id: str):
    path = project.model_dir(model_id) / "anomaly.pt"
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    with _cache_lock:
        hit = _loaded.get(str(path))
        if hit and hit[0] == mtime:
            return hit[1], hit[2]
    ckpt = load_checkpoint(path)
    extractor, _ = build_extractor(ckpt["arch"])
    with _cache_lock:
        _loaded.clear()
        _loaded[str(path)] = (mtime, extractor, ckpt)
    return extractor, ckpt


def anomaly_maps(project: Project, model_id: str, views_bgr: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """Cartes de distance ``[V, h, w]`` (normalisées par le max de l'œuf) et scores calibrés par vue."""
    import torch
    from eggvision_shared.preprocess import PreprocessConfig, preprocess_views

    extractor, ckpt = _load(project, model_id)
    x = torch.from_numpy(preprocess_views(views_bgr, PreprocessConfig(input_size=int(ckpt["input_size"]))))
    with torch.no_grad():
        f = patch_features(extractor, x)
        n, d, h, w = f.shape
        dist = nn_distance(f.permute(0, 2, 3, 1).reshape(-1, d), ckpt["bank"]).reshape(n, h, w).numpy()
    scores = calibrate(dist.reshape(n, -1).max(axis=1), ckpt["calibration"]).tolist()
    lo, hi = float(dist.min()), float(dist.max())
    return (dist - lo) / ((hi - lo) or 1.0), scores
