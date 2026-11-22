"""Cartes de chaleur par vue (Grad-CAM), à la demande, avec cache disque.

Pour un classifieur : Grad-CAM du logit de CHAQUE vue par rapport à la dernière
carte de caractéristiques du backbone (``forward_features``). Le logit d'une vue
ne dépend que de cette vue (BatchNorm en mode évaluation) : un seul passage
arrière sur la somme des logits donne les gradients de toutes les vues.
Les cartes d'un même œuf sont normalisées par leur maximum commun : l'intensité
reste comparable d'une vue à l'autre.

Pour un modèle d'anomalies : la carte est la distance de chaque patch au plus
proche voisin de la banque mémoire (cf. :mod:`.anomaly`).

Cache : ``models/<id>/cam/<track_id>/<vue:02d>.jpg`` + ``meta.json`` (scores).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from eggvision_shared.formats import encode_jpeg
from eggvision_shared.preprocess import PreprocessConfig, load_view, preprocess_views
from sqlalchemy import select

from ..db import models as m
from ..project import Project
from .store import get_model_row, load_classifier

_lock = threading.Lock()
CAM_VERSION = 1


def gradcam(model: Any, views_bgr: list[np.ndarray], input_size: int) -> tuple[np.ndarray, list[float]]:
    """Cartes ``[V, h, w]`` dans [0, 1] (normalisées par le max de l'œuf) et scores par vue."""
    import torch

    x = torch.from_numpy(preprocess_views(views_bgr, PreprocessConfig(input_size=input_size)))
    model.eval()
    with torch.enable_grad():
        feats = model.backbone.forward_features(x)
        feats = feats.detach().requires_grad_(True)
        pooled = model.backbone.forward_head(feats)
        logits = model.head(pooled).reshape(-1)
        (grads,) = torch.autograd.grad(logits.sum(), feats)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * feats).sum(dim=1)).detach().numpy()
    peak = float(cams.max())
    cams = cams / peak if peak > 0 else np.zeros_like(cams)
    scores = torch.sigmoid(logits).detach().numpy().astype(float).tolist()
    return cams, scores


def overlay(view_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la carte (palette « jet ») à la vue."""
    h, w = view_bgr.shape[:2]
    heat = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0.0, 1.0)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (alpha * heat[..., None]).astype(np.float32) + 0.15 * (heat[..., None] > 0.05)
    a = np.clip(a, 0.0, 0.7)
    return (view_bgr.astype(np.float32) * (1 - a) + colored.astype(np.float32) * a).astype(np.uint8)


def cam_dir(project: Project, model_id: str, track_id: int) -> Path:
    return project.model_dir(model_id) / "cam" / str(track_id)


def track_cams(project: Project, model_id: str, track_id: int) -> dict[str, Any]:
    """Calcule (ou relit du cache) les cartes de toutes les vues d'une piste."""
    d = cam_dir(project, model_id, track_id)
    meta_path = d / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") == CAM_VERSION and all((d / f"{v['index']:02d}.jpg").is_file() for v in meta["views"]):
            return meta
    with project.session() as s:
        row = get_model_row(s, model_id)
        kind, input_size, status = row.kind, row.input_size, row.status
        views = s.execute(select(m.View.index, m.View.path).where(m.View.track_id == track_id).order_by(m.View.index)).all()
    if not views:
        raise LookupError("Piste introuvable ou sans vue.")
    if kind not in ("classifier", "anomaly"):
        raise ValueError("Cartes de chaleur indisponibles pour ce type de modèle.")
    imgs = [load_view(str(project.abspath(p))) for _, p in views]
    with _lock:
        if kind == "anomaly":
            from .anomaly import anomaly_maps

            cams, scores = anomaly_maps(project, model_id, imgs)
            method = "distance_patch"
        else:
            if status not in ("trained", "cancelled", "failed") or not (project.model_dir(model_id) / "best.pt").is_file():
                raise ValueError("Ce modèle n'a pas encore de poids entraînés.")
            cams, scores = gradcam(load_classifier(project, model_id), imgs, input_size)
            method = "grad_cam"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for (idx, _), img, cam, score in zip(views, imgs, cams, scores, strict=False):
        (d / f"{idx:02d}.jpg").write_bytes(encode_jpeg(overlay(img, cam), 88))
        out.append({"index": int(idx), "score": round(float(score), 6), "peak": round(float(cam.max()), 4)})
    meta = {"version": CAM_VERSION, "model_id": model_id, "track_id": track_id, "method": method, "views": out}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def cam_image_path(project: Project, model_id: str, track_id: int, view_index: int) -> Path:
    track_cams(project, model_id, track_id)
    p = cam_dir(project, model_id, track_id) / f"{view_index:02d}.jpg"
    if not p.is_file():
        raise LookupError("Vue introuvable.")
    return p


def clear_cams(project: Project, model_id: str) -> None:
    import shutil

    shutil.rmtree(project.model_dir(model_id) / "cam", ignore_errors=True)
