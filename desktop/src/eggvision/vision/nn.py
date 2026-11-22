"""Détecteur entraînable (option ``method: "nn"``) : petit réseau mono-classe de type CenterNet.

* **Réseau** (:class:`EggNet`, ≈ 0,1 M paramètres) : mini-U-Net à pas de sortie 4 ; trois têtes —
  carte de chaleur des centres (sigmoïde), taille de la boîte (log largeur / hauteur, en pixels
  d'entrée) et décalage sous-pixel du centre. Entrée = image sous-échantillonnée à
  ``input_width`` pixels de large.
* **Annotations** (``models/detector/annotations.json``) : un petit jeu d'images de lots dont les
  boîtes sont générées automatiquement par la méthode classique, puis corrigées à la main
  (ajout, suppression, déplacement de boîtes) depuis l'écran « Détecteur entraînable ».
* **Entraînement** : tâche ``detector_training`` (groupe ``gpu``, :func:`eggvision.device.torch_device`) ;
  poids et métadonnées dans ``models/detector/<id>/`` (``weights.pt``, ``meta.json``).
* **Inférence** : sur CPU (petit réseau, sûr en multi-fil), dans :func:`nn_candidates`, avec le même
  post-traitement que la méthode classique (couloirs, partiels, netteté).

Les boîtes sont alignées sur les axes de l'image ; l'ellipse déduite a pour axes la largeur et la
hauteur de la boîte (le grand axe d'un œuf sur une table à rouleaux est parallèle aux rouleaux).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from eggvision_shared.formats import atomic_write_text, read_image

from ..project import Project
from .params import DetectionParams
from .scene import SceneGeometry

ANNOTATIONS_FILE = "annotations.json"
ARCH = "eggnet-centernet-s4"
STRIDE = 4
_LOCK = threading.RLock()


class DetectorError(ValueError):
    """Erreur présentable (message en français)."""


def detector_root(project: Project) -> Path:
    root = project.models_dir / "detector"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def _load_annotations(project: Project) -> dict[str, Any]:
    path = detector_root(project) / ANNOTATIONS_FILE
    if not path.is_file():
        return {"version": 1, "images": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("images", [])
    return data


def _save_annotations(project: Project, data: dict[str, Any]) -> None:
    atomic_write_text(
        detector_root(project) / ANNOTATIONS_FILE, json.dumps(data, ensure_ascii=False, indent=1)
    )


def annotation_id(lot_id: str, frame_index: int) -> str:
    return f"{lot_id}:{int(frame_index)}"


def ellipse_to_box(d: dict[str, Any], width: int, height: int) -> dict[str, float] | None:
    """Boîte englobante (bornée à l'image) d'une détection elliptique."""
    a, b = d["major"] / 2, d["minor"] / 2
    t = math.radians(d.get("angle", 0.0))
    hx = math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
    hy = math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
    x0, y0 = max(0.0, d["cx"] - hx), max(0.0, d["cy"] - hy)
    x1, y1 = min(float(width), d["cx"] + hx), min(float(height), d["cy"] + hy)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return {"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)}


def _validate_boxes(boxes: list[dict[str, Any]], width: int, height: int) -> list[dict[str, float]]:
    out = []
    for b in boxes:
        try:
            x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorError("Boîte invalide : x, y, w, h numériques attendus.") from exc
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise DetectorError("Boîte vide ou hors de l'image.")
        out.append({"x": round(x0, 1), "y": round(y0, 1), "w": round(x1 - x0, 1), "h": round(y1 - y0, 1)})
    return out


def list_annotations(project: Project, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    with _LOCK:
        images = _load_annotations(project)["images"]
    items = [
        {k: v for k, v in im.items() if k != "boxes"} | {"n_boxes": len(im["boxes"])}
        for im in images[offset : offset + limit]
    ]
    return {
        "total": len(images),
        "items": items,
        "n_boxes": sum(len(im["boxes"]) for im in images),
        "n_corrected": sum(1 for im in images if im.get("source") == "corrige"),
    }


def get_annotation(project: Project, ann_id: str) -> dict[str, Any]:
    with _LOCK:
        for im in _load_annotations(project)["images"]:
            if im["id"] == ann_id:
                return im
    raise DetectorError("Image d'annotation introuvable.")


def add_annotation_images(
    project: Project,
    lot_id: str,
    frame_indices: list[int] | None = None,
    n: int = 10,
    scene: dict[str, Any] | None = None,
    params: DetectionParams | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Ajoute des images d'annotation avec boîtes générées automatiquement par la méthode classique.

    Sans ``frame_indices``, ``n`` images sont prises régulièrement dans le lot.
    """
    from sqlalchemy import select

    from ..db import models as m
    from .detect import detect_frame

    with project.session() as s:
        indices = [
            int(i)
            for i in s.scalars(select(m.Frame.index).where(m.Frame.lot_id == lot_id).order_by(m.Frame.index))
        ]
    if not indices:
        raise DetectorError("Lot introuvable ou sans image.")
    if frame_indices is None:
        n = max(1, min(int(n), len(indices)))
        # Répartition régulière en évitant le tout début et la toute fin du lot (souvent sans œuf).
        lo, hi = (len(indices) - 1) * 0.1, (len(indices) - 1) * 0.9
        frame_indices = list(dict.fromkeys(indices[int(round(k))] for k in np.linspace(lo, hi, n)))
    p = DetectionParams.model_validate(
        {
            **(params.model_dump() if isinstance(params, DetectionParams) else (params or {})),
            "method": "classic",
        }
    )
    if scene is None:
        scene = project.get_config("scene")
    added = []
    with _LOCK:
        data = _load_annotations(project)
        known = {im["id"] for im in data["images"]}
        for fi in frame_indices:
            aid = annotation_id(lot_id, fi)
            if aid in known:
                continue
            path = project.frame_path(lot_id, fi)
            if not path.is_file():
                raise DetectorError(f"Image introuvable : lot {lot_id}, rang {fi}.")
            img = read_image(path)
            h, w = img.shape[:2]
            dets = detect_frame(img, scene, p)
            assert isinstance(dets, list)
            boxes = [b for b in (ellipse_to_box(d, w, h) for d in dets) if b is not None]
            item = {
                "id": aid,
                "lot_id": lot_id,
                "frame_index": int(fi),
                "width": w,
                "height": h,
                "boxes": boxes,
                "source": "auto",
                "updated_at": _now(),
            }
            data["images"].append(item)
            known.add(aid)
            added.append(item)
        _save_annotations(project, data)
    return added


def set_annotation_boxes(project: Project, ann_id: str, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """Remplace les boîtes d'une image d'annotation (correction manuelle)."""
    with _LOCK:
        data = _load_annotations(project)
        for im in data["images"]:
            if im["id"] == ann_id:
                im["boxes"] = _validate_boxes(boxes, im["width"], im["height"])
                im["source"] = "corrige"
                im["updated_at"] = _now()
                _save_annotations(project, data)
                return im
    raise DetectorError("Image d'annotation introuvable.")


def delete_annotation(project: Project, ann_id: str) -> None:
    with _LOCK:
        data = _load_annotations(project)
        before = len(data["images"])
        data["images"] = [im for im in data["images"] if im["id"] != ann_id]
        if len(data["images"]) == before:
            raise DetectorError("Image d'annotation introuvable.")
        _save_annotations(project, data)


# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------
def _build_model(width: int = 16):
    import torch
    from torch import nn

    def block(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class EggNet(nn.Module):
        """Mini-U-Net à pas de sortie 4 : carte de chaleur, taille, décalage."""

        def __init__(self, c: int = width) -> None:
            super().__init__()
            self.s2 = nn.Sequential(block(3, c, 2), block(c, c))
            self.s4 = nn.Sequential(block(c, 2 * c, 2), block(2 * c, 2 * c))
            self.s8 = nn.Sequential(
                block(2 * c, 4 * c, 2), block(4 * c, 4 * c), block(4 * c, 4 * c, dilation=2)
            )
            self.s16 = nn.Sequential(
                block(4 * c, 4 * c, 2), block(4 * c, 4 * c, dilation=2), block(4 * c, 4 * c, dilation=4)
            )
            self.up8 = block(8 * c, 4 * c)
            self.up4 = block(6 * c, 2 * c)
            self.heat = nn.Conv2d(2 * c, 1, 1)
            self.size = nn.Conv2d(2 * c, 2, 1)
            self.offset = nn.Conv2d(2 * c, 2, 1)
            nn.init.constant_(self.heat.bias, -2.19)  # probabilité initiale ≈ 0,1 (CenterNet)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            f4 = self.s4(self.s2(x))
            f8 = self.s8(f4)
            f16 = self.s16(f8)
            u8 = self.up8(
                torch.cat([f8, nn.functional.interpolate(f16, size=f8.shape[-2:], mode="nearest")], 1)
            )
            u4 = self.up4(
                torch.cat([f4, nn.functional.interpolate(u8, size=f4.shape[-2:], mode="nearest")], 1)
            )
            return self.heat(u4), self.size(u4), self.offset(u4)

    return EggNet()


def _input_size(width: int, height: int, input_width: int) -> tuple[int, int]:
    """Taille d'entrée (multiple de 16) conservant le rapport largeur / hauteur."""
    iw = max(32, int(round(input_width / 16)) * 16)
    ih = max(32, int(round(iw * height / width / 16)) * 16)
    return iw, ih


def _to_tensor(img_bgr: np.ndarray, iw: int, ih: int) -> np.ndarray:
    small = cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    return (small[:, :, ::-1].astype(np.float32) / 255.0 - 0.45).transpose(2, 0, 1) / 0.25


def _targets(boxes: list[dict[str, float]], sx: float, sy: float, ow: int, oh: int) -> tuple[np.ndarray, ...]:
    """Cibles CenterNet à la résolution de sortie (``ow`` × ``oh``)."""
    heat = np.zeros((oh, ow), dtype=np.float32)
    size = np.zeros((2, oh, ow), dtype=np.float32)
    off = np.zeros((2, oh, ow), dtype=np.float32)
    mask = np.zeros((oh, ow), dtype=np.float32)
    ys, xs = np.mgrid[0:oh, 0:ow].astype(np.float32)
    for b in boxes:
        cx = (b["x"] + b["w"] / 2) * sx / STRIDE
        cy = (b["y"] + b["h"] / 2) * sy / STRIDE
        w, h = b["w"] * sx, b["h"] * sy  # pixels d'entrée
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < ow and 0 <= iy < oh):
            continue
        sigma = max(0.8, min(w, h) / STRIDE / 6.0)
        heat = np.maximum(heat, np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)))
        size[:, iy, ix] = (math.log(max(w, 1.0)), math.log(max(h, 1.0)))
        off[:, iy, ix] = (cx - ix, cy - iy)
        mask[iy, ix] = 1.0
    return heat, size, off, mask


def _augment(
    img: np.ndarray, boxes: list[dict[str, float]], rng: random.Random
) -> tuple[np.ndarray, list[dict[str, float]]]:
    h, w = img.shape[:2]
    boxes = [dict(b) for b in boxes]
    if rng.random() < 0.5:
        img = img[:, ::-1]
        for b in boxes:
            b["x"] = w - b["x"] - b["w"]
    if rng.random() < 0.5:
        img = img[::-1]
        for b in boxes:
            b["y"] = h - b["y"] - b["h"]
    gain = rng.uniform(0.75, 1.25)
    bias = rng.uniform(-20, 20)
    img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), boxes


def _focal_loss(pred_logits, target):
    import torch

    p = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = (target > 0.99).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p**2 * (1 - target) ** 4 * neg
    n_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def train_detector(
    project: Project,
    *,
    epochs: int = 60,
    input_width: int = 256,
    batch_size: int = 8,
    lr: float = 2e-3,
    name: str = "",
    seed: int = 0,
    device: Any = None,
    progress: Callable[[float, str | None], None] = lambda *_: None,
    log_fn: Callable[[str], None] = lambda *_: None,
    check_cancelled: Callable[[], None] = lambda: None,
    publish: Callable[..., None] = lambda **_: None,
) -> dict[str, Any]:
    """Entraîne un détecteur sur les images d'annotation ; retourne ses métadonnées (``meta.json``)."""
    import torch

    from ..device import torch_device

    with _LOCK:
        images = [im for im in _load_annotations(project)["images"]]
    images = [im for im in images if project.frame_path(im["lot_id"], im["frame_index"]).is_file()]
    if len(images) < 2:
        raise DetectorError("Au moins deux images d'annotation sont nécessaires pour entraîner le détecteur.")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    order = list(range(len(images)))
    rng.shuffle(order)
    n_val = max(1, len(images) // 5) if len(images) >= 5 else 0
    val_ids, train_ids = order[:n_val], order[n_val:]
    w0, h0 = images[0]["width"], images[0]["height"]
    iw, ih = _input_size(w0, h0, input_width)
    frames = {}
    for k, im in enumerate(images):
        img = read_image(project.frame_path(im["lot_id"], im["frame_index"]))
        frames[k] = img
    dev = device or torch_device()
    model = _build_model()
    # Initialisation de la tête de taille sur la taille moyenne des boîtes : la régression part du bon ordre de grandeur.
    all_boxes = [b for im in images for b in im["boxes"]]
    if all_boxes:
        sx, sy = iw / w0, ih / h0
        with torch.no_grad():
            model.size.bias[0] = float(np.mean([math.log(max(1.0, b["w"] * sx)) for b in all_boxes]))
            model.size.bias[1] = float(np.mean([math.log(max(1.0, b["h"] * sy)) for b in all_boxes]))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Petit jeu d'annotations : chaque époque tire au moins 4 lots d'images (avec remise).
    samples_per_epoch = max(len(train_ids), 4 * batch_size)
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.15
    )
    ow, oh = iw // STRIDE, ih // STRIDE
    history = []
    log_fn(
        f"Détecteur : {len(train_ids)} images d'entraînement, {n_val} de validation, entrée {iw}×{ih}, appareil {dev}"
    )
    t0 = time.time()
    for epoch in range(epochs):
        check_cancelled()
        model.train()
        epoch_ids = [train_ids[i % len(train_ids)] for i in range(samples_per_epoch)]
        rng.shuffle(epoch_ids)
        total = 0.0
        for b0 in range(0, samples_per_epoch, batch_size):
            batch = epoch_ids[b0 : b0 + batch_size]
            xs, hs, ss, os_, ms = [], [], [], [], []
            for k in batch:
                im = images[k]
                img, boxes = _augment(frames[k], im["boxes"], rng)
                hh, ww = img.shape[:2]
                xs.append(_to_tensor(img, iw, ih))
                tgt = _targets(boxes, iw / ww, ih / hh, ow, oh)
                hs.append(tgt[0])
                ss.append(tgt[1])
                os_.append(tgt[2])
                ms.append(tgt[3])
            x = torch.from_numpy(np.stack(xs)).to(dev)
            th = torch.from_numpy(np.stack(hs))[:, None].to(dev)
            ts = torch.from_numpy(np.stack(ss)).to(dev)
            to = torch.from_numpy(np.stack(os_)).to(dev)
            tm = torch.from_numpy(np.stack(ms))[:, None].to(dev)
            ph, ps, po = model(x)
            n_pos = tm.sum().clamp(min=1.0)
            loss = (
                _focal_loss(ph, th)
                + 0.5 * (torch.abs(ps - ts) * tm).sum() / n_pos
                + (torch.abs(torch.sigmoid(po) - to) * tm).sum() / n_pos
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach().cpu())
        entry = {"epoch": epoch + 1, "loss": round(total / steps_per_epoch, 4)}
        history.append(entry)
        publish(history=history[-200:])
        progress((epoch + 1) / epochs, f"Époque {epoch + 1}/{epochs} — perte {entry['loss']:.3f}")
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_fn(f"Époque {epoch + 1}/{epochs} : perte {entry['loss']:.4f}")

    model.eval().cpu()
    det_id = datetime.now().strftime("det-%Y%m%d-%H%M%S")
    out_dir = detector_root(project) / det_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    meta = {
        "id": det_id,
        "name": name or det_id,
        "arch": ARCH,
        "created_at": _now(),
        "input_width": iw,
        "input_height": ih,
        "image_width": w0,
        "image_height": h0,
        "epochs": epochs,
        "n_images": len(images),
        "n_train": len(train_ids),
        "n_val": n_val,
        "n_boxes": sum(len(im["boxes"]) for im in images),
        "train_seconds": round(time.time() - t0, 1),
        "device": str(dev),
        "history": history,
    }
    # Évaluation rapide : rappel / précision sur la validation (ou l'entraînement à défaut).
    eval_ids = val_ids or train_ids
    meta["metrics"] = _evaluate(model, [images[k] for k in eval_ids], [frames[k] for k in eval_ids], meta)
    meta["metrics"]["subset"] = "validation" if val_ids else "entrainement"
    atomic_write_text(out_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=1))
    log_fn(
        f"Détecteur {det_id} enregistré : rappel {meta['metrics']['recall']:.2f}, précision {meta['metrics']['precision']:.2f}"
    )
    return meta


def _evaluate(
    model: Any,
    images: list[dict[str, Any]],
    frames: list[np.ndarray],
    meta: dict[str, Any],
    thr: float = 0.35,
) -> dict[str, Any]:
    tp = fp = fn = 0
    for im, img in zip(images, frames):
        boxes = _infer_boxes(model, meta, img, thr)
        gt = [(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, min(b["w"], b["h"])) for b in im["boxes"]]
        used = set()
        for bx in boxes:
            cx, cy = bx["x"] + bx["w"] / 2, bx["y"] + bx["h"] / 2
            best = None
            for k, (gx, gy, gs) in enumerate(gt):
                if k not in used and math.hypot(cx - gx, cy - gy) < 0.3 * gs:
                    best = k
                    break
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
        fn += len(gt) - len(used)
    return {
        "recall": round(tp / max(1, tp + fn), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, tuple[float, Any, dict[str, Any]]] = {}


def list_detectors(project: Project) -> list[dict[str, Any]]:
    out = []
    for d in sorted(detector_root(project).iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if d.is_dir() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            meta.pop("history", None)
            out.append(meta)
    return out


def delete_detector(project: Project, det_id: str) -> None:
    d = detector_root(project) / det_id
    if not det_id or "/" in det_id or "\\" in det_id or not (d / "meta.json").is_file():
        raise DetectorError("Détecteur introuvable.")
    shutil.rmtree(d)


def _load(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    weights = model_dir / "weights.pt"
    if not weights.is_file():
        raise DetectorError(f"Détecteur introuvable : {model_dir.name}")
    key = str(weights)
    mtime = weights.stat().st_mtime
    with _LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        model = _build_model()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        _MODEL_CACHE[key] = (mtime, model, meta)
        return model, meta


def _infer_boxes(
    model: Any, meta: dict[str, Any], img: np.ndarray, thr: float, max_det: int = 64
) -> list[dict[str, float]]:
    import torch

    h, w = img.shape[:2]
    iw, ih = _input_size(w, h, meta["input_width"])
    x = torch.from_numpy(_to_tensor(img, iw, ih)[None])
    with torch.no_grad():
        ph, ps, po = model(x)
        heat = torch.sigmoid(ph)
        peaks = (heat == torch.nn.functional.max_pool2d(heat, 3, 1, 1)) & (heat > thr)
    heat_np = heat[0, 0].numpy()
    ys, xs = np.nonzero(peaks[0, 0].numpy())
    order = np.argsort(-heat_np[ys, xs])[:max_det]
    size, off = ps[0].numpy(), torch.sigmoid(po[0]).numpy()
    sx, sy = w / iw, h / ih
    out = []
    for k in order:
        y, xx = int(ys[k]), int(xs[k])
        cx = (xx + off[0, y, xx]) * STRIDE * sx
        cy = (y + off[1, y, xx]) * STRIDE * sy
        bw, bh = math.exp(size[0, y, xx]) * sx, math.exp(size[1, y, xx]) * sy
        out.append({"x": cx - bw / 2, "y": cy - bh / 2, "w": bw, "h": bh, "score": float(heat_np[y, xx])})
    return out


def nn_candidates(
    image_bgr: np.ndarray, p: DetectionParams, geo: SceneGeometry
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Candidats du détecteur entraînable, au format de la méthode classique."""
    if not p.nn_model:
        raise DetectorError("Aucun détecteur entraînable n'est sélectionné (paramètre « nn_model »).")
    direct = Path(p.nn_model)
    if direct.is_absolute():
        model_dir = direct
    else:
        from .. import state

        project = state.projects.current
        if project is None:
            raise DetectorError("Aucun projet n'est ouvert : détecteur entraînable indisponible.")
        model_dir = detector_root(project) / p.nn_model
    model, meta = _load(model_dir)
    return candidates_from_boxes(
        _infer_boxes(model, meta, image_bgr, p.nn_threshold), image_bgr.shape[1], image_bgr.shape[0], p, geo
    ), {"model": p.nn_model}


def candidates_from_boxes(
    boxes: list[dict[str, float]], w: int, h: int, p: DetectionParams, geo: SceneGeometry
) -> list[dict[str, Any]]:
    """Boîtes → ellipses ; les boîtes coupées par un bord sont complétées avec la taille typique de l'image."""
    margin = max(1.0, p.partial_margin_mm / geo.mm_per_px)
    items = []
    for b in boxes:
        x0, y0, x1, y1 = (
            max(0.0, b["x"]),
            max(0.0, b["y"]),
            min(float(w), b["x"] + b["w"]),
            min(float(h), b["y"] + b["h"]),
        )
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        touch = [
            s
            for s, c in (
                ("left", x0 <= margin),
                ("top", y0 <= margin),
                ("right", x1 >= w - margin),
                ("bottom", y1 >= h - margin),
            )
            if c
        ]
        items.append((x0, y0, x1, y1, touch, b.get("score", 1.0)))
    full = [(x1 - x0, y1 - y0) for x0, y0, x1, y1, touch, _ in items if not touch]
    if full:
        tw, th = float(np.median([f[0] for f in full])), float(np.median([f[1] for f in full]))
    else:
        big, small = 58.0 / geo.mm_per_px, 43.0 / geo.mm_per_px
        tw, th = (big, small) if geo.direction in ("up", "down") else (small, big)
    out = []
    for x0, y0, x1, y1, touch, score in items:
        bw, bh = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if touch:
            if "left" in touch and "right" not in touch:
                bw = max(bw, tw)
                cx = x1 - bw / 2
            elif "right" in touch and "left" not in touch:
                bw = max(bw, tw)
                cx = x0 + bw / 2
            if "top" in touch and "bottom" not in touch:
                bh = max(bh, th)
                cy = y1 - bh / 2
            elif "bottom" in touch and "top" not in touch:
                bh = max(bh, th)
                cy = y0 + bh / 2
        major, minor = max(bw, bh), min(bw, bh)
        major_mm, minor_mm = major * geo.mm_per_px, minor * geo.mm_per_px
        if not touch and (
            not (p.min_major_mm <= major_mm <= p.max_major_mm)
            or not (p.min_minor_mm <= minor_mm <= p.max_minor_mm)
        ):
            continue
        out.append(
            {
                "cx": cx,
                "cy": cy,
                "major": major,
                "minor": minor,
                "angle": 0.0 if bw >= bh else 90.0,
                "touch": touch,
                "score": round(float(score), 4),
            }
        )
    return out


def submit_training(
    project: Project, *, epochs: int = 60, input_width: int = 256, name: str = "", activate: bool = False
) -> Any:
    """Lance l'entraînement du détecteur dans une tâche de fond ``detector_training`` (groupe ``gpu``)."""
    from .. import state
    from ..jobs import JobContext
    from .params import save_params

    with _LOCK:
        n = len(_load_annotations(project)["images"])
    if n < 2:
        raise DetectorError("Au moins deux images d'annotation sont nécessaires pour entraîner le détecteur.")

    def run(ctx: JobContext) -> dict[str, Any]:
        meta = train_detector(
            project,
            epochs=epochs,
            input_width=input_width,
            name=name,
            progress=ctx.progress,
            log_fn=ctx.log,
            check_cancelled=ctx.check_cancelled,
            publish=ctx.publish,
        )
        if activate:
            save_params(project, "detection", {"nn_model": meta["id"]})
            ctx.log(f"Détecteur {meta['id']} sélectionné pour la méthode « nn ».")
        return {"detector": {k: v for k, v in meta.items() if k != "history"}}

    return state.jobs.submit(
        "detector_training",
        f"Entraînement du détecteur ({n} images)",
        run,
        params={"epochs": epochs, "input_width": input_width, "name": name},
        project=project,
        group="gpu",
    )
