"""Jeu de données : œufs utilisables, sélection des vues, augmentations, chargement.

* Un œuf (une piste) est utilisable si son verdict est ``viable`` ou ``non_viable``
  (les ``incertain``, ``exclu`` et pistes sans label sont écartés ; option « labels
  relus uniquement »). Cible : 1 = non viable.
* Chaque œuf fournit exactement ``n_views`` vues : sous-échantillonnage régulier
  (aléatoire à l'entraînement) si la piste en a plus, complément par répétition
  cyclique si elle en a moins (sans effet sur une agrégation « max »).
* Les augmentations s'appliquent aux vues **BGR uint8** puis le prétraitement
  partagé ``eggvision_shared.preprocess.preprocess_views`` produit le tenseur :
  c'est exactement le code exécuté sur le Raspberry Pi.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from eggvision_shared.constants import VERDICT_NON_VIABLE
from eggvision_shared.preprocess import PreprocessConfig, load_view, preprocess_views, select_view_indices
from sqlalchemy import select

from ..db import models as m
from ..project import Project
from .splits import usable_verdict_filter

_SQL_CHUNK = 500


@dataclass
class EggRecord:
    """Un œuf utilisable et les chemins (absolus) de ses vues, dans l'ordre."""

    track_id: int
    lot_id: str
    subset: str
    target: int  # 1 = non viable
    defect: str | None
    reviewed: bool
    view_paths: list[str]


def load_records(
    project: Project,
    assignments: dict[str, str],
    subsets: Iterable[str] | None = None,
    reviewed_only: bool = False,
) -> list[EggRecord]:
    """Œufs utilisables des lots affectés (``assignments`` : ``{lot_id: subset}``), triés par piste."""
    wanted = set(subsets) if subsets is not None else None
    lot_ids = [lot for lot, sub in assignments.items() if wanted is None or sub in wanted]
    records: list[EggRecord] = []
    with project.session() as s:
        for i in range(0, len(lot_ids), _SQL_CHUNK):
            chunk = lot_ids[i : i + _SQL_CHUNK]
            rows = s.execute(
                select(m.Track.id, m.Track.lot_id, m.Label.verdict, m.Label.defect, m.Label.reviewed)
                .join(m.Label, m.Label.track_id == m.Track.id)
                .where(m.Track.lot_id.in_(chunk), *usable_verdict_filter(reviewed_only))
                .order_by(m.Track.id)
            ).all()
            by_track = {r[0]: r for r in rows}
            ids = list(by_track)
            paths: dict[int, list[str]] = {}
            for j in range(0, len(ids), _SQL_CHUNK):
                for track_id, path in s.execute(
                    select(m.View.track_id, m.View.path)
                    .where(m.View.track_id.in_(ids[j : j + _SQL_CHUNK]))
                    .order_by(m.View.track_id, m.View.index)
                ):
                    paths.setdefault(track_id, []).append(str(project.abspath(path)))
            for tid in ids:
                if tid not in paths:
                    continue  # piste sans vue : inutilisable
                _, lot_id, verdict, defect, reviewed = by_track[tid]
                records.append(
                    EggRecord(
                        track_id=tid,
                        lot_id=lot_id,
                        subset=assignments[lot_id],
                        target=1 if verdict == VERDICT_NON_VIABLE else 0,
                        defect=defect if verdict == VERDICT_NON_VIABLE else None,
                        reviewed=bool(reviewed),
                        view_paths=paths[tid],
                    )
                )
    records.sort(key=lambda r: r.track_id)
    return records


def count_records(records: Sequence[EggRecord]) -> dict[str, Any]:
    """Effectifs par sous-ensemble (n, viables, non viables, par défaut)."""
    out: dict[str, Any] = {}
    for r in records:
        e = out.setdefault(r.subset, {"n": 0, "viable": 0, "non_viable": 0, "defects": {}})
        e["n"] += 1
        if r.target:
            e["non_viable"] += 1
            key = r.defect or "non_precise"
            e["defects"][key] = e["defects"].get(key, 0) + 1
        else:
            e["viable"] += 1
    return out


def pick_view_indices(n_available: int, n_views: int, rng: np.random.Generator | None = None) -> list[int]:
    """``n_views`` indices parmi ``n_available`` (régulier, ou aléatoire trié si ``rng``), complétés cycliquement."""
    if n_available <= 0:
        raise ValueError("Piste sans vue")
    if n_available >= n_views:
        if rng is not None and n_available > n_views:
            return sorted(rng.choice(n_available, size=n_views, replace=False).tolist())
        return select_view_indices(n_available, n_views)
    base = list(range(n_available))
    return [base[i % n_available] for i in range(n_views)]


# ---------------------------------------------------------------------------
# Cache des vues décodées
# ---------------------------------------------------------------------------
class ViewCache:
    """Cache LRU (borné en octets) des vues décodées ; ``max_bytes = 0`` le désactive."""

    def __init__(self, max_bytes: int = 0) -> None:
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def __getstate__(self) -> dict[str, Any]:
        """Copie vers un processus de chargement (``num_workers > 0``) : cache vide, verrou recréé."""
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, st: dict[str, Any]) -> None:
        self.__init__(st["max_bytes"])  # type: ignore[misc]

    def get(self, path: str) -> np.ndarray:
        if self.max_bytes <= 0:
            return load_view(path)
        with self._lock:
            img = self._data.get(path)
            if img is not None:
                self._data.move_to_end(path)
                self.hits += 1
                return img
        img = load_view(path)
        with self._lock:
            self.misses += 1
            if img.nbytes <= self.max_bytes and path not in self._data:
                self._data[path] = img
                self._size += img.nbytes
                while self._size > self.max_bytes and self._data:
                    _, old = self._data.popitem(last=False)
                    self._size -= old.nbytes
        return img


# ---------------------------------------------------------------------------
# Augmentations (vues BGR uint8, grand axe de l'œuf horizontal)
# ---------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """Augmentations légères, cohérentes avec la prise de vue (œuf couché, grand axe horizontal).

    * retournements horizontal et vertical (l'œuf reste couché) ;
    * petite rotation (± ``rotate_deg``) et léger décalage / zoom (imprécision du recadrage) ;
    * luminosité / contraste (± ``brightness``, ± ``contrast``) et gain par canal (± ``color``) ;
    * flou de bouger léger (probabilité ``blur_p``, longueur ≤ ``blur_max_px``) dans une direction proche
      du défilement (verticale dans la vue).
    """

    hflip: bool = True
    vflip: bool = True
    rotate_deg: float = 10.0
    shift: float = 0.04
    scale: float = 0.05
    brightness: float = 0.10
    contrast: float = 0.10
    color: float = 0.04
    blur_p: float = 0.25
    blur_max_px: int = 7

    @classmethod
    def preset(cls, name: str) -> AugmentConfig | None:
        if name in ("none", "aucune", "", None):
            return None
        if name == "light":
            return cls(rotate_deg=5.0, shift=0.02, scale=0.03, brightness=0.06, contrast=0.06, color=0.02, blur_p=0.15, blur_max_px=5)
        if name == "standard":
            return cls()
        raise ValueError(f"Niveau d'augmentation inconnu : {name!r} (none, light, standard)")


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2.0
    dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    for t in np.linspace(-c, c, length * 3):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        k[y, x] = 1.0
    return k / k.sum()


def augment_view(img: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """Augmente une vue BGR uint8 (même taille en sortie)."""
    out = img
    if cfg.hflip and rng.random() < 0.5:
        out = out[:, ::-1]
    if cfg.vflip and rng.random() < 0.5:
        out = out[::-1, :]
    h, w = out.shape[:2]
    angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg) if cfg.rotate_deg else 0.0
    scale = 1.0 + (rng.uniform(-cfg.scale, cfg.scale) if cfg.scale else 0.0)
    tx = rng.uniform(-cfg.shift, cfg.shift) * w if cfg.shift else 0.0
    ty = rng.uniform(-cfg.shift, cfg.shift) * h if cfg.shift else 0.0
    if angle or scale != 1.0 or tx or ty:
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty
        out = cv2.warpAffine(np.ascontiguousarray(out), mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if cfg.blur_p and rng.random() < cfg.blur_p:
        length = int(rng.integers(3, max(4, cfg.blur_max_px + 1)))
        out = cv2.filter2D(np.ascontiguousarray(out), -1, motion_blur_kernel(length, 90.0 + rng.uniform(-20, 20)))
    if cfg.brightness or cfg.contrast or cfg.color:
        f = out.astype(np.float32)
        mean = f.mean()
        contrast = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        bright = 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
        gains = 1.0 + rng.uniform(-cfg.color, cfg.color, size=3).astype(np.float32)
        f = ((f - mean) * contrast + mean) * bright * gains[None, None, :]
        out = np.clip(f, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class EggViewsDataset:
    """``Dataset`` PyTorch : ``(vues [V, 3, S, S] float32, cible float32, rang)``.

    Les vues passent par :func:`eggvision_shared.preprocess.preprocess_views`.
    Les tirages aléatoires dépendent de ``(seed, epoch, rang)`` : reproductibles,
    y compris avec plusieurs processus de chargement.
    """

    def __init__(
        self,
        records: Sequence[EggRecord],
        n_views: int,
        preprocess: PreprocessConfig,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        cache: ViewCache | None = None,
        random_views: bool = False,
    ) -> None:
        self.records = list(records)
        self.n_views = int(n_views)
        self.preprocess = preprocess
        self.augment = augment
        self.seed = int(seed)
        self.cache = cache or ViewCache(0)
        self.random_views = random_views
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def raw_views(self, i: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
        rec = self.records[i]
        idx = pick_view_indices(len(rec.view_paths), self.n_views, rng if self.random_views else None)
        return [self.cache.get(rec.view_paths[k]) for k in idx]

    def __getitem__(self, i: int):
        import torch

        rng = np.random.default_rng([self.seed, self.epoch, i]) if (self.augment or self.random_views) else None
        views = self.raw_views(i, rng)
        if self.augment is not None and rng is not None:
            views = [augment_view(v, self.augment, rng) for v in views]
        x = preprocess_views(views, self.preprocess)
        return torch.from_numpy(x), torch.tensor(float(self.records[i].target)), i


def class_balanced_weights(records: Sequence[EggRecord]) -> list[float]:
    """Poids d'échantillonnage : classes équilibrées, et défauts équilibrés entre eux parmi les non viables."""
    n_pos = sum(r.target for r in records)
    n_neg = len(records) - n_pos
    by_defect: dict[str | None, int] = {}
    for r in records:
        if r.target:
            by_defect[r.defect] = by_defect.get(r.defect, 0) + 1
    weights = []
    for r in records:
        if r.target:
            weights.append(0.5 / max(1, len(by_defect)) / by_defect[r.defect])
        else:
            weights.append(0.5 / max(1, n_neg))
    if n_pos == 0 or n_neg == 0:
        return [1.0] * len(records)
    return weights
