"""Prétraitement des vues d'œufs — SOURCE UNIQUE.

Ce fichier est utilisé tel quel par :
  * l'entraînement et l'évaluation (application de bureau, PyTorch) ;
  * la simulation (application de bureau, ONNX Runtime) ;
  * le Raspberry Pi (il est copié sans modification dans le paquet de déploiement).

Il ne dépend que de NumPy et d'OpenCV et ne doit importer aucun autre module du
projet : toute modification ici change le comportement des trois environnements
à la fois, ce qui est précisément le but (aucun écart possible).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

PREPROCESS_VERSION = 1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    """Paramètres du prétraitement, sérialisés dans ``inference_config.json``."""

    input_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    crop_margin: float = 1.15
    version: int = field(default=PREPROCESS_VERSION)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PreprocessConfig:
        if not d:
            return cls()
        version = int(d.get("version", PREPROCESS_VERSION))
        if version != PREPROCESS_VERSION:
            raise ValueError(
                f"Version de prétraitement incompatible : {version} (attendu {PREPROCESS_VERSION})"
            )
        return cls(
            input_size=int(d.get("input_size", 224)),
            mean=tuple(float(x) for x in d.get("mean", IMAGENET_MEAN)),  # type: ignore[arg-type]
            std=tuple(float(x) for x in d.get("std", IMAGENET_STD)),  # type: ignore[arg-type]
            crop_margin=float(d.get("crop_margin", 1.15)),
            version=version,
        )


DEFAULT_CONFIG = PreprocessConfig()


# ---------------------------------------------------------------------------
# Recadrage d'un œuf dans une image complète
# ---------------------------------------------------------------------------
def crop_box(cx: float, cy: float, major: float, minor: float, margin: float = 1.15) -> tuple[float, float, float]:
    """Carré de recadrage (x0, y0, côté) centré sur l'œuf.

    ``major`` et ``minor`` sont les longueurs complètes (en pixels) des axes de
    l'ellipse ajustée. Le côté du carré vaut ``margin`` × le plus grand axe.
    """
    side = float(max(major, minor)) * float(margin)
    side = max(side, 8.0)
    return float(cx) - side / 2.0, float(cy) - side / 2.0, side


def extract_crop(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    out_size: int = 224,
    margin: float = 1.15,
) -> np.ndarray:
    """Extrait la vue carrée d'un œuf (BGR uint8, ``out_size`` × ``out_size``).

    Le recadrage n'est pas tourné : l'orientation de l'œuf dans l'image est
    conservée. Les zones hors image sont complétées par réplication du bord.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("extract_crop attend une image BGR à 3 canaux")
    x0, y0, side = crop_box(cx, cy, major, minor, margin)
    scale = out_size / side
    # Transformation affine : (x - x0) * scale -> coordonnée de sortie (centre de pixel aligné).
    m = np.array(
        [[scale, 0.0, (0.5 - x0) * scale - 0.5], [0.0, scale, (0.5 - y0) * scale - 0.5]],
        dtype=np.float64,
    )
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.warpAffine(
        frame_bgr, m, (out_size, out_size), flags=interp, borderMode=cv2.BORDER_REPLICATE
    )


# ---------------------------------------------------------------------------
# Vue (BGR uint8) -> tenseur du modèle
# ---------------------------------------------------------------------------
def preprocess_view(view_bgr: np.ndarray, config: PreprocessConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Convertit une vue BGR uint8 en tenseur float32 ``(3, S, S)`` normalisé (RGB)."""
    if view_bgr.dtype != np.uint8:
        raise ValueError("preprocess_view attend une image uint8")
    if view_bgr.ndim == 2:
        view_bgr = cv2.cvtColor(view_bgr, cv2.COLOR_GRAY2BGR)
    size = config.input_size
    h, w = view_bgr.shape[:2]
    if (h, w) != (size, size):
        interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
        view_bgr = cv2.resize(view_bgr, (size, size), interpolation=interp)
    rgb = view_bgr[:, :, ::-1].astype(np.float32) / np.float32(255.0)
    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def preprocess_views(views_bgr: Sequence[np.ndarray], config: PreprocessConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Empile les vues d'un œuf en un tenseur float32 ``(V, 3, S, S)``."""
    if len(views_bgr) == 0:
        return np.zeros((0, 3, config.input_size, config.input_size), dtype=np.float32)
    return np.stack([preprocess_view(v, config) for v in views_bgr], axis=0)


def decode_image(data: bytes) -> np.ndarray:
    """Décode une image encodée (JPEG/PNG) en BGR uint8."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image illisible")
    return img


def load_view(path: str) -> np.ndarray:
    """Lit une vue sur disque (BGR uint8). Gère les chemins non ASCII (Windows)."""
    with open(path, "rb") as f:
        return decode_image(f.read())


def select_view_indices(n_available: int, n_views: int) -> list[int]:
    """Indices régulièrement répartis pour retenir ``n_views`` parmi ``n_available``.

    Utilisé partout où l'on sous-échantillonne les détections d'une piste pour
    couvrir une rotation complète. Retourne tous les indices s'il y en a moins
    que demandé.
    """
    if n_available <= 0 or n_views <= 0:
        return []
    if n_available <= n_views:
        return list(range(n_available))
    pos = np.linspace(0, n_available - 1, n_views)
    return [int(round(p)) for p in pos]


def sharpness(gray_or_bgr: np.ndarray) -> float:
    """Indice de netteté : variance du Laplacien (plus grand = plus net)."""
    img = gray_or_bgr
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def iter_chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
