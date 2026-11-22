"""Modèle multi-vues : un score de défaut par vue, agrégé en un score par œuf.

Contrat (identique au graphe ONNX exporté) ::

    forward(views [B, V, 3, S, S]) -> (egg_scores [B], view_scores [B, V])   # probabilités

* backbone timm léger (MobileNetV3-Small / Large, EfficientNet-Lite0) partagé par toutes les vues ;
* tête : ``Dropout`` + ``Linear`` → un logit par vue ;
* agrégation dans le graphe : ``max`` des logits des vues (défaut ; équivaut au max des
  probabilités, la sigmoïde étant croissante) ou ``attention`` (pooling appris : moyenne
  des logits pondérée par un softmax sur les vues). B et V restent dynamiques.

Poids pré-entraînés : UNIQUEMENT les fichiers locaux
``<assets>/pretrained/<arch>.safetensors`` (cf. ``scripts/fetch_pretrained.py``) ;
jamais d'accès réseau (``timm.create_model(pretrained=False)`` puis chargement du
dictionnaire d'état).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .. import settings

log = logging.getLogger(__name__)

ARCHITECTURES: dict[str, dict[str, Any]] = {
    "mobilenetv3_small_100": {
        "label": "MobileNetV3-Small",
        "description": "Très léger (≈ 1,5 M paramètres) : le plus rapide en FP32. Se dégrade en INT8 (déployer en FP32).",
        "params_m": 1.5,
        "int8_friendly": False,
    },
    "mobilenetv3_large_100": {
        "label": "MobileNetV3-Large",
        "description": "Plus précis (≈ 4,2 M paramètres), environ 3 × plus lent que Small. Se dégrade en INT8.",
        "params_m": 4.2,
        "int8_friendly": False,
    },
    "efficientnet_lite0": {
        "label": "EfficientNet-Lite0",
        "description": "≈ 3,4 M paramètres, conçu pour la quantification : en INT8, latence proche de Small en FP32.",
        "params_m": 3.4,
        "int8_friendly": True,
    },
}
AGGREGATIONS = ("max", "attention")


def pretrained_dir() -> Path:
    return settings.assets_dir() / "pretrained"


def pretrained_path(arch: str) -> Path:
    return pretrained_dir() / f"{arch}.safetensors"


def pretrained_status(arch: str) -> dict[str, Any]:
    """Présence et description du fichier de poids pré-entraînés local."""
    path = pretrained_path(arch)
    info: dict[str, Any] = {"available": path.is_file(), "file": path.name}
    if path.is_file():
        info["size_bytes"] = path.stat().st_size
        meta = path.with_suffix(".json")
        if meta.is_file():
            try:
                info.update({k: v for k, v in json.loads(meta.read_text(encoding="utf-8")).items() if k in ("source", "sha256", "fetched_at", "n_params")})
            except Exception:  # métadonnées facultatives
                pass
    return info


def load_pretrained(backbone: Any, arch: str) -> dict[str, Any]:
    """Charge les poids locaux dans ``backbone`` (clés et formes compatibles uniquement).

    Retourne ``{"loaded": bool, "warning": str | None, "n_tensors": int, "missing": int}``.
    """
    path = pretrained_path(arch)
    if not path.is_file():
        msg = (
            f"Poids pré-entraînés absents ({path.name}) : initialisation aléatoire. "
            "Exécuter scripts/fetch_pretrained.py sur une machine connectée puis reconstruire l'application."
        )
        log.warning(msg)
        return {"loaded": False, "warning": msg, "n_tensors": 0, "missing": 0}
    from safetensors.torch import load_file

    state = load_file(str(path))
    own = backbone.state_dict()
    usable = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    missing = [k for k in own if k not in usable and not k.endswith("num_batches_tracked")]
    backbone.load_state_dict(usable, strict=False)
    warning = None
    if missing:
        warning = f"{len(missing)} tenseur(s) du backbone non trouvés dans les poids pré-entraînés (ex. {missing[0]})."
        log.warning(warning)
    return {"loaded": True, "warning": warning, "n_tensors": len(usable), "missing": len(missing), "file": path.name}


def build_backbone(arch: str, **kwargs: Any) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Architecture non prise en charge : {arch}")
    import timm

    return timm.create_model(arch, pretrained=False, num_classes=0, **kwargs)


class MultiViewClassifier(nn.Module):
    """Classifieur multi-vues (voir la documentation du module)."""

    def __init__(self, arch: str, aggregation: str = "max", dropout: float = 0.2, attention_dim: int = 128) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"Agrégation inconnue : {aggregation}")
        self.arch = arch
        self.aggregation = aggregation
        self.backbone = build_backbone(arch)
        n_feat = int(getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features)
        self.n_features = n_feat
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, 1))
        self.attention = (
            nn.Sequential(nn.Linear(n_feat, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))
            if aggregation == "attention"
            else None
        )

    def view_features(self, views: torch.Tensor) -> torch.Tensor:
        return self.backbone(views.flatten(0, 1))  # [B*V, F]

    def aggregate(self, feats: torch.Tensor, view_logits: torch.Tensor) -> torch.Tensor:
        """Logit de l'œuf à partir des logits des vues ``[B, V]`` (et de leurs caractéristiques)."""
        if self.attention is None:
            return view_logits.max(dim=1).values
        w = torch.softmax(self.attention(feats).reshape(view_logits.shape), dim=1)
        return (w * view_logits).sum(dim=1)

    def forward_logits(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits ``(œuf [B], vues [B, V])`` — utilisés par la perte d'entraînement."""
        feats = self.view_features(views)
        view_logits = self.head(feats).reshape(views.shape[:2])
        return self.aggregate(feats, view_logits), view_logits

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        egg_logits, view_logits = self.forward_logits(views)
        return torch.sigmoid(egg_logits), torch.sigmoid(view_logits)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)


def build_model(
    arch: str, aggregation: str = "max", pretrained: bool = True, dropout: float = 0.2
) -> tuple[MultiViewClassifier, dict[str, Any]]:
    """Modèle + rapport de chargement des poids pré-entraînés."""
    model = MultiViewClassifier(arch, aggregation=aggregation, dropout=dropout)
    if pretrained:
        info = load_pretrained(model.backbone, arch)
    else:
        info = {"loaded": False, "warning": None, "n_tensors": 0, "missing": 0, "disabled": True}
    return model, info
