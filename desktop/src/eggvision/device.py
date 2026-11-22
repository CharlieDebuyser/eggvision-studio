"""Détection de l'accélérateur de calcul (CUDA, MPS, sinon CPU)."""

from __future__ import annotations

import functools
import platform
from typing import Any


@functools.lru_cache(maxsize=1)
def detect_device() -> dict[str, Any]:
    """Retourne ``{"kind": "cuda" | "mps" | "cpu", "name": str}`` sans jamais lever d'exception."""
    try:
        import torch

        if torch.cuda.is_available():
            return {"kind": "cuda", "name": torch.cuda.get_device_name(0)}
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return {"kind": "mps", "name": f"Apple Silicon ({platform.machine()})"}
        return {"kind": "cpu", "name": platform.processor() or platform.machine()}
    except Exception as exc:  # torch absent ou cassé
        return {"kind": "cpu", "name": f"CPU (PyTorch indisponible : {exc})"}


def torch_device(prefer: str | None = None):
    """Objet ``torch.device`` à utiliser ; ``prefer`` force ``"cpu"`` par exemple."""
    import torch

    kind = prefer or detect_device()["kind"]
    return torch.device(kind)
