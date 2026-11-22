"""API : écran Diagnostic (versions, accélérateur, agent, journaux)."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

from fastapi import APIRouter

from .. import __version__, settings, state
from ..device import detect_device

router = APIRouter(prefix="/api", tags=["diagnostic"])

PACKAGES = ("torch", "torchvision", "timm", "onnx", "onnxruntime", "opencv-python-headless", "numpy", "fastapi", "sqlalchemy", "pywebview")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/diagnostics")
def diagnostics() -> dict[str, Any]:
    project = state.projects.current
    out: dict[str, Any] = {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _version(name) for name in PACKAGES},
        "device": detect_device(),
        "agent": None,
        "project": {"path": str(project.root), "name": project.name} if project else None,
        "log_file": str(settings.log_file()),
        "data_dir": str(settings.data_dir()),
    }
    for key, provider in state.diagnostic_providers.items():
        try:
            out[key] = provider()  # type: ignore[operator]
        except Exception as exc:  # un fournisseur défaillant ne casse pas l'écran
            out[key] = {"connected": False, "detail": f"Erreur : {exc}"}
    return out


@router.get("/diagnostics/log")
def diagnostics_log(lines: int = 300) -> dict[str, str]:
    path = settings.log_file()
    if not path.exists():
        return {"text": ""}
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400_000))
        text = f.read().decode("utf-8", errors="replace")
    return {"text": "\n".join(text.splitlines()[-max(1, min(lines, 5000)) :])}
