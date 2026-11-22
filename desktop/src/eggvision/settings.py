"""Réglages de l'application (indépendants du projet ouvert)."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

APP_NAME = "EggVision Studio"
DEFAULT_PORT = 8765
_lock = threading.Lock()


def data_dir() -> Path:
    """Dossier de données de l'application (journaux, projets récents). Surchargé par EGGVISION_DATA_DIR."""
    override = os.environ.get("EGGVISION_DATA_DIR")
    path = Path(override) if override else Path(user_data_dir("EggVisionStudio", appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file() -> Path:
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "eggvision.log"


def resource_dir() -> Path:
    """Dossier des ressources embarquées (frontend compilé, poids pré-entraînés), y compris sous PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "eggvision"
    return Path(__file__).resolve().parent


def web_dir() -> Path:
    return resource_dir() / "web"


def assets_dir() -> Path:
    return resource_dir() / "assets"


def _state_path() -> Path:
    return data_dir() / "state.json"


def load_state() -> dict[str, Any]:
    with _lock:
        try:
            return json.loads(_state_path().read_text(encoding="utf-8"))
        except Exception:
            return {}


def save_state(state: dict[str, Any]) -> None:
    with _lock:
        tmp = _state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, _state_path())


def update_state(**changes: Any) -> dict[str, Any]:
    state = load_state()
    state.update(changes)
    save_state(state)
    return state
