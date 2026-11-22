"""Configuration de l'agent (ligne de commande ou variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import DEFAULT_PORT

CAMERA_CHOICES = ("auto", "picamera2", "simulated")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class AgentConfig:
    """Paramètres de l'agent. Chaque champ a une variable d'environnement ``EGGVISION_AGENT_*``."""

    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    storage: Path = field(default_factory=lambda: Path.home() / "eggvision-data")
    camera: str = "auto"  # auto | picamera2 | simulated
    mdns: bool = True
    # capture
    queue_size: int = 45  # ≈ 1,5 s à 30 images/s (≈ 215 Mo en pleine résolution)
    n_writers: int = 3
    min_free_bytes: int = 2_000_000_000  # arrêt automatique sous 2 Go libres
    disk_check_interval: float = 1.0
    # aperçu
    preview_width: int = 640
    # source simulée
    sim_scale: float = 1.0
    sim_folder: Path | None = None
    sim_cycle_s: float = 1.0
    # caméra réelle
    camera_num: int = 0
    lores_width: int = 640
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> AgentConfig:
        cfg = cls()
        cfg.host = _env("EGGVISION_AGENT_HOST", cfg.host) or cfg.host
        cfg.port = int(_env("EGGVISION_AGENT_PORT", str(cfg.port)) or cfg.port)
        storage = _env("EGGVISION_AGENT_STORAGE")
        if storage:
            cfg.storage = Path(storage)
        cfg.camera = (_env("EGGVISION_AGENT_CAMERA", cfg.camera) or cfg.camera).lower()
        cfg.mdns = (_env("EGGVISION_AGENT_MDNS", "1") or "1").lower() not in ("0", "false", "non", "no")
        cfg.queue_size = int(_env("EGGVISION_AGENT_QUEUE_SIZE", str(cfg.queue_size)) or cfg.queue_size)
        cfg.n_writers = int(_env("EGGVISION_AGENT_WRITERS", str(cfg.n_writers)) or cfg.n_writers)
        min_free_gb = _env("EGGVISION_AGENT_MIN_FREE_GB")
        if min_free_gb:
            cfg.min_free_bytes = int(float(min_free_gb) * 1e9)
        cfg.sim_scale = float(_env("EGGVISION_AGENT_SIM_SCALE", str(cfg.sim_scale)) or cfg.sim_scale)
        folder = _env("EGGVISION_AGENT_SIM_FOLDER")
        if folder:
            cfg.sim_folder = Path(folder)
        cfg.sim_cycle_s = float(_env("EGGVISION_AGENT_SIM_CYCLE", str(cfg.sim_cycle_s)) or cfg.sim_cycle_s)
        return cfg

    def validate(self) -> None:
        if self.camera not in CAMERA_CHOICES:
            raise ValueError(f"Caméra inconnue : {self.camera!r} (choix : {', '.join(CAMERA_CHOICES)})")
        if not (0 <= self.port < 65536):
            raise ValueError(f"Port invalide : {self.port}")

    @property
    def lots_root(self) -> Path:
        return Path(self.storage) / "lots"
