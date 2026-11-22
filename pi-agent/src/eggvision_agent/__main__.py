"""Lancement de l'agent : ``eggvision-agent`` ou ``python -m eggvision_agent``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import CAMERA_CHOICES, AgentConfig


def build_config(argv: list[str] | None = None) -> AgentConfig:
    cfg = AgentConfig.from_env()
    p = argparse.ArgumentParser(prog="eggvision-agent", description=f"Agent de capture EggVision {__version__} (Raspberry Pi 5)")
    p.add_argument("--host", default=cfg.host, help="Adresse d'écoute (défaut : %(default)s)")
    p.add_argument("--port", type=int, default=cfg.port, help="Port HTTP (défaut : %(default)s)")
    p.add_argument("--storage", type=Path, default=cfg.storage, help="Dossier de stockage des lots, sur le SSD (défaut : %(default)s)")
    p.add_argument("--camera", choices=CAMERA_CHOICES, default=cfg.camera, help="Caméra : auto | picamera2 | simulated (défaut : %(default)s)")
    p.add_argument("--no-mdns", action="store_true", help="Ne pas annoncer l'agent sur le réseau local (mDNS)")
    p.add_argument("--min-free-gb", type=float, default=cfg.min_free_bytes / 1e9, help="Arrêt automatique de la capture sous ce seuil d'espace libre (Go)")
    p.add_argument("--queue-size", type=int, default=cfg.queue_size, help="Capacité de la file d'enregistrement (images)")
    p.add_argument("--writers", type=int, default=cfg.n_writers, help="Nombre de fils d'encodage JPEG")
    p.add_argument("--sim-scale", type=float, default=cfg.sim_scale, help="Caméra simulée : échelle de l'image (1 = 1456 × 1088)")
    p.add_argument("--sim-folder", type=Path, default=cfg.sim_folder, help="Caméra simulée : dossier d'images à relire en boucle")
    p.add_argument("--sim-cycle", type=float, default=cfg.sim_cycle_s, help="Caméra simulée : durée du cycle pré-rendu (s)")
    p.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"))
    args = p.parse_args(argv)
    cfg.host, cfg.port, cfg.storage, cfg.camera = args.host, args.port, args.storage, args.camera
    cfg.mdns = cfg.mdns and not args.no_mdns
    cfg.min_free_bytes = int(args.min_free_gb * 1e9)
    cfg.queue_size, cfg.n_writers = args.queue_size, args.writers
    cfg.sim_scale, cfg.sim_folder, cfg.sim_cycle_s = args.sim_scale, args.sim_folder, args.sim_cycle
    cfg.log_level = args.log_level
    cfg.validate()
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = build_config(argv)
    level = cfg.log_level
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    import uvicorn

    from .api import create_app

    app = create_app(cfg)
    # timeout_graceful_shutdown : les flux d'aperçu ouverts ne bloquent pas l'arrêt (systemd stop).
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=level, timeout_graceful_shutdown=3, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
