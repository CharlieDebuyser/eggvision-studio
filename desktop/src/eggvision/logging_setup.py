"""Journalisation : fichier tournant + console."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from . import settings

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s — %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    file_handler = RotatingFileHandler(settings.log_file(), maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    for noisy in ("uvicorn.access", "httpx", "httpcore", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
