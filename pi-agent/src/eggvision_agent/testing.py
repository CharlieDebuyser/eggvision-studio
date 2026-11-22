"""Aides pour les tests : agent simulé dans un vrai serveur uvicorn, en fil d'exécution."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import AgentConfig

# Ports réservés aux essais (évite les collisions avec d'autres services de développement).
TEST_PORTS = range(8820, 8830)


def free_port(candidates: range = TEST_PORTS) -> int:
    for port in list(candidates) + [0]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Aucun port libre")


def make_test_config(storage: str | Path, **overrides: Any) -> AgentConfig:
    """Configuration rapide pour les tests : caméra simulée basse résolution, sans mDNS."""
    cfg = AgentConfig(
        host="127.0.0.1",
        port=0,
        storage=Path(storage),
        camera="simulated",
        mdns=False,
        sim_scale=0.25,
        sim_cycle_s=0.5,
        min_free_bytes=0,
        disk_check_interval=0.2,
        preview_width=200,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@contextmanager
def running_agent(storage: str | Path, ready_timeout: float = 60.0, **overrides: Any) -> Iterator[tuple[str, Any]]:
    """Démarre un agent (serveur uvicorn réel) et retourne ``(url, service)``."""
    import uvicorn

    from .api import create_app
    from .service import AgentService

    cfg = make_test_config(storage, **overrides)
    if not cfg.port:
        cfg.port = free_port()
    service = AgentService(cfg)
    app = create_app(cfg, service)
    server = uvicorn.Server(uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="warning", timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, name="test-agent", daemon=True)
    thread.start()
    deadline = time.time() + ready_timeout
    while not (server.started and service.backend.ready):
        if time.time() > deadline or not thread.is_alive():
            server.should_exit = True
            raise RuntimeError("L'agent de test n'a pas démarré")
        time.sleep(0.05)
    try:
        yield f"http://{cfg.host}:{cfg.port}", service
    finally:
        server.should_exit = True
        thread.join(15)
