"""Lancement d'EggVision Studio : serveur local + fenêtre dédiée (pywebview) ou navigateur."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

# À fixer avant tout import de PyTorch / timm : repli CPU des opérations absentes de MPS, et aucun accès réseau.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from . import __version__, settings  # noqa: E402

log = logging.getLogger("eggvision")


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Aucun port disponible")


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eggvision-studio", description=f"EggVision Studio {__version__}")
    parser.add_argument("--project", help="Dossier d'un projet à ouvrir au démarrage")
    parser.add_argument("--port", type=int, default=settings.DEFAULT_PORT)
    parser.add_argument("--no-window", action="store_true", help="Ne pas ouvrir de fenêtre (serveur seul)")
    parser.add_argument("--browser", action="store_true", help="Ouvrir dans le navigateur par défaut plutôt que dans une fenêtre dédiée")
    args = parser.parse_args(argv)

    import uvicorn

    from . import state
    from .app import create_app

    app = create_app()
    if args.project:
        project = state.projects.open(args.project)
        state.jobs.mark_stale(project)

    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=3)
    server = uvicorn.Server(config)

    if args.no_window:
        log.info("Serveur sur %s", url)
        server.run()
        return 0

    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()
    if not _wait_ready(port):
        log.error("Le serveur local n'a pas démarré.")
        return 1

    if not args.browser:
        try:
            import webview

            webview.create_window("EggVision Studio", url, width=1440, height=900, min_size=(1100, 700))
            webview.start()
            server.should_exit = True
            thread.join(timeout=10)
            return 0
        except Exception as exc:  # pywebview indisponible : repli sur le navigateur
            log.warning("Fenêtre dédiée indisponible (%s) : ouverture dans le navigateur.", exc)

    webbrowser.open(url)
    print(f"EggVision Studio est ouvert dans votre navigateur : {url}\nCtrl+C pour quitter.")
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
