"""Annonce mDNS / DNS-SD de l'agent (``_eggvision._tcp.local.``)."""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any

from . import API_VERSION, SERVICE_TYPE, __version__, system

log = logging.getLogger(__name__)


def local_ipv4_addresses() -> list[str]:
    """Adresses IPv4 non locales de la machine (Wi-Fi, Ethernet, point d'accès)."""
    out: list[str] = []
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith("127.") and not a.address.startswith("169.254."):
                    out.append(a.address)
    except Exception:
        log.exception("Liste des interfaces réseau")
    return sorted(set(out))


class MdnsAnnouncer:
    """Publie le service sur le réseau local ; l'application de bureau le découvre sans saisie d'adresse."""

    def __init__(self, port: int, host: str = "0.0.0.0") -> None:
        self.port = port
        self.host = host
        self._zc: Any = None
        self._info: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._register, name="mdns", daemon=True)
        self._thread.start()

    def _register(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf

            name = system.hostname()
            if self.host not in ("0.0.0.0", "", "::"):
                addresses = [self.host]
            else:
                addresses = local_ipv4_addresses() or ["127.0.0.1"]
            self._info = ServiceInfo(
                SERVICE_TYPE,
                f"EggVision {name}.{SERVICE_TYPE}",
                port=self.port,
                parsed_addresses=addresses,
                server=f"{name}.local.",
                properties={"version": __version__, "api": str(API_VERSION), "hostname": name, "path": "/api/status"},
            )
            self._zc = Zeroconf()
            self._zc.register_service(self._info, allow_name_change=True)
            log.info("Annonce mDNS : %s sur %s:%d", SERVICE_TYPE, ", ".join(addresses), self.port)
        except Exception:
            log.exception("Annonce mDNS impossible (l'agent reste joignable par son adresse IP)")

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(5.0)
        if self._zc is not None:
            try:
                if self._info is not None:
                    self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                log.exception("Arrêt de l'annonce mDNS")
            self._zc = None
