"""Découverte des agents Pi sur le réseau local (mDNS / DNS-SD, ``_eggvision._tcp.local.``)."""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

SERVICE_TYPE = "_eggvision._tcp.local."


def discover_agents(timeout: float = 2.0) -> list[dict[str, Any]]:
    """Agents annoncés : ``[{name, hostname, addresses, port, version, url}]`` (vide si rien trouvé).

    Écoute pendant ``timeout`` secondes. Ne lève pas d'exception : en cas
    d'échec du multicast (pare-feu, VPN…), retourne une liste vide ; l'adresse
    peut toujours être saisie à la main.
    """
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except ImportError:  # pragma: no cover
        return []

    found: set[str] = set()

    class _Listener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            found.add(name)

        def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            found.add(name)

        def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            found.discard(name)

    out: list[dict[str, Any]] = []
    zc = None
    try:
        zc = Zeroconf()
        browser = ServiceBrowser(zc, SERVICE_TYPE, _Listener())
        time.sleep(max(0.2, timeout))
        for name in sorted(found):
            info = zc.get_service_info(SERVICE_TYPE, name, timeout=1500)
            if info is None:
                continue
            props = {
                (k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else v)
                for k, v in (info.properties or {}).items()
            }
            addresses = [a for a in info.parsed_addresses() if ":" not in a] or info.parsed_addresses()
            if not addresses:
                continue
            host = addresses[0]
            out.append(
                {
                    "name": name.replace("." + SERVICE_TYPE, ""),
                    "hostname": props.get("hostname") or (info.server or "").rstrip("."),
                    "addresses": addresses,
                    "port": info.port,
                    "version": props.get("version"),
                    "api": props.get("api"),
                    "url": f"http://{host}:{info.port}",
                }
            )
        browser.cancel()
    except Exception:
        log.exception("Découverte mDNS impossible")
    finally:
        if zc is not None:
            try:
                zc.close()
            except Exception:
                pass
    return out
