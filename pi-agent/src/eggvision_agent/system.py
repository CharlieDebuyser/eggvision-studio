"""Informations système : modèle de Pi, température, disque."""

from __future__ import annotations

import os
import platform
import shutil
import socket
from pathlib import Path
from typing import Any


def pi_model() -> str:
    """Modèle de la carte (``Raspberry Pi 5 Model B Rev 1.0``) ou, à défaut, description de la machine."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            text = Path(path).read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
            if text:
                return text
        except OSError:
            continue
    return f"{platform.system()} {platform.machine()} (hors Raspberry Pi)"


def is_raspberry_pi() -> bool:
    return pi_model().lower().startswith("raspberry pi")


def cpu_temperature_c() -> float | None:
    """Température du processeur (°C), ou None si indisponible."""
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return round(int(raw) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    try:
        import psutil

        temps = getattr(psutil, "sensors_temperatures", lambda: {})() or {}
        for key in ("cpu_thermal", "coretemp", "k10temp"):
            if temps.get(key):
                return round(float(temps[key][0].current), 1)
    except Exception:
        pass
    return None


def disk_info(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    usage = shutil.disk_usage(p)
    return {"path": str(path), "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def cpu_count() -> int:
    return os.cpu_count() or 1


def device_info() -> dict[str, Any]:
    return {"model": pi_model(), "cpu_count": cpu_count(), "temperature_c": cpu_temperature_c()}
