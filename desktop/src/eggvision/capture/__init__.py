"""Module 1 — Prise de vue : sources de caméra, lots, synchronisation avec l'agent Pi.

Voir ``docs/api/capture.md`` pour l'API HTTP et ``docs/pi-agent.md`` pour l'agent.
"""

from .base import CameraSource, SourceError
from .pi import PiAgentSource, normalize_agent_url
from .simulated import SimulatedSource
from .sync import SyncError, sync_lot
from .usb import UsbCameraSource

__all__ = [
    "CameraSource",
    "PiAgentSource",
    "SimulatedSource",
    "SourceError",
    "SyncError",
    "UsbCameraSource",
    "normalize_agent_url",
    "sync_lot",
]
