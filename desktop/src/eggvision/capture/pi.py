"""Source « agent Raspberry Pi » : client HTTP de l'agent de capture (``eggvision_agent``)."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from eggvision_shared.formats import CameraSettings, LotManifest

from .base import CameraSource, SourceError

log = logging.getLogger(__name__)

DEFAULT_AGENT_PORT = 8770


def normalize_agent_url(address: str) -> str:
    """``192.168.4.1`` → ``http://192.168.4.1:8770`` ; ``hote:9000`` → ``http://hote:9000``."""
    address = (address or "").strip().rstrip("/")
    if not address:
        raise SourceError(400, "Adresse de l'agent vide.")
    if not re.match(r"^https?://", address):
        address = "http://" + address
    scheme, rest = address.split("://", 1)
    host = rest.split("/", 1)[0]
    if host.startswith("["):  # IPv6 littérale
        has_port = "]:" in host
    else:
        has_port = ":" in host
    if not has_port:
        host = f"{host}:{DEFAULT_AGENT_PORT}"
    if not re.match(r"^[A-Za-z0-9.\-_\[\]:%]+$", host):
        raise SourceError(400, f"Adresse invalide : {address}")
    return f"{scheme}://{host}"


class PiAgentSource(CameraSource):
    kind = "pi"

    def __init__(self, url: str, *, timeout: float = 5.0, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__()
        self.url = normalize_agent_url(url)
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.url, timeout=timeout, transport=transport)
        self._status: dict[str, Any] | None = None
        self._status_at = 0.0
        self._lock = threading.Lock()
        self.version: str | None = None
        self.hostname: str | None = None

    @property
    def label(self) -> str:
        return f"Agent Pi {self.hostname or ''} ({self.url})".replace("  ", " ")

    def describe(self) -> dict[str, Any]:
        return {"kind": "pi", "label": self.label, "agent_url": self.url, "agent_version": self.version}

    # -- HTTP -----------------------------------------------------------------
    def request(self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any) -> httpx.Response:
        try:
            r = self._client.request(method, path, timeout=timeout or self.timeout, **kwargs)
        except httpx.TimeoutException as exc:
            raise SourceError(504, f"L'agent ne répond pas ({self.url}) : délai dépassé.") from exc
        except httpx.HTTPError as exc:
            raise SourceError(503, f"Agent injoignable ({self.url}) : {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = None
            raise SourceError(r.status_code, str(detail or f"Erreur {r.status_code} de l'agent"))
        return r

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self.request(method, path, **kwargs).json()

    # -- cycle de vie ------------------------------------------------------------
    def open(self) -> None:
        st = self.raw_status(max_age=0)
        if st.get("agent") != "eggvision-agent":
            raise SourceError(502, f"{self.url} ne répond pas comme un agent EggVision.")

    def close(self) -> None:
        self._client.close()

    # -- état ----------------------------------------------------------------------
    def raw_status(self, max_age: float = 0.5, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            if self._status is not None and time.monotonic() - self._status_at < max_age:
                return self._status
        st = self._json("GET", "/api/status", timeout=timeout)
        with self._lock:
            self._status, self._status_at = st, time.monotonic()
        self.version = st.get("version")
        self.hostname = st.get("hostname")
        return st

    def status(self, max_age: float = 0.5) -> dict[str, Any]:
        st = self.raw_status(max_age)
        cam = st.get("camera") or {}
        return {
            "kind": "pi",
            "label": self.label,
            "url": self.url,
            "connected": True,
            "ready": bool(cam.get("ready")),
            "error": cam.get("error"),
            "version": st.get("version"),
            "hostname": st.get("hostname"),
            "camera": cam,
            "actual_fps": cam.get("actual_fps", 0.0),
            "device": st.get("device"),
            "storage": st.get("storage"),
            "agent_time": st.get("time"),
            "clock_offset_s": round(float(st.get("time", time.time())) - time.time(), 3),
            "capturing": bool(st.get("capturing")),
            "capture": st.get("capture"),
            "last_capture": st.get("last_capture"),
        }

    def output_size(self) -> tuple[int, int]:
        cam = self.raw_status().get("camera") or {}
        return int(cam.get("output_width") or 0), int(cam.get("output_height") or 0)

    # -- réglages -----------------------------------------------------------------
    def get_settings(self) -> CameraSettings:
        return CameraSettings.model_validate(self._json("GET", "/api/camera/settings"))

    def set_settings(self, changes: dict[str, Any]) -> CameraSettings:
        out = CameraSettings.model_validate(self._json("PUT", "/api/camera/settings", json=changes))
        self._status = None
        return out

    def calibrate_awb(self, duration_s: float = 2.0) -> dict[str, Any]:
        return self._json("POST", "/api/camera/awb", json={"duration_s": duration_s}, timeout=duration_s + 15)

    # -- aperçu --------------------------------------------------------------------
    def preview_jpeg(self, width: int | None = None, quality: int = 80) -> bytes:
        params: dict[str, Any] = {"quality": quality}
        if width:
            params["width"] = width
        return self.request("GET", "/api/preview.jpg", params=params).content

    def mjpeg_url(self) -> str:
        return f"{self.url}/api/preview.mjpeg"

    # -- lots -------------------------------------------------------------------------
    def start_lot(self, manifest: LotManifest, lot_dir: Path | None, min_free_bytes: int) -> dict[str, Any]:
        body = {
            "lot_id": manifest.lot_id,
            "name": manifest.name,
            "type": manifest.type,
            "farm": manifest.farm,
            "operator": manifest.operator,
            "comment": manifest.comment,
            "created_at": manifest.created_at,
            "camera": manifest.camera.model_dump() if manifest.camera else None,
            "min_free_bytes": int(min_free_bytes),
            "source": {k: v for k, v in manifest.source.items() if k != "kind"},
        }
        out = self._json("POST", "/api/lots", json=body, timeout=30)
        self._status = None
        return out

    def stop_lot(self) -> dict[str, Any]:
        out = self._json("POST", "/api/lots/stop", timeout=120)
        self._status = None
        return out

    def capture_stats(self) -> dict[str, Any] | None:
        return self._json("GET", "/api/capture")

    @property
    def capturing(self) -> bool:
        try:
            return bool(self.raw_status(max_age=0.5).get("capturing"))
        except SourceError:
            return False

    def list_lots(self, limit: int = 1000, offset: int = 0) -> dict[str, Any]:
        return self._json("GET", "/api/lots", params={"limit": limit, "offset": offset}, timeout=30)

    def lot_info(self, lot_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/lots/{lot_id}")

    def delete_lot(self, lot_id: str) -> dict[str, Any]:
        return self._json("DELETE", f"/api/lots/{lot_id}", timeout=120)
