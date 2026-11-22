"""Format des données d'un lot, commun à l'agent Pi et à l'application de bureau.

Un lot est un dossier ::

    <lot_id>/
      lot.json          manifeste + réglages caméra figés
      metadata.jsonl    une ligne JSON par image enregistrée
      frames/000000.jpg ...

``metadata.jsonl`` est écrit au fil de l'eau (une ligne par image, vidée sur
disque régulièrement) : un lot interrompu reste lisible. Les lignes peuvent être
légèrement désordonnées (plusieurs fils d'écriture) : :func:`read_metadata` les trie.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .constants import (
    DEFAULT_EXPOSURE_US,
    DEFAULT_FPS,
    FORMAT_VERSION,
    JPEG_QUALITY,
    LOT_STATUS_DONE,
    LOT_STATUS_INTERRUPTED,
    LOT_STATUS_RUNNING,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    parse_lot_type,
)

LOT_MANIFEST = "lot.json"
LOT_METADATA = "metadata.jsonl"
FRAMES_DIR = "frames"


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def frame_filename(index: int) -> str:
    return f"{index:06d}.jpg"


def frame_relpath(index: int) -> str:
    return f"{FRAMES_DIR}/{frame_filename(index)}"


def new_lot_id(name: str = "") -> str:
    """Identifiant de lot lisible, triable par date et sûr comme nom de dossier."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", _strip_accents(name.lower())).strip("-")[:32]
    suffix = uuid.uuid4().hex[:4]
    return "-".join(p for p in (stamp, slug, suffix) if p)


def _strip_accents(text: str) -> str:
    import unicodedata

    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


class CameraSettings(BaseModel):
    """Réglages manuels de la caméra. Tous les automatismes sont coupés en capture."""

    exposure_us: int = Field(DEFAULT_EXPOSURE_US, ge=20, le=1_000_000, description="Temps d'exposition (µs)")
    analogue_gain: float = Field(1.0, ge=1.0, le=16.0, description="Gain analogique")
    red_gain: float = Field(2.0, ge=0.1, le=8.0, description="Gain de couleur rouge")
    blue_gain: float = Field(1.8, ge=0.1, le=8.0, description="Gain de couleur bleu")
    fps: float = Field(DEFAULT_FPS, gt=0, le=60.0, description="Cadence (images/s)")
    roi: tuple[int, int, int, int] | None = Field(
        None, description="Zone de recadrage capteur (x, y, largeur, hauteur) ; None = plein capteur"
    )

    @field_validator("roi")
    @classmethod
    def _check_roi(cls, v: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if v is None:
            return v
        x, y, w, h = v
        if x < 0 or y < 0 or w < 16 or h < 16:
            raise ValueError("ROI invalide")
        return v

    def output_size(self, sensor_width: int = SENSOR_WIDTH, sensor_height: int = SENSOR_HEIGHT) -> tuple[int, int]:
        """(largeur, hauteur) des images produites avec ces réglages."""
        if self.roi is None:
            return sensor_width, sensor_height
        x, y, w, h = self.roi
        return min(w, sensor_width - x), min(h, sensor_height - y)


class LotStats(BaseModel):
    frames_written: int = 0
    frames_dropped: int = 0
    duration_s: float = 0.0
    measured_fps: float = 0.0
    bytes_written: int = 0


class LotManifest(BaseModel):
    """Contenu de ``lot.json``."""

    format_version: int = FORMAT_VERSION
    lot_id: str
    name: str
    type: str = Field(description="viable | declasse | declasse:<défaut> | production")
    farm: str = ""
    operator: str = ""
    comment: str = ""
    created_at: str = Field(default_factory=utcnow_iso)
    started_at: str | None = None
    ended_at: str | None = None
    status: str = LOT_STATUS_RUNNING
    source: dict[str, Any] = Field(default_factory=dict, description="{kind: pi|usb|simulated|import, ...}")
    camera: CameraSettings = Field(default_factory=CameraSettings)
    width: int = SENSOR_WIDTH
    height: int = SENSOR_HEIGHT
    jpeg_quality: int = JPEG_QUALITY
    stats: LotStats = Field(default_factory=LotStats)

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        parse_lot_type(v)
        return v

    def save(self, lot_dir: str | os.PathLike[str]) -> None:
        atomic_write_text(Path(lot_dir) / LOT_MANIFEST, self.model_dump_json(indent=2))

    @classmethod
    def load(cls, lot_dir: str | os.PathLike[str]) -> LotManifest:
        return cls.model_validate_json((Path(lot_dir) / LOT_MANIFEST).read_text(encoding="utf-8"))


class FrameMeta(BaseModel):
    """Une ligne de ``metadata.jsonl``."""

    index: int = Field(description="Rang de l'image enregistrée (0, 1, 2…) = nom du fichier")
    seq: int = Field(description="Numéro de séquence capteur, images perdues comprises")
    file: str
    sensor_ts_ns: int = Field(description="Horodatage capteur (ns)")
    system_ts: float = Field(description="Heure système (secondes epoch)")
    encoder_count: int | None = Field(None, description="Réservé au futur codeur ; null pour l'instant")
    size: int | None = None
    sha256: str | None = None


def atomic_write_text(path: Path, text: str) -> None:
    """Écrit un fichier texte de façon atomique (fichier temporaire + remplacement)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def encode_jpeg(image_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Échec de l'encodage JPEG")
    return buf.tobytes()


def read_image(path: str | os.PathLike[str]) -> np.ndarray:
    """Lit une image en BGR uint8 (compatible chemins non ASCII sous Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image illisible : {path}")
    return img


def write_image(path: str | os.PathLike[str], image_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> int:
    """Écrit une image JPEG ; retourne sa taille en octets."""
    data = encode_jpeg(image_bgr, quality)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


class LotWriter:
    """Écriture d'un lot au fil de l'eau (thread-safe).

    Usage ::

        writer = LotWriter(lot_dir, manifest)
        writer.write_frame(image_bgr, sensor_ts_ns, seq)   # autant de fois que nécessaire
        writer.close()                                      # statut « termine »

    Si le processus meurt avant ``close()``, ``lot.json`` reste au statut
    ``en_cours`` : :func:`mark_interrupted_lots` le passera à ``interrompu``.
    """

    FLUSH_EVERY = 15

    def __init__(self, lot_dir: str | os.PathLike[str], manifest: LotManifest) -> None:
        self.lot_dir = Path(lot_dir)
        self.manifest = manifest
        (self.lot_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_index = 0
        self._bytes = 0
        self._dropped = 0
        self._first_ts_ns: int | None = None
        self._last_ts_ns: int | None = None
        self._t0 = time.time()
        self._closed = False
        if self.manifest.started_at is None:
            self.manifest.started_at = utcnow_iso()
        self.manifest.status = LOT_STATUS_RUNNING
        self.manifest.save(self.lot_dir)
        self._meta = open(self.lot_dir / LOT_METADATA, "a", encoding="utf-8")
        self._since_flush = 0

    # -- écriture ---------------------------------------------------------
    def reserve_index(self) -> int:
        """Réserve le prochain rang d'image (pour des fils d'écriture multiples)."""
        with self._lock:
            idx = self._next_index
            self._next_index += 1
            return idx

    def write_frame(
        self,
        image_bgr: np.ndarray,
        sensor_ts_ns: int,
        seq: int | None = None,
        system_ts: float | None = None,
        index: int | None = None,
    ) -> FrameMeta:
        data = encode_jpeg(image_bgr, self.manifest.jpeg_quality)
        return self.write_encoded(data, sensor_ts_ns, seq=seq, system_ts=system_ts, index=index)

    def write_encoded(
        self,
        jpeg: bytes,
        sensor_ts_ns: int,
        seq: int | None = None,
        system_ts: float | None = None,
        index: int | None = None,
    ) -> FrameMeta:
        if index is None:
            index = self.reserve_index()
        rel = frame_relpath(index)
        with open(self.lot_dir / rel, "wb") as f:
            f.write(jpeg)
        meta = FrameMeta(
            index=index,
            seq=index if seq is None else seq,
            file=rel,
            sensor_ts_ns=int(sensor_ts_ns),
            system_ts=time.time() if system_ts is None else system_ts,
            encoder_count=None,
            size=len(jpeg),
            sha256=hashlib.sha256(jpeg).hexdigest(),
        )
        line = meta.model_dump_json() + "\n"
        with self._lock:
            self._meta.write(line)
            self._bytes += len(jpeg)
            if self._first_ts_ns is None or sensor_ts_ns < self._first_ts_ns:
                self._first_ts_ns = int(sensor_ts_ns)
            if self._last_ts_ns is None or sensor_ts_ns > self._last_ts_ns:
                self._last_ts_ns = int(sensor_ts_ns)
            self._since_flush += 1
            if self._since_flush >= self.FLUSH_EVERY:
                self._meta.flush()
                self._since_flush = 0
        return meta

    def count_dropped(self, n: int = 1) -> None:
        with self._lock:
            self._dropped += n

    # -- état -------------------------------------------------------------
    def stats(self) -> LotStats:
        with self._lock:
            written = self._next_index
            duration = 0.0
            if self._first_ts_ns is not None and self._last_ts_ns is not None:
                duration = (self._last_ts_ns - self._first_ts_ns) / 1e9
            total = written + self._dropped
            fps = (total - 1) / duration if duration > 0 and total > 1 else 0.0
            return LotStats(
                frames_written=written,
                frames_dropped=self._dropped,
                duration_s=round(duration, 3),
                measured_fps=round(fps, 3),
                bytes_written=self._bytes,
            )

    def close(self, status: str = LOT_STATUS_DONE) -> LotManifest:
        if self._closed:
            return self.manifest
        self._closed = True
        with self._lock:
            self._meta.flush()
            try:
                os.fsync(self._meta.fileno())
            except OSError:
                pass
            self._meta.close()
        self.manifest.stats = self.stats()
        self.manifest.status = status
        self.manifest.ended_at = utcnow_iso()
        self.manifest.save(self.lot_dir)
        return self.manifest

    def __enter__(self) -> LotWriter:
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        self.close(LOT_STATUS_DONE if exc_type is None else LOT_STATUS_INTERRUPTED)


def read_metadata(lot_dir: str | os.PathLike[str]) -> list[FrameMeta]:
    """Lit ``metadata.jsonl`` (trié par rang ; une dernière ligne tronquée est ignorée)."""
    path = Path(lot_dir) / LOT_METADATA
    if not path.exists():
        return []
    out: list[FrameMeta] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(FrameMeta.model_validate_json(line))
            except Exception:
                continue  # ligne tronquée par une coupure
    out.sort(key=lambda m: m.index)
    return out


def iter_lot_dirs(lots_root: str | os.PathLike[str]) -> Iterator[Path]:
    root = Path(lots_root)
    if not root.is_dir():
        return
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / LOT_MANIFEST).exists():
            yield p


def mark_interrupted_lots(lots_root: str | os.PathLike[str]) -> list[str]:
    """Passe à ``interrompu`` les lots restés ``en_cours`` (après un arrêt brutal).

    Recalcule leurs statistiques à partir de ``metadata.jsonl``. Retourne les
    identifiants des lots modifiés.
    """
    changed: list[str] = []
    for lot_dir in iter_lot_dirs(lots_root):
        try:
            manifest = LotManifest.load(lot_dir)
        except Exception:
            continue
        if manifest.status != LOT_STATUS_RUNNING:
            continue
        metas = read_metadata(lot_dir)
        metas = [m for m in metas if (lot_dir / m.file).exists()]
        stats = LotStats(frames_written=len(metas))
        if metas:
            duration = (metas[-1].sensor_ts_ns - metas[0].sensor_ts_ns) / 1e9
            total = metas[-1].seq - metas[0].seq + 1
            stats.frames_dropped = max(0, total - len(metas))
            stats.duration_s = round(duration, 3)
            stats.measured_fps = round((total - 1) / duration, 3) if duration > 0 and total > 1 else 0.0
            stats.bytes_written = sum(m.size or 0 for m in metas)
        manifest.stats = stats
        manifest.status = LOT_STATUS_INTERRUPTED
        manifest.ended_at = manifest.ended_at or utcnow_iso()
        manifest.save(lot_dir)
        changed.append(manifest.lot_id)
    return changed


def lot_file_list(lot_dir: str | os.PathLike[str], with_checksums: bool = True) -> list[dict[str, Any]]:
    """Liste des fichiers d'un lot pour la synchronisation : ``[{path, size, sha256}]``.

    Les sommes des images viennent de ``metadata.jsonl`` quand elles y figurent
    (calculées à la capture), sinon elles sont calculées ici.
    """
    lot_dir = Path(lot_dir)
    known = {m.file: m for m in read_metadata(lot_dir)}
    files: list[dict[str, Any]] = []
    for rel in (LOT_MANIFEST, LOT_METADATA):
        p = lot_dir / rel
        if p.exists():
            files.append(
                {"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p) if with_checksums else None}
            )
    frames_dir = lot_dir / FRAMES_DIR
    if frames_dir.is_dir():
        for p in sorted(frames_dir.iterdir()):
            if not p.is_file() or p.name.endswith(".tmp") or p.name.endswith(".part"):
                continue
            rel = f"{FRAMES_DIR}/{p.name}"
            size = p.stat().st_size
            meta = known.get(rel)
            digest = None
            if with_checksums:
                digest = meta.sha256 if (meta and meta.sha256 and meta.size == size) else sha256_file(p)
            files.append({"path": rel, "size": size, "sha256": digest})
    return files


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
