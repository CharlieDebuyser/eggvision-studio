"""Moteur d'enregistrement d'un lot, commun à l'agent Pi et aux sources locales du PC.

Principe (producteur / consommateur) ::

    recorder = CaptureRecorder(lot_dir, manifest, queue_size=45, n_writers=3, min_free_bytes=2e9)
    recorder.start()
    # boucle d'acquisition (fil de la caméra) :
    recorder.submit(frame_bgr, sensor_ts_ns)      # NE BLOQUE JAMAIS ; False = image perdue
    ...
    manifest = recorder.stop()                     # vide la file, ferme le lot (« termine »)

* ``submit`` dépose l'image dans une file bornée. Si la file est pleine (disque
  ou processeur trop lents), l'image est abandonnée et comptée comme perdue :
  la capture n'attend jamais l'écriture.
* Plusieurs fils d'écriture encodent en JPEG (``cv2.imencode`` relâche le GIL)
  et écrivent au fil de l'eau via :class:`~eggvision_shared.formats.LotWriter`.
* Le numéro de séquence ``seq`` compte toutes les images vues par le capteur,
  perdues comprises : un trou dans ``seq`` = une image perdue.
* L'espace disque est surveillé ; sous le seuil, la capture s'arrête d'elle-même
  (lot clos proprement, statut ``termine``, ``stop_reason = "disk_low"``).
* L'image passée à ``submit`` ne doit plus être modifiée par l'appelant (elle est
  encodée plus tard, par un autre fil) : fournir une nouvelle image à chaque appel.

Ce module contient aussi :class:`SyntheticFrameSource` (images synthétiques
pré-rendues ou relecture d'un dossier, avec application des réglages caméra) et
:class:`PacedLoop` (boucle à cadence fixe imitant un capteur), utilisés par les
sources simulées de l'agent et du PC.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .constants import LOT_STATUS_DONE, LOT_STATUS_INTERRUPTED, LOT_STATUS_RUNNING
from .formats import CameraSettings, LotManifest, LotStats, LotWriter

log = logging.getLogger(__name__)

STOP_USER = "user"
STOP_DISK_LOW = "disk_low"
STOP_WRITE_ERROR = "write_error"
STOP_SHUTDOWN = "shutdown"

_SENTINEL = object()


def disk_free_bytes(path: str | os.PathLike[str]) -> int:
    """Espace libre (octets) du système de fichiers contenant ``path``."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int(shutil.disk_usage(p).free)


@dataclass
class CapturedFrame:
    """Une image livrée par une caméra."""

    image: np.ndarray  # BGR uint8, pleine résolution (après ROI)
    sensor_ts_ns: int  # horodatage capteur (ns, horloge monotone)
    system_ts: float  # heure système (secondes epoch)
    seq: int  # numéro de séquence capteur (images perdues comprises)
    preview: np.ndarray | None = None  # image basse résolution (flux « lores »), si disponible
    metadata: dict[str, Any] = field(default_factory=dict)


class CaptureRecorder:
    """Enregistre un lot à partir d'un flux d'images, sans jamais bloquer le producteur.

    Paramètres :
      * ``queue_size`` : capacité de la file (images) ; au-delà, les images sont perdues ;
      * ``n_writers`` : nombre de fils d'encodage / écriture ;
      * ``min_free_bytes`` : seuil d'arrêt automatique sur l'espace disque (0 = désactivé) ;
      * ``disk_check_interval`` : période de surveillance du disque (s) ;
      * ``disk_free_fn`` : fonction ``() -> octets libres`` (remplaçable dans les tests) ;
      * ``on_finished`` : rappel ``(manifest, stop_reason)`` appelé une fois le lot clos,
        y compris après un arrêt automatique ;
      * ``write_delay_s`` : ralentissement artificiel de l'écriture (tests uniquement).
    """

    def __init__(
        self,
        lot_dir: str | os.PathLike[str],
        manifest: LotManifest,
        *,
        queue_size: int = 45,
        n_writers: int = 3,
        min_free_bytes: int = 0,
        disk_check_interval: float = 1.0,
        disk_free_fn: Callable[[], int] | None = None,
        on_finished: Callable[[LotManifest, str], None] | None = None,
        write_delay_s: float = 0.0,
    ) -> None:
        if queue_size < 1 or n_writers < 1:
            raise ValueError("queue_size et n_writers doivent être ≥ 1")
        self.lot_dir = Path(lot_dir)
        self.manifest = manifest
        self.queue_size = int(queue_size)
        self.n_writers = int(n_writers)
        self.min_free_bytes = int(min_free_bytes)
        self.disk_check_interval = float(disk_check_interval)
        self._disk_free_fn = disk_free_fn or (lambda: disk_free_bytes(self.lot_dir))
        self._on_finished = on_finished
        self._write_delay_s = float(write_delay_s)

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
        self._writer: LotWriter | None = None
        self._threads: list[threading.Thread] = []
        self._submit_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._accepting = False
        self._started = False
        self._finished = threading.Event()
        self._stop_requested = threading.Event()
        self.stop_reason: str | None = None
        self.error: str | None = None

        # compteurs
        self._seq_next = 0  # prochain seq attendu (si l'appelant ne fournit pas seq)
        self._seq0: int | None = None
        self._last_seq: int | None = None
        self._submitted = 0
        self._dropped_queue = 0  # perdues car file pleine
        self._dropped_upstream = 0  # trous de séquence (perdues en amont, capteur / pilote)
        self._written = 0
        self._write_errors = 0
        self._bytes = 0
        self._first_ts: int | None = None
        self._last_ts: int | None = None
        self._recent_ts: collections.deque[int] = collections.deque(maxlen=256)
        self._max_fill = 0
        self._disk_free: int | None = None
        self._t_start = 0.0
        self._t_end: float | None = None

    # ------------------------------------------------------------------ cycle de vie
    def start(self) -> CaptureRecorder:
        if self._started:
            raise RuntimeError("Enregistrement déjà démarré")
        self._disk_free = self._safe_disk_free()
        if self.min_free_bytes and self._disk_free is not None and self._disk_free < self.min_free_bytes:
            raise DiskFullError(
                f"Espace disque insuffisant : {self._disk_free / 1e9:.2f} Go libres, "
                f"seuil {self.min_free_bytes / 1e9:.2f} Go"
            )
        self._writer = LotWriter(self.lot_dir, self.manifest)
        self._started = True
        self._t_start = time.monotonic()
        self._accepting = True
        for i in range(self.n_writers):
            t = threading.Thread(target=self._writer_loop, name=f"lot-writer-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        mon = threading.Thread(target=self._monitor_loop, name="lot-disk-monitor", daemon=True)
        mon.start()
        self._monitor_thread = mon
        return self

    @property
    def running(self) -> bool:
        return self._started and not self._finished.is_set()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Attend la clôture du lot (utile après un arrêt automatique)."""
        return self._finished.wait(timeout)

    # ------------------------------------------------------------------ production
    def submit(
        self,
        frame_bgr: np.ndarray,
        sensor_ts_ns: int,
        system_ts: float | None = None,
        seq: int | None = None,
    ) -> bool:
        """Dépose une image (non bloquant). Retourne False si elle est perdue ou refusée.

        ``seq`` : numéro de séquence capteur absolu ; s'il saute, les images
        manquantes sont comptées perdues. Sans ``seq``, un compteur interne est utilisé.
        """
        if not self._accepting:
            return False
        if system_ts is None:
            system_ts = time.time()
        with self._submit_lock:
            if not self._accepting:
                return False
            if seq is None:
                seq = self._seq_next
            if self._seq0 is None:
                self._seq0 = int(seq)
            rel_seq = int(seq) - self._seq0
            if self._last_seq is not None:
                gap = rel_seq - self._last_seq - 1
                if gap > 0:
                    self._dropped_upstream += gap
                    assert self._writer is not None
                    self._writer.count_dropped(gap)
                elif gap < 0:
                    # séquence non croissante : on l'ignore pour la détection de trous
                    rel_seq = self._last_seq + 1
            self._last_seq = rel_seq
            self._seq_next = int(seq) + 1
            ts = int(sensor_ts_ns)
            with self._stats_lock:
                self._submitted += 1
                if self._first_ts is None:
                    self._first_ts = ts
                self._last_ts = ts
                self._recent_ts.append(ts)
            if self._queue.qsize() >= self.queue_size:
                self._count_queue_drop()
                return False
            assert self._writer is not None
            index = self._writer.reserve_index()
            try:
                self._queue.put_nowait((index, frame_bgr, ts, float(system_ts), rel_seq))
            except queue.Full:  # ne devrait pas arriver : seuls les producteurs ajoutent, sous verrou
                self._count_queue_drop()
                return False
            fill = self._queue.qsize()
            if fill > self._max_fill:
                self._max_fill = fill
            return True

    def _count_queue_drop(self) -> None:
        with self._stats_lock:
            self._dropped_queue += 1
        assert self._writer is not None
        self._writer.count_dropped(1)

    # ------------------------------------------------------------------ consommation
    def _writer_loop(self) -> None:
        assert self._writer is not None
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            index, image, ts, system_ts, rel_seq = item
            try:
                if self._write_delay_s > 0:
                    time.sleep(self._write_delay_s)
                meta = self._writer.write_frame(image, ts, seq=rel_seq, system_ts=system_ts, index=index)
                with self._stats_lock:
                    self._written += 1
                    self._bytes += int(meta.size or 0)
            except Exception as exc:  # disque plein, support retiré…
                with self._stats_lock:
                    self._write_errors += 1
                if self.error is None:
                    self.error = f"Erreur d'écriture : {exc}"
                    log.exception("Erreur d'écriture de l'image %s", index)
                    self._request_auto_stop(STOP_WRITE_ERROR)

    def _monitor_loop(self) -> None:
        while not self._stop_requested.wait(self.disk_check_interval):
            free = self._safe_disk_free()
            self._disk_free = free
            if self.min_free_bytes and free is not None and free < self.min_free_bytes:
                log.warning("Espace disque sous le seuil (%.2f Go) : arrêt automatique de la capture", free / 1e9)
                self._request_auto_stop(STOP_DISK_LOW)
                return

    def _safe_disk_free(self) -> int | None:
        try:
            return int(self._disk_free_fn())
        except Exception:
            return None

    def _request_auto_stop(self, reason: str) -> None:
        if self._stop_requested.is_set():
            return
        threading.Thread(target=self.stop, kwargs={"reason": reason}, name="lot-auto-stop", daemon=True).start()

    # ------------------------------------------------------------------ arrêt
    def stop(self, reason: str = STOP_USER, status: str | None = None, timeout: float = 120.0) -> LotManifest:
        """Arrête la capture, vide la file (toutes les images acceptées sont écrites) et clôt le lot.

        Statut final : ``termine`` (défaut), ``interrompu`` si ``status`` le demande,
        en cas d'arrêt du service (``reason="shutdown"``) ou d'erreur d'écriture.
        Idempotent : un second appel attend la fin du premier et retourne le manifeste.
        """
        with self._stop_lock:
            if self._stop_requested.is_set():
                pass
            else:
                self._stop_requested.set()
                self.stop_reason = reason
                with self._submit_lock:
                    self._accepting = False
                if self._started:
                    self._finalize(reason, status, timeout)
                else:
                    self._finished.set()
        self._finished.wait(timeout)
        return self.manifest

    def _finalize(self, reason: str, status: str | None, timeout: float) -> None:
        for _ in self._threads:
            self._queue.put(_SENTINEL)
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
        if status is None:
            status = LOT_STATUS_INTERRUPTED if reason in (STOP_SHUTDOWN, STOP_WRITE_ERROR) else LOT_STATUS_DONE
        assert self._writer is not None
        try:
            manifest = self._writer.close(status)
            # LotWriter compte les rangs réservés ; on corrige si des écritures ont échoué.
            final = self.stats()
            manifest.stats = LotStats(
                frames_written=final["frames_written"],
                frames_dropped=final["frames_dropped"],
                duration_s=final["duration_s"],
                measured_fps=final["measured_fps"],
                bytes_written=final["bytes_written"],
            )
            manifest.save(self.lot_dir)
            self.manifest = manifest
        except Exception as exc:
            self.error = self.error or f"Erreur à la clôture du lot : {exc}"
            log.exception("Clôture du lot %s", self.lot_dir)
        self._t_end = time.monotonic()
        self._finished.set()
        if self._on_finished is not None:
            try:
                self._on_finished(self.manifest, reason)
            except Exception:
                log.exception("Rappel de fin de lot en échec")

    # ------------------------------------------------------------------ statistiques
    def stats(self) -> dict[str, Any]:
        """Statistiques en direct (sérialisables en JSON)."""
        with self._stats_lock:
            submitted = self._submitted
            dropped = self._dropped_queue + self._dropped_upstream
            total = submitted + self._dropped_upstream
            duration = (self._last_ts - self._first_ts) / 1e9 if self._first_ts is not None and self._last_ts is not None else 0.0
            measured = (total - 1) / duration if duration > 0 and total > 1 else 0.0
            recent = list(self._recent_ts)
            written = self._written
            nbytes = self._bytes
            errors = self._write_errors
        instant = 0.0
        if len(recent) >= 2:
            # cadence sur la dernière seconde (horloge capteur)
            window = [t for t in recent if t >= recent[-1] - 1_000_000_000]
            if len(window) >= 2 and window[-1] > window[0]:
                instant = (len(window) - 1) / ((window[-1] - window[0]) / 1e9)
        elapsed = ((self._t_end or time.monotonic()) - self._t_start) if self._started else 0.0
        return {
            "lot_id": self.manifest.lot_id,
            "status": self.manifest.status if self.finished else LOT_STATUS_RUNNING,
            "running": self.running,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "elapsed_s": round(elapsed, 3),
            "duration_s": round(duration, 3),
            "frames_received": total,
            "frames_written": written,
            "frames_queued": self._queue.qsize() if not self.finished else 0,
            "frames_dropped": dropped,
            "frames_dropped_queue": self._dropped_queue,
            "frames_dropped_upstream": self._dropped_upstream,
            "write_errors": errors,
            "drop_ratio": round(dropped / total, 6) if total else 0.0,
            "measured_fps": round(measured, 3),
            "instant_fps": round(instant, 3),
            "bytes_written": nbytes,
            "queue_size": self.queue_size,
            "queue_fill": round(self._queue.qsize() / self.queue_size, 3) if not self.finished else 0.0,
            "queue_max_fill": round(self._max_fill / self.queue_size, 3),
            "disk_free_bytes": self._disk_free,
            "min_free_bytes": self.min_free_bytes,
        }


class DiskFullError(RuntimeError):
    """Espace disque sous le seuil au démarrage d'une capture."""


# ---------------------------------------------------------------------------
# Réglages caméra appliqués en logiciel (sources simulées, gains couleur USB)
# ---------------------------------------------------------------------------
NEUTRAL_RED_GAIN = 2.0
NEUTRAL_BLUE_GAIN = 1.8
NEUTRAL_EXPOSURE_US = 500


def settings_lut(settings: CameraSettings, wb: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray | None:
    """Table de correspondance BGR (256 × 1 × 3) simulant exposition, gain et gains couleur.

    ``wb`` : facteurs (b, g, r) de dominante de la scène (balance des blancs « vraie »).
    Retourne ``None`` si la table est l'identité (aucun calcul à faire).
    """
    k = (settings.exposure_us / NEUTRAL_EXPOSURE_US) * settings.analogue_gain
    fb = k * wb[0] * settings.blue_gain / NEUTRAL_BLUE_GAIN
    fg = k * wb[1]
    fr = k * wb[2] * settings.red_gain / NEUTRAL_RED_GAIN
    if all(abs(f - 1.0) < 1e-6 for f in (fb, fg, fr)):
        return None
    x = np.arange(256, dtype=np.float32)
    lut = np.stack([np.clip(x * f, 0, 255) for f in (fb, fg, fr)], axis=-1).astype(np.uint8)
    return lut.reshape(256, 1, 3)


def apply_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = roi
    H, W = image.shape[:2]
    x, y = min(max(0, x), W - 1), min(max(0, y), H - 1)
    return np.ascontiguousarray(image[y : min(H, y + h), x : min(W, x + w)])


def gray_world_gains(image_bgr: np.ndarray, current: CameraSettings) -> tuple[float, float]:
    """Gains rouge / bleu qui neutralisent la dominante d'une image (carte grise)."""
    small = cv2.resize(image_bgr, (max(16, image_bgr.shape[1] // 4), max(16, image_bgr.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype(np.float64)
    # on écarte les pixels saturés ou trop sombres
    ok = (px.max(axis=1) < 250) & (px.min(axis=1) > 5)
    if ok.sum() > 100:
        px = px[ok]
    b, g, r = px.mean(axis=0)
    red = float(np.clip(current.red_gain * g / max(r, 1e-3), 0.1, 8.0))
    blue = float(np.clip(current.blue_gain * g / max(b, 1e-3), 0.1, 8.0))
    return round(red, 4), round(blue, 4)


# ---------------------------------------------------------------------------
# Images simulées
# ---------------------------------------------------------------------------
class SyntheticFrameSource:
    """Images simulées : cycle d'images synthétiques pré-rendues, ou relecture d'un dossier.

    Le rendu synthétique pleine résolution coûte ~45 ms par image : un cycle de
    ``cycle_s`` secondes est pré-rendu en mémoire (``prepare``) pour pouvoir tenir
    30 images/s. La scène est périodique (une colonne d'œufs toutes les 0,5 s), le
    bouclage est donc sans saut visible des rouleaux.

    ``frame(i, settings)`` applique les réglages (exposition, gain, gains couleur,
    ROI) en logiciel ; la scène a une légère dominante (``wb``) pour que la
    calibration de la balance des blancs ait un effet mesurable.
    """

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        *,
        scale: float = 1.0,
        fps: float = 30.0,
        cycle_s: float = 1.0,
        folder: str | os.PathLike[str] | None = None,
        max_folder_frames: int = 120,
        lot_type: str = "production",
        seed: int = 0,
        wb: tuple[float, float, float] = (0.92, 1.0, 1.08),
    ) -> None:
        self.scale = float(scale)
        self.fps = float(fps)
        self.cycle_s = float(cycle_s)
        self.folder = Path(folder) if folder else None
        self.max_folder_frames = int(max_folder_frames)
        self.lot_type = lot_type
        self.seed = int(seed)
        self.wb = wb
        self.frames: list[np.ndarray] = []
        self.ready = threading.Event()
        self.error: str | None = None
        self._lut_key: Any = None
        self._lut: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        if self.frames:
            h, w = self.frames[0].shape[:2]
            return w, h
        from .constants import SENSOR_HEIGHT, SENSOR_WIDTH

        if self.scale == 1.0:
            return SENSOR_WIDTH, SENSOR_HEIGHT
        from .synth import SceneConfig

        cfg = SceneConfig.scaled(self.scale)
        return cfg.width, cfg.height

    def prepare(self, progress: Callable[[float], None] | None = None) -> None:
        """Pré-rend (ou charge) les images du cycle. Peut être appelé dans un fil séparé."""
        try:
            frames = self._load_folder() if self.folder else self._render_cycle(progress)
            if not frames:
                raise ValueError("Aucune image disponible pour la source simulée")
            for f in frames:
                f.setflags(write=False)  # partagées entre fils : lecture seule
            self.frames = frames
        except Exception as exc:
            self.error = str(exc)
            log.exception("Préparation de la source simulée")
        finally:
            self.ready.set()

    def _render_cycle(self, progress: Callable[[float], None] | None) -> list[np.ndarray]:
        from .synth import SceneConfig, SyntheticScene

        if self.scale == 1.0:
            cfg = SceneConfig(fps=self.fps, lot_type=self.lot_type, seed=self.seed)
        else:
            cfg = SceneConfig.scaled(self.scale, fps=self.fps, lot_type=self.lot_type, seed=self.seed)
        # cycle = multiple de la période d'une colonne (bouclage sans saut des rouleaux)
        periods = max(1, round(self.cycle_s / cfg.column_period_s))
        n = max(2, int(round(periods * cfg.column_period_s * self.fps)))
        scene = SyntheticScene(cfg)
        t0 = 8.0  # la table est déjà pleine
        out = []
        for i in range(n):
            img, _ = scene.render_at(t0 + i / self.fps, noise_index=i)
            out.append(img)
            if progress:
                progress((i + 1) / n)
        return out

    def _load_folder(self) -> list[np.ndarray]:
        from .formats import read_image

        assert self.folder is not None
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {self.folder}")
        root = self.folder / "frames" if (self.folder / "frames").is_dir() else self.folder
        files = sorted(p for p in root.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"Aucune image dans {root}")
        if len(files) > self.max_folder_frames:
            files = files[: self.max_folder_frames]
        frames = [read_image(p) for p in files]
        h, w = frames[0].shape[:2]
        return [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)) for f in frames]

    def raw(self, index: int) -> np.ndarray:
        return self.frames[index % len(self.frames)]

    def frame(self, index: int, settings: CameraSettings) -> np.ndarray:
        """Image n° ``index`` (cyclique) avec les réglages appliqués. Ne pas modifier le résultat."""
        img = self.raw(index)
        key = (settings.exposure_us, settings.analogue_gain, settings.red_gain, settings.blue_gain)
        if key != self._lut_key:
            self._lut = settings_lut(settings, self.wb)
            self._lut_key = key
        lut = self._lut
        img = apply_roi(img, settings.roi)
        if lut is not None:
            img = cv2.LUT(img, lut)
        return img


# ---------------------------------------------------------------------------
# Boucle à cadence fixe
# ---------------------------------------------------------------------------
class PacedLoop:
    """Appelle ``fn(seq, sensor_ts_ns)`` à cadence fixe dans un fil dédié, comme un capteur.

    Les horodatages sont ceux d'un capteur idéal (``t0 + seq × période``). Comme
    une vraie caméra, la boucle dispose de ``buffer_frames`` tampons : un retard
    ponctuel inférieur est rattrapé (images livrées en rafale, horodatages
    intacts) ; au-delà, les créneaux en trop sont sautés et ``seq`` fait un bond,
    exactement comme une image perdue par le capteur. ``fps`` est modifiable à
    chaud (``set_fps``).
    """

    def __init__(self, fn: Callable[[int, int], None], fps: float, name: str = "paced-loop", buffer_frames: int = 4) -> None:
        self._fn = fn
        self.buffer_frames = max(1, int(buffer_frames))
        self._fps = float(fps)
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._changed = threading.Event()
        self.skipped = 0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def set_fps(self, fps: float) -> None:
        if abs(float(fps) - self._fps) > 1e-9:
            self._fps = float(fps)
            self._changed.set()

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            self._changed.clear()
            period_ns = int(round(1e9 / self._fps))
            t0 = time.monotonic_ns() - seq * period_ns
            while not self._stop.is_set() and not self._changed.is_set():
                due = t0 + seq * period_ns
                now = time.monotonic_ns()
                if due > now:
                    time.sleep((due - now) / 1e9)
                    now = time.monotonic_ns()
                late = (now - due) // period_ns
                if late >= self.buffer_frames:  # tampons épuisés : créneaux sautés (images perdues)
                    skip = int(late) - self.buffer_frames + 1
                    seq += skip
                    self.skipped += skip
                    due = t0 + seq * period_ns
                try:
                    self._fn(seq, due)
                except Exception:
                    log.exception("Erreur dans la boucle d'acquisition")
                    time.sleep(0.05)
                seq += 1


def fps_from_timestamps(ts_ns: list[int] | collections.deque[int]) -> float:
    """Cadence moyenne (images/s) d'une suite d'horodatages en ns."""
    if len(ts_ns) < 2:
        return 0.0
    span = (ts_ns[-1] - ts_ns[0]) / 1e9
    return (len(ts_ns) - 1) / span if span > 0 else 0.0


def nominal_seq(sensor_ts_ns: int, first_ts_ns: int, fps: float) -> int:
    """Numéro de séquence déduit de l'horodatage capteur à cadence fixe (images perdues comprises)."""
    return int(math.floor((sensor_ts_ns - first_ts_ns) * fps / 1e9 + 0.5))
