"""Synchronisation d'un lot depuis l'agent Pi vers ``lots/<id>/`` du projet.

Robuste aux coupures :
  * chaque fichier est téléchargé dans ``<fichier>.part`` puis renommé une fois
    sa somme SHA-256 vérifiée ; une coupure laisse le ``.part`` en place et la
    reprise continue à partir de sa taille (en-tête HTTP ``Range``) ;
  * les fichiers déjà vérifiés sont notés dans ``.sync-state.json`` (taille,
    date, SHA-256) : une reprise ultérieure ne retélécharge ni ne rehache rien ;
  * un fichier dont la somme ne correspond pas est supprimé puis retéléchargé
    (nouvelles tentatives avec attente croissante) ; s'il reste faux, la
    synchronisation échoue en le nommant ;
  * ``lot.json`` est écrit en dernier : un lot local n'a de manifeste que
    lorsque toutes ses images sont présentes et vérifiées.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx
from eggvision_shared.formats import LOT_MANIFEST, LOT_METADATA, atomic_write_text, sha256_file

log = logging.getLogger(__name__)

STATE_FILE = ".sync-state.json"

ProgressFn = Callable[[dict[str, Any]], None]


class SyncError(Exception):
    """Échec de la synchronisation (message en français)."""


class _Stop(Exception):
    pass


def _noop(*_: Any, **__: Any) -> None:
    return None


class LotSync:
    """Synchronise un lot ; voir :func:`sync_lot`."""

    def __init__(
        self,
        agent_url: str,
        lot_id: str,
        dest_dir: str | os.PathLike[str],
        *,
        progress: ProgressFn | None = None,
        check_cancelled: Callable[[], None] | None = None,
        log_fn: Callable[[str], None] | None = None,
        retries: int = 5,
        concurrency: int = 4,
        timeout: float = 30.0,
        backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.agent_url = agent_url.rstrip("/")
        self.lot_id = lot_id
        self.dest = Path(dest_dir)
        self.progress = progress or _noop
        self.check_cancelled = check_cancelled or _noop
        self.log = log_fn or (lambda text: log.info("%s", text))
        self.retries = max(1, retries)
        self.concurrency = max(1, concurrency)
        self.backoff_s = backoff_s
        self.client = httpx.Client(base_url=self.agent_url, timeout=timeout, transport=transport)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state: dict[str, dict[str, Any]] = {}
        self._state_dirty = 0
        # compteurs
        self.files_total = 0
        self.bytes_total = 0
        self.files_done = 0
        self.bytes_completed = 0
        self._inflight: dict[str, int] = {}
        self.bytes_downloaded = 0
        self.files_skipped = 0
        self.n_retries = 0
        self.n_corrupted = 0
        self._t0 = time.monotonic()
        self._last_progress = 0.0

    # ------------------------------------------------------------------ état local
    def _state_path(self) -> Path:
        return self.dest / STATE_FILE

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
            if data.get("lot_id") == self.lot_id:
                self.state = dict(data.get("files", {}))
        except (OSError, ValueError):
            self.state = {}

    def _save_state(self, force: bool = False) -> None:
        with self._lock:
            self._state_dirty += 1
            if not force and self._state_dirty < 100:
                return
            self._state_dirty = 0
            payload = json.dumps({"lot_id": self.lot_id, "files": self.state}, separators=(",", ":"))
        atomic_write_text(self._state_path(), payload)

    def _record(self, rel: str, path: Path, digest: str) -> None:
        st = path.stat()
        with self._lock:
            self.state[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        self._save_state()

    def _is_verified(self, entry: dict[str, Any]) -> bool:
        rel = entry["path"]
        path = self.dest / rel
        if not path.is_file():
            return False
        st = path.stat()
        if st.st_size != entry["size"]:
            return False
        known = self.state.get(rel)
        if known and known.get("sha256") == entry["sha256"] and known.get("size") == st.st_size and known.get("mtime_ns") == st.st_mtime_ns:
            return True
        if sha256_file(path) == entry["sha256"]:  # fichier présent mais non noté : on vérifie
            self._record(rel, path, entry["sha256"])
            return True
        return False

    # ------------------------------------------------------------------ progression
    def _report(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_progress < 0.25:
            return
        self._last_progress = now
        elapsed = max(1e-6, now - self._t0)
        with self._lock:
            done = self.bytes_completed + sum(self._inflight.values())
        self.progress(
            {
                "files_done": self.files_done,
                "files_total": self.files_total,
                "bytes_done": done,
                "bytes_total": self.bytes_total,
                "bytes_downloaded": self.bytes_downloaded,
                "rate_mb_s": round(self.bytes_downloaded / elapsed / 1e6, 2),
                "retries": self.n_retries,
                "corrupted": self.n_corrupted,
            }
        )

    def _checkpoint(self) -> None:
        if self._stop.is_set():
            raise _Stop()
        self.check_cancelled()

    # ------------------------------------------------------------------ téléchargement
    def _set_inflight(self, rel: str, n: int | None) -> None:
        with self._lock:
            if n is None:
                self._inflight.pop(rel, None)
            else:
                self._inflight[rel] = n

    def _download(self, entry: dict[str, Any]) -> None:
        rel, size, expected = entry["path"], int(entry["size"]), entry["sha256"]
        final = self.dest / rel
        part = final.with_name(final.name + ".part")
        final.parent.mkdir(parents=True, exist_ok=True)
        last_error = ""
        for attempt in range(self.retries):
            self._checkpoint()
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > size:
                    part.unlink()
                    offset = 0
                self._set_inflight(rel, offset)
                if offset < size:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with self.client.stream("GET", f"/api/lots/{self.lot_id}/files/{rel}", headers=headers) as r:
                        if r.status_code == 416:  # .part incohérent : on repart de zéro
                            part.unlink(missing_ok=True)
                            raise httpx.HTTPError("plage refusée (416)")
                        if r.status_code >= 400:
                            r.read()
                            raise SyncError(f"Téléchargement de {rel} refusé par l'agent ({r.status_code}) : {r.text[:200]}")
                        if offset and r.status_code != 206:  # plage ignorée : fichier complet renvoyé
                            offset = 0
                        received = offset
                        with open(part, "ab" if offset else "wb") as f:
                            for block in r.iter_bytes():  # sans tampon : tout octet reçu est écrit (reprise exacte)
                                self._checkpoint()
                                f.write(block)
                                received += len(block)
                                with self._lock:
                                    self.bytes_downloaded += len(block)
                                    self._inflight[rel] = received
                                self._report()
                # vérification de la somme SHA-256
                if part.stat().st_size == size and sha256_file(part) == expected:
                    os.replace(part, final)
                    self._record(rel, final, expected)
                    with self._lock:
                        self._inflight.pop(rel, None)
                        self.files_done += 1
                        self.bytes_completed += size
                    self._report()
                    return
                with self._lock:
                    self.n_corrupted += 1
                last_error = f"somme de contrôle incorrecte (taille reçue {part.stat().st_size} / {size})"
                self.log(f"{rel} : {last_error}, nouveau téléchargement")
                part.unlink(missing_ok=True)
                self._set_inflight(rel, 0)
            except (_Stop, SyncError):
                raise
            except (httpx.HTTPError, OSError) as exc:
                with self._lock:
                    self.n_retries += 1
                last_error = str(exc) or exc.__class__.__name__
                # la partie déjà reçue reste dans .part : la tentative suivante reprend à sa taille
                self._set_inflight(rel, part.stat().st_size if part.exists() else 0)
                self.log(f"{rel} : coupure ({last_error}) — reprise (tentative {attempt + 2}/{self.retries})")
            # attente croissante, interrompue par une annulation
            if self._stop.wait(min(5.0, self.backoff_s * (2**attempt))):
                raise _Stop()
        raise SyncError(f"Échec de la synchronisation de {rel} après {self.retries} tentatives : {last_error}")

    # ------------------------------------------------------------------ orchestration
    def run(self) -> dict[str, Any]:
        try:
            r = self.client.get(f"/api/lots/{self.lot_id}/files", timeout=120)
        except httpx.HTTPError as exc:
            raise SyncError(f"Agent injoignable ({self.agent_url}) : {exc}") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:200]
            raise SyncError(f"Manifeste du lot {self.lot_id} indisponible ({r.status_code}) : {detail}")
        listing = r.json()
        files: list[dict[str, Any]] = listing["files"]
        self.files_total = len(files)
        self.bytes_total = int(sum(int(f["size"]) for f in files))
        self.dest.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self.log(f"Lot {self.lot_id} : {self.files_total} fichiers, {self.bytes_total / 1e6:.1f} Mo")

        todo: list[dict[str, Any]] = []
        for entry in files:
            if entry["path"] == LOT_MANIFEST:
                continue
            if self._is_verified(entry):
                self.files_done += 1
                self.bytes_completed += int(entry["size"])
                self.files_skipped += 1
            else:
                part = self.dest / (entry["path"] + ".part")
                if part.exists():  # reprise : les octets déjà reçus comptent
                    self._inflight[entry["path"]] = min(part.stat().st_size, int(entry["size"]))
                todo.append(entry)
        if self.files_skipped:
            self.log(f"{self.files_skipped} fichiers déjà présents et vérifiés")
        self._report(force=True)

        frames = [e for e in todo if e["path"] != LOT_METADATA]
        last = [e for e in todo if e["path"] == LOT_METADATA] + [e for e in files if e["path"] == LOT_MANIFEST]
        try:
            self._run_parallel(frames)
            for entry in last:  # metadata.jsonl puis lot.json, en dernier
                if entry["path"] == LOT_MANIFEST and self._is_verified(entry):
                    self.files_done += 1
                    self.bytes_completed += int(entry["size"])
                    self.files_skipped += 1
                    continue
                self._download(entry)
        finally:
            self._save_state(force=True)
            self.client.close()
        self._report(force=True)
        elapsed = time.monotonic() - self._t0
        return {
            "lot_id": self.lot_id,
            "files": self.files_total,
            "bytes": self.bytes_total,
            "bytes_downloaded": self.bytes_downloaded,
            "files_skipped": self.files_skipped,
            "retries": self.n_retries,
            "corrupted": self.n_corrupted,
            "duration_s": round(elapsed, 2),
            "rate_mb_s": round(self.bytes_downloaded / max(elapsed, 1e-6) / 1e6, 2),
            "verified": True,
        }

    def _run_parallel(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="sync") as pool:
            it = iter(entries)
            pending = set()
            for entry in it:
                pending.add(pool.submit(self._download, entry))
                if len(pending) >= self.concurrency * 4:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for fut in done:
                    exc = fut.exception()
                    if exc is not None:
                        self._stop.set()
                        wait(pending)
                        if isinstance(exc, _Stop):
                            self.check_cancelled()
                        raise exc
                for entry in it:
                    pending.add(pool.submit(self._download, entry))
                    if len(pending) >= self.concurrency * 4:
                        break


def sync_lot(agent_url: str, lot_id: str, dest_dir: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """Télécharge et vérifie le lot ``lot_id`` de l'agent dans ``dest_dir`` (reprise automatique).

    ``kwargs`` : ``progress(dict)``, ``check_cancelled()`` (lève une exception pour
    annuler), ``log_fn(str)``, ``retries``, ``concurrency``, ``timeout``,
    ``backoff_s``, ``transport`` (httpx, pour les tests). Lève :class:`SyncError`.
    """
    return LotSync(agent_url, lot_id, dest_dir, **kwargs).run()


def verify_lot_dir(lot_dir: str | os.PathLike[str], files: list[dict[str, Any]]) -> list[str]:
    """Chemins dont la taille ou la somme SHA-256 locale diffère du manifeste (vide = tout est conforme)."""
    bad = []
    for entry in files:
        p = Path(lot_dir) / entry["path"]
        if not p.is_file() or p.stat().st_size != entry["size"]:
            bad.append(entry["path"])
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != entry["sha256"]:
            bad.append(entry["path"])
    return bad
