#!/usr/bin/env python3
"""Critère d'acceptation 1 — capture par l'agent Pi puis synchronisation vérifiée (mode simulé).

Rejoue le scénario « 10 min à 30 images/s en 1456 × 1088, moins de 0,1 % d'images
perdues, puis synchronisation vérifiée sur le PC » :

  1. lance l'agent (``python -m eggvision_agent --camera simulated``) en sous-processus ;
  2. crée un projet temporaire, règle la caméra et démarre un lot via l'API de l'agent ;
  3. suit la capture (images, pertes, cadence, file, disque, charge de l'agent) ;
  4. arrête le lot, le synchronise (reprise + SHA-256) dans le projet et l'enregistre ;
  5. vérifie : checksums, continuité des numéros de séquence, régularité des horodatages,
     nombre d'images en base ; imprime un rapport chiffré et un verdict.

Usage ::

    uv run python scripts/acceptance_capture.py                  # 600 s (critère complet)
    uv run python scripts/acceptance_capture.py --duration 60    # essai court

Sur un vrai Raspberry Pi, ``--agent-url http://<ip>:8770`` saute le lancement
local et utilise l'agent indiqué (caméra réelle).
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} Go" if n >= 1e9 else f"{n / 1e6:.1f} Mo"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=600.0, help="Durée de capture en secondes (défaut : 600)")
    p.add_argument("--fps", type=float, default=30.0, help="Cadence (défaut : 30)")
    p.add_argument("--scale", type=float, default=1.0, help="Échelle de l'image simulée (1 = 1456 × 1088)")
    p.add_argument("--port", type=int, default=8828, help="Port de l'agent lancé localement (défaut : 8828)")
    p.add_argument("--agent-url", help="Utiliser un agent déjà démarré (ex. vrai Pi) au lieu d'en lancer un")
    p.add_argument("--workdir", type=Path, help="Dossier de travail (défaut : dossier temporaire supprimé à la fin)")
    p.add_argument("--keep", action="store_true", help="Conserver les données (lot de l'agent et projet)")
    p.add_argument("--writers", type=int, default=3, help="Fils d'encodage JPEG de l'agent lancé localement")
    p.add_argument("--max-drop", type=float, default=0.001, help="Taux de pertes maximal accepté (défaut : 0,1 %%)")
    args = p.parse_args(argv)

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="eggvision-acceptance-"))
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EGGVISION_DATA_DIR", str(workdir / "appdata"))

    from eggvision.capture.pi import PiAgentSource
    from eggvision.capture.sync import sync_lot, verify_lot_dir
    from eggvision.project import Project
    from eggvision_shared.formats import LotManifest, new_lot_id, read_metadata

    agent_proc: subprocess.Popen | None = None
    url = args.agent_url
    report: dict[str, object] = {}
    ok = False
    try:
        if url is None:
            url = f"http://127.0.0.1:{args.port}"
            cmd = [
                sys.executable, "-m", "eggvision_agent",
                "--host", "127.0.0.1", "--port", str(args.port),
                "--storage", str(workdir / "agent"), "--camera", "simulated",
                "--sim-scale", str(args.scale), "--no-mdns", "--min-free-gb", "1",
                "--writers", str(args.writers), "--log-level", "warning",
            ]  # fmt: skip
            print(f"Lancement de l'agent : {' '.join(cmd)}")
            agent_proc = subprocess.Popen(cmd)
        client = httpx.Client(base_url=url, timeout=10)
        t0 = time.time()
        while True:  # attente de l'agent et de sa caméra (pré-rendu du cycle simulé)
            try:
                st = client.get("/api/status").json()
                if st["camera"]["ready"] and st["camera"]["actual_fps"] > 0:
                    break
            except httpx.HTTPError:
                pass
            if agent_proc is not None and agent_proc.poll() is not None:
                raise RuntimeError("L'agent s'est arrêté au démarrage")
            if time.time() - t0 > 180:
                raise RuntimeError("Agent non prêt après 180 s")
            time.sleep(0.5)
        cam = st["camera"]
        print(f"Agent {st['version']} prêt en {time.time() - t0:.1f} s — caméra {cam['model']} {cam['output_width']}×{cam['output_height']}, {st['device']['model']}")

        project = Project.create(workdir / "projet", "Acceptation capture")
        src = PiAgentSource(url, timeout=30)
        src.open()
        settings = src.set_settings({"exposure_us": 500, "fps": args.fps, "analogue_gain": 1.0, "roi": None})
        time.sleep(1.0)  # la cadence se stabilise après le changement
        lot_id = new_lot_id("acceptation")
        manifest = LotManifest(lot_id=lot_id, name="Acceptation capture", type="production", operator="script", camera=settings)
        try:
            import psutil

            agent_ps = psutil.Process(agent_proc.pid) if agent_proc is not None else None
            if agent_ps is not None:
                agent_ps.cpu_percent(None)
        except Exception:
            agent_ps = None
        src.start_lot(manifest, None, int(1e9))
        print(f"\nCapture du lot {lot_id} pendant {args.duration:.0f} s à {args.fps:g} images/s…")
        t_start = time.monotonic()
        cpu_samples: list[float] = []
        rss_max = 0
        fill_max = 0.0
        next_print = 0.0
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= args.duration:
                break
            time.sleep(min(1.0, args.duration - elapsed))
            live = src.capture_stats() or {}
            if not live.get("active"):
                raise RuntimeError(f"La capture s'est arrêtée d'elle-même : {live}")
            fill_max = max(fill_max, float(live.get("queue_fill") or 0))
            if agent_ps is not None:
                cpu_samples.append(agent_ps.cpu_percent(None))
                rss_max = max(rss_max, agent_ps.memory_info().rss)
            if elapsed >= next_print:
                next_print += 10
                print(
                    f"  t={elapsed:5.0f} s  images={live['frames_written']:6d}  perdues={live['frames_dropped']:4d}  "
                    f"cadence={live['instant_fps']:5.2f}  file={live['queue_fill'] * 100:4.0f} %  "
                    f"disque libre={_fmt_bytes(live.get('disk_free_bytes') or 0)}"
                )
        final = src.stop_lot()
        cap = final["capture"]
        m = final["manifest"]
        print(f"Lot clos : {m['status']}, {m['stats']['frames_written']} images, {m['stats']['frames_dropped']} perdues")

        print("\nSynchronisation vers le projet…")
        t_sync = time.monotonic()
        last = [0.0]

        def progress(pr: dict) -> None:
            now = time.monotonic()
            if now - last[0] > 5:
                last[0] = now
                print(f"  {pr['files_done']}/{pr['files_total']} fichiers — {pr['bytes_done'] / 1e6:.0f}/{pr['bytes_total'] / 1e6:.0f} Mo ({pr['rate_mb_s']:.0f} Mo/s)")

        sync_res = sync_lot(url, lot_id, project.lot_dir(lot_id), progress=progress, log_fn=lambda s: None)
        sync_s = time.monotonic() - t_sync
        files = client.get(f"/api/lots/{lot_id}/files", timeout=120).json()["files"]
        t_verify = time.monotonic()
        bad = verify_lot_dir(project.lot_dir(lot_id), files)
        verify_s = time.monotonic() - t_verify
        lot = project.register_lot(lot_id, "pi", sync_status="done")
        metas = read_metadata(project.lot_dir(lot_id))
        seqs = np.array([x.seq for x in metas])
        ts = np.array([x.sensor_ts_ns for x in metas], dtype=np.int64)
        dt_ms = np.diff(ts) / 1e6 if len(ts) > 1 else np.array([0.0])
        gaps = int(np.sum(np.diff(seqs) - 1)) if len(seqs) > 1 else 0
        written, dropped = m["stats"]["frames_written"], m["stats"]["frames_dropped"]
        total = written + dropped
        drop_ratio = dropped / total if total else 1.0
        duration = m["stats"]["duration_s"]
        nbytes = m["stats"]["bytes_written"]
        report = {
            "images enregistrées": written,
            "images perdues": dropped,
            "taux de pertes": f"{drop_ratio * 100:.4f} % (seuil {args.max_drop * 100:.2f} %)",
            "  dont file pleine / en amont": f"{cap.get('frames_dropped_queue')} / {cap.get('frames_dropped_upstream')}",
            "trous de séquence (metadata)": gaps,
            "durée capteur": f"{duration:.1f} s",
            "cadence mesurée": f"{m['stats']['measured_fps']:.3f} images/s (cible {args.fps:g})",
            "intervalle entre images": f"moy. {dt_ms.mean():.2f} ms, max {dt_ms.max():.2f} ms",
            "remplissage max. de la file": f"{max(fill_max, float(cap.get('queue_max_fill') or 0)) * 100:.0f} %",
            "volume du lot": f"{_fmt_bytes(nbytes)} ({nbytes / max(written, 1) / 1e3:.0f} Ko/image)",
            "débit d'écriture": f"{nbytes / max(duration, 1e-6) / 1e6:.1f} Mo/s",
            "CPU agent (moy. / max)": f"{np.mean(cpu_samples):.0f} % / {np.max(cpu_samples):.0f} %" if cpu_samples else "n/d",
            "mémoire agent (max)": _fmt_bytes(rss_max) if rss_max else "n/d",
            "synchronisation": f"{sync_res['files']} fichiers, {_fmt_bytes(sync_res['bytes'])} en {sync_s:.1f} s ({sync_res['bytes'] / max(sync_s, 1e-6) / 1e6:.0f} Mo/s)",
            "reprises / fichiers corrompus": f"{sync_res['retries']} / {sync_res['corrupted']}",
            "vérification SHA-256 indépendante": f"{len(files) - len(bad)}/{len(files)} conformes ({verify_s:.1f} s)",
            "images en base (projet)": lot.frame_count,
        }
        ok = (
            m["status"] == "termine"
            and drop_ratio < args.max_drop
            and not bad
            and lot.frame_count == written
            and abs(m["stats"]["measured_fps"] - args.fps) / args.fps < 0.01
        )
    finally:
        if agent_proc is not None:
            agent_proc.send_signal(signal.SIGTERM if os.name != "nt" else signal.CTRL_BREAK_EVENT)
            try:
                agent_proc.wait(20)
            except subprocess.TimeoutExpired:
                agent_proc.kill()
        if not args.keep and args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)
        elif args.keep:
            print(f"\nDonnées conservées dans {workdir}")

    print("\n" + "=" * 72)
    print("RAPPORT — critère d'acceptation 1 (capture agent Pi + synchronisation)")
    print("=" * 72)
    width = max(len(k) for k in report)
    for k, v in report.items():
        print(f"{k.ljust(width)} : {v}")
    print("-" * 72)
    print("VERDICT :", "CONFORME" if ok else "NON CONFORME")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
