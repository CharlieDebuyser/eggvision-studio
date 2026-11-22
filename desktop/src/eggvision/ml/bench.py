"""Banc de mesure de latence (tâche ``kind="bench"``).

* cible ``local`` : le paquet exporté est chargé sur ce PC par
  :class:`eggvision_shared.runtime.EggClassifier` (ONNX Runtime CPU) ;
* cible ``pi`` : le zip du paquet est envoyé à l'agent Raspberry Pi ::

      POST {agent_url}/api/bench     (multipart/form-data)
        package    fichier zip du paquet
        variant    fp32 | int8
        n_columns  nombre de colonnes mesurées
        threads    nombre de fils ONNX Runtime (1–64 ; champ absent = défaut)
      → 200 JSON : dictionnaire de EggClassifier.benchmark()
                   + "parity" (résultat de verify_reference) + "device" + "onnxruntime"

Chaque résultat est ajouté à ``Model.export["bench"]`` (50 derniers conservés).
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from typing import Any

from .. import state
from ..jobs import JobContext
from ..project import Project
from .store import get_model_row

MAX_HISTORY = 50


class BenchError(RuntimeError):
    """Erreur de banc de mesure présentable à l'utilisateur."""


def _package(project: Project, model_id: str) -> tuple[dict[str, Any], str]:
    with project.session() as s:
        row = get_model_row(s, model_id)
        export = dict(row.export or {})
        name = row.name
    if not export.get("zip") or not project.abspath(export["zip"]).is_file():
        raise BenchError("Le modèle n'a pas encore été exporté : lancer l'export d'abord.")
    return export, name


def start_bench(
    project: Project,
    model_id: str,
    target: str = "local",
    variant: str = "fp32",
    n_columns: int = 50,
    threads: int | None = None,
    agent_url: str | None = None,
) -> dict[str, str]:
    if target not in ("local", "pi"):
        raise BenchError("Cible inconnue : « local » ou « pi ».")
    export, name = _package(project, model_id)
    if variant not in export.get("variants", []):
        raise BenchError(f"Variante « {variant} » absente du paquet (disponibles : {', '.join(export.get('variants', []))}).")
    if not 1 <= int(n_columns) <= 10_000:
        raise BenchError("Le nombre de colonnes doit être compris entre 1 et 10 000.")
    url = None
    if target == "pi":
        url = (agent_url or state.agent_url or "").rstrip("/")
        if not url:
            raise BenchError("Aucun agent Raspberry Pi connecté : se connecter à l'agent (module Prise de vue) ou indiquer son adresse.")
    params = {"model_id": model_id, "target": target, "variant": variant, "n_columns": int(n_columns), "threads": threads, "agent_url": url}
    label = "Raspberry Pi" if target == "pi" else "ce PC"
    job = state.jobs.submit(
        "bench", f"Banc de mesure ({label}, {variant.upper()}) — {name}",
        lambda ctx: run_bench(ctx, project, params),
        params=params, project=project,
        group="gpu" if target == "local" else "pi-bench",  # local : jamais en même temps qu'un entraînement
    )
    return {"model_id": model_id, "job_id": job.id}


def local_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"host": platform.node(), "system": f"{platform.system()} {platform.release()}", "machine": platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import psutil

        info["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # facultatif
        pass
    proc = platform.processor()
    if proc:
        info["processor"] = proc
    return info


def bench_local(package_dir: str, variant: str, n_columns: int, threads: int | None) -> dict[str, Any]:
    import onnxruntime
    from eggvision_shared.runtime import EggClassifier

    clf = EggClassifier.load(package_dir, variant=variant, threads=threads or None)
    result = clf.benchmark(n_columns=n_columns)
    result["parity"] = clf.verify_reference(package_dir)
    result["device"] = local_device_info()
    result["onnxruntime"] = onnxruntime.__version__
    return result


def bench_pi(agent_url: str, zip_path: str, variant: str, n_columns: int, threads: int | None, ctx: JobContext | None = None) -> dict[str, Any]:
    import httpx

    data = {"variant": variant, "n_columns": str(n_columns)}
    if threads:  # absent = défaut d'ONNX Runtime sur l'agent (l'agent refuse 0)
        data["threads"] = str(threads)
    timeout = httpx.Timeout(connect=10.0, read=max(120.0, n_columns * 5.0), write=120.0, pool=10.0)
    try:
        with open(zip_path, "rb") as f:
            r = httpx.post(f"{agent_url}/api/bench", data=data, files={"package": (os.path.basename(zip_path), f, "application/zip")}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise BenchError(f"Agent injoignable ({agent_url}) : {exc}") from exc
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:300]
        raise BenchError(f"L'agent a refusé le banc de mesure (HTTP {r.status_code}) : {detail}")
    result = r.json()
    if "latency_ms" not in result:
        raise BenchError("Réponse de l'agent inattendue (latency_ms manquant).")
    return result


def run_bench(ctx: JobContext, project: Project, params: dict[str, Any]) -> dict[str, Any]:
    model_id = params["model_id"]
    export, _ = _package(project, model_id)
    ctx.progress(0.05, "Préparation")
    if params["target"] == "local":
        ctx.progress(0.1, f"Mesure sur ce PC ({params['n_columns']} colonnes)")
        result = bench_local(str(project.abspath(export["package_dir"])), params["variant"], params["n_columns"], params.get("threads"))
    else:
        ctx.progress(0.1, "Envoi du paquet à l'agent et mesure sur le Raspberry Pi")
        ctx.log(f"Agent : {params['agent_url']}")
        result = bench_pi(params["agent_url"], str(project.abspath(export["zip"])), params["variant"], params["n_columns"], params.get("threads"), ctx)
    ctx.check_cancelled()
    lat = result.get("latency_ms", {})
    parity = result.get("parity") or {}
    ctx.log(
        f"Latence par colonne : moyenne {lat.get('mean', float('nan')):.1f} ms, p95 {lat.get('p95', float('nan')):.1f} ms"
        + (f" ; parité : écart {parity.get('max_abs_diff'):.2e} ({'OK' if parity.get('ok') else 'hors tolérance'})" if parity else "")
    )
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "target": params["target"],
        "variant": params["variant"],
        "n_columns": params["n_columns"],
        "threads": params.get("threads"),
        "agent_url": params.get("agent_url"),
        "export_version": export.get("version"),
        "result": result,
    }
    with project.session() as s:
        row = get_model_row(s, model_id)
        current = dict(row.export or {})
        current["bench"] = (list(current.get("bench") or []) + [entry])[-MAX_HISTORY:]
        row.export = current
    return {"bench": entry}


def bench_history(project: Project, model_id: str) -> list[dict[str, Any]]:
    with project.session() as s:
        return list(reversed((get_model_row(s, model_id).export or {}).get("bench") or []))
