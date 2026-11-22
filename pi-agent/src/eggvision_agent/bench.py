"""Banc de mesure d'un paquet de déploiement sur le Pi (``POST /api/bench``)."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from . import system

CONFIG_FILENAME = "inference_config.json"
MAX_UNCOMPRESSED = 2_000_000_000  # garde-fou contre les archives piégées


def _safe_extract(zip_path: Path, dest: Path) -> None:
    from .service import AgentError

    try:
        with zipfile.ZipFile(zip_path) as zf:
            total = 0
            dest_resolved = dest.resolve()
            for info in zf.infolist():
                total += info.file_size
                if total > MAX_UNCOMPRESSED:
                    raise AgentError(413, "Paquet trop volumineux une fois décompressé.")
                target = (dest / info.filename).resolve()
                if target != dest_resolved and dest_resolved not in target.parents:
                    raise AgentError(400, f"Chemin interdit dans le paquet : {info.filename}")
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise AgentError(400, "Le fichier « package » n'est pas une archive zip valide.") from exc


def find_package_dir(root: Path) -> Path:
    """Dossier contenant ``inference_config.json`` (racine du zip ou un sous-dossier unique)."""
    from .service import AgentError

    if (root / CONFIG_FILENAME).is_file():
        return root
    candidates = sorted(p.parent for p in root.rglob(CONFIG_FILENAME))
    if not candidates:
        raise AgentError(400, f"Paquet invalide : {CONFIG_FILENAME} introuvable.")
    return min(candidates, key=lambda p: len(p.parts))


def onnxruntime_version() -> str | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    import onnxruntime

    return str(onnxruntime.__version__)


def run_bench(package_zip: Path, *, variant: str = "fp32", n_columns: int = 50, threads: int | None = None) -> dict[str, Any]:
    """Dézippe le paquet, mesure la latence par colonne et vérifie la parité avec la référence."""
    from .service import AgentError

    ort_version = onnxruntime_version()
    if ort_version is None:
        raise AgentError(
            501,
            "ONNX Runtime n'est pas installé sur l'agent : banc de mesure indisponible "
            "(installer l'option « bench » : pip install 'eggvision-agent[bench]').",
        )
    if variant not in ("fp32", "int8"):
        raise AgentError(422, f"Variante inconnue : {variant!r} (fp32 ou int8)")
    if not (1 <= n_columns <= 10_000):
        raise AgentError(422, "n_columns doit être compris entre 1 et 10 000.")
    if threads is not None and not (1 <= threads <= 64):
        raise AgentError(422, "threads doit être compris entre 1 et 64.")
    from eggvision_shared.runtime import EggClassifier

    with tempfile.TemporaryDirectory(prefix="eggvision-bench-") as tmp:
        tmp_path = Path(tmp)
        _safe_extract(package_zip, tmp_path)
        package_dir = find_package_dir(tmp_path)
        try:
            clf = EggClassifier.load(os.fspath(package_dir), variant=variant, threads=threads)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            raise AgentError(422, f"Paquet inutilisable : {exc}") from exc
        result: dict[str, Any] = dict(clf.benchmark(n_columns))
        result["parity"] = clf.verify_reference(os.fspath(package_dir))
    result["device"] = system.device_info()
    result["onnxruntime"] = ort_version
    return result
