"""Aides pour les tests automatisés du module d'apprentissage.

``trained_template`` entraîne UNE fois par session de tests un petit modèle
(MobileNetV3-Small, 96 px, 6 vues) sur une copie du projet de démonstration
et renvoie le dossier du projet obtenu ; chaque test en fait ensuite une copie.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

TINY_HPARAMS: dict[str, Any] = {
    "arch": "mobilenetv3_small_100",
    "input_size": 96,
    "n_views": 6,
    "epochs": 6,
    "batch_size": 8,
    "lr": 2e-3,
    "freeze_epochs": 1,
    "patience": 10,
    "augment": "light",
    "warmup_epochs": 0.5,
    "seed": 7,
}

_lock = threading.Lock()
_cache: dict[str, Path] = {}


def trained_template(demo_template: Path, workdir: Path) -> Path:
    """Projet (copie de ``demo_template``) avec un découpage par lot et un modèle entraîné."""
    from .. import state
    from ..project import Project
    from . import splits
    from .training import start_training

    key = str(demo_template)
    with _lock:
        if key in _cache and _cache[key].exists():
            return _cache[key]
        root = Path(workdir) / "projet_entraine"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(demo_template, root)
        project = Project(root)
        try:
            split = splits.create_split(project, "Découpage de test", splits.SplitParams(group_by="lot", seed=1))
            out = start_training(project, split["id"], None, TINY_HPARAMS, name="Modèle de test")
            job = state.jobs.wait(out["job_id"], 900)
            if job["status"] != "done":
                raise RuntimeError(f"Entraînement de test en échec : {job['error']}")
        finally:
            project.close()
        _cache[key] = root
        return root


def copy_project(template: Path, tmp_path: Path) -> Path:
    target = Path(tmp_path) / "projet"
    shutil.copytree(template, target)
    return target
