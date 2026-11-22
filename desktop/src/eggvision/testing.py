"""Aides pour les tests automatisés (aucune dépendance à un module métier)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from eggvision_shared import runtime as shared_runtime
from eggvision_shared import source_file
from eggvision_shared.preprocess import PreprocessConfig


def make_dummy_package(
    package_dir: str | Path,
    *,
    n_views: int = 12,
    n_lanes: int = 6,
    input_size: int = 64,
    threshold: float = 0.5,
    seed: int = 0,
) -> Path:
    """Fabrique un paquet de déploiement minimal mais conforme (modèle ONNX non entraîné).

    Le modèle factice répond « défaut » quand la vue contient beaucoup de rouge :
    son score dépend réellement des images, ce qui permet des tests de bout en bout
    (simulation, banc de mesure) sans entraînement. Respecte le contrat décrit dans
    :mod:`eggvision_shared.runtime`.
    """
    import torch
    from torch import nn

    class Dummy(nn.Module):
        def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # views : [B, V, 3, S, S] normalisé ImageNet (canal 0 = rouge)
            red = views[:, :, 0].mean(dim=(2, 3))
            green = views[:, :, 1].mean(dim=(2, 3))
            view_scores = torch.sigmoid((red - green) * 6.0 - 1.0)
            egg_scores = view_scores.max(dim=1).values
            return egg_scores, view_scores

    torch.manual_seed(seed)
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(2, n_views, 3, input_size, input_size)
    torch.onnx.export(
        Dummy(),
        (example,),
        str(package_dir / "model_fp32.onnx"),
        input_names=["views"],
        output_names=["egg_scores", "view_scores"],
        dynamic_axes={"views": {0: "batch", 1: "n_views"}, "egg_scores": {0: "batch"}, "view_scores": {0: "batch", 1: "n_views"}},
        opset_version=17,
        dynamo=False,
    )
    config: dict[str, Any] = {
        "format_version": shared_runtime.PACKAGE_FORMAT_VERSION,
        "model_id": "factice",
        "model_name": "Modèle factice (tests)",
        "version": "0",
        "engine": "onnxruntime",
        "model_files": {"fp32": "model_fp32.onnx"},
        "default_variant": "fp32",
        "preprocess": PreprocessConfig(input_size=input_size).to_dict(),
        "n_views": n_views,
        "n_lanes": n_lanes,
        "aggregation": "max",
        "threshold": threshold,
        "scene": None,
        "metrics": {},
    }
    (package_dir / shared_runtime.CONFIG_FILENAME).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(source_file("preprocess.py"), package_dir / "preprocess.py")
    shutil.copyfile(source_file("runtime.py"), package_dir / "infer.py")
    return package_dir


def wait_job(job_id: str, timeout: float = 300.0, expect: str = "done") -> dict[str, Any]:
    """Attend la fin d'une tâche et retourne son état ; échoue si son statut final n'est pas ``expect``."""
    from . import state

    job = state.jobs.wait(job_id, timeout)
    assert job["status"] == expect, f"tâche {job['kind']} : {job['status']} — {job['error']}"
    return job
