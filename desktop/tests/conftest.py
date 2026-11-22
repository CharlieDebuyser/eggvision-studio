"""Fixtures communes aux tests de l'application de bureau.

* ``client``            : TestClient FastAPI, sans projet ouvert, données d'application isolées.
* ``demo_template``     : (session) projet synthétique SANS images complètes — pistes, vues, labels (rapide).
* ``project``           : copie fraîche de ``demo_template``, ouverte dans l'application (modifiable librement).
* ``frames_template``   : (session) projet avec de vraies images de lots et sans pistes (pour la détection).
* ``frames_project``    : copie fraîche de ``frames_template``, ouverte dans l'application.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_app_data(tmp_path_factory: pytest.TempPathFactory):
    import os

    data = tmp_path_factory.mktemp("appdata")
    os.environ["EGGVISION_DATA_DIR"] = str(data)
    yield


@pytest.fixture(scope="session")
def demo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """≈ 216 œufs : 3 sessions × (1 lot viable de 4 colonnes + 6 lots déclassés de 1 colonne) + 1 lot production."""
    from eggvision.demo import build_demo_project

    root = tmp_path_factory.mktemp("demo") / "projet"
    project = build_demo_project(root, viable_columns=4, defect_columns=1, production_columns=3, with_frames=False, scale=0.5)
    project.close()
    return root


@pytest.fixture(scope="session")
def frames_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Deux lots bruts avec images (728 × 544) : un viable (5 colonnes) et un ``declasse:souillure`` (3 colonnes)."""
    from eggvision.demo import add_synthetic_lot
    from eggvision.project import Project
    from eggvision_shared.synth import SceneConfig, SyntheticScene

    root = tmp_path_factory.mktemp("frames") / "projet"
    project = Project.create(root, "Projet images")
    project.update_config({"scene": SyntheticScene(SceneConfig.scaled(0.5)).calibration()})
    add_synthetic_lot(project, "Viables", "viable", 5, seed=1, with_tracks=False, lot_id="lot-viable")
    add_synthetic_lot(project, "Souillures", "declasse:souillure", 3, seed=2, with_tracks=False, lot_id="lot-souillure")
    project.close()
    return root


def _copy(template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "projet"
    shutil.copytree(template, target)
    return target


@pytest.fixture()
def client():
    from eggvision import state
    from eggvision.app import create_app
    from fastapi.testclient import TestClient

    state.projects.close()
    with TestClient(create_app()) as c:
        yield c
    state.jobs.cancel_all(wait=10)
    state.projects.close()


@pytest.fixture()
def project(client, demo_template: Path, tmp_path: Path):
    from eggvision import state

    yield state.projects.open(_copy(demo_template, tmp_path))


@pytest.fixture()
def frames_project(client, frames_template: Path, tmp_path: Path):
    from eggvision import state

    yield state.projects.open(_copy(frames_template, tmp_path))
