from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from eggvision_shared import constants as c
from eggvision_shared.formats import (
    CameraSettings,
    LotManifest,
    LotWriter,
    lot_file_list,
    mark_interrupted_lots,
    read_image,
    read_metadata,
    sha256_file,
)
from eggvision_shared.preprocess import (
    PreprocessConfig,
    extract_crop,
    preprocess_view,
    preprocess_views,
    select_view_indices,
)
from eggvision_shared.synth import SceneConfig, SyntheticScene


def test_lot_types():
    assert c.parse_lot_type("declasse:felure") == ("declasse", "felure")
    assert c.prelabel_for_lot_type("viable") == ("viable", None)
    assert c.prelabel_for_lot_type("declasse") == ("non_viable", None)
    assert c.prelabel_for_lot_type("declasse:sang") == ("non_viable", "sang")
    assert c.prelabel_for_lot_type("production") == (None, None)
    for bad in ("autre", "viable:felure", "declasse:inconnu"):
        with pytest.raises(ValueError):
            c.parse_lot_type(bad)


def test_preprocess_is_deterministic_and_normalised():
    rng = np.random.default_rng(0)
    view = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    a, b = preprocess_view(view), preprocess_view(view.copy())
    assert a.shape == (3, 224, 224) and a.dtype == np.float32 and np.array_equal(a, b)
    # canal 0 du tenseur = rouge = canal 2 de l'image BGR
    cfg = PreprocessConfig()
    expected = (view[:, :, 2].astype(np.float32) / 255 - cfg.mean[0]) / cfg.std[0]
    assert np.allclose(a[0], expected, atol=1e-6)
    assert preprocess_views([view, view]).shape == (2, 3, 224, 224)
    assert preprocess_view(rng.integers(0, 256, (300, 280, 3), dtype=np.uint8)).shape == (3, 224, 224)
    assert PreprocessConfig.from_dict(cfg.to_dict()) == cfg


def test_extract_crop_centres_egg_and_handles_borders():
    frame = np.zeros((400, 600, 3), np.uint8)
    frame[190:210, 290:310] = 255
    crop = extract_crop(frame, 300, 200, 100, 80, out_size=224)
    assert crop.shape == (224, 224, 3)
    ys, xs = np.nonzero(crop[:, :, 0] > 128)
    assert abs(xs.mean() - 111.5) < 1.5 and abs(ys.mean() - 111.5) < 1.5
    assert extract_crop(frame, 5, 5, 100, 80).shape == (224, 224, 3)


def test_select_view_indices():
    assert select_view_indices(5, 12) == [0, 1, 2, 3, 4]
    idx = select_view_indices(47, 12)
    assert len(idx) == 12 and idx[0] == 0 and idx[-1] == 46 and idx == sorted(set(idx))
    assert select_view_indices(0, 12) == []


def test_lot_writer_roundtrip_and_interruption(tmp_path):
    lots = tmp_path / "lots"
    manifest = LotManifest(lot_id="a", name="A", type="declasse:plume", camera=CameraSettings(fps=30))
    img = np.full((48, 64, 3), 120, np.uint8)
    with LotWriter(lots / "a", manifest) as w:
        for i in range(5):
            w.write_frame(img, 1_000_000_000 + i * 33_333_333, seq=i)
        w.count_dropped(1)
    done = LotManifest.load(lots / "a")
    assert done.status == "termine" and done.stats.frames_written == 5 and done.stats.frames_dropped == 1
    metas = read_metadata(lots / "a")
    assert [m.index for m in metas] == list(range(5)) and metas[0].encoder_count is None
    assert metas[2].sha256 == sha256_file(lots / "a" / metas[2].file)
    assert read_image(lots / "a" / metas[0].file).shape == (48, 64, 3)
    files = lot_file_list(lots / "a")
    assert {f["path"] for f in files} >= {"lot.json", "metadata.jsonl", "frames/000004.jpg"} and all(f["sha256"] for f in files)

    # Lot abandonné en cours d'écriture (coupure) : dernière ligne tronquée, statut resté « en_cours ».
    w2 = LotWriter(lots / "b", LotManifest(lot_id="b", name="B", type="viable"))
    for i in range(20):
        w2.write_frame(img, i * 33_333_333, seq=i)
    w2._meta.flush()
    with open(lots / "b" / "metadata.jsonl", "a", encoding="utf-8") as f:
        f.write('{"index": 20, "seq": 2')
    assert mark_interrupted_lots(lots) == ["b"]
    interrupted = LotManifest.load(lots / "b")
    assert interrupted.status == "interrompu" and interrupted.stats.frames_written == 20
    assert mark_interrupted_lots(lots) == []


def test_synthetic_scene_is_deterministic_and_consistent():
    cfg = SceneConfig.scaled(0.25, lot_type="declasse:sang", seed=5)
    a, b = SyntheticScene(cfg), SyntheticScene(cfg)
    img_a, truth_a = a.frame(40)
    img_b, _ = b.frame(40)
    assert img_a.shape == (cfg.height, cfg.width, 3) and np.array_equal(img_a, img_b)
    assert truth_a and all(t.defect == "sang" and not t.viable for t in truth_a)
    assert {t.lane for t in truth_a} == set(range(6))
    # Les œufs sont nettement plus clairs que les rouleaux.
    t = next(t for t in truth_a if t.fully_visible)
    assert img_a[int(t.cy), int(t.cx)].mean() > 110 and img_a[int(t.cy), int(t.cx - a.cfg.lane_width_px / 2)].mean() < 90
    # Chaque œuf d'une colonne est vu en entier sur assez d'images pour 12 vues.
    n = a.frames_for_columns(2)
    seen: dict[int, int] = {}
    for i in range(n):
        for tr in a.truth_at(a.frame_time(i)):
            if tr.fully_visible and tr.column < 2:
                seen[tr.egg_id] = seen.get(tr.egg_id, 0) + 1
    assert len(seen) == 12 and min(seen.values()) >= 20
    prod = SyntheticScene(SceneConfig.scaled(0.25, lot_type="production", defect_rate=0.3, seed=1))
    eggs = [e for col in range(40) for e in prod.column(col)]
    rate = sum(not e.viable for e in eggs) / len(eggs)
    assert 0.2 < rate < 0.4
    assert len(a.calibration()["lanes"]) == 6


def test_runtime_package_python_and_cli(tmp_path):
    pytest.importorskip("torch")
    from eggvision.testing import make_dummy_package
    from eggvision_shared.runtime import EggClassifier

    pkg = make_dummy_package(tmp_path / "pkg", n_views=4, input_size=32)
    clf = EggClassifier.load(str(pkg))
    red = np.zeros((32, 32, 3), np.uint8)
    red[:, :, 2] = 255
    grey = np.full((32, 32, 3), 128, np.uint8)
    result = clf.predict_column([[grey] * 4, [grey, red, grey, grey], None, [grey] * 2, [], [red] * 4])
    assert result.decisions == [0, 1, 0, 0, 0, 1]
    assert result.scores[2] is None and result.scores[4] is None and result.n_views == [4, 4, 0, 2, 0, 4]
    assert result.view_scores[1][1] > 0.9 > 0.5 > result.view_scores[1][0]
    assert result.latency_ms > 0
    bench = clf.benchmark(3, warmup=1)
    assert bench["n_columns"] == 3 and bench["latency_ms"]["mean"] > 0

    # Le paquet est autonome : infer.py tourne hors du projet, depuis son propre dossier.
    out = subprocess.run([sys.executable, "infer.py", "--bench", "2"], cwd=pkg, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["n_lanes"] == 6
    col = tmp_path / "col"
    import cv2

    for lane, img in ((0, grey), (1, red)):
        (col / f"lane_{lane}").mkdir(parents=True)
        for v in range(4):
            cv2.imwrite(str(col / f"lane_{lane}" / f"{v:02d}.png"), img)
    out = subprocess.run([sys.executable, str(pkg / "infer.py"), "--column", str(col)], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["decisions"] == [0, 1, 0, 0, 0, 0]
