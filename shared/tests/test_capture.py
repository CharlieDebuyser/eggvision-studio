"""Tests du moteur d'enregistrement commun (``eggvision_shared.capture``)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from eggvision_shared.capture import (
    STOP_DISK_LOW,
    CaptureRecorder,
    DiskFullError,
    PacedLoop,
    SyntheticFrameSource,
    gray_world_gains,
    nominal_seq,
    settings_lut,
)
from eggvision_shared.formats import (
    CameraSettings,
    LotManifest,
    lot_file_list,
    read_image,
    read_metadata,
    sha256_file,
)


def _manifest(lot_id: str = "lot-test") -> LotManifest:
    return LotManifest(lot_id=lot_id, name="Essai", type="viable", width=160, height=120, source={"kind": "simulated"})


def _frame(i: int) -> np.ndarray:
    img = np.full((120, 160, 3), (i * 7) % 255, np.uint8)
    img[10:30, 10 + (i % 50) : 40 + (i % 50)] = 255
    return img


def test_recorder_writes_conformant_lot(tmp_path):
    lot_dir = tmp_path / "lot-test"
    rec = CaptureRecorder(lot_dir, _manifest(), queue_size=64, n_writers=3).start()
    period = 33_333_333
    for i in range(60):
        assert rec.submit(_frame(i), 1_000_000_000 + i * period, system_ts=1e9 + i / 30)
    manifest = rec.stop()
    assert manifest.status == "termine"
    assert manifest.stats.frames_written == 60 and manifest.stats.frames_dropped == 0
    assert manifest.stats.measured_fps == pytest.approx(30.0, rel=1e-3)
    metas = read_metadata(lot_dir)
    assert [m.index for m in metas] == list(range(60))
    assert [m.seq for m in metas] == list(range(60))  # ordre conservé malgré 3 fils d'écriture
    assert all(m.encoder_count is None for m in metas)
    assert metas[5].sensor_ts_ns == 1_000_000_000 + 5 * period
    files = lot_file_list(lot_dir)
    assert len(files) == 62
    f = next(x for x in files if x["path"] == "frames/000010.jpg")
    assert f["sha256"] == sha256_file(lot_dir / f["path"])
    assert read_image(lot_dir / "frames/000010.jpg").shape == (120, 160, 3)
    stats = rec.stats()
    assert stats["frames_written"] == 60 and stats["bytes_written"] > 0 and not stats["running"]
    # idempotent
    assert rec.stop().status == "termine"
    assert not rec.submit(_frame(0), 0)


def test_full_queue_drops_without_blocking_producer(tmp_path):
    """Écrivain ralenti : les images en trop sont perdues, le producteur n'attend jamais."""
    rec = CaptureRecorder(tmp_path / "lot", _manifest(), queue_size=4, n_writers=1, write_delay_s=0.05).start()
    accepted, worst = 0, 0.0
    for i in range(80):
        t = time.perf_counter()
        ok = rec.submit(_frame(i), i * 10_000_000, seq=1000 + i)
        worst = max(worst, time.perf_counter() - t)
        accepted += ok
    assert worst < 0.02, f"submit a bloqué {worst * 1000:.1f} ms"
    live = rec.stats()
    assert live["frames_dropped_queue"] == 80 - accepted and live["frames_dropped"] > 50
    assert live["queue_max_fill"] == 1.0
    manifest = rec.stop()
    assert manifest.stats.frames_written == accepted
    assert manifest.stats.frames_dropped == 80 - accepted
    metas = read_metadata(tmp_path / "lot")
    assert len(metas) == accepted
    # seq relatif au début du lot, avec des trous là où des images ont été perdues
    assert metas[0].seq == 0 and metas[-1].seq <= 79 and len({m.seq for m in metas}) == accepted


def test_sequence_gaps_count_as_upstream_drops(tmp_path):
    rec = CaptureRecorder(tmp_path / "lot", _manifest(), queue_size=50).start()
    for seq in (10, 11, 12, 15, 16, 20):  # 2 + 3 images perdues en amont
        rec.submit(_frame(seq), seq * 33_333_333, seq=seq)
    m = rec.stop()
    assert m.stats.frames_written == 6 and m.stats.frames_dropped == 5
    assert rec.stats()["frames_dropped_upstream"] == 5
    assert [x.seq for x in read_metadata(tmp_path / "lot")] == [0, 1, 2, 5, 6, 10]


def test_disk_threshold_auto_stop(tmp_path):
    free = {"bytes": 10_000_000_000}
    finished = []
    rec = CaptureRecorder(
        tmp_path / "lot",
        _manifest(),
        min_free_bytes=1_000_000_000,
        disk_check_interval=0.02,
        disk_free_fn=lambda: free["bytes"],
        on_finished=lambda m, reason: finished.append((m.status, reason)),
    ).start()
    for i in range(10):
        assert rec.submit(_frame(i), i * 33_333_333)
    free["bytes"] = 500_000_000  # le disque se remplit
    assert rec.wait(5.0)
    assert rec.stop_reason == STOP_DISK_LOW and finished == [("termine", STOP_DISK_LOW)]
    assert not rec.submit(_frame(99), 99)
    assert rec.manifest.stats.frames_written == 10
    assert LotManifest.load(tmp_path / "lot").status == "termine"


def test_disk_threshold_refuses_start(tmp_path):
    rec = CaptureRecorder(tmp_path / "lot", _manifest(), min_free_bytes=10, disk_free_fn=lambda: 5)
    with pytest.raises(DiskFullError):
        rec.start()


def test_shutdown_marks_interrupted(tmp_path):
    rec = CaptureRecorder(tmp_path / "lot", _manifest()).start()
    rec.submit(_frame(0), 0)
    assert rec.stop(reason="shutdown").status == "interrompu"


def test_synthetic_source_and_settings(tmp_path):
    src = SyntheticFrameSource(scale=0.125, fps=30, cycle_s=0.5)
    src.prepare()
    assert src.error is None and len(src.frames) == 15
    w, h = src.size
    neutral = CameraSettings()
    assert settings_lut(CameraSettings(red_gain=2.0, blue_gain=1.8)) is None
    a = src.frame(3, neutral)
    bright = src.frame(3, CameraSettings(exposure_us=1000))
    assert bright.mean() > a.mean() * 1.3
    roi = src.frame(3, CameraSettings(roi=(10, 8, 64, 48)))
    assert roi.shape == (48, 64, 3) and roi.flags["C_CONTIGUOUS"]
    # calibration : les gains trouvés neutralisent la dominante
    red, blue = gray_world_gains(src.frame(0, neutral), neutral)
    calibrated = CameraSettings(red_gain=red, blue_gain=blue)
    b, g, r = src.frame(0, calibrated).reshape(-1, 3).mean(axis=0)
    b0, g0, r0 = src.frame(0, neutral).reshape(-1, 3).mean(axis=0)
    assert abs(r - g) < abs(r0 - g0) and abs(b - g) < abs(b0 - g0)
    # relecture d'un dossier
    rec = CaptureRecorder(tmp_path / "lot", _manifest()).start()
    for i in range(5):
        rec.submit(src.frame(i, neutral), i)
    rec.stop()
    replay = SyntheticFrameSource(folder=tmp_path / "lot")
    replay.prepare()
    assert len(replay.frames) == 5 and replay.size == (w, h)


def test_paced_loop_cadence_and_skips():
    seen: list[tuple[int, int]] = []

    def fn(seq: int, ts: int) -> None:
        seen.append((seq, ts))
        if seq == 10:
            time.sleep(0.12)  # retard : quelques créneaux sautés

    loop = PacedLoop(fn, fps=100)
    loop.start()
    time.sleep(0.5)
    loop.stop()
    seqs = [s for s, _ in seen]
    assert seqs == sorted(seqs) and len(seqs) > 25
    assert loop.skipped >= 6  # 12 périodes de retard − 4 tampons
    ts = [t for _, t in seen]
    assert nominal_seq(ts[-1], ts[0], 100) == seqs[-1] - seqs[0]
