# -*- mode: python ; coding: utf-8 -*-
"""Spécification PyInstaller d'EggVision Studio (Windows et macOS).

Construire avec :  uv run python scripts/build_desktop.py
(le script compile d'abord le frontend et vérifie la présence des poids pré-entraînés).
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
PKG = ROOT / "desktop" / "src" / "eggvision"
SHARED = ROOT / "shared" / "src" / "eggvision_shared"

datas = [
    (str(PKG / "web"), "eggvision/web"),
    (str(PKG / "assets"), "eggvision/assets"),
    # Sources copiées telles quelles dans les paquets de déploiement Raspberry Pi.
    (str(SHARED / "preprocess.py"), "eggvision_shared"),
    (str(SHARED / "runtime.py"), "eggvision_shared"),
]
binaries = []
hiddenimports = collect_submodules("eggvision") + collect_submodules("eggvision_shared") + [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
]
# torchvision doit être collecté en entier (extension _C : sinon « operator torchvision::nms does not exist »).
for package in ("torchvision", "timm", "onnxruntime", "onnx", "onnxscript", "imageio_ffmpeg", "zeroconf", "webview"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
# Métadonnées lues par l'écran Diagnostic (versions).
for dist in ("torch", "torchvision", "opencv-python-headless", "fastapi", "sqlalchemy", "numpy", "timm", "onnx", "onnxruntime", "pywebview"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

a = Analysis(
    [str(Path(SPECPATH) / "launcher.py")],
    pathex=[str(ROOT / "desktop" / "src"), str(ROOT / "shared" / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "eggvision_agent"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EggVisionStudio",
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="EggVisionStudio")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="EggVision Studio.app",
        bundle_identifier="fr.noxiom.eggvision",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "EggVision Studio utilise la caméra USB pour la prise de vue des œufs.",
            "NSLocalNetworkUsageDescription": "EggVision Studio communique avec l'agent de capture Raspberry Pi sur le réseau local.",
            "NSBonjourServices": ["_eggvision._tcp"],
        },
    )
