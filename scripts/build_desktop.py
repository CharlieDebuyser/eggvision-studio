#!/usr/bin/env python3
"""Construit l'exécutable d'EggVision Studio pour la plateforme courante (Windows ou macOS).

    uv sync --group build
    uv run python scripts/build_desktop.py            # frontend + PyInstaller (+ zip / dmg)
    uv run python scripts/build_desktop.py --skip-frontend

Résultat dans ``dist/`` :
  * Windows : ``dist/EggVisionStudio/EggVisionStudio.exe`` + ``dist/EggVisionStudio-windows-<arch>.zip``
  * macOS   : ``dist/EggVision Studio.app``             + ``dist/EggVisionStudio-macos-<arch>.dmg`` (ou .zip)

L'exécutable est autonome (Python, PyTorch, ONNX Runtime, frontend, poids pré-entraînés) et
fonctionne sans connexion Internet. PyInstaller ne fait pas de compilation croisée : lancer ce
script sur chaque plateforme cible (voir ``.github/workflows/build.yml``).
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "desktop" / "frontend"
WEB = ROOT / "desktop" / "src" / "eggvision" / "web"
PRETRAINED = ROOT / "desktop" / "src" / "eggvision" / "assets" / "pretrained"
SPEC = ROOT / "desktop" / "packaging" / "eggvision.spec"
DIST = ROOT / "dist"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("→", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32" and cmd[0] in ("npm", "npx")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-frontend", action="store_true", help="Ne pas recompiler le frontend")
    parser.add_argument("--no-archive", action="store_true", help="Ne pas produire le zip / dmg final")
    args = parser.parse_args()

    if not args.skip_frontend:
        run(["npm", "ci"], cwd=FRONTEND)
        run(["npm", "run", "build"], cwd=FRONTEND)
    if not (WEB / "index.html").is_file():
        print("ERREUR : frontend non compilé (desktop/src/eggvision/web/index.html absent).")
        return 1
    weights = list(PRETRAINED.glob("*.safetensors")) if PRETRAINED.is_dir() else []
    if not weights:
        print("ATTENTION : aucun poids pré-entraîné dans assets/pretrained/. Lancer d'abord\n"
              "  uv run python scripts/fetch_pretrained.py\n"
              "sinon les modèles partiront d'une initialisation aléatoire.")

    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST),
         "--workpath", str(ROOT / "build"), str(SPEC)])

    if args.no_archive:
        return 0
    arch = platform.machine().lower() or "x64"
    if sys.platform == "darwin":
        app = DIST / "EggVision Studio.app"
        dmg = DIST / f"EggVisionStudio-macos-{arch}.dmg"
        dmg.unlink(missing_ok=True)
        try:
            run(["hdiutil", "create", "-volname", "EggVision Studio", "-srcfolder", str(app), "-ov", "-format", "UDZO", str(dmg)])
            print(f"Image disque : {dmg}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            archive = shutil.make_archive(str(DIST / f"EggVisionStudio-macos-{arch}"), "zip", DIST, "EggVision Studio.app")
            print(f"Archive : {archive}")
    else:
        system = "windows" if sys.platform == "win32" else sys.platform
        archive = shutil.make_archive(str(DIST / f"EggVisionStudio-{system}-{arch}"), "zip", DIST, "EggVisionStudio")
        print(f"Archive : {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
