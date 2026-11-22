#!/usr/bin/env python3
"""Télécharge UNE FOIS les poids pré-entraînés (ImageNet) des architectures prises en charge.

À lancer sur une machine de développement ou de build connectée à Internet ::

    uv run python scripts/fetch_pretrained.py            # les trois architectures
    uv run python scripts/fetch_pretrained.py mobilenetv3_small_100

Les poids du backbone (sans la tête de classification ImageNet) sont écrits dans
``desktop/src/eggvision/assets/pretrained/<arch>.safetensors``, accompagnés d'un
fichier ``<arch>.json`` (source, empreinte SHA-256, nombre de paramètres).

À l'exécution, l'application ne lit QUE ces fichiers locaux (aucun accès réseau) :
voir ``eggvision.ml.network.load_pretrained``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ARCHS = ("mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_lite0")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "desktop" / "src" / "eggvision" / "assets" / "pretrained"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(arch: str, out_dir: Path) -> dict:
    import timm
    from safetensors.torch import save_file

    model = timm.create_model(arch, pretrained=True, num_classes=0)
    state = {k: v.detach().contiguous().clone() for k, v in model.state_dict().items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arch}.safetensors"
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    save_file(state, str(path), metadata={"arch": arch, "source": str(cfg.get("hf_hub_id") or cfg.get("url") or "")})
    info = {
        "arch": arch,
        "timm_version": timm.__version__,
        "source": cfg.get("hf_hub_id") or cfg.get("url") or "",
        "tag": cfg.get("tag", ""),
        "num_features": int(model.num_features),
        "n_params": int(sum(v.numel() for v in state.values())),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / f"{arch}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archs", nargs="*", default=list(ARCHS), help="Architectures timm (défaut : toutes)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Dossier de sortie")
    args = parser.parse_args(argv)
    for arch in args.archs:
        if arch not in ARCHS:
            print(f"Architecture non prise en charge : {arch}", file=sys.stderr)
            return 2
        info = fetch(arch, args.out)
        print(f"{arch}: {info['n_params']:,} paramètres, {info['size_bytes'] / 1e6:.1f} Mo → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
