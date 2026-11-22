"""Entraînement du classifieur multi-vues (tâche ``kind="training"``, groupe ``"gpu"``).

Déroulement d'une tâche :

1. chargement des œufs utilisables du découpage (labels de l'œuf, cible 1 = non viable) ;
2. modèle : poids pré-entraînés locaux, ou reprise de ``last.pt`` (même modèle,
   entraînement interrompu), ou initialisation depuis ``best.pt`` d'un modèle
   parent (prolongation → nouveau modèle, ``parent_id``) ;
3. boucle d'époques : AdamW + échauffement linéaire puis décroissance cosinus,
   gel du backbone pendant les ``freeze_epochs`` premières époques (seule la tête
   apprend, statistiques BatchNorm figées), puis dégel avec un taux d'apprentissage
   réduit (``backbone_lr_mult``) ; déséquilibre traité par ``pos_weight`` ou par un
   échantillonneur pondéré ; précision mixte sur CUDA ; ``ctx.check_cancelled()`` à
   chaque lot ; courbes publiées en direct et enregistrées dans ``Model.history`` ;
   points de contrôle ``last.pt`` (chaque époque) et ``best.pt`` (meilleure
   validation) ; arrêt anticipé sur ``monitor`` ;
4. évaluation automatique : prédictions de toutes les pistes du découpage, rapport.

Préréglages (``PRESETS``) :

``rapide``    MobileNetV3-Small, 160 px, 8 vues, 8 époques, augmentations légères —
              quelques minutes : vérifier qu'un jeu de données « apprend ».
``int8``      « standard » avec EfficientNet-Lite0, pour un déploiement INT8.
``standard``  MobileNetV3-Small, 224 px, 12 vues, 30 époques (arrêt anticipé,
              patience 6), augmentations standard — modèle à déployer. Small est
              retenu par défaut pour tenir la cadence sur le Raspberry Pi 5
              (6 œufs × 12 vues par colonne) en FP32. MobileNetV3 se dégrade en INT8
              (quantification post-entraînement) : pour un déploiement INT8, choisir
              ``efficientnet_lite0`` (conçu pour la quantification ; en INT8, latence
              proche de Small en FP32). Large reste disponible en réglage personnalisé.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .. import state
from ..db import models as m
from ..device import detect_device
from ..jobs import JobCancelled, JobContext
from ..project import Project
from . import metrics as mt
from .dataset import (
    AugmentConfig,
    EggViewsDataset,
    ViewCache,
    class_balanced_weights,
    count_records,
    load_records,
)
from .evaluation import predict_records, refresh_report, write_predictions
from .network import AGGREGATIONS, ARCHITECTURES
from .store import (
    get_model_row,
    load_checkpoint,
    model_dir,
    new_model_id,
    next_model_number,
    read_report,
    report_path,
    save_checkpoint,
    write_json,
)

log = logging.getLogger(__name__)

MONITORS = ("val_auc", "val_ap", "val_loss")
IMBALANCE = ("pos_weight", "sampler", "none")


class TrainingError(ValueError):
    """Erreur de paramétrage présentable à l'utilisateur."""


@dataclass
class TrainParams:
    """Hyperparamètres (valeurs par défaut = préréglage « standard »)."""

    arch: str = "mobilenetv3_small_100"
    aggregation: str = "max"  # max | attention
    pretrained: bool = True
    input_size: int = 224
    n_views: int = 12
    epochs: int = 30
    batch_size: int = 8  # en œufs (× n_views images)
    lr: float = 5e-4  # tête ; backbone : lr × backbone_lr_mult
    backbone_lr_mult: float = 0.3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.02
    freeze_epochs: int = 2
    dropout: float = 0.2
    imbalance: str = "pos_weight"  # pos_weight | sampler | none
    augment: str = "standard"  # none | light | standard
    monitor: str = "val_auc"  # val_auc | val_ap | val_loss
    patience: int = 6
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu
    amp: bool = True  # précision mixte (CUDA uniquement)
    num_workers: int = 0
    cache_mb: int = 512
    reviewed_only: bool = False
    grad_clip: float = 5.0

    def validate(self) -> TrainParams:
        if self.arch not in ARCHITECTURES:
            raise TrainingError(f"Architecture inconnue : {self.arch}")
        if self.aggregation not in AGGREGATIONS:
            raise TrainingError(f"Agrégation inconnue : {self.aggregation}")
        if self.imbalance not in IMBALANCE:
            raise TrainingError(f"Gestion du déséquilibre inconnue : {self.imbalance}")
        if self.monitor not in MONITORS:
            raise TrainingError(f"Métrique d'arrêt anticipé inconnue : {self.monitor}")
        if self.device not in ("auto", "cuda", "mps", "cpu"):
            raise TrainingError(f"Appareil inconnu : {self.device}")
        AugmentConfig.preset(self.augment)
        checks = [
            (32 <= self.input_size <= 512, "La taille d'entrée doit être comprise entre 32 et 512 px."),
            (1 <= self.n_views <= 64, "Le nombre de vues doit être compris entre 1 et 64."),
            (1 <= self.epochs <= 1000, "Le nombre d'époques doit être compris entre 1 et 1000."),
            (1 <= self.batch_size <= 512, "La taille de lot doit être comprise entre 1 et 512."),
            (0 < self.lr < 1, "Taux d'apprentissage invalide."),
            (self.patience >= 1, "La patience doit être ≥ 1."),
            (0 <= self.num_workers <= 32, "Nombre de processus de chargement invalide."),
            (0 <= self.dropout < 1, "Dropout invalide."),
        ]
        for ok, msg in checks:
            if not ok:
                raise TrainingError(msg)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TrainParams:
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names - {"preset"}
        if unknown:
            raise TrainingError("Hyperparamètre(s) inconnu(s) : " + ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in (d or {}).items() if k in names}).validate()


PRESETS: dict[str, dict[str, Any]] = {
    "rapide": {
        "label": "Rapide",
        "description": "MobileNetV3-Small, 160 px, 8 vues, 8 époques, augmentations légères : vérifier en quelques minutes que le jeu de données apprend.",
        "hparams": {"arch": "mobilenetv3_small_100", "input_size": 160, "n_views": 8, "epochs": 8, "batch_size": 8, "lr": 1e-3, "freeze_epochs": 1, "patience": 3, "augment": "light", "warmup_epochs": 0.5},
    },
    "standard": {
        "label": "Standard",
        "description": "MobileNetV3-Small, 224 px, 12 vues, jusqu'à 30 époques avec arrêt anticipé, augmentations standard : modèle à déployer en FP32.",
        "hparams": {},
    },
    "int8": {
        "label": "Déploiement INT8",
        "description": "EfficientNet-Lite0 (conçu pour la quantification), sinon comme « standard » : à choisir pour déployer la variante INT8 sur le Pi.",
        "hparams": {"arch": "efficientnet_lite0"},
    },
}


def resolve_hparams(preset: str | None, overrides: dict[str, Any] | None) -> TrainParams:
    base: dict[str, Any] = {}
    if preset:
        if preset not in PRESETS:
            raise TrainingError(f"Préréglage inconnu : {preset}")
        base.update(PRESETS[preset]["hparams"])
    base.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return TrainParams.from_dict(base)


def presets_payload() -> list[dict[str, Any]]:
    defaults = TrainParams().to_dict()
    return [
        {"id": k, "label": v["label"], "description": v["description"], "hparams": {**defaults, **v["hparams"]}}
        for k, v in PRESETS.items()
    ]


# ---------------------------------------------------------------------------
# Lancement / reprise
# ---------------------------------------------------------------------------
def _split_for(session: Any, split_id: int) -> m.DatasetSplit:
    split = session.get(m.DatasetSplit, split_id)
    if split is None:
        raise LookupError("Découpage introuvable.")
    if not split.assignments:
        raise TrainingError("Ce découpage n'affecte aucun lot.")
    return split


def _default_name(hp: TrainParams, split: m.DatasetSplit, number: int) -> str:
    return f"#{number} {ARCHITECTURES[hp.arch]['label']} · {split.name} v{split.version}"


def start_training(
    project: Project,
    split_id: int,
    preset: str | None = "standard",
    overrides: dict[str, Any] | None = None,
    name: str | None = None,
) -> dict[str, str]:
    """Crée le modèle et lance la tâche d'entraînement ; retourne ``{"model_id", "job_id"}``."""
    hp = resolve_hparams(preset, overrides)
    with project.session() as s:
        split = _split_for(s, split_id)
        number = next_model_number(s)
        model_id = new_model_id(hp.arch)
        row = m.Model(
            id=model_id,
            name=(name or "").strip() or _default_name(hp, split, number),
            kind="classifier",
            arch=hp.arch,
            status="created",
            split_id=split.id,
            hparams={**hp.to_dict(), "preset": preset},
            n_views=hp.n_views,
            input_size=hp.input_size,
            threshold=0.5,
            extra={"number": number},
        )
        s.add(row)
    model_dir(project, model_id)
    return _submit(project, model_id, mode="new")


def resume_training(
    project: Project,
    model_id: str,
    mode: str = "continue",
    epochs: int | None = None,
    overrides: dict[str, Any] | None = None,
    name: str | None = None,
) -> dict[str, str]:
    """Reprend un entraînement.

    * ``continue`` : modèle interrompu (annulé / en échec) — reprise exacte depuis
      ``last.pt`` sur le même modèle ; ``epochs`` permet d'augmenter le total prévu.
    * ``extend`` : prolonge un modèle (souvent terminé) — crée un NOUVEAU modèle
      (``parent_id``) initialisé avec les meilleurs poids du parent, même découpage,
      ``epochs`` époques supplémentaires (défaut : moitié du parent), hyperparamètres
      du parent modifiables par ``overrides``.
    """
    if state.jobs and _active_job(model_id):
        raise TrainingError("Un entraînement est déjà en cours pour ce modèle.")
    with project.session() as s:
        row = get_model_row(s, model_id)
        if row.kind != "classifier":
            raise TrainingError("Seuls les classifieurs peuvent être repris.")
        if mode == "continue":
            if row.status not in ("cancelled", "failed"):
                raise TrainingError("Seul un entraînement interrompu (annulé ou en échec) peut être poursuivi ; utiliser « extend » pour prolonger.")
            if not (project.model_dir(model_id) / "last.pt").is_file():
                raise TrainingError("Aucun point de contrôle : l'entraînement s'est arrêté avant la fin de la première époque.")
            hp = TrainParams.from_dict({**(row.hparams or {}), **(overrides or {}), **({"epochs": epochs} if epochs else {})})
            if overrides and set(overrides) & {"arch", "aggregation", "input_size", "n_views"}:
                raise TrainingError("L'architecture, l'agrégation, la taille d'entrée et le nombre de vues ne peuvent pas changer en cours d'entraînement.")
            row.hparams = {**(row.hparams or {}), **hp.to_dict()}
            row.status = "created"
            s.flush()
            target_id = model_id
        elif mode == "extend":
            if not (project.model_dir(model_id) / "best.pt").is_file():
                raise TrainingError("Ce modèle n'a pas de poids enregistrés.")
            parent_hp = dict(row.hparams or {})
            extra_epochs = int(epochs or max(1, int(parent_hp.get("epochs", 10)) // 2))
            blocked = set(overrides or {}) & {"arch", "aggregation"}
            if blocked:
                raise TrainingError("L'architecture et l'agrégation d'un modèle prolongé sont celles du parent.")
            hp = TrainParams.from_dict({**parent_hp, "freeze_epochs": 0, "warmup_epochs": 0.0, **(overrides or {}), "epochs": extra_epochs})
            split = _split_for(s, row.split_id) if row.split_id else None
            if split is None:
                raise TrainingError("Le modèle parent n'a pas de découpage.")
            number = next_model_number(s)
            target_id = new_model_id(hp.arch)
            s.add(
                m.Model(
                    id=target_id,
                    name=(name or "").strip() or f"#{number} {_strip_number(row.name)} (prolongé)",
                    kind="classifier",
                    arch=hp.arch,
                    status="created",
                    split_id=row.split_id,
                    parent_id=row.id,
                    hparams={**hp.to_dict(), "preset": (row.hparams or {}).get("preset")},
                    n_views=hp.n_views,
                    input_size=hp.input_size,
                    threshold=row.threshold,
                    extra={"number": number},
                )
            )
        else:
            raise TrainingError("Mode de reprise inconnu : « continue » ou « extend ».")
    model_dir(project, target_id)
    return _submit(project, target_id, mode=mode)


def _strip_number(name: str) -> str:
    """« #4 Mon modèle » → « Mon modèle » (évite « #5 #4 … » pour un modèle prolongé)."""
    return re.sub(r"^#\d+\s+", "", name)


def _active_job(model_id: str) -> Any:
    from .store import active_job_for_model

    return active_job_for_model(model_id)


def _submit(project: Project, model_id: str, mode: str) -> dict[str, str]:
    with project.session() as s:
        row = get_model_row(s, model_id)
        title = {"new": "Entraînement", "continue": "Reprise de l'entraînement", "extend": "Prolongation"}[mode] + f" — {row.name}"
    job = state.jobs.submit(
        "training",
        title,
        lambda ctx: run_training(ctx, project, model_id, mode),
        params={"model_id": model_id, "mode": mode},
        project=project,
        group="gpu",
    )
    with project.session() as s:
        row = get_model_row(s, model_id)
        row.job_id = job.id
    return {"model_id": model_id, "job_id": job.id}


# ---------------------------------------------------------------------------
# Tâche
# ---------------------------------------------------------------------------
def resolve_device(pref: str):
    import torch

    kind = detect_device()["kind"] if pref == "auto" else pref
    if kind == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA demandé mais indisponible.")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise TrainingError("MPS demandé mais indisponible.")
    return torch.device(kind)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def labels_fingerprint(records: list) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(f"{r.track_id}:{r.subset}:{r.target}:{r.defect};".encode())
    return h.hexdigest()[:16]


def _lr_lambda(total_steps: int, warmup_steps: int, min_ratio: float):
    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def _set_status(project: Project, model_id: str, **fields_: Any) -> None:
    with project.session() as s:
        row = get_model_row(s, model_id)
        extra_updates = fields_.pop("extra_updates", None)
        for k, v in fields_.items():
            setattr(row, k, v)
        if extra_updates:
            row.extra = {**(row.extra or {}), **extra_updates}


def _score(entry: dict[str, Any], monitor: str) -> float:
    if monitor == "val_loss":
        v = entry.get("val_loss")
        return -float(v) if v is not None else -math.inf
    v = entry.get(monitor)
    return float(v) if v is not None else -math.inf


def run_training(ctx: JobContext, project: Project, model_id: str, mode: str) -> dict[str, Any]:
    import torch
    from eggvision_shared.preprocess import PreprocessConfig
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from .network import build_model
    from .store import classifier_from_checkpoint, forget_cached

    t_start = time.perf_counter()
    with project.session() as s:
        row = get_model_row(s, model_id)
        hp = TrainParams.from_dict({k: v for k, v in (row.hparams or {}).items() if k != "preset"})
        split = _split_for(s, row.split_id)
        assignments = dict(split.assignments)
        split_info = {"split_id": split.id, "split_name": split.name, "split_version": split.version, "group_by": (split.strategy or {}).get("group_by")}
        parent_id = row.parent_id
        prev_extra = dict(row.extra or {})
        row.status = "training"
        row.job_id = ctx.job_id
    warnings: list[str] = list(prev_extra.get("warnings", [])) if mode == "continue" else []
    mdir = model_dir(project, model_id)
    forget_cached(model_id)

    try:
        ctx.progress(0.0, "Chargement des données")
        records = load_records(project, assignments, reviewed_only=hp.reviewed_only)
        train = [r for r in records if r.subset == "train"]
        val = [r for r in records if r.subset == "val"]
        test = [r for r in records if r.subset == "test"]
        n_pos = sum(r.target for r in train)
        if not train or n_pos == 0 or n_pos == len(train):
            raise TrainingError("Le jeu d'entraînement doit contenir des œufs viables ET non viables.")
        if not val:
            warnings.append("Jeu de validation vide : sélection et arrêt anticipé sur la perte d'entraînement.")
        if not test:
            warnings.append("Jeu de test vide : pas d'évaluation finale.")
        counts = count_records(records)
        lots_by_subset: dict[str, list[str]] = {}
        for lot, sub in sorted(assignments.items()):
            lots_by_subset.setdefault(sub, []).append(lot)
        data_summary = {
            **split_info,
            "reviewed_only": hp.reviewed_only,
            "counts": counts,
            "lots": lots_by_subset,
            "n_eggs": len(records),
            "labels_fingerprint": labels_fingerprint(records),
        }
        ctx.log(f"Données : {len(train)} œufs d'entraînement ({n_pos} non viables), {len(val)} validation, {len(test)} test.")

        seed_everything(hp.seed)
        device = resolve_device(hp.device)
        use_amp = bool(hp.amp and device.type == "cuda")
        dev_info = {**detect_device(), "used": device.type, "amp": use_amp}
        ctx.log(f"Appareil : {device.type} ({dev_info.get('name')}){' — précision mixte' if use_amp else ''}")

        # -- modèle -----------------------------------------------------------
        last_ckpt: dict[str, Any] | None = None
        pretrained_info: dict[str, Any]
        if mode == "continue":
            last_ckpt = load_checkpoint(mdir / "last.pt")
            model = classifier_from_checkpoint(last_ckpt)
            pretrained_info = prev_extra.get("pretrained") or {}
            ctx.log(f"Reprise après l'époque {last_ckpt['epoch']}.")
        elif mode == "extend":
            parent = load_checkpoint(project.model_dir(parent_id) / "best.pt")  # type: ignore[arg-type]
            model = classifier_from_checkpoint(parent)
            pretrained_info = {"loaded": False, "init_from": parent_id}
            ctx.log(f"Initialisation depuis le modèle {parent_id}.")
        else:
            model, pretrained_info = build_model(hp.arch, hp.aggregation, hp.pretrained, hp.dropout)
            if pretrained_info.get("warning"):
                warnings.append(pretrained_info["warning"])
                ctx.log("AVERTISSEMENT : " + pretrained_info["warning"])
            elif pretrained_info.get("loaded"):
                ctx.log(f"Poids pré-entraînés chargés ({pretrained_info['n_tensors']} tenseurs, {pretrained_info.get('file')}).")
        model.to(device)

        _set_status(project, model_id, data_summary=data_summary, extra_updates={"pretrained": pretrained_info, "device": dev_info, "warnings": warnings})

        # -- données ------------------------------------------------------------
        pre = PreprocessConfig(input_size=hp.input_size)
        cache = ViewCache(hp.cache_mb * 1024 * 1024)
        train_ds = EggViewsDataset(train, hp.n_views, pre, augment=AugmentConfig.preset(hp.augment), seed=hp.seed, cache=cache, random_views=True)
        gen = torch.Generator()
        if hp.imbalance == "sampler":
            sampler = WeightedRandomSampler(class_balanced_weights(train), num_samples=len(train), replacement=True, generator=gen)
            train_loader = DataLoader(train_ds, batch_size=hp.batch_size, sampler=sampler, num_workers=hp.num_workers, drop_last=len(train) > hp.batch_size)
        else:
            train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, generator=gen, num_workers=hp.num_workers, drop_last=len(train) > hp.batch_size)
        pos_weight = None
        if hp.imbalance == "pos_weight":
            pos_weight = torch.tensor([min(10.0, max(0.1, (len(train) - n_pos) / n_pos))], device=device)
            ctx.log(f"Pondération des non viables : pos_weight = {pos_weight.item():.3f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # -- optimiseur et planification ----------------------------------------------
        head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        groups = [
            {"params": list(model.backbone.parameters()), "lr": hp.lr * hp.backbone_lr_mult, "name": "backbone"},
            {"params": head_params, "lr": hp.lr, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(groups, lr=hp.lr, weight_decay=hp.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * hp.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(total_steps, int(hp.warmup_epochs * steps_per_epoch), hp.min_lr_ratio))
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        start_epoch = 0
        history: list[dict[str, Any]] = []
        best_score, best_loss, best_epoch, bad_epochs = -math.inf, math.inf, None, 0
        if last_ckpt is not None:
            optimizer.load_state_dict(last_ckpt["optimizer"])
            for g in optimizer.param_groups:  # le taux de base suit les hyperparamètres (éventuellement modifiés)
                g["initial_lr"] = hp.lr * (hp.backbone_lr_mult if g.get("name") == "backbone" else 1.0)
            scheduler.load_state_dict(last_ckpt["scheduler"])
            scheduler.lr_lambdas = [_lr_lambda(total_steps, int(hp.warmup_epochs * steps_per_epoch), hp.min_lr_ratio)] * len(optimizer.param_groups)
            scheduler.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
            start_epoch = int(last_ckpt["epoch"])
            history = list(last_ckpt.get("history", []))
            best_score = float(last_ckpt.get("best_score", -math.inf))
            best_loss = float(last_ckpt.get("best_loss", math.inf))
            best_epoch = last_ckpt.get("best_epoch")
            bad_epochs = int(last_ckpt.get("bad_epochs", 0))
            if last_ckpt.get("torch_rng") is not None:
                torch.set_rng_state(last_ckpt["torch_rng"])
        monitor = hp.monitor if val else "train_loss"
        ctx.publish(model_id=model_id, history=history, epochs=hp.epochs, device=device.type)

        # -- boucle -------------------------------------------------------------
        stopped_early = False
        epoch = start_epoch
        for epoch in range(start_epoch, hp.epochs):
            t0 = time.perf_counter()
            frozen = epoch < hp.freeze_epochs
            model.train()
            model.set_backbone_trainable(not frozen)
            if frozen:
                model.backbone.eval()  # BatchNorm figée tant que le backbone est gelé
            train_ds.set_epoch(epoch)
            gen.manual_seed(hp.seed * 1000 + epoch)
            tot_loss, n_seen = 0.0, 0
            for b, (x, y, _) in enumerate(train_loader):
                ctx.check_cancelled()
                x = x.to(device, non_blocking=True)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        egg_logits, _ = model.forward_logits(x)
                        loss = criterion(egg_logits.float(), y)
                    scaler.scale(loss).backward()  # type: ignore[union-attr]
                    scaler.unscale_(optimizer)  # type: ignore[union-attr]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                    scaler.step(optimizer)  # type: ignore[union-attr]
                    scaler.update()  # type: ignore[union-attr]
                else:
                    egg_logits, _ = model.forward_logits(x)
                    loss = criterion(egg_logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                    optimizer.step()
                scheduler.step()
                tot_loss += float(loss.detach().cpu()) * len(y)
                n_seen += len(y)
                frac = (epoch - start_epoch + (b + 1) / steps_per_epoch) / max(1, hp.epochs - start_epoch)
                ctx.progress(0.92 * frac, f"Époque {epoch + 1}/{hp.epochs} — lot {b + 1}/{steps_per_epoch}")
            train_time = time.perf_counter() - t0
            entry: dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": round(tot_loss / max(1, n_seen), 6),
                "lr": optimizer.param_groups[1]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "backbone_frozen": frozen,
            }
            if val:
                vs, _ = predict_records(model, val, hp.n_views, hp.input_size, device, batch_size=max(1, hp.batch_size * 2), num_workers=hp.num_workers, check_cancelled=ctx.check_cancelled, cache=cache)
                yv = np.asarray([r.target for r in val])
                p = np.clip(vs, 1e-7, 1 - 1e-7)
                at = mt.confusion(yv, vs, 0.5)
                entry.update(
                    {
                        "val_loss": round(float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))), 6),
                        "val_auc": _round(mt.roc_auc(yv, vs)),
                        "val_ap": _round(mt.average_precision(yv, vs)),
                        "val_recall": _round(at["recall"]),
                        "val_false_reject_rate": _round(at["false_reject_rate"]),
                    }
                )
            entry["duration_s"] = round(time.perf_counter() - t0, 2)
            entry["images_per_s"] = round(n_seen * hp.n_views / max(train_time, 1e-6), 1)
            history.append(entry)

            score = _score(entry, monitor) if monitor != "train_loss" else -entry["train_loss"]
            loss_now = entry.get("val_loss", entry["train_loss"])
            improved = score > best_score + hp.min_delta or (abs(score - best_score) <= hp.min_delta and loss_now < best_loss - 1e-6)
            if improved:
                best_score, best_loss, best_epoch, bad_epochs = score, loss_now, epoch + 1, 0
                entry["best"] = True
                save_checkpoint(mdir / "best.pt", _weights_ckpt(model, hp, epoch + 1))
            else:
                bad_epochs += 1
            save_checkpoint(
                mdir / "last.pt",
                {
                    **_weights_ckpt(model, hp, epoch + 1),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_score": best_score,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad_epochs,
                    "torch_rng": torch.get_rng_state(),
                },
            )
            _set_status(project, model_id, history=history, extra_updates={"best_epoch": best_epoch})
            ctx.publish(history=history, best_epoch=best_epoch)
            ctx.log(
                f"Époque {epoch + 1}/{hp.epochs} : perte {entry['train_loss']:.4f}"
                + (f", val. perte {entry['val_loss']:.4f}, AUC {entry['val_auc']}" if val else "")
                + f" ({entry['duration_s']} s){' ★' if improved else ''}"
            )
            if bad_epochs >= hp.patience and epoch + 1 < hp.epochs:
                stopped_early = True
                ctx.log(f"Arrêt anticipé : pas d'amélioration de {monitor} depuis {hp.patience} époque(s).")
                break

        # -- évaluation finale ----------------------------------------------------------
        ctx.check_cancelled()
        ctx.progress(0.93, "Évaluation sur toutes les pistes du découpage")
        best = classifier_from_checkpoint(load_checkpoint(mdir / "best.pt")).to(device)
        scores, view_scores = predict_records(
            best, records, hp.n_views, hp.input_size, device, batch_size=max(1, hp.batch_size * 2), num_workers=hp.num_workers,
            check_cancelled=ctx.check_cancelled, progress=lambda f: ctx.progress(0.93 + 0.06 * f), cache=cache,
        )
        n_pred = write_predictions(project, model_id, records, scores, view_scores)
        duration = round(time.perf_counter() - t_start + float(prev_extra.get("duration_s", 0) if mode == "continue" else 0), 1)
        _set_status(
            project, model_id, status="trained",
            extra_updates={"best_epoch": best_epoch, "stopped_early": stopped_early, "duration_s": duration, "warnings": warnings, "interrupted": False},
        )
        report = refresh_report(project, model_id)
        report["training"] = {
            "epochs_done": len(history),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "monitor": monitor,
            "duration_s": duration,
            "device": dev_info,
            "warnings": warnings,
        }
        write_json(report_path(project, model_id), report)
        forget_cached(model_id)
        test_metrics = (report.get("subsets") or {}).get("test", {})
        ctx.log(f"Terminé : {n_pred} prédictions ; AUC test = {test_metrics.get('auc_roc')} ; meilleure époque {best_epoch}.")
        return {"model_id": model_id, "best_epoch": best_epoch, "epochs_done": len(history), "test_auc": test_metrics.get("auc_roc"), "n_predictions": n_pred}
    except JobCancelled:
        _set_status(project, model_id, status="cancelled", extra_updates={"warnings": warnings, "duration_s": round(time.perf_counter() - t_start, 1)})
        ctx.log("Entraînement annulé : reprise possible depuis la dernière époque terminée.")
        raise
    except Exception:
        _set_status(project, model_id, status="failed", extra_updates={"warnings": warnings})
        raise


def _round(x: Any) -> Any:
    return None if x is None else round(float(x), 6)


def _weights_ckpt(model: Any, hp: TrainParams, epoch: int) -> dict[str, Any]:
    return {
        "kind": "classifier",
        "arch": hp.arch,
        "aggregation": hp.aggregation,
        "dropout": hp.dropout,
        "n_views": hp.n_views,
        "input_size": hp.input_size,
        "epoch": epoch,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }


def model_report(project: Project, model_id: str) -> dict[str, Any] | None:
    return read_report(project, model_id)
