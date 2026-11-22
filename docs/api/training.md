# API — Entraînement et évaluation (`/api/training`)

Code : `desktop/src/eggvision/api/training.py`, logique dans `desktop/src/eggvision/ml/`.
Toutes les routes exigent un projet ouvert (sinon **409** `{"detail": "Aucun projet n'est ouvert."}`).
Erreurs : **400** paramètre invalide / règle métier, **404** ressource introuvable, **409** conflit (tâche en cours), **422** validation du corps (FastAPI). Corps d'erreur : `{"detail": "message en français"}`.

Conventions : cible **1 = non viable** ; décision **éjecter si `score >= seuil`** ; « faux rejets » = part des viables éjectés à tort. Sous-ensembles : `train`, `val`, `test`. Les tâches longues renvoient `job_id` (suivi via `/api/jobs/{id}`, journal via `/api/jobs/{id}/log`).

---

## Options

### `GET /api/training/options`
```jsonc
{
  "architectures": [
    {"id": "mobilenetv3_small_100", "label": "MobileNetV3-Small", "description": "…", "params_m": 1.5, "int8_friendly": false,
     "pretrained": {"available": true, "file": "mobilenetv3_small_100.safetensors", "size_bytes": 6141832, "source": "timm/…", "sha256": "…", "fetched_at": "…", "n_params": 1530002}},
    … mobilenetv3_large_100, efficientnet_lite0 (int8_friendly: true)
  ],
  "presets": [{"id": "rapide", "label": "Rapide", "description": "…", "hparams": {…hyperparamètres complets…}},
              {"id": "standard", …}],
  "defaults": {…TrainParams…},            // voir « Hyperparamètres »
  "anomaly_defaults": {…AnomalyParams…},
  "choices": {"aggregation": ["max","attention"], "imbalance": ["pos_weight","sampler","none"],
              "monitor": ["val_auc","val_ap","val_loss"], "augment": ["none","light","standard"],
              "device": ["auto","cuda","mps","cpu"]},
  "device": {"kind": "mps", "name": "Apple Silicon (arm64)"},
  "split": {"farms": [{"farm": "…", "n_eggs": 60}], "dates": [{"date": "2026-11-17", "n_eggs": 60}],
            "lots": [LotRow sans "subset"], "defaults": {…SplitParams…}}
}
```
`pretrained.available = false` → l'entraînement démarre en initialisation aléatoire (avertissement dans le journal et `Model.extra.warnings`).

### Hyperparamètres (`TrainParams`, défauts = préréglage « standard »)
| Clé | Défaut | Rôle |
|---|---|---|
| `arch` | `mobilenetv3_small_100` | backbone timm (3 choix) |
| `aggregation` | `max` | `max` ou `attention` (pooling appris) |
| `pretrained` | `true` | poids ImageNet locaux |
| `input_size` | 224 | taille d'entrée (px, 32–512) |
| `n_views` | 12 | vues par œuf (1–64) |
| `epochs` | 30 | époques max |
| `batch_size` | 8 | œufs par lot (× `n_views` images) |
| `lr` | 5e-4 | tête ; backbone : `lr × backbone_lr_mult` |
| `backbone_lr_mult` | 0.3 | |
| `weight_decay` | 1e-4 | AdamW |
| `warmup_epochs` | 1.0 | échauffement linéaire, puis cosinus jusqu'à `lr × min_lr_ratio` |
| `min_lr_ratio` | 0.02 | |
| `freeze_epochs` | 2 | backbone gelé (BN figée) pendant ces époques |
| `dropout` | 0.2 | |
| `imbalance` | `pos_weight` | `pos_weight` (n_viables / n_non_viables, borné [0,1 ; 10]), `sampler` (échantillonnage équilibré classes + défauts), `none` |
| `augment` | `standard` | `none`, `light`, `standard` |
| `monitor` | `val_auc` | métrique d'arrêt anticipé / sélection de `best.pt` (`val_auc`, `val_ap`, `val_loss`) |
| `patience` | 6 | époques sans amélioration avant arrêt |
| `min_delta` | 1e-4 | |
| `seed` | 42 | graine (données, augmentations, initialisation) |
| `device` | `auto` | CUDA → MPS → CPU |
| `amp` | `true` | précision mixte (CUDA seulement) |
| `num_workers` | 0 | processus de chargement (0 = compatible PyInstaller/Windows) |
| `cache_mb` | 512 | cache mémoire des vues décodées (0 = désactivé) |
| `reviewed_only` | `false` | labels relus uniquement |
| `grad_clip` | 5.0 | |

Préréglage **rapide** : `input_size 160, n_views 8, epochs 8, lr 1e-3, freeze_epochs 1, patience 3, augment light, warmup 0.5`.
Préréglage **standard** : valeurs par défaut ci-dessus.
Préréglage **int8** (« Déploiement INT8 ») : standard avec `arch: efficientnet_lite0`. MobileNetV3 se quantifie mal en INT8 (déployer en FP32) ; pour un déploiement INT8 choisir `efficientnet_lite0`.

---

## Découpages

`SplitParams` (corps) : `{"group_by": "session"|"lot", "ratios": [0.7,0.15,0.15], "holdout_farms": [], "holdout_dates": ["AAAA-MM-JJ"], "seed": 0, "reviewed_only": false, "exclude_lots": []}`.
Unité d'affectation : le lot, ou tous les lots d'une même `session_date` (`group_by: "session"`). Un groupe contenant un élevage / une journée réservé(e) va entièrement en test.

**LotRow** : `{"id","name","type","farm","session_date","n_eggs","viable","non_viable","defects":{"felure":6,…},"subset":"train"|"val"|"test"|null}`

**Summary** :
```jsonc
{"subsets": {"train": {"n_lots","n_eggs","viable","non_viable","defects":{…},"fraction"}, "val": {…}, "test": {…}},
 "total_eggs", "total_lots", "classes": ["coulure",…,"viable"],
 "leakage_check": {"unit": "lot"|"session", "lots_in_one_subset": true, "shared_sessions": ["2026-11-18"], "ok": true, "holdout_respected": true},
 "warnings": ["Aucun œuf « felure » dans le jeu de validation.", "Ratios non atteints — …", "Fuite possible : …"]}
```
(`shared_sessions` est informatif en regroupement par lot ; il rend `ok=false` en regroupement par session.)

**Split** : `{"id","name","version","created_at","strategy":{…SplitParams, "manual_changes"?, "derived_from"?},"summary","used_by":["model_id"],"locked":bool,"assignments":{lot_id: subset}}`

| Route | Corps | Réponse |
|---|---|---|
| `GET /splits?limit=50&offset=0` | — | `{"total", "items": [Split sans assignments]}` (récents d'abord) |
| `POST /splits/preview` | SplitParams | `{"assignments", "summary", "lots": [LotRow]}` (rien n'est enregistré) |
| `POST /splits` | SplitParams + `"name"` | Split. Nom existant → nouvelle version |
| `GET /splits/{id}` | — | Split + `"current_summary"` (labels actuels), `"lots": [LotRow]`, `"unassigned_lots": [LotRow]` (lots apparus depuis) |
| `PATCH /splits/{id}/assignments` | `{"changes": {lot_id: "train"\|"val"\|"test"\|null}, "move_group": true}` | Split + `"new_version": bool`. Découpage utilisé par un modèle → **nouvelle version** (nouvel `id`, `strategy.derived_from`). `move_group` (regroupement par session) déplace toute la session. `null` retire le lot |
| `DELETE /splits/{id}` | — | `{"deleted": id}` ; 400 si utilisé |

---

## Entraînement

| Route | Corps | Réponse |
|---|---|---|
| `POST /train` | `{"split_id", "preset": "rapide"\|"standard"\|"int8"\|null, "hparams": {…surcharges}, "name"?}` | `{"model_id", "job_id"}` — tâche `kind="training"`, groupe `gpu` |
| `POST /anomaly` | `{"split_id", "params": {…AnomalyParams}, "name"?}` | `{"model_id", "job_id"}` — tâche `kind="anomaly"` |
| `POST /models/{id}/resume` | `{"mode": "continue"\|"extend", "epochs"?, "hparams"?, "name"?}` | `{"model_id", "job_id"}` |
| `POST /models/{id}/cancel` | — | `{"job_id", "status": "cancelling"}` ; 409 si aucune tâche |

* `continue` : modèle `cancelled`/`failed` avec `last.pt` (`resumable: true`) → reprise exacte sur le **même** modèle ; `epochs` = nouveau total. Architecture, agrégation, taille et vues non modifiables.
* `extend` : crée un **nouveau** modèle (`parent_id`) initialisé avec `best.pt` du parent, même découpage, `epochs` époques (défaut : moitié du parent), sans gel ni échauffement.
* Annulation : prise en compte au lot suivant ; l'époque en cours est perdue, `last.pt` = dernière époque terminée.

Pendant la tâche, `job.result` = `{"model_id", "history": [Epoch…], "epochs", "device", "best_epoch"}` (direct).
**Epoch** : `{"epoch", "train_loss", "lr", "lr_backbone", "backbone_frozen", "val_loss", "val_auc", "val_ap", "val_recall", "val_false_reject_rate", "duration_s", "images_per_s", "best"?: true}` (métriques val au seuil 0,5).

`AnomalyParams` : `{"arch": "mobilenetv3_large_100", "input_size": 224, "n_views": 12, "patches_per_view": 48, "bank_size": 4000, "coreset": "greedy"|"random", "projection_dim": 64, "max_train_eggs": 400, "batch_size": 8, "seed": 42, "device": "auto", "reviewed_only": false, "cache_mb": 256}`.

---

## Modèles

**ModelSummary** :
```jsonc
{"id", "name", "kind": "classifier"|"anomaly", "arch", "status": "created"|"training"|"trained"|"failed"|"cancelled",
 "number": 3, "created_at", "split_id", "parent_id", "threshold", "n_views", "input_size",
 "metrics": MetricsSummary, "epochs_done", "best_epoch", "device": {"kind","name","used","amp"}, "duration_s",
 "warnings": [], "pretrained": {"loaded", "warning", "n_tensors", "missing", "file"},
 "job_id", "job_active", "resumable", "exported", "export_outdated"}
```
`export_outdated` : le seuil a changé depuis l'export (ré-exporter). Un modèle resté `training` sans tâche (arrêt brutal) est présenté `cancelled` (reprise possible).

**MetricsSummary** (`Model.metrics`) : `{"threshold", "generated_at", "train"|"val"|"test": {"n","n_pos","n_neg","auc_roc","average_precision","recall","false_reject_rate","precision","tp","fn","fp","tn","recall_by_defect"? (test)}, "test_set": {"n","fingerprint"}}`.

| Route | Réponse |
|---|---|
| `GET /models?limit&offset&kind&status` | `{"total", "items": [ModelSummary]}` (récents d'abord) |
| `GET /models/{id}` | ModelSummary + `"hparams"`, `"data_summary"` (`split_id/name/version`, `group_by`, `reviewed_only`, `counts` par sous-ensemble, `lots` par sous-ensemble, `n_eggs`, `labels_fingerprint`), `"history"`, `"export"`, `"extra"` |
| `PATCH /models/{id}` `{"name"}` | ModelSummary |
| `DELETE /models/{id}` | `{"deleted"}` — supprime dossier et prédictions ; 409 si tâche active |
| `GET /models/{id}/history` | `{"model_id","status","epochs","history":[Epoch],"job": Job\|null}` (direct pendant l'entraînement) |

### Rapport
`GET /models/{id}/report` → `models/<id>/report.json` (404 si pas encore évalué) :
```jsonc
{"format": 1, "model_id", "generated_at", "threshold", "decision_rule": "éjecter si score ≥ seuil",
 "subsets": {
   "train": {"n","n_pos","n_neg","auc_roc","average_precision","at_threshold": Confusion},
   "val"|"test": {"n","n_pos","n_neg","auc_roc","average_precision","at_threshold": Confusion,
                  "roc": {"fpr": [...], "tpr": [...], "thresholds": [null, …]},      // ≤ 200 points ; null = +∞
                  "pr":  {"recall": [...], "precision": [...], "thresholds": [...]},
                  "histogram": {"edges": [0,…,1], "viable": [...], "non_viable": [...]},
                  "recall_by_defect": {"felure": {"n","detected","recall"}, …, "non_precise"?}}},
 "test_set": {"n", "fingerprint"},
 "threshold_suggestions": [Suggestion pour rappel ≥ 95 %, ≥ 99 %, faux rejets ≤ 2 %, ≤ 5 % (calculées sur val)],
 "training": {"epochs_done","best_epoch","stopped_early","monitor","duration_s","device","warnings"},   // classifieur
 "anomaly": {"bank_size","bank_eggs","calibration","duration_s","warnings"}}                           // anomalies
```
**Confusion** : `{"threshold","tp","fn","fp","tn","n","n_pos","n_neg","recall","false_reject_rate","precision","specificity","accuracy"}` (null si indéfini).

`POST /models/{id}/report/refresh` → rapport recalculé depuis la table `Prediction` et les **labels actuels** (après relecture) ; met à jour `Model.metrics`.

### Seuil
| Route | Corps | Réponse |
|---|---|---|
| `POST /models/{id}/threshold/suggest` | `{"objective": "recall"\|"false_reject", "target": 0.95, "subset": "val"\|"test"}` | **Suggestion** `{"objective","target","computed_on","threshold","val": Confusion+{"objective_met"},"test": …}` |
| `GET /models/{id}/threshold/evaluate?threshold=0.4` | — | `{"threshold", "val": Confusion+{"recall_by_defect"}, "test": …}` (curseur de seuil, rien n'est modifié) |
| `PUT /models/{id}/threshold` | `{"threshold": 0.37}` | `{"threshold","metrics": MetricsSummary,"export_outdated","report_generated_at"}` (rapport régénéré) |

`recall ≥ x` → seuil le **plus haut** atteignant x ; `false_reject ≤ y` → seuil le **plus bas** respectant y (juste au-dessus d'un score viable). 400 si la classe nécessaire est absente du sous-ensemble.

### Explorateur d'erreurs
`GET /models/{id}/errors?kind=fn|fp&subset=test&threshold=<défaut: seuil du modèle>&limit=50&offset=0`
```jsonc
{"kind","subset","threshold","total","limit","offset",
 "items": [{"track_id","lot_id","lot_name","lane","subset","score","severity","verdict","defect","reviewed","n_views",
            "views": [{"index": 0, "score": 0.12}, …]}]}
```
`fn` = non viables sous le seuil, score croissant (plus grave d'abord) ; `fp` = viables au-dessus, score décroissant. `views[].index` = rang de la vue de la piste (image : `/api/media/crop/{track_id}/{index}`). Lien labellisation : `track_id`.

`GET /models/{id}/predictions?subset=&order=uncertain|score_desc|track&limit=100&offset=0` → `{"total","threshold","items":[{"track_id","subset","score","view_scores"}]}` (`uncertain` = plus proche du seuil d'abord).

### Cartes de chaleur
* `GET /models/{id}/gradcam/{track_id}` → `{"version","model_id","track_id","method": "grad_cam"|"distance_patch","views": [{"index","score","peak","url"}]}` — calcul au premier appel (≈ 0,1–1 s, CPU), cache `models/<id>/cam/<track_id>/`. Toutes les vues de la piste ; intensités normalisées par le maximum de l'œuf (`peak` ∈ [0,1]).
* `GET /models/{id}/gradcam/{track_id}/{view_index}` → `image/jpeg` (vue + carte superposée).

### Comparaison
`GET /compare?ids=a&ids=b[&ids=…]&objective=recall|false_reject&target=0.95&subset=val|test`
```jsonc
{"objective": {"objective","target","computed_on"}, "same_test_set": bool, "warnings": ["Les jeux de test diffèrent…"],
 "models": [{"id","name","kind","arch","split_id","number","threshold","status",
             "test_set": {"n","n_pos","fingerprint"}, "auc_roc","average_precision",
             "own_threshold": Confusion, "recall_by_defect": {…},
             "common_objective": {"threshold","computed_on","test": Confusion+{"objective_met"}} | {"error"}}
            | {…, "error": "Aucune prédiction sur le jeu de test."}]}
```
