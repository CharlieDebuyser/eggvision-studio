# Architecture et conventions de développement

## Monorepo

```
shared/      eggvision_shared   format des lots, prétraitement, générateur synthétique, moteur d'inférence
desktop/     eggvision          backend FastAPI (src/eggvision), frontend React (frontend/), tests (tests/)
pi-agent/    eggvision_agent    service de capture pour Raspberry Pi 5
docs/        documentation
```

Espace de travail `uv` (Python 3.11) : `uv sync` à la racine, `uv run pytest`, `uv run eggvision-studio`.

## Code partagé (`shared/src/eggvision_shared`)

| Module | Rôle |
|---|---|
| `constants.py` | Défauts (`DEFECT_TYPES`), verdicts, types de lot, `parse_lot_type`, `prelabel_for_lot_type` |
| `formats.py` | `CameraSettings`, `LotManifest` (`lot.json`), `FrameMeta` (`metadata.jsonl`), `LotWriter` (écriture au fil de l'eau, thread-safe), `read_metadata`, `mark_interrupted_lots`, `lot_file_list` (manifeste de synchronisation avec SHA-256), `read_image` / `write_image` |
| `preprocess.py` | **Source unique du prétraitement** : `extract_crop` (image → vue 224×224), `preprocess_view(s)` (vue BGR → tenseur), `select_view_indices`, `sharpness`. NumPy + OpenCV uniquement. Copié tel quel dans le paquet de déploiement |
| `runtime.py` | Moteur d'inférence autonome : `EggClassifier.load(dossier_paquet)`, `predict_column`, `benchmark`, `verify_reference` ; interface `InferenceEngine` (ONNX Runtime ; Hailo/TFLite prévus). Copié dans le paquet sous le nom `infer.py` |
| `capture.py` | `CaptureRecorder` (file bornée, fils d'écriture JPEG, images perdues comptées, arrêt sur seuil disque), `SyntheticFrameSource`, `PacedLoop` — commun à l'agent Pi et aux sources locales du PC |
| `synth.py` | `SyntheticScene` : images de table à rouleaux déterministes avec vérité terrain |

### Contrat du modèle ONNX
Entrée `views` float32 `[B, V, 3, S, S]` ; sorties `egg_scores` `[B]` et `view_scores` `[B, V]`, probabilités de défaut dans [0, 1]. B et V dynamiques ; l'agrégation multi-vues est dans le graphe.

### Paquet de déploiement (dossier ou zip)
```
inference_config.json   format_version, model_id, model_name, version, engine, model_files{fp32,int8},
                        default_variant, preprocess{…}, n_views, n_lanes, aggregation, threshold,
                        scene{lanes, direction, mm_per_px…}, metrics{…}
model_fp32.onnx  model_int8.onnx
preprocess.py    infer.py            (copies de shared/preprocess.py et shared/runtime.py)
reference/expected.json + reference/egg_XX/NN.png   (scores PyTorch attendus, pour la vérification de parité)
```

### Calibration de scène (`config.yaml` → `scene`)
```yaml
scene:
  image_width: 1456
  image_height: 1088
  direction: down            # sens de défilement dans l'image : up | down | left | right
  mm_per_px: 0.27
  lanes:                     # un quadrilatère par couloir, dans l'ordre des couloirs
    - {index: 0, polygon: [[x, y], [x, y], [x, y], [x, y]]}
```

## Backend (`desktop/src/eggvision`)

| Fichier | Rôle |
|---|---|
| `project.py` | `Project` (dossier projet : chemins, `session()`, `load_config/update_config/get_config/set_config`, `register_lot`) et `ProjectManager` |
| `db/models.py` | Schéma SQLite : `Lot`, `Frame`, `Track`, `Detection`, `View`, `Label`, `LabelHistory`, `DatasetSplit`, `Model`, `Prediction`, `Simulation`, `CameraPreset`, `Job` |
| `jobs.py` | `JobManager.submit(kind, title, fn, params, project, group)` ; `JobContext` : `progress`, `log`, `publish`, `check_cancelled` |
| `state.py` | Singletons `projects`, `jobs` ; dépendance `get_project` ; `agent_url` ; `diagnostic_providers` |
| `prelabel.py` | `apply_prelabels(session, lot_id)` : pré-label = type du lot |
| `demo.py` | `build_demo_project`, `add_synthetic_lot` (projet synthétique) |
| `testing.py` | `make_dummy_package` (paquet ONNX factice conforme), `wait_job` |
| `device.py` | `detect_device()`, `torch_device()` : CUDA → MPS → CPU |
| `api/<module>.py` | Un routeur FastAPI par module, monté par `app.py` (crochet facultatif `setup(app)`) |

### Règles
- **Chemins** stockés en base : relatifs au dossier du projet (`project.relpath` / `project.abspath`). Un projet doit rester copiable.
- **Tâches longues** : toujours via `state.jobs.submit(...)`, avec `ctx.progress`, `ctx.check_cancelled()` régulier et `ctx.log`. Les tâches lourdes en calcul utilisent `group="gpu"` (entraînement, évaluation) ou `group="cpu"`.
- **Sessions SQL** : `with project.session() as s:` (commit à la sortie). Ne pas garder de session ouverte pendant un long calcul ; écrire par paquets.
- **Erreurs API** : `HTTPException(status, "message en français")`.
- **Volumétrie** : toute liste est paginée (`limit`/`offset`) ; aucune requête ne charge 100 000 lignes d'un coup.
- **Hors-ligne** : aucun accès Internet à l'exécution.
- **Label d'une piste** : table `Label` (une ligne par piste). Un œuf est utilisable pour l'apprentissage ou la simulation si son verdict est `viable` ou `non_viable`.
- **Découpage** : `DatasetSplit.assignments = {lot_id: "train" | "val" | "test"}` — toujours par lot.
- **Modèle exporté** : `Model.export = {"package_dir": "models/<id>/package", "zip": "models/<id>/….zip", "variants": [...], "parity": {...}, "exported_at": "…"}`.

## Frontend (`desktop/frontend`)
Vite + React + TypeScript, HashRouter, TanStack Query. Voir `desktop/frontend/src/pages/README.md`.
Le build (`npm run build`) écrit dans `desktop/src/eggvision/web/`, servi par le backend.

## Tests
`uv run pytest` (fixtures dans `desktop/tests/conftest.py` : `client`, `project`, `frames_project`). Tout tourne sans matériel grâce au générateur synthétique.
