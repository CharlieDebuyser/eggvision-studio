# API — Export vers Raspberry Pi (`/api/export`)

Code : `desktop/src/eggvision/api/export.py`, logique dans `desktop/src/eggvision/ml/{export,bench}.py`.
Projet ouvert requis (sinon 409). Erreurs : `{"detail": "…"}` — 400 (règle métier), 404, 422.

## Export

### `POST /api/export/models/{model_id}`
Corps facultatif :
```json
{"variants": ["fp32", "int8"], "n_reference": 6, "n_calibration_eggs": 32, "n_parity_eggs": 16}
```
→ `{"model_id", "job_id"}` — tâche `kind="export"`, groupe `cpu`.
400 si : modèle d'anomalies (« L'export ONNX du modèle de référence par anomalies n'est pas pris en charge. »), modèle non entraîné, `fp32` absent des variantes, export déjà en cours.

Étapes (progression et journal de la tâche) :
1. ONNX FP32 (opset 17 ; entrée `views` `[B, V, 3, S, S]`, sorties `egg_scores` `[B]`, `view_scores` `[B, V]` ; B et V dynamiques) ;
2. **parité** PyTorch ↔ ONNX FP32 sur `n_parity_eggs` œufs réels (test, sinon val/train) pour V ∈ {n_views, 6, 12} : écart max ≤ **1e-4**, sinon **échec de l'export** ;
3. INT8 (`int8` demandé) : quantification statique ONNX Runtime QDQ (poids INT8 par canal, activations UINT8, MinMax), calibrée sur `n_calibration_eggs` œufs d'entraînement ; tête (après la mise en commun globale) en flottant. Écart à FP32 + métriques test FP32 vs INT8 ; **alerte** si rappel ou AUC baisse de plus de 2 points. Échec de quantification → paquet FP32 seul + avertissement ;
4. paquet (`inference_config.json`, modèles, `preprocess.py`, `infer.py`, `reference/`, `README.txt`) ;
5. `python infer.py --verify` en sous-processus dans le dossier du paquet (bloquant pour FP32 ; INT8 informatif) ;
6. zip `models/<id>/eggvision_<id>_v<version>.zip` (version = `<n° du modèle>.<n° d'export>`).

Résultat de la tâche : `{"model_id","zip","version","parity_max_abs_diff","variants","warnings"}`.

### `GET /api/export/models/{model_id}`
```jsonc
{"model_id", "exportable": true, "exported": true,
 "outdated": false,              // seuil du modèle modifié depuis l'export → ré-exporter
 "current_threshold": 0.42,
 "export": Export,               // {} si jamais exporté
 "active_jobs": [Job]}           // exports / bancs en cours pour ce modèle
```
**Export** (`Model.export`) :
```jsonc
{"package_dir": "models/<id>/package", "zip": "models/<id>/eggvision_<id>_v1.1.zip", "zip_size_bytes",
 "variants": ["fp32","int8"], "files": {"fp32": {"file","size_bytes"}, "int8": {…}},
 "parity": {
   "fp32": {"tolerance": 1e-4, "max_abs_diff", "ok", "n_eggs", "by_n_views": {"6": {"n_eggs","max_abs_diff","max_abs_diff_views"}, "12": {…}}},
   "int8": {"format":"QDQ","activations","weights","calibration","n_calibration_eggs","excluded_nodes":[…],"size_bytes",
            "vs_fp32": {"max_abs_diff","mean_abs_diff","max_abs_diff_views"},
            "test": {"n","fp32": {"auc_roc","recall","false_reject_rate","tp","fn","fp","tn"},"int8": {…},"max_abs_diff"},
            "alert"?: "INT8 dégrade les performances…"} | {"error"} | null,
   "package_verify": {"fp32": VerifyReport, "int8": VerifyReport|null}},
 "threshold", "version", "export_count", "opset": 17, "n_views", "input_size", "exported_at", "duration_s",
 "warnings": [], "bench": [BenchEntry]}
```
**VerifyReport** (= `EggClassifier.verify_reference`) : `{"n_eggs","max_abs_diff","max_abs_diff_views","tolerance","ok","scores"}`.

### `GET /api/export/models/{model_id}/download`
→ `application/zip` (nom de fichier du paquet). 404 si non exporté / fichier absent.

### Contenu du paquet
```
inference_config.json  format_version 1, model_id, model_name, version, exported_at, engine "onnxruntime",
                       model_files {fp32, int8?}, default_variant "fp32", preprocess {input_size, mean, std, crop_margin, version},
                       n_views, n_lanes, aggregation, arch, threshold, scene (copie de config.yaml), metrics {test, threshold, int8}, io
model_fp32.onnx  model_int8.onnx
preprocess.py  infer.py          copies exactes de eggvision_shared/preprocess.py et runtime.py
reference/expected.json          {"tolerance", "source": "pytorch", "n_views", "eggs": [{"dir","score","view_scores","track_id","target","defect"}]}
reference/egg_XX/NN.png          vues des œufs de référence (scores PyTorch calculés à partir de ces PNG)
README.txt                       usage sur le Pi
```

## Banc de mesure

### `POST /api/export/models/{model_id}/bench`
```json
{"target": "local" | "pi", "variant": "fp32" | "int8", "n_columns": 50, "threads": null, "agent_url": null}
```
→ `{"model_id","job_id"}` — tâche `kind="bench"` (`local` : groupe `gpu`, jamais en parallèle d'un entraînement ; `pi` : groupe `pi-bench`).
400 si : non exporté, variante absente du paquet, `target: "pi"` sans `agent_url` ni agent connecté (`state.agent_url`).

* `local` : `EggClassifier.load(paquet, variant, threads).benchmark(n_columns)` + `verify_reference` sur ce PC.
* `pi` : envoi du zip à l'agent (contrat ci-dessous). Agent injoignable / réponse ≠ 200 → tâche `failed` avec message explicite.

Contrat agent : `POST {agent_url}/api/bench`, multipart : fichier `package` (zip), champs `variant`, `n_columns`, `threads` (1–64 ; champ omis = défaut ONNX Runtime) → JSON = `EggClassifier.benchmark()` + `"parity"` (VerifyReport) + `"device"` (objet libre) + `"onnxruntime"` (version).

### `GET /api/export/models/{model_id}/bench?limit=50`
```jsonc
{"agent_url": "http://…" | null,
 "items": [   // plus récent d'abord, 50 max
   {"at","target","variant","n_columns","threads","agent_url","export_version",
    "result": {"n_columns","n_lanes","n_views","input_size","variant","engine","threads",
               "latency_ms": {"mean","median","p95","p99","min","max"},
               "preprocess_ms_mean","inference_ms_mean","columns_per_second",
               "parity": VerifyReport, "device": {…}, "onnxruntime": "1.x"}}]}
```
La latence est mesurée par colonne (n_lanes œufs × n_views vues), prétraitement compris.
